// pwc_net_imgpair.h
// Header-only PwcGraph implementation.
// Dimensions are read from NBG at runtime via ta_runtime_query -- no hardcoding needed.

#ifndef PWC_NET_IMGPAIR_H
#define PWC_NET_IMGPAIR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include "ta-runtime-api.h"

namespace pwc_net {

// ── half-float helpers ────────────────────────────────────────────────────────
namespace detail {
static inline uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t fraction = bits & 0x007fffffu;
    if (exponent == 0xffu) return static_cast<uint16_t>(sign | (fraction ? 0x7e00u : 0x7c00u));
    int half_exp = static_cast<int>(exponent) - 127 + 15;
    if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<uint16_t>(sign);
        fraction |= 0x00800000u;
        const int shift = 14 - half_exp;
        uint32_t half_frac = fraction >> shift;
        const uint32_t rem = fraction & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half_frac & 1u)))
            ++half_frac;
        return static_cast<uint16_t>(sign | half_frac);
    }
    uint32_t half_frac = fraction >> 13;
    const uint32_t rem = fraction & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_frac & 1u))) {
        ++half_frac;
        if (half_frac == 0x400u) { half_frac = 0; ++half_exp; }
        if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10) | half_frac);
}

static inline float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    const uint32_t fraction = h & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) bits = sign;
        else {
            uint32_t f = fraction; int e = -1;
            do { f <<= 1; --e; } while ((f & 0x0400u) == 0);
            f &= 0x03ffu;
            bits = sign | static_cast<uint32_t>(e + 127) << 23 | f << 13;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | fraction << 13;
    } else {
        bits = sign | (exponent + 112u) << 23 | fraction << 13;
    }
    float value; std::memcpy(&value, &bits, sizeof(value)); return value;
}

static inline size_t tensor_elements(const taconn_inout_attr_t& attr) {
    size_t count = 1;
    for (unsigned i = 0; i < attr.dim_count; ++i) count *= attr.dim_size[i];
    return count;
}

static inline size_t tensor_bytes(const taconn_inout_attr_t& attr) {
    size_t el = 1;
    switch (attr.data_format) {
    case TACONN_DATA_FORMAT_FP16: case TACONN_DATA_FORMAT_BFP16:
    case TACONN_DATA_FORMAT_INT16:  case TACONN_DATA_FORMAT_UINT16: el = 2; break;
    case TACONN_DATA_FORMAT_FP32: case TACONN_DATA_FORMAT_INT32:
    case TACONN_DATA_FORMAT_UINT32: el = 4; break;
    case TACONN_DATA_FORMAT_FP64: el = 8; break;
    default: break;
    }
    return tensor_elements(attr) * el;
}
}  // namespace detail

// ── PwcGraph ─────────────────────────────────────────────────────────────────
// Reads all shape/size info from NBG at init() time -- no compile-time constants needed.
class PwcGraph {
public:
    ~PwcGraph() { close(); }

    // Return the runtime network dimensions (valid after init() succeeds).
    int net_w()  const { return net_w_; }
    int net_h()  const { return net_h_; }
    int flow_w() const { return flow_w_; }
    int flow_h() const { return flow_h_; }

    bool init(const std::string& model_path) {
        close();

        taco_status_t st = ta_runtime_load_model_from_file(&ctx_, model_path.c_str(), 0);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_load_model_from_file", st);

        taconn_input_output_num_t io{};
        st = ta_runtime_query(&ctx_, TACONN_QUERY_IN_OUT_NUM, &io);
        if (st != NNRT_SUCCESS || io.input_num != 1 || io.output_num != 1)
            return fail("ta_runtime_query(IN_OUT_NUM)", st);

        // Query input attr
        input_attr_.index = 0;
        st = ta_runtime_query(&ctx_, TACONN_QUERY_INPUT_ATTR, &input_attr_);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_query(INPUT_ATTR)", st);

        // Query output attr
        output_attr_.index = 0;
        st = ta_runtime_query(&ctx_, TACONN_QUERY_OUTPUT_ATTR, &output_attr_);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_query(OUTPUT_ATTR)", st);

        if (input_attr_.dim_count < 3 || output_attr_.dim_count < 3)
            return fail("unexpected dim_count (< 3)", NNRT_ILLEGAL_ARGUMENTS);

        // NBG metadata declares NCHW: input=[1,6,512,960], output=[1,2,128,240]
        // But ta_runtime_query may return dimensions in a different order.
        // We verified empirically that:
        //   runtime dims[0] = W (image width = 960)
        //   runtime dims[1] = H (image height = 512)
        //   runtime dims[2] = C (channels = 6 for input, 2 for output)
        // So we use NCHW positional mapping:
        //   net_w = dim[0], net_h = dim[1], channels = dim[2]
        //   flow_w = dim[0], flow_h = dim[1]
        channels_ = static_cast<int>(input_attr_.dim_size[2]);   // C = 6
        net_h_    = static_cast<int>(input_attr_.dim_size[1]);   // H = 512
        net_w_    = static_cast<int>(input_attr_.dim_size[0]);   // W = 960
        flow_h_   = static_cast<int>(output_attr_.dim_size[1]);  // H = 128
        flow_w_   = static_cast<int>(output_attr_.dim_size[0]);  // W = 240

        input_bytes_ = detail::tensor_bytes(input_attr_);
        output_bytes_ = detail::tensor_bytes(output_attr_);

        if (input_attr_.data_format != TACONN_DATA_FORMAT_FP16 ||
            output_attr_.data_format != TACONN_DATA_FORMAT_FP16)
            return fail("NBG requires FP16 IO", NNRT_NOT_SUPPORT);

        std::cerr << "[PwcGraph] NBG loaded: input [" << channels_ << "," << net_h_
                  << "," << net_w_ << "] output [" << flow_h_ << "," << flow_w_ << "]\n";

        if (posix_memalign(&input_memory_, 256, input_bytes_) != 0)
            return fail("posix_memalign(input)", NNRT_NOMEM);
        std::memset(input_memory_, 0, input_bytes_);
        input_.index = 0; input_.data = input_memory_;
        input_.size = static_cast<uint32_t>(input_bytes_);

        st = ta_runtime_create_buffer(&ctx_, static_cast<uint32_t>(output_bytes_), &output_);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_create_buffer(output)", st);
        output_created_ = true; ready_ = true;
        return true;
    }

    // Run inference on two BGR images (net_w x net_h resized).
    // Output: flow_w x flow_h CV_32FC2 flow (scaled by 20x to match training target).
    bool run(const cv::Mat& bgr0, const cv::Mat& bgr1,
             cv::Mat& flow, double& run_ms) {
        if (!ready_) return false;

        // Build FP16 input tensor [channels, net_h, net_w]
        std::vector<float> inp(static_cast<size_t>(channels_) * net_h_ * net_w_);
        fill_frame(bgr0, inp.data(), 0);
        fill_frame(bgr1, inp.data(), 3);  // offset=3 → second image's channels
        uint16_t* fp16_in = static_cast<uint16_t*>(input_memory_);
        for (size_t i = 0; i < inp.size(); ++i)
            fp16_in[i] = detail::float_to_half(inp[i]);

        taco_status_t st = ta_runtime_set_input_cva(&ctx_, 1, &input_);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_set_input_cva", st);
        st = ta_runtime_set_output(&ctx_, 1, &output_);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_set_output", st);

        auto t0 = std::chrono::steady_clock::now();
        st = ta_runtime_run_network(&ctx_);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_run_network", st);
        st = ta_runtime_invalidate_buffer(&ctx_, &output_);
        if (st != NNRT_SUCCESS) return fail("ta_runtime_invalidate_buffer", st);
        run_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        // Decode FP16 output [2, flow_h, flow_w] → CV_32FC2
        flow.create(flow_h_, flow_w_, CV_32FC2);
        const size_t plane = static_cast<size_t>(flow_h_) * flow_w_;
        const uint16_t* raw = static_cast<const uint16_t*>(output_.data);
        for (int y = 0; y < flow_h_; ++y) {
            cv::Vec2f* row = flow.ptr<cv::Vec2f>(y);
            for (int x = 0; x < flow_w_; ++x) {
                row[x][0] = detail::half_to_float(raw[static_cast<size_t>(y) * flow_w_ + x]) * 20.0f;
                row[x][1] = detail::half_to_float(raw[plane + static_cast<size_t>(y) * flow_w_ + x]) * 20.0f;
            }
        }
        return true;
    }

private:
    void fill_frame(const cv::Mat& bgr, float* dst, int offset) const {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(net_w_, net_h_), 0.0, 0.0, cv::INTER_LINEAR);
        const size_t plane = static_cast<size_t>(net_h_) * net_w_;
        for (int y = 0; y < net_h_; ++y) {
            const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < net_w_; ++x)
                for (int c = 0; c < 3; ++c)
                    dst[static_cast<size_t>(offset + c) * plane + static_cast<size_t>(y) * net_w_ + x] =
                        static_cast<float>(row[x][c]) / 255.0f;
        }
    }

    bool fail(const char* where, taco_status_t st = NNRT_UNKNOWN) {
        std::cerr << "[PwcGraph] " << where << " failed (0x" << std::hex << st << std::dec << ")\n";
        close(); return false;
    }

    void close() {
        ready_ = false;
        if (output_created_) {
            ta_runtime_destroy_buffer(&ctx_, &output_);
            output_ = taconn_buffer_t{}; output_created_ = false;
        }
        if (input_memory_) { free(input_memory_); input_memory_ = nullptr; }
        if (ctx_) { ta_runtime_destroy_context(&ctx_); ctx_ = 0; }
    }

    ta_runtime_context ctx_ = 0;
    taconn_inout_attr_t input_attr_{}, output_attr_{};
    taconn_input_t input_{};
    taconn_buffer_t output_{};
    void* input_memory_ = nullptr;
    size_t input_bytes_ = 0, output_bytes_ = 0;
    bool output_created_ = false, ready_ = false;

    // Runtime dimensions -- read from NBG, not hardcoded
    int net_h_ = 0, net_w_ = 0;
    int flow_h_ = 0, flow_w_ = 0;
    int channels_ = 0;  // should be 6 (img0+img1 RGB)
};

}  // namespace pwc_net

#endif  // PWC_NET_IMGPAIR_H
