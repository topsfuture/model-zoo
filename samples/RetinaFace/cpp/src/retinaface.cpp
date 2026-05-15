#include "retinaface.hpp"
#include "timer.hpp"
#include <iostream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ta-runtime-api.h>
#include <riscv_vector.h>

// COCO颜色，用于画框可视化
static const std::vector<cv::Scalar> COCO_COLORS = {
    {128, 56, 0, 255}, {128, 226, 255, 0}, {128, 0, 94, 255}, {128, 0, 37, 255}, {128, 0, 255, 94}, 
    {128, 255, 226, 0}, {128, 0, 18, 255}, {128, 255, 151, 0}, {128, 170, 0, 255}, {128, 0, 255, 56}
};


inline void dequantize_int8_to_fp32_rvv(const int8_t* src, float* dst, size_t n, int32_t zp, float scale) {
    size_t vl;
    for (; n > 0; n -= vl, src += vl, dst += vl) {
        vl = __riscv_vsetvl_e8m1(n);
        vint8m1_t v_src = __riscv_vle8_v_i8m1(src, vl);
        vint16m2_t v_src_16 = __riscv_vsext_vf2_i16m2(v_src, vl);
        vint32m4_t v_src_32 = __riscv_vsext_vf2_i32m4(v_src_16, vl);
        vint32m4_t v_sub = __riscv_vsub_vx_i32m4(v_src_32, zp, vl);
        vfloat32m4_t v_f32 = __riscv_vfcvt_f_x_v_f32m4(v_sub, vl);
        vfloat32m4_t v_dst = __riscv_vfmul_vf_f32m4(v_f32, scale, vl);
        __riscv_vse32_v_f32m4(dst, v_dst, vl);
    }
}

inline void dequantize_uint8_to_fp32_rvv(const uint8_t* src, float* dst, size_t n, int32_t zp, float scale) {
    size_t vl;
    for (; n > 0; n -= vl, src += vl, dst += vl) {
        vl = __riscv_vsetvl_e8m1(n);
        vuint8m1_t v_src = __riscv_vle8_v_u8m1(src, vl);
        vuint16m2_t v_src_16 = __riscv_vzext_vf2_u16m2(v_src, vl);
        vuint32m4_t v_src_32 = __riscv_vzext_vf2_u32m4(v_src_16, vl);
        vint32m4_t v_src_32_i = __riscv_vreinterpret_v_u32m4_i32m4(v_src_32);
        vint32m4_t v_sub = __riscv_vsub_vx_i32m4(v_src_32_i, zp, vl);
        vfloat32m4_t v_f32 = __riscv_vfcvt_f_x_v_f32m4(v_sub, vl);
        vfloat32m4_t v_dst = __riscv_vfmul_vf_f32m4(v_f32, scale, vl);
        __riscv_vse32_v_f32m4(dst, v_dst, vl);
    }
}

// inline void dequantize_fp16_to_fp32_rvv(const uint16_t* src, float* dst, size_t n) {
//     size_t vl;
//     for (; n > 0; n -= vl, src += vl, dst += vl) {
//         vl = __riscv_vsetvl_e16m2(n);
//         vuint16m2_t v_src_u16 = __riscv_vle16_v_u16m2(src, vl);
//         vfloat16m2_t v_src_f16 = __riscv_vreinterpret_v_u16m2_f16m2(v_src_u16);
//         vfloat32m4_t v_dst = __riscv_vfwcvt_f_f_v_f32m4(v_src_f16, vl);
//         __riscv_vse32_v_f32m4(dst, v_dst, vl);
//     }
// }
inline void dequantize_fp16_to_fp32_rvv(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint16_t value = src[i];
        uint32_t sign = (value >> 15) & 0x1;
        uint32_t exponent = (value >> 10) & 0x1F;
        uint32_t mantissa = value & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                uint32_t f = (sign << 31);
                dst[i] = *reinterpret_cast<float*>(&f);
            } else {
                const uint32_t exp_offset = 103;
                uint32_t f = (sign << 31) | (exp_offset << 23) | (mantissa << 13);
                dst[i] = *reinterpret_cast<float*>(&f);
            }
        } 
        else if (exponent == 31) {
            uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
            dst[i] = *reinterpret_cast<float*>(&f);
        } 
        else {
            exponent += (127 - 15);
            uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
            dst[i] = *reinterpret_cast<float*>(&f);
        }
    }
}

std::vector<int> rvv_nms(const std::vector<facebox>& boxes, float nms_thresh) {
    int num_boxes = boxes.size();
    if (num_boxes == 0) return {};

    std::vector<int> order(num_boxes);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&boxes](int a, int b) {
        return boxes[a].score > boxes[b].score;
    });
    std::vector<float> sx1(num_boxes), sy1(num_boxes), sx2(num_boxes), sy2(num_boxes), sarea(num_boxes);
    std::vector<uint8_t> suppressed(num_boxes, 0);

    for (int i = 0; i < num_boxes; i++) {
        int idx = order[i];
        sx1[i] = boxes[idx].x1; sy1[i] = boxes[idx].y1;
        sx2[i] = boxes[idx].x2; sy2[i] = boxes[idx].y2;
        sarea[i] = (sx2[i] - sx1[i]) * (sy2[i] - sy1[i]);
    }

    std::vector<int> keep;
    for (int i = 0; i < num_boxes; i++) {
        if (suppressed[i]) continue;
        keep.push_back(order[i]);

        float cx1 = sx1[i], cy1 = sy1[i], cx2 = sx2[i], cy2 = sy2[i], carea = sarea[i];
        int j = i + 1;
        int n_left = num_boxes - j;

        size_t vl;
        for (; n_left > 0; n_left -= vl, j += vl) {
            vl = __riscv_vsetvl_e32m4(n_left);

            vfloat32m4_t vx1 = __riscv_vle32_v_f32m4(&sx1[j], vl);
            vfloat32m4_t vy1 = __riscv_vle32_v_f32m4(&sy1[j], vl);
            vfloat32m4_t vx2 = __riscv_vle32_v_f32m4(&sx2[j], vl);
            vfloat32m4_t vy2 = __riscv_vle32_v_f32m4(&sy2[j], vl);
            vfloat32m4_t varea = __riscv_vle32_v_f32m4(&sarea[j], vl);

            vfloat32m4_t vxx1 = __riscv_vfmax_vf_f32m4(vx1, cx1, vl);
            vfloat32m4_t vyy1 = __riscv_vfmax_vf_f32m4(vy1, cy1, vl);
            vfloat32m4_t vxx2 = __riscv_vfmin_vf_f32m4(vx2, cx2, vl);
            vfloat32m4_t vyy2 = __riscv_vfmin_vf_f32m4(vy2, cy2, vl);

            vfloat32m4_t vw = __riscv_vfmax_vf_f32m4(__riscv_vfsub_vv_f32m4(vxx2, vxx1, vl), 0.0f, vl);
            vfloat32m4_t vh = __riscv_vfmax_vf_f32m4(__riscv_vfsub_vv_f32m4(vyy2, vyy1, vl), 0.0f, vl);

            vfloat32m4_t v_inter = __riscv_vfmul_vv_f32m4(vw, vh, vl);
            vfloat32m4_t v_union = __riscv_vfsub_vv_f32m4(__riscv_vfadd_vf_f32m4(varea, carea, vl), v_inter, vl);
            vfloat32m4_t v_iou = __riscv_vfdiv_vv_f32m4(v_inter, v_union, vl);

            vbool8_t v_mask = __riscv_vmfgt_vf_f32m4_b8(v_iou, nms_thresh, vl);

            vuint8m1_t v_supp = __riscv_vle8_v_u8m1(&suppressed[j], vl);
            v_supp = __riscv_vmerge_vxm_u8m1(v_supp, 1, v_mask, vl);
            __riscv_vse8_v_u8m1(&suppressed[j], v_supp, vl);
        }
    }
    return keep;
}


RetinaFace::RetinaFace() : initialized_(false), input_height_(640), input_width_(640), ts_(nullptr) {
    ctx_.nnrt_context = 0;
    ctx_.input_tensors = nullptr;
    ctx_.output_buffer = nullptr;
    ctx_.input_num = 0;
    ctx_.output_num = 0;
}

RetinaFace::~RetinaFace() {
    if (initialized_) deinit();
}

void RetinaFace::enableProfile(TimeStamp *ts) {
    ts_ = ts;
}

bool RetinaFace::init(const std::string& model_path) {
    if (initialized_) return false;

    int status = ta_runtime_init();
    if (status != 0) return false;

    ctx_.nnrt_context = 0;
    status = ta_runtime_load_model_from_file(&ctx_.nnrt_context, model_path.c_str(), 0);
    if (status != 0) return false;

    taconn_input_output_num_t num = {0};
    ta_runtime_query(&ctx_.nnrt_context, TACONN_QUERY_IN_OUT_NUM, &num);
    ctx_.input_num = num.input_num;
    ctx_.output_num = num.output_num;

    for (int i = 0; i < ctx_.input_num; i++) {
        taconn_inout_attr_t attr = {0};
        attr.index = i;
        ta_runtime_query(&ctx_.nnrt_context, TACONN_QUERY_INPUT_ATTR, &attr);
        ctx_.ins_attr.push_back(attr);
    }
    
    if (ctx_.input_num > 0 && ctx_.ins_attr[0].dim_count >= 3) {
        input_height_ = ctx_.ins_attr[0].dim_size[1];
        input_width_ = ctx_.ins_attr[0].dim_size[0];
    }

    for (int i = 0; i < ctx_.output_num; i++) {
        taconn_inout_attr_t attr = {0};
        attr.index = i;
        ta_runtime_query(&ctx_.nnrt_context, TACONN_QUERY_OUTPUT_ATTR, &attr);
        ctx_.outs_attr.push_back(attr);
    }

    ctx_.input_tensors = (taconn_input_t*)malloc(sizeof(taconn_input_t) * ctx_.input_num);
    for (int i = 0; i < ctx_.input_num; i++) {
        taconn_data_format_t dtype = static_cast<taconn_data_format_t>(ctx_.ins_attr[i].data_format);
        size_t buf_size = calculate_buffer_size(dtype, get_element_num(ctx_.ins_attr[i]));

        ctx_.input_tensors[i].index = i;
        ctx_.input_tensors[i].size = buf_size;
        posix_memalign((void**)&ctx_.input_tensors[i].data, 256, buf_size);
        memset(ctx_.input_tensors[i].data, 0, buf_size);
    }
    ta_runtime_set_input_cva(&ctx_.nnrt_context, ctx_.input_num, ctx_.input_tensors);

    ctx_.output_buffer = (taconn_buffer_t*)malloc(sizeof(taconn_buffer_t) * ctx_.output_num);
    for (int i = 0; i < ctx_.output_num; i++) {
        taconn_data_format_t dtype = static_cast<taconn_data_format_t>(ctx_.outs_attr[i].data_format);
        size_t buf_size = calculate_buffer_size(dtype, get_element_num(ctx_.outs_attr[i]));
        ta_runtime_create_buffer(&ctx_.nnrt_context, buf_size, &ctx_.output_buffer[i]);
    }
    ta_runtime_set_output(&ctx_.nnrt_context, ctx_.output_num, ctx_.output_buffer);

    initialized_ = true;
    return true;
}

void RetinaFace::deinit() {
    if (!initialized_) return;
    for (int i = 0; i < ctx_.output_num; i++) ta_runtime_destroy_buffer(&ctx_.nnrt_context, &ctx_.output_buffer[i]);
    if (ctx_.input_tensors) {
        for (int i = 0; i < ctx_.input_num; i++) free(ctx_.input_tensors[i].data);
        free(ctx_.input_tensors);
    }
    if (ctx_.output_buffer) free(ctx_.output_buffer);
    ta_runtime_destroy_context(&ctx_.nnrt_context);
    ta_runtime_deinit();
    initialized_ = false;
}


bool RetinaFace::detect_and_save(const cv::Mat& image, const std::string& output_path,
                                 std::vector<facebox>& objects, float conf_thresh, float nms_thresh) {
    if (!initialized_) return false;

    if (ts_) ts_->start();
    PreprocessParams pre_params;
    cv::Mat processed_image;
    preprocess_image(image, processed_image, pre_params);
    if (ts_) ts_->time_accumulation("pre_time");

    if (ts_) ts_->start();
    if (ta_runtime_run_network(&ctx_.nnrt_context) != 0) return false;
    ta_runtime_invalidate_buffer(&ctx_.nnrt_context, ctx_.output_buffer);
    if (ts_) ts_->time_accumulation("infer_time");

    if (ts_) ts_->start();
    
    std::cout << std::endl;
    std::vector<float> loc_data = get_real_float_data(ctx_.output_buffer[0].data, ctx_.outs_attr[0]);
    std::vector<float> conf_data = get_real_float_data(ctx_.output_buffer[1].data, ctx_.outs_attr[1]);
    std::vector<float> landm_data = get_real_float_data(ctx_.output_buffer[2].data, ctx_.outs_attr[2]);
    
    // float max_score = -999, min_score = 999;
    // int num_anchors = conf_data.size() / 2;
    // for (int i = 0; i < num_anchors; i++) {
    //     float s = conf_data[i * 2 + 1];
    //     max_score = std::max(max_score, s);
    //     min_score = std::min(min_score, s);
    // }
    // std::cout << "conf_data size: " << conf_data.size() << std::endl;
    // std::cout << "num_anchors: " << num_anchors << std::endl;
    // std::cout << "conf score range: [" << min_score << ", " << max_score << "]" << std::endl;

    // // 2. 打印前10个原始 conf 值
    // std::cout << "First 10 conf values (raw): ";
    // for (int i = 0; i < 20 && i < conf_data.size(); i++) {
    //     std::cout << conf_data[i] << " ";
    // }

    objects = postprocess(loc_data, conf_data, landm_data, conf_thresh, nms_thresh, pre_params.ratio);
    if (ts_) ts_->time_accumulation("post_time");

    return true;
}


void RetinaFace::preprocess_image(const cv::Mat& src, cv::Mat& dst, PreprocessParams& params) {
    std::cout <<"preprocess_image"<<std::endl;
    // 计算缩放比例并 Resize
    params.ratio = float(input_height_) / float(std::max(src.cols, src.rows));
    cv::Mat resized_image;
    cv::resize(src, resized_image, cv::Size(), params.ratio, params.ratio, cv::INTER_LINEAR);
    
    // 补边 (Padding)，保持为 uint8 类型
    cv::copyMakeBorder(resized_image, dst, 0, input_height_ - resized_image.rows, 
                      0, input_width_ - resized_image.cols, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    // 直接将 BGR 打散为三个 uint8 通道，不进行中间 float 转换
    std::vector<cv::Mat> channels(3);
    cv::split(dst, channels);

    // 调用优化后的量化函数，内部处理减均值逻辑
    normalize_and_quantize(channels, ctx_.input_tensors[0].data, ctx_.ins_attr[0]);
}

std::vector<facebox> RetinaFace::postprocess(const std::vector<float>& loc_data, const std::vector<float>& conf_data_raw,
                                             const std::vector<float>& landm_data, float score_thresh,
                                             float nms_thresh, float resize_ratio) {
    // 对 conf_data 做 softmax (模型输出是 logits)
    std::vector<float> conf_data(conf_data_raw.size());
    int num_conf = conf_data_raw.size() / 2;
    for (int i = 0; i < num_conf; i++) {
        float c0 = conf_data_raw[i * 2 + 0];  // 背景类
        float c1 = conf_data_raw[i * 2 + 1];  // 人脸类
        float max_c = std::max(c0, c1);
        float exp0 = std::exp(c0 - max_c);
        float exp1 = std::exp(c1 - max_c);
        float sum = exp0 + exp1;
        conf_data[i * 2 + 0] = exp0 / sum;
        conf_data[i * 2 + 1] = exp1 / sum;
    }

    std::vector<int> steps = {8, 16, 32};
    std::vector<std::vector<int>> min_sizes = {{16, 32}, {64, 128}, {256, 512}};
    const float variance_loc[2] = {0.1f, 0.2f};
    
    std::vector<std::vector<float>> priors;
    for (size_t i = 0; i < steps.size(); i++) {
        int feature_h = std::ceil(float(input_height_) / steps[i]);
        int feature_w = std::ceil(float(input_width_) / steps[i]);
        for (int j = 0; j < feature_h; j++) {
            for (int k = 0; k < feature_w; k++) {
                for (int min_size : min_sizes[i]) {
                    float s_kx = (float)min_size / input_width_;
                    float s_ky = (float)min_size / input_height_;
                    float dense_cx = (k + 0.5f) * steps[i] / input_width_;
                    float dense_cy = (j + 0.5f) * steps[i] / input_height_;
                    priors.push_back({dense_cx, dense_cy, s_kx, s_ky});
                }
            }
        }
    }

    std::vector<facebox> candidate_boxes;
    int num_anchors = priors.size();

    for (int i = 0; i < num_anchors; i++) {
        float score = conf_data[i * 2 + 1];
        if (score > score_thresh) {
            float P_cx = priors[i][0], P_cy = priors[i][1], P_w = priors[i][2], P_h = priors[i][3];

            facebox box;
            box.score = score;
            box.class_id = 1;

            float B_cx = P_cx + loc_data[i * 4 + 0] * variance_loc[0] * P_w;
            float B_cy = P_cy + loc_data[i * 4 + 1] * variance_loc[0] * P_h;
            float B_w  = P_w * std::exp(loc_data[i * 4 + 2] * variance_loc[1]);
            float B_h  = P_h * std::exp(loc_data[i * 4 + 3] * variance_loc[1]);

            box.x1 = (B_cx - B_w / 2.0f) * input_width_ / resize_ratio;
            box.y1 = (B_cy - B_h / 2.0f) * input_height_ / resize_ratio;
            box.x2 = (B_cx + B_w / 2.0f) * input_width_ / resize_ratio;
            box.y2 = (B_cy + B_h / 2.0f) * input_height_ / resize_ratio;

            for (int j = 0; j < 5; j++) {
                box.landmarks[j * 2] = (P_cx + landm_data[i * 10 + j * 2] * variance_loc[0] * P_w) * input_width_ / resize_ratio;
                box.landmarks[j * 2 + 1] = (P_cy + landm_data[i * 10 + j * 2 + 1] * variance_loc[0] * P_h) * input_height_ / resize_ratio;
            }
            candidate_boxes.push_back(box);
        }
    }

    std::vector<int> keep_indices = rvv_nms(candidate_boxes, nms_thresh);
    std::vector<facebox> final_results;
    for (int idx : keep_indices) final_results.push_back(candidate_boxes[idx]);
    
    return final_results;
}


std::vector<float> RetinaFace::get_real_float_data(void* raw_data, const taconn_inout_attr_t& attr) {
    size_t num_elements = get_element_num(attr);
    std::vector<float> real_data(num_elements);
    
    float scale = 1.0f;
    int32_t zp = 0;
    
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        scale = attr.quant_data.affine.tf_scale;
        zp = attr.quant_data.affine.tf_zero_point;
    } 

    switch (attr.data_format) {
        case taconn_data_format_e::TACONN_DATA_FORMAT_INT8:
            dequantize_int8_to_fp32_rvv(static_cast<const int8_t*>(raw_data), real_data.data(), num_elements, zp, scale);
            break;
        case taconn_data_format_e::TACONN_DATA_FORMAT_UINT8:
            dequantize_uint8_to_fp32_rvv(static_cast<const uint8_t*>(raw_data), real_data.data(), num_elements, zp, scale);
            break;
        case taconn_data_format_e::TACONN_DATA_FORMAT_FP16:
            dequantize_fp16_to_fp32_rvv(static_cast<const uint16_t*>(raw_data), real_data.data(), num_elements);
            break;
        case taconn_data_format_e::TACONN_DATA_FORMAT_FP32:
            memcpy(real_data.data(), raw_data, num_elements * sizeof(float));
            break;
        default:
            std::cerr << "Unsupported TACO Output format for RVV: " << attr.data_format << std::endl;
            break;
    }

    return real_data;
}

size_t RetinaFace::get_element_num(const taconn_inout_attr_t& input_attr) {
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i) size *= input_attr.dim_size[i];
    return size;
}

size_t RetinaFace::calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
    switch(format) {
        case TACONN_DATA_FORMAT_FP32: return sizeof(uint32_t) * element_count;
        case TACONN_DATA_FORMAT_FP16: return sizeof(uint16_t) * element_count;
        case TACONN_DATA_FORMAT_UINT8:
        case TACONN_DATA_FORMAT_INT8: return sizeof(uint8_t) * element_count;
        default: return element_count;
    }
}

void RetinaFace::normalize_and_quantize(const std::vector<cv::Mat>& channels, void* dst, const taconn_inout_attr_t& attr) {
    size_t channel_size = input_height_ * input_width_;
    const float means[3] = {104.0f, 117.0f, 123.0f};

    // --- 1. 量化模式 (INT8 / UINT8) ---
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_ptr = static_cast<int8_t*>(dst);
            for (int c = 0; c < 3; ++c) {
                const uint8_t* src_data = channels[c].data;
                int8_t* current_dst = dst_ptr + (c * channel_size);
                float sub_offset = (means[c] / scale) - static_cast<float>(zero_point);
                for (size_t i = 0; i < channel_size; ++i) {
                    int32_t quantized = std::round(static_cast<float>(src_data[i]) / scale - sub_offset);
                    current_dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
                }
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
            for (int c = 0; c < 3; ++c) {
                const uint8_t* src_data = channels[c].data;
                uint8_t* current_dst = dst_ptr + (c * channel_size);
                float sub_offset = (means[c] / scale) - static_cast<float>(zero_point);
                for (size_t i = 0; i < channel_size; ++i) {
                    int32_t quantized = std::round(static_cast<float>(src_data[i]) / scale - sub_offset);
                    current_dst[i] = static_cast<uint8_t>(std::max(0, std::min(255, quantized)));
                }
            }
        }
    } 
    // --- 2. 浮点模式 (FP32) ---
    else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
        float* dst_ptr = static_cast<float*>(dst);
        for (int c = 0; c < 3; ++c) {
            const uint8_t* src_data = channels[c].data;
            float* current_dst = dst_ptr + (c * channel_size);
            for (size_t i = 0; i < channel_size; ++i) {
                current_dst[i] = static_cast<float>(src_data[i]) - means[c];
            }
        }
    }
    // --- 3. 半精度模式 (FP16) ---
    else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
        uint16_t* dst_ptr = static_cast<uint16_t*>(dst);
        for (int c = 0; c < 3; ++c) {
            const uint8_t* src_data = channels[c].data;
            uint16_t* current_dst = dst_ptr + (c * channel_size);
            for (size_t i = 0; i < channel_size; ++i) {
                // 计算 float 值
                float val = static_cast<float>(src_data[i]) - means[c];
                
                // Float32 to Float16 转换逻辑
                uint32_t f = *reinterpret_cast<uint32_t*>(&val);
                uint32_t sign = (f >> 16) & 0x8000;
                int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
                uint32_t mantissa = f & 0x7FFFFF;

                if (exponent <= 0) {
                    current_dst[i] = (uint16_t)sign;  
                } else if (exponent >= 31) {
                    current_dst[i] = (uint16_t)(sign | 0x7C00);  
                } else {
                    current_dst[i] = (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
                }
            }
        }
    }
}


void RetinaFace::draw_objects(const cv::Mat& bgr, const std::vector<facebox>& objects, const std::string& output_name) {
    cv::Mat image = bgr.clone();
    std::cout<<"objects.size(): "<<objects.size()<<std::endl;
    for (size_t i = 0; i < objects.size(); i++) {
        const facebox& obj = objects[i];
        cv::Scalar color = COCO_COLORS[i % COCO_COLORS.size()];
        cv::rectangle(image, cv::Point((int)obj.x1, (int)obj.y1), cv::Point((int)obj.x2, (int)obj.y2), color, 2);
        for(int j = 0; j < 5; j++) {
            cv::circle(image, cv::Point((int)obj.landmarks[j * 2], (int)obj.landmarks[j * 2 + 1]), 2, color, -1);
        }
        char text[256];
        sprintf(text, "%.1f%%", obj.score * 100.0f);
        cv::putText(image, text, cv::Point((int)obj.x1, (int)obj.y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    cv::imwrite(output_name, image);
}