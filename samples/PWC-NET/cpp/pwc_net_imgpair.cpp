// pwc_net_imgpair
// Reads two images, runs PWC-Net NBG inference,
// writes the result as a .flo binary file to stdout.
// Usage: pwc_net_imgpair --model MODEL.nb img0.png img1.png > flow.flo

#include "pwc_net_imgpair.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::string model;
    std::string img0;
    std::string img1;
    std::string out_path;  // empty = stdout
};

static bool parse_opts(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) o.model = argv[++i];
        else if (a == "-o" && i + 1 < argc) o.out_path = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " --model MODEL.nb img0.png img1.png [-o output.flo]\n";
            return false;
        } else if (o.img0.empty()) o.img0 = a;
        else if (o.img1.empty()) o.img1 = a;
    }
    if (o.model.empty() || o.img0.empty() || o.img1.empty()) {
        std::cerr << "--model, img0, img1 are required\n"; return false;
    }
    return true;
}

static bool read_image(const std::string& path, cv::Mat& img, double& decode_ms) {
    const auto begin = Clock::now();
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    if (size <= 0) return false;
    file.seekg(0, std::ios::beg);
    std::vector<uchar> buf(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buf.data()), size);
    if (!file) return false;
    // Match the board examples' decoder mode.  Keeping the byte-buffer path
    // avoids the cv::imread(std::string, ...) ABI mismatch seen on the board.
    img = cv::imdecode(buf, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
    decode_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - begin).count();
    return !img.empty();
}

static bool write_flo(const std::string& path, const cv::Mat& flow) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const char* tag = "PIEH";
    int32_t w = flow.cols, h = flow.rows;
    out.write(tag, 4);
    out.write(reinterpret_cast<const char*>(&w), sizeof(w));
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(flow.data),
              static_cast<std::streamsize>(flow.total() * sizeof(cv::Vec2f)));
    return static_cast<bool>(out);
}

}  // anonymous namespace

int main(int argc, char** argv) {
    const auto overall_begin = Clock::now();
    Options opts;
    if (!parse_opts(argc, argv, opts)) return 2;

    const auto init_begin = Clock::now();
    ta_runtime_init();
    pwc_net::PwcGraph net;
    if (!net.init(opts.model)) return 1;
    const double init_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - init_begin).count();

    cv::Mat img0, img1;
    double img0_ms = 0.0, img1_ms = 0.0;
    if (!read_image(opts.img0, img0, img0_ms)) {
        std::cerr << "Cannot read image: " << opts.img0 << "\n"; return 1;
    }
    if (!read_image(opts.img1, img1, img1_ms)) {
        std::cerr << "Cannot read image: " << opts.img1 << "\n"; return 1;
    }

    const auto run_begin = Clock::now();
    cv::Mat flow; double nbg_ms = 0.0;
    if (!net.run(img0, img1, flow, nbg_ms)) return 1;
    const double run_total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - run_begin).count();

    const auto write_begin = Clock::now();
    bool write_ok = false;
    if (!opts.out_path.empty()) {
        write_ok = write_flo(opts.out_path, flow);
    } else {
        const char* tag = "PIEH";
        int32_t w = flow.cols, h = flow.rows;
        std::cout.write(tag, 4);
        std::cout.write(reinterpret_cast<const char*>(&w), sizeof(w));
        std::cout.write(reinterpret_cast<const char*>(&h), sizeof(h));
        std::cout.write(reinterpret_cast<const char*>(flow.data),
                        static_cast<std::streamsize>(flow.total() * sizeof(cv::Vec2f)));
        write_ok = static_cast<bool>(std::cout);
    }
    const double write_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - write_begin).count();
    if (!write_ok) {
        std::cerr << "Cannot write flow output\n";
        return 1;
    }

    std::cerr << std::fixed << std::setprecision(3)
              << "pwc_init_ms=" << init_ms << "\n"
              << "pwc_img0_decode_ms=" << img0_ms << "\n"
              << "pwc_img1_decode_ms=" << img1_ms << "\n"
              << "pwc_run_total_ms=" << run_total_ms << "\n"
              << "pwc_nbg_ms=" << nbg_ms << "\n"
              << "pwc_write_ms=" << write_ms << "\n"
              << "pwc_total_ms=" << std::chrono::duration<double, std::milli>(
                     Clock::now() - overall_begin).count()
              << " flow_size=" << flow.cols << "x" << flow.rows << "\n";
    return 0;
}
