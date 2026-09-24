// farneback_imgpair
// Reads two images, runs cv::calcOpticalFlowFarneback,
// writes the result as a .flo binary file to stdout.
// Usage: farneback_imgpair img0.png img1.png > flow.flo
// The .flo format matches Middlebury's standard: "PIEH" + w + h + dx/dy floats.

#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

static double elapsed_ms(const Clock::time_point& begin,
                         const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

// Use the same soft-decoder mode as the YOLOv5 board examples.  The image is
// decoded from memory instead of calling cv::imread(path, ...): some board
// images carry an OpenCV library with a different std::string ABI and fail
// during dynamic symbol lookup for cv::imread.  imdecode has no std::string
// OpenCV ABI boundary and is also the path used by pwc_net_imgpair.
static bool read_image(const std::string& path, cv::Mat& image,
                       double& decode_ms) {
    const auto begin = Clock::now();
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) return false;
    file.seekg(0, std::ios::beg);
    std::vector<uchar> bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!file) return false;
    image = cv::imdecode(bytes, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
    decode_ms = elapsed_ms(begin, Clock::now());
    return !image.empty();
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

static bool write_flo_stdout(const cv::Mat& flow) {
    const char* tag = "PIEH";
    int32_t w = flow.cols, h = flow.rows;
    std::cout.write(tag, 4);
    std::cout.write(reinterpret_cast<const char*>(&w), sizeof(w));
    std::cout.write(reinterpret_cast<const char*>(&h), sizeof(h));
    std::cout.write(reinterpret_cast<const char*>(flow.data),
                    static_cast<std::streamsize>(flow.total() * sizeof(cv::Vec2f)));
    return static_cast<bool>(std::cout);
}

static bool parse_opts(int argc, char** argv,
                       std::string& img0, std::string& img1, std::string& out_path) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " img0.png img1.png [-o output.flo]\n"
                      << "  If -o is omitted, writes to stdout in .flo binary format.\n";
            return false;
        } else if (img0.empty()) img0 = a;
        else if (img1.empty()) img1 = a;
    }
    if (img0.empty() || img1.empty()) {
        std::cerr << "Need two image arguments.\n"; return false;
    }
    return true;
}
}  // anonymous namespace

int main(int argc, char** argv) {
    const auto overall_begin = Clock::now();
    std::string img0_path, img1_path, out_path;
    if (!parse_opts(argc, argv, img0_path, img1_path, out_path)) return 2;

    cv::Mat img0;
    double img0_ms = 0.0, img1_ms = 0.0;
    if (!read_image(img0_path, img0, img0_ms)) {
        std::cerr << "Cannot read image: " << img0_path << "\n";
        return 1;
    }
    cv::Mat img1;
    if (!read_image(img1_path, img1, img1_ms)) {
        std::cerr << "Cannot read images\n"; return 1;
    }

    const auto preprocess_begin = Clock::now();
    cv::Mat gray0, gray1;
    cv::cvtColor(img0, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    const double preprocess_ms = elapsed_ms(preprocess_begin, Clock::now());

    cv::Mat flow;
    const auto farneback_begin = Clock::now();
    cv::calcOpticalFlowFarneback(gray0, gray1, flow,
                                 0.5, 3, 15, 3, 5, 1.2, 0);
    const double farneback_ms = elapsed_ms(farneback_begin, Clock::now());

    const auto write_begin = Clock::now();
    bool write_ok = false;
    if (!out_path.empty()) {
        write_ok = write_flo(out_path, flow);
    } else {
        write_ok = write_flo_stdout(flow);
    }
    const double write_ms = elapsed_ms(write_begin, Clock::now());
    if (!write_ok) {
        std::cerr << "Cannot write flow output\n";
        return 1;
    }

    std::cerr << std::fixed << std::setprecision(3)
              << "farneback_img0_decode_ms=" << img0_ms << "\n"
              << "farneback_img1_decode_ms=" << img1_ms << "\n"
              << "farneback_preprocess_ms=" << preprocess_ms << "\n"
              << "farneback_calc_ms=" << farneback_ms << "\n"
              << "farneback_write_ms=" << write_ms << "\n"
              << "farneback_total_ms=" << elapsed_ms(overall_begin, Clock::now())
              << " flow_size=" << flow.cols << "x" << flow.rows << "\n";
    return 0;
}
