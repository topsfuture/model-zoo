// pwc_net_video
// Reads a video using OpenCV VideoCapture, computes PWC-Net optical flow
// between consecutive frames, and outputs per-frame of_speed.
// No FFmpeg dependency = no linking headaches.

#include "pwc_net_imgpair.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::string model;
    std::string video;
    int max_frames = 0;
};

static bool parse_opts(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) o.model = argv[++i];
        else if (a == "--video" && i + 1 < argc) o.video = argv[++i];
        else if (a == "--max-frames" && i + 1 < argc) o.max_frames = std::max(0, std::atoi(argv[++i]));
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " --model MODEL.nb --video INPUT.mp4 [--max-frames N]\n";
            return false;
        } else { std::cerr << "Unknown: " << a << "\n"; return false; }
    }
    if (o.model.empty() || o.video.empty()) { std::cerr << "--model and --video required\n"; return false; }
    return true;
}

// Magnitude-based optical flow speed, scaled to display resolution.
static double of_speed(const cv::Mat& flow, float sx, float sy) {
    double sum = 0.0;
    for (int y = 0; y < flow.rows; ++y) {
        const cv::Vec2f* row = flow.ptr<cv::Vec2f>(y);
        for (int x = 0; x < flow.cols; ++x) {
            float dx = row[x][0] * sx;
            float dy = row[x][1] * sy;
            sum += std::sqrt(dx*dx + dy*dy);
        }
    }
    return sum / static_cast<double>(flow.total());
}

}  // anonymous namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_opts(argc, argv, opts)) return 2;

    // Init ta_runtime and load NBG
    ta_runtime_init();
    pwc_net::PwcGraph net;
    if (!net.init(opts.model)) return 1;

    // Open video via OpenCV (uses TA SDK's FFmpeg backend internally)
    cv::VideoCapture cap(opts.video);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << opts.video << "\n"; return 1;
    }

    cv::Mat prev_bgr, curr_bgr;
    std::vector<double> history;
    double smoothed = 0.0;
    int frame_idx = 0;

    while (true) {
        if (opts.max_frames > 0 && frame_idx >= opts.max_frames) break;
        if (!cap.read(curr_bgr) || curr_bgr.empty()) break;

        const auto frame_begin = Clock::now();
        if (prev_bgr.empty()) {
            std::cout << "frame=1 first_frame=1 frame_overall_ms="
                      << std::fixed << std::setprecision(3)
                      << std::chrono::duration<double, std::milli>(
                             Clock::now() - frame_begin).count() << "\n";
        } else {
            cv::Mat flow_low; double nbg_ms = 0.0;
            if (!net.run(prev_bgr, curr_bgr, flow_low, nbg_ms)) return 1;

            const float sx = static_cast<float>(curr_bgr.cols) / net.net_w();
            const float sy = static_cast<float>(curr_bgr.rows) / net.net_h();
            const double disp = of_speed(flow_low, sx, sy);

            double instant = disp / 8.0 / 60.0 * 44.0;
            double conf = 0.0;
            if (!history.empty()) {
                double denom = 0.0;
                for (size_t i = 0; i < history.size(); ++i) denom += static_cast<double>(i + 1);
                double w = 0.0;
                for (size_t i = 0; i < history.size(); ++i)
                    w += history[i] * static_cast<double>(i + 1) / denom;
                conf = w;
            }
            smoothed = history.empty() ? 0.0 : (0.6 * instant + 0.4 * conf);
            history.push_back(smoothed);
            if (history.size() > 30) history.erase(history.begin());

            std::cout << "frame=" << (frame_idx + 1)
                      << " pwc_nbg_ms=" << nbg_ms
                      << " frame_overall_ms=" << std::fixed << std::setprecision(3)
                      << std::chrono::duration<double, std::milli>(
                             Clock::now() - frame_begin).count()
                      << " displacement=" << disp << " of_speed=" << smoothed << "\n";
        }
        prev_bgr = std::move(curr_bgr);
        ++frame_idx;
    }

    std::cout << "summary_frames=" << frame_idx << "\n";
    return 0;
}
