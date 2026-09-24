// pwc_net_eval
// Reads a predicted .flo and a ground-truth .flo file,
// computes EPE, mean angular error, and bad-pixel rates.
// Supports resolution mismatch by keeping pred at its native resolution and
// resizing/scaling GT to pred resolution before evaluation.
// Usage: pwc_net_eval --pred predicted.flo --gt ground_truth.flo [--summary out.csv]
//        pwc_net_eval --list manifest.txt [--summary out.csv]
//   manifest format (per line): pred_path gt_path

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Metrics {
    double epe_sum = 0.0, angle_sum = 0.0;
    uint64_t pixels = 0, bad3 = 0, bad5 = 0;
};

static bool read_flo(const std::string& path, cv::Mat& flow) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    char magic[5] = {};
    int32_t w = 0, h = 0;
    file.read(magic, 4);
    file.read(reinterpret_cast<char*>(&w), sizeof(w));
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!file || std::string(magic, 4) != "PIEH" || w <= 0 || h <= 0) return false;
    flow.create(h, w, CV_32FC2);
    file.read(reinterpret_cast<char*>(flow.data),
              static_cast<std::streamsize>(flow.total() * sizeof(cv::Vec2f)));
    return static_cast<bool>(file);
}

// Resize a flow field to target_size and scale its vectors by (sx, sy).
// For native-resolution evaluation this is used on GT, not pred: GT vectors
// are expressed in source-image pixels, so their x/y components must be
// scaled together with the spatial resolution.
static cv::Mat resize_flow(const cv::Mat& flow, const cv::Size& target_size,
                           double sx, double sy) {
    std::vector<cv::Mat> ch(2), ch_out(2);
    cv::split(flow, ch);
    for (int c = 0; c < 2; ++c) {
        cv::Mat ch_f;
        ch[c].convertTo(ch_f, CV_64F);
        cv::Mat resized;
        cv::resize(ch_f, resized, target_size, 0, 0, cv::INTER_LINEAR);
        ch_out[c] = resized * (c == 0 ? sx : sy);
    }
    cv::Mat out_f;
    cv::merge(ch_out, out_f);
    out_f.convertTo(out_f, CV_32FC2);
    return out_f;
}

static void accumulate(const cv::Mat& pred, const cv::Mat& truth, Metrics& m) {
    CV_Assert(pred.size() == truth.size());
    CV_Assert(pred.type() == CV_32FC2 && truth.type() == CV_32FC2);
    for (int y = 0; y < truth.rows; ++y) {
        const float* rp = pred.ptr<float>() + y * truth.cols * 2;
        const float* rg = truth.ptr<float>() + y * truth.cols * 2;
        for (int x = 0; x < truth.cols; ++x) {
            float pdx = rp[x*2], pdy = rp[x*2+1];
            float gdx = rg[x*2], gdy = rg[x*2+1];
            if (!std::isfinite(gdx) || !std::isfinite(gdy)) continue;
            double du = static_cast<double>(pdx) - gdx;
            double dv = static_cast<double>(pdy) - gdy;
            double epe = std::sqrt(du*du + dv*dv);

            double dot = static_cast<double>(pdx)*gdx + static_cast<double>(pdy)*gdy + 1.0;
            double pn = std::sqrt(static_cast<double>(pdx)*pdx + static_cast<double>(pdy)*pdy + 1.0);
            double gn = std::sqrt(static_cast<double>(gdx)*gdx + static_cast<double>(gdy)*gdy + 1.0);
            double cosine = std::max(-1.0, std::min(1.0, dot/(pn*gn)));
            m.epe_sum += epe;
            m.angle_sum += std::acos(cosine) * 180.0 / M_PI;
            m.pixels++;
            if (epe > 3.0) m.bad3++;
            if (epe > 5.0) m.bad5++;
        }
    }
}

static void print_summary(const Metrics& m, int count) {
    const double px = static_cast<double>(m.pixels);
    std::cout << std::fixed << std::setprecision(6)
              << "pairs=" << count << "\n"
              << "mean_epe=" << (px ? m.epe_sum / px : 0.0) << "\n"
              << "mean_angular_error_deg=" << (px ? m.angle_sum / px : 0.0) << "\n"
              << "bad3_percent=" << (px ? 100.0 * m.bad3 / px : 0.0) << "\n"
              << "bad5_percent=" << (px ? 100.0 * m.bad5 / px : 0.0) << "\n";
}

struct Opts {
    std::string pred, gt, summary_csv;
    bool list_mode = false;
    std::string list_file;
};

static bool parse_opts(int argc, char** argv, Opts& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pred" && i+1 < argc) o.pred = argv[++i];
        else if (a == "--gt" && i+1 < argc) o.gt = argv[++i];
        else if (a == "--list" && i+1 < argc) {
            o.list_mode = true; o.list_file = argv[++i];
        } else if (a == "--summary" && i+1 < argc) o.summary_csv = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage:\n"
                      << "  Single: " << argv[0]
                      << " --pred PRED.flo --gt GT.flo [--summary out.csv]\n"
                      << "  Batch:  " << argv[0]
                      << " --list manifest.txt [--summary out.csv]\n"
                      << "  Note: GT is resized+scaled to pred resolution if sizes differ.\n";
            return false;
        }
    }
    return !o.list_mode || !o.list_file.empty();
}

static bool read_list(const std::string& path,
                      std::vector<std::pair<std::string, std::string>>& pairs) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        std::string pred_path, gt_path;
        if (!(in >> pred_path >> gt_path)) continue;
        pairs.emplace_back(pred_path, gt_path);
    }
    return !pairs.empty();
}

}  // anonymous namespace

int main(int argc, char** argv) {
    Opts opts;
    if (!parse_opts(argc, argv, opts)) return 2;

    Metrics total;
    int count = 0;
    std::ofstream csv_out;
    if (!opts.summary_csv.empty()) {
        csv_out.open(opts.summary_csv);
        csv_out << "pred,gt,epe,angular_error,bad3_pct,bad5_pct\n";
    }

    std::vector<std::pair<std::string, std::string>> workloads;
    if (opts.list_mode) {
        if (!read_list(opts.list_file, workloads)) {
            std::cerr << "Cannot read list: " << opts.list_file << "\n"; return 1;
        }
    } else {
        if (opts.pred.empty() || opts.gt.empty()) {
            std::cerr << "Need --pred and --gt\n"; return 2;
        }
        workloads.emplace_back(opts.pred, opts.gt);
    }

    for (auto& [pred_path, gt_path] : workloads) {
        cv::Mat pred_flow, gt_flow;
        if (!read_flo(pred_path, pred_flow)) {
            std::cerr << "Cannot read pred: " << pred_path << "\n"; continue;
        }
        if (!read_flo(gt_path, gt_flow)) {
            std::cerr << "Cannot read gt: " << gt_path << "\n"; continue;
        }

        if (pred_flow.size() != gt_flow.size()) {
            const double sx = static_cast<double>(pred_flow.cols) / gt_flow.cols;
            const double sy = static_cast<double>(pred_flow.rows) / gt_flow.rows;
            std::cerr << "Downsampling GT " << gt_flow.cols << "x" << gt_flow.rows
                      << " -> " << pred_flow.cols << "x" << pred_flow.rows
                      << " (flow scale " << sx << "x" << sy << ")\n";
            gt_flow = resize_flow(gt_flow, pred_flow.size(), sx, sy);
        }

        Metrics m;
        accumulate(pred_flow, gt_flow, m);
        const double px = static_cast<double>(m.pixels);
        double epe = px ? m.epe_sum / px : 0.0;
        double ang = px ? m.angle_sum / px : 0.0;
        double b3  = px ? 100.0 * m.bad3 / px : 0.0;
        double b5  = px ? 100.0 * m.bad5 / px : 0.0;

        std::cout << std::fixed << std::setprecision(4)
                  << "pair=" << (count+1) << " pred=" << pred_path
                  << " epe=" << epe << " ang=" << ang
                  << " bad3=" << b3 << "% bad5=" << b5 << "%\n";

        if (csv_out) {
            csv_out << std::fixed << std::setprecision(6)
                    << pred_path << "," << gt_path << ","
                    << epe << "," << ang << "," << b3 << "," << b5 << "\n";
        }

        total.epe_sum += m.epe_sum; total.angle_sum += m.angle_sum;
        total.pixels += m.pixels; total.bad3 += m.bad3; total.bad5 += m.bad5;
        ++count;
    }

    if (count > 0) {
        std::cout << "\n--- SUMMARY (" << count << " pairs) ---\n";
        print_summary(total, count);
    }
    if (csv_out) csv_out.close();
    return 0;
}
