#include "yolo11_seg.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <algorithm>

struct CmdParams {
    std::string input;
    std::string model;
    std::string output;
    float conf_thresh;
    float nms_thresh;
    int max_det;
    bool json_mode;
    std::string mask_dir;
    bool is_dir_input;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -m <model.nb> -i <image_or_dir> -o <output> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Single image mode:  -i input.jpg -o output.jpg" << std::endl;
    std::cout << "Batch dir mode:     -i input_dir/ -o output_dir/" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m <path>        Model file (.nb)" << std::endl;
    std::cout << "  -i <path>        Input image file or directory" << std::endl;
    std::cout << "  -o <path>        Output image file or directory" << std::endl;
    std::cout << "  --conf <float>   Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  --nms <float>    NMS threshold (default: 0.45)" << std::endl;
    std::cout << "  --max-det <int>  Max detections per image, -1=no limit (default: -1)" << std::endl;
    std::cout << "  --json           Output JSON results (batch mode)" << std::endl;
    std::cout << "  --mask-dir <dir> Save mask PNGs to this directory" << std::endl;
    std::cout << "  -h, --help       Show this help" << std::endl;
}

static bool is_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}
// Recursively create directories (like mkdir -p)
static bool mkdir_p(const std::string& path) {
    if (path.empty()) return true;
    if (is_directory(path)) return true;
    size_t pos = path.rfind('/');
    if (pos != std::string::npos && pos > 0) {
        if (!mkdir_p(path.substr(0, pos))) return false;
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}


static bool has_image_ext(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower.size() > 4 &&
            (lower.substr(lower.size()-4) == ".jpg" ||
             lower.substr(lower.size()-5) == ".jpeg" ||
             lower.substr(lower.size()-4) == ".png" ||
             lower.substr(lower.size()-4) == ".bmp"));
}

static std::vector<std::string> list_images(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) return files;
    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr) {
        std::string name = entry->d_name;
        if (has_image_ext(name)) {
            files.push_back(name);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

static std::string strip_ext(const std::string& name) {
    size_t dot = name.rfind('.');
    if (dot == std::string::npos) return name;
    return name.substr(0, dot);
}



CmdParams parse_arguments(int argc, char** argv) {
    CmdParams config;
    config.model = "";
    config.input = "";
    config.output = "output.jpg";
    config.conf_thresh = 0.25f;
    config.nms_thresh = 0.45f;
    config.max_det = -1;
    config.json_mode = false;
    config.mask_dir = "";
    config.is_dir_input = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" && i + 1 < argc) {
            config.model = argv[++i];
        } else if (arg == "-i" && i + 1 < argc) {
            config.input = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            config.output = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            config.conf_thresh = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            config.nms_thresh = std::stof(argv[++i]);
        } else if (arg == "--max-det" && i + 1 < argc) {
            config.max_det = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            config.json_mode = true;
        } else if (arg == "--mask-dir" && i + 1 < argc) {
            config.mask_dir = argv[++i];
        }
    }

    if (config.model.empty() || config.input.empty()) {
        std::cerr << "Error: -m and -i are required" << std::endl;
        print_usage(argv[0]);
        exit(1);
    }

    // Detect directory input
    config.is_dir_input = is_directory(config.input);

    return config;
}

// ========== Single image mode ==========
int run_single(const CmdParams& config) {
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "YOLO11s-seg Inference (single image)" << std::endl;
    std::cout << "Model: " << config.model << std::endl;
    std::cout << "Input: " << config.input << std::endl;
    std::cout << "Output: " << config.output << std::endl;
    std::cout << "Conf: " << config.conf_thresh
              << "  NMS: " << config.nms_thresh
              << "  MaxDet: " << config.max_det << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    YOLO11SegDetector detector;
    TimeStamp tick;
    detector.enableProfile(&tick);

    if (!detector.init(config.model)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }

    tick.start();
    cv::Mat image = cv::imread(config.input, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
    tick.time_accumulation("imread_time");

    if (image.empty()) {
        std::cerr << "Failed to load image: " << config.input << std::endl;
        return -1;
    }

    std::vector<Object> objects;
    if (!detector.detect_and_save(image, config.output, objects,
                                   config.conf_thresh, config.nms_thresh,
                                   config.max_det)) {
        std::cerr << "Detection failed" << std::endl;
        return -1;
    }

    std::cout << "Detected " << objects.size() << " objects" << std::endl;
    std::cout << "\n===== Time Statistics =====" << std::endl;
    std::cout << "Image read:    " << std::fixed << std::setprecision(2)
              << tick.time_map_lab["imread_time"] << " ms" << std::endl;
    std::cout << "Preprocess:    " << tick.time_map_lab["pre_time"] << " ms" << std::endl;
    std::cout << "Inference:     " << tick.time_map_lab["infer_time"] << " ms" << std::endl;
    std::cout << "Postprocess:   " << tick.time_map_lab["post_time"] << " ms" << std::endl;
    float total = tick.time_map_lab["imread_time"] + tick.time_map_lab["pre_time"] +
                  tick.time_map_lab["infer_time"] + tick.time_map_lab["post_time"];
    std::cout << "Total:         " << total << " ms" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Output saved to: " << config.output << std::endl;

    return 0;
}

// ========== Batch directory mode ==========
int run_batch(const CmdParams& config) {
    std::string in_dir = config.input;
    std::string out_dir = config.output;

    // Ensure output dir ends with /
    if (!out_dir.empty() && out_dir.back() != '/') out_dir += "/";
    // Ensure mask dir ends with /
    std::string mask_dir = config.mask_dir;
    if (!mask_dir.empty() && mask_dir.back() != '/') mask_dir += "/";

    // Create output and mask directories if they don't exist
    if (!out_dir.empty() && !mkdir_p(out_dir.substr(0, out_dir.size() - 1))) {
        std::cerr << "Warning: could not create output dir: " << out_dir << std::endl;
    }
    if (!mask_dir.empty() && !mkdir_p(mask_dir.substr(0, mask_dir.size() - 1))) {
        std::cerr << "Warning: could not create mask dir: " << mask_dir << std::endl;
    }

    std::vector<std::string> images = list_images(in_dir);
    if (images.empty()) {
        std::cerr << "No images found in: " << in_dir << std::endl;
        return -1;
    }

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "YOLO11s-seg Batch Inference" << std::endl;
    std::cout << "Model: " << config.model << std::endl;
    std::cout << "Input dir: " << in_dir << " (" << images.size() << " images)" << std::endl;
    std::cout << "Output dir: " << out_dir << std::endl;
    std::cout << "Conf: " << config.conf_thresh
              << "  NMS: " << config.nms_thresh
              << "  MaxDet: " << config.max_det << std::endl;
    if (!mask_dir.empty())
        std::cout << "Mask dir: " << mask_dir << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    YOLO11SegDetector detector;
    TimeStamp tick;
    detector.enableProfile(&tick);

    if (!detector.init(config.model)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }

    // JSON output accumulator
    std::ofstream json_file;
    if (config.json_mode) {
        std::string json_path = out_dir + "results.json";
        json_file.open(json_path);
        if (!json_file.is_open()) {
            std::cerr << "Cannot open JSON output: " << json_path << std::endl;
            return -1;
        }
        json_file << "{\"results\": [" << std::endl;
    }

    int total_dets = 0;
    int fail_count = 0;
    float total_infer_time = 0.0f;

    for (size_t idx = 0; idx < images.size(); idx++) {
        const std::string& img_name = images[idx];
        std::string img_path = in_dir + "/" + img_name;
        std::string out_path = out_dir + img_name;
        std::string stem = strip_ext(img_name);

        tick.start();
        cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
        tick.time_accumulation("imread_time");

        if (image.empty()) {
            std::cerr << "[" << (idx+1) << "/" << images.size() << "] SKIP (empty): " << img_name << std::endl;
            fail_count++;
            continue;
        }

        std::vector<Object> objects;
        if (!detector.detect_and_save(image, out_path, objects,
                                       config.conf_thresh, config.nms_thresh,
                                       config.max_det)) {
            std::cerr << "[" << (idx+1) << "/" << images.size() << "] FAIL: " << img_name << std::endl;
            fail_count++;
            continue;
        }

        total_dets += objects.size();
        total_infer_time += tick.time_map_lab["infer_time"];

        // Print progress every 10 images
        if ((idx + 1) % 10 == 0 || idx == 0 || idx == images.size() - 1) {
            std::cout << "[" << (idx+1) << "/" << images.size() << "] "
                      << img_name << ": " << objects.size() << " dets, "
                      << std::fixed << std::setprecision(1) << tick.time_map_lab["infer_time"] << "ms infer"
                      << std::endl;
        }

        // Save masks with image-name prefix
        if (!mask_dir.empty()) {
            for (size_t i = 0; i < objects.size(); i++) {
                if (!objects[i].mask.empty()) {
                    char mask_path[512];
                    snprintf(mask_path, sizeof(mask_path), "%s%s_mask_%04zu.png",
                             mask_dir.c_str(), stem.c_str(), i);
                    cv::Mat mask_u8;
                    objects[i].mask.convertTo(mask_u8, CV_8U, 255.0);
                    cv::imwrite(mask_path, mask_u8);
                }
            }
        }

        // JSON output per image
        if (config.json_mode && json_file.is_open()) {
            if (idx > 0) json_file << "," << std::endl;
            json_file << "  {\"image\": \"" << img_name
                      << "\", \"detections\": [" << std::endl;
            for (size_t i = 0; i < objects.size(); i++) {
                const Object& obj = objects[i];
                json_file << "    {\"category_id\": " << (obj.class_id + 1)
                          << ", \"score\": " << std::fixed << std::setprecision(6) << obj.prob
                          << ", \"bbox\": [" << std::fixed << std::setprecision(2)
                          << obj.box.left << ", " << obj.box.top
                          << ", " << obj.box.width << ", " << obj.box.height << "]";
                if (!obj.mask.empty() && !mask_dir.empty()) {
                    char mask_rel[512];
                    snprintf(mask_rel, sizeof(mask_rel), "%s_mask_%04zu.png", stem.c_str(), i);
                    json_file << ", \"mask_file\": \"" << mask_dir << mask_rel
                              << "\", \"mask_size\": [" << obj.mask.rows << ", " << obj.mask.cols << "]";
                }
                if (i < objects.size() - 1)
                    json_file << "}," << std::endl;
                else
                    json_file << "}" << std::endl;
            }
            json_file << "  ]}" ;
        }
    }

    // Close JSON
    if (config.json_mode && json_file.is_open()) {
        json_file << std::endl << "]}" << std::endl;
        json_file.close();
        std::cout << "JSON results saved to: " << out_dir << "results.json" << std::endl;
    }

    // Summary
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Batch complete: " << images.size() << " images" << std::endl;
    std::cout << "Total detections: " << total_dets << std::endl;
    std::cout << "Failures: " << fail_count << std::endl;
    std::cout << "Avg inference: " << std::fixed << std::setprecision(1)
              << (total_infer_time / (images.size() - fail_count)) << " ms/image" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    return 0;
}

int main(int argc, char** argv) {
    CmdParams config = parse_arguments(argc, argv);

    // Check model exists
    if (!YOLO11SegDetector::file_exists(config.model)) {
        std::cerr << "Model file does not exist: " << config.model << std::endl;
        return -1;
    }

    if (config.is_dir_input) {
        return run_batch(config);
    } else {
        return run_single(config);
    }
}
