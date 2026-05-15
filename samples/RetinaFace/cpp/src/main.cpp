#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "retinaface.hpp"
#include "timer.hpp"

#define USE_JSON 1
#ifdef USE_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif


// 检查文件是否存在
bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// 解析命令行参数
CmdParams parse_arguments(int argc, char** argv) {
    const char* keys = 
        "{help h usage ? |      | print help message   }"
        "{input i      | test.jpg  | input image file or directory for batch mode    }"
        "{model m      | retinaface_int8.nb  | model file           }"
        "{output o | output.jpg | output image file or json file for batch mode    }"
        "{conf_thresh | 0.5 | confidence threshold for filter boxes}"
        "{nms_thresh | 0.3 | iou threshold for nms}"
        "{batch b | false | batch mode for dataset validation}"
        "{save_result | false | save detection result images in batch mode}";
        
    cv::CommandLineParser parser(argc, argv, keys);
    
    if (parser.get<bool>("help")) {
        parser.printMessage();
        exit(0);
    }
    
    CmdParams config;
    config.input = parser.get<std::string>("input");
    config.model = parser.get<std::string>("model");
    config.output = parser.get<std::string>("output");
    config.conf_thresh = parser.get<float>("conf_thresh");
    config.nms_thresh = parser.get<float>("nms_thresh");
    config.batch_mode = parser.get<bool>("batch");
    config.save_result = parser.get<bool>("save_result");

    return config;
}

// 获取目录下所有图片文件
std::vector<std::string> get_image_files(const std::string& directory) {
    std::vector<std::string> files;
    namespace fs = std::filesystem;
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                files.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    std::sort(files.begin(), files.end());
    return files;
}

// 单图推理模式
int run_single_image_inference(const CmdParams& config) {
    std::cout << "--------------------------------------\n";
    std::cout << "Single Image Inference Mode\n";
    std::cout << "Model: " << config.model << "\n";
    std::cout << "Input: " << config.input << "\n";
    std::cout << "--------------------------------------\n";

    if (!file_exists(config.model) || !file_exists(config.input)) {
        std::cerr << "Model or Image file does not exist.\n";
        return -1;
    }

    RetinaFace detector;
    TimeStamp tick;
    detector.enableProfile(&tick);
    
    if (!detector.init(config.model)) {
        std::cerr << "Failed to initialize detector\n";
        return -1;
    }

    tick.start();
    cv::Mat image = cv::imread(config.input, cv::IMREAD_COLOR|cv::IMREAD_RETRY_SOFTDEC);
    tick.time_accumulation("imread_time");
    
    if (image.empty()) return -1;

    std::vector<facebox> objects;
    if (!detector.detect_and_save(image, config.output, objects, config.conf_thresh, config.nms_thresh)) {
        std::cerr << "Detection failed\n";
        return -1;
    }

    // 画图并保存
    if (config.save_result || config.output.find(".jpg") != std::string::npos) {
        detector.draw_objects(image, objects, config.output);
    }

    std::cout << "\n===== Time Statistics =====\n";
    std::cout << "Total objects:      " << objects.size() << "\n";
    std::cout << "Image read time:    " << tick.time_map_lab["imread_time"] << " ms\n";
    std::cout << "Preprocess time:    " << tick.time_map_lab["pre_time"] << " ms\n";
    std::cout << "Inference time:     " << tick.time_map_lab["infer_time"] << " ms\n";
    std::cout << "Postprocess time:   " << tick.time_map_lab["post_time"] << " ms\n";
    std::cout << "===========================\n";
    
    return 0;
}

// 批量推理模式
int run_batch_inference(const CmdParams& config) {
    std::cout << "--------------------------------------\nBatch Inference Mode\n";
    
    RetinaFace detector;
    TimeStamp tick;
    detector.enableProfile(&tick);
    if (!detector.init(config.model)) return -1;

    auto image_files = get_image_files(config.input);
    if (image_files.empty()) return -1;

    std::string output_dir = config.save_result ? "results" : "";
    if (config.save_result) std::filesystem::create_directories(output_dir);

    json results_json = json::array();
    
    for (const auto& image_file : image_files) {
        cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
        if (image.empty()) continue;

        std::vector<facebox> objects;
        detector.detect_and_save(image, "", objects, config.conf_thresh, config.nms_thresh);

        size_t pos = image_file.find_last_of("/\\");
        std::string image_name = (pos == std::string::npos) ? image_file : image_file.substr(pos + 1);

        if (config.save_result) {
            detector.draw_objects(image, objects, output_dir + "/" + image_name);
        }

        // 保存 JSON 结果
        json image_result;
        image_result["image_name"] = image_name;
        json bboxes_json = json::array();
        for (const auto& obj : objects) {
            bboxes_json.push_back({
                {"score", obj.score},
                {"bbox", {obj.x1, obj.y1, obj.x2, obj.y2}},
                {"landmarks", {obj.landmarks[0], obj.landmarks[1], obj.landmarks[2], obj.landmarks[3], obj.landmarks[4],
                               obj.landmarks[5], obj.landmarks[6], obj.landmarks[7], obj.landmarks[8], obj.landmarks[9]}}
            });
        }
        image_result["bboxes"] = bboxes_json;
        results_json.push_back(image_result);
    }

    std::ofstream ofs(config.output);
    ofs << std::setw(4) << results_json << std::endl;
    std::cout << "Batch validation done. JSON saved to: " << config.output << "\n";
    
    return 0;
}

int main(int argc, char* argv[]) {
    CmdParams config = parse_arguments(argc, argv);
    if (config.batch_mode) return run_batch_inference(config);
    return run_single_image_inference(config);
}