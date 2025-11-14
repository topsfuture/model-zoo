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
#include "yolo11.hpp"
#include "timer.hpp"

#define USE_JSON 1
#ifdef USE_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

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
    
    // 按文件名排序
    std::sort(files.begin(), files.end());
    
    return files;
}

// 单图推理模式
int run_single_image_inference(const CmdParams& config) {
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Single Image Inference Mode" << std::endl;
    std::cout << "Model: " << config.model << std::endl;
    std::cout << "Input: " << config.input << std::endl;
    std::cout << "Output: " << config.output << std::endl;
    std::cout << "Conf thresh: " << config.conf_thresh << std::endl;
    std::cout << "NMS thresh: " << config.nms_thresh << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    // 检查文件是否存在
    if (!file_exists(config.model)) {
        std::cerr << "Model file does not exist: " << config.model << std::endl;
        return -1;
    }
    if (!file_exists(config.input)) {
        std::cerr << "Image file does not exist: " << config.input << std::endl;
        return -1;
    }

    // 创建检测器
    YOLO11Detector detector;
    TimeStamp tick;
    TimeStamp *ts = &tick;
    detector.enableProfile(&tick);
    
    // 初始化模型
    if (!detector.init(config.model)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }

    // 读取图像
    ts->start();
    cv::Mat image = cv::imread(config.input, cv::IMREAD_COLOR |cv::IMREAD_RETRY_SOFTDEC);
    ts->time_accumulation("imread_time");
    
    if (image.empty()) {
        std::cerr << "Failed to load image: " << config.input << std::endl;
        return -1;
    }

    // 执行推理
    std::vector<Object> objects;
    
    if (!detector.detect_and_save(image, config.output, objects, config.conf_thresh, config.nms_thresh)) {
        std::cerr << "Detection failed" << std::endl;
        return -1;
    }

    // 打印性能信息
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Detected " << objects.size() << " objects" << std::endl;
    
    std::cout << "\n===== Time Statistics =====" << std::endl;
    std::cout << "Image read time:    " << std::fixed << std::setprecision(2) 
              << ts->time_map_lab["imread_time"] << " ms" << std::endl;
    std::cout << "Preprocess time:    " << ts->time_map_lab["pre_time"] << " ms" << std::endl;
    std::cout << "Inference time:     " << ts->time_map_lab["infer_time"] << " ms" << std::endl;
    std::cout << "Postprocess time:   " << ts->time_map_lab["post_time"] << " ms" << std::endl;
    
    float total_time = ts->time_map_lab["imread_time"] + ts->time_map_lab["pre_time"] + 
                       ts->time_map_lab["infer_time"] + ts->time_map_lab["post_time"];
    std::cout << "Total time:         " << total_time << " ms" << std::endl;
    std::cout << "============================" << std::endl;
    
    std::cout << "Output saved to: " << config.output << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    

    return 0;
}

// 批量推理模式
int run_batch_inference(const CmdParams& config) {
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Batch Inference Mode" << std::endl;
    std::cout << "Model: " << config.model << std::endl;
    std::cout << "Dataset: " << config.input << std::endl;
    std::cout << "Output: " << config.output << std::endl;
    std::cout << "Conf thresh: " << config.conf_thresh << std::endl;
    std::cout << "NMS thresh: " << config.nms_thresh << std::endl;
    std::cout << "Save result images: " << (config.save_result ? "Yes" : "No") << std::endl;
    std::cout << "--------------------------------------" << std::endl;


    if (!file_exists(config.model)) {
        std::cerr << "Model file does not exist: " << config.model << std::endl;
        return -1;
    }

    struct stat info;
    if (stat(config.input.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
        std::cerr << "Dataset directory does not exist: " << config.input << std::endl;
        return -1;
    }


    YOLO11Detector detector;
    TimeStamp tick;
    TimeStamp *ts = &tick;
    detector.enableProfile(&tick);
    // 初始化模型
    if (!detector.init(config.model)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }


    std::vector<std::string> image_files = get_image_files(config.input);
    if (image_files.empty()) {
        std::cerr << "No images found in directory: " << config.input << std::endl;
        return -1;
    }

    std::cout << "Found " << image_files.size() << " images for inference" << std::endl;


    std::string output_dir = "";
    if (config.save_result) {
        output_dir = "results";

        std::filesystem::create_directories(output_dir);
        std::cout << "Result images will be saved to: " << output_dir << std::endl;
    }

   
    std::vector<YOLO11Detector::BatchResult> results;
    int total_objects = 0;
    int processed_count = 0;
    
    for (const auto& image_file : image_files) {
        if (processed_count >= 1000) {
            break;
        }

        // 读取图像 - 统计imread时间
        ts->start();
        cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
        ts->time_accumulation("imread_time");
        
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_file << std::endl;
            continue;
        }

   
        size_t pos = image_file.find_last_of("/\\");
        std::string image_name = (pos == std::string::npos) ? image_file : image_file.substr(pos + 1);

        YOLO11Detector::BatchResult result;
        result.image_name = image_name;

 
        std::string output_path = "";
        if (config.save_result) {
            output_path = output_dir + "/" + image_name;
        }

 
        std::vector<Object> objects;
        if (!detector.detect_and_save(image, output_path, objects, config.conf_thresh, config.nms_thresh)) {
            std::cerr << "Detection failed for: " << image_file << std::endl;
            continue;
        }
        
        result.objects = objects;
        results.push_back(result);
        total_objects += objects.size();
        processed_count++;


        if (processed_count % 10 == 0 || processed_count == image_files.size()) {
            std::cout << "Processed " << processed_count << "/" << image_files.size() 
                     << " images (" << objects.size() << " objects detected)" << std::endl;
        }
    }


    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Processing completed" << std::endl;
    std::cout << "Total images processed: " << results.size() << std::endl;
    std::cout << "Total objects detected: " << total_objects << std::endl;
    std::cout << "\n===== Time Statistics (Accumulated) =====" << std::endl;
    std::cout << "Image read time:    " << std::fixed << std::setprecision(2) 
              << ts->time_map_lab["imread_time"] << " ms (avg: " 
              << ts->time_map_lab["imread_time"] / results.size() << " ms/image)" << std::endl;
    std::cout << "Preprocess time:    " << ts->time_map_lab["pre_time"] << " ms (avg: " 
              << ts->time_map_lab["pre_time"] / results.size() << " ms/image)" << std::endl;
    std::cout << "Inference time:     " << ts->time_map_lab["infer_time"] << " ms (avg: " 
              << ts->time_map_lab["infer_time"] / results.size() << " ms/image)" << std::endl;
    std::cout << "Postprocess time:   " << ts->time_map_lab["post_time"] << " ms (avg: " 
              << ts->time_map_lab["post_time"] / results.size() << " ms/image)" << std::endl;
    
    float total_time = ts->time_map_lab["imread_time"] + ts->time_map_lab["pre_time"] + 
                       ts->time_map_lab["infer_time"] + ts->time_map_lab["post_time"];
    std::cout << "Total time:         " << total_time << " ms (avg: " 
              << total_time / results.size() << " ms/image)" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    if (config.save_result) {
        std::cout << "Result images saved to: " << output_dir << std::endl;
    }
    std::cout << "--------------------------------------" << std::endl;

#ifdef USE_JSON

    json results_json = json::array();
    for (const auto& result : results) {
        json image_result;
        image_result["image_name"] = result.image_name;
        
        json bboxes_json = json::array();
        for (const auto& obj : result.objects) {
            json bbox_json;
            bbox_json["category_id"] = obj.class_id;
            bbox_json["score"] = obj.prob;
            bbox_json["bbox"] = {obj.box.left, obj.box.top, obj.box.width, obj.box.height};
            bboxes_json.push_back(bbox_json);
        }
        
        image_result["bboxes"] = bboxes_json;
        results_json.push_back(image_result);
    }

    std::ofstream ofs(config.output);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << config.output << std::endl;
        return -1;
    }
    
    ofs << std::setw(4) << results_json << std::endl;
    ofs.close();
    std::cout << "Results saved to: " << config.output << std::endl;
#else
    //文本
    std::ofstream ofs(config.output);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << config.output << std::endl;
        return -1;
    }

    for (const auto& result : results) {
        ofs << "Image: " << result.image_name << std::endl;
        ofs << "Objects: " << result.objects.size() << std::endl;
        for (const auto& obj : result.objects) {
            ofs << "  class_id=" << obj.class_id 
                << " score=" << obj.prob
                << " bbox=[" << obj.box.left << "," << obj.box.top << ","
                << obj.box.width << "," << obj.box.height << "]" << std::endl;
        }
        ofs << std::endl;
    }
    ofs.close();
    std::cout << "Results saved to: " << config.output << std::endl;
#endif

    return 0;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    CmdParams config = parse_arguments(argc, argv);

    // 选择单图推理还是多图推理
    if (config.batch_mode) {
        return run_batch_inference(config);
    } else {
        return run_single_image_inference(config);
    }
}

