#include "segformer.hpp"

// 命令行参数结构体
struct CmdParams {
    std::string dataset_path;
    std::string model_path;
    std::string output_dir = "./outputs";
    int core_id = 0;
    bool batch_mode = false;
};



namespace fs = std::filesystem;
using json = nlohmann::json;


struct JsonParams {
    std::string data_root = "cityscapes";               // 数据集根目录
    std::string img_dir_suffix =  "leftImg8bit/val";     // 图像目录后缀
    std::string ann_dir_suffix =  "gtFine/val";          // 标注目录后缀
    std::string img_suffix = "_leftImg8bit";            // 图像文件名后缀（不带扩展名）
    std::string seg_map_suffix = "_gtFine_labelTrainIds.png"; // 标注文件后缀
    bool save_result_images = true;                     // 是否保存结果图像
    bool generate_json = true;                          // 是否生成JSON文件
};

// 单张图像的结果信息
struct SingleImageResult {
    std::string filename;               // 相对于图像目录的相对路径
    std::string result_image_path;      // 彩色分割结果图像路径
    std::string prediction_path;        // 原始预测图路径（分类图）
    std::string annotation_seg_map;     // 标注文件的相对路径
    
    // 构造函数
    SingleImageResult(const std::string& fn = "",
                     const std::string& rip = "",
                     const std::string& pp = "",
                     const std::string& asm_path = "")
        : filename(fn), result_image_path(rip), prediction_path(pp), annotation_seg_map(asm_path) {}
};

// ==================== 辅助函数 ====================

// 检查路径是否存在
bool check_path_exists(const std::string& path) {
    struct stat info;
    return (stat(path.c_str(), &info) == 0);
}

// 判断是否为图像文件
bool is_image_file(const std::string& filename) {
    if (filename.empty()) return false;
    
    std::string ext = fs::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
           ext == ".bmp" || ext == ".tiff" || ext == ".tif";
}

// 递归获取目录下所有图像文件（与Python代码一致）
std::vector<std::string> get_image_files_recursive(const std::string& img_dir) {
    std::vector<std::string> filepaths;
    
    try {
        for (const auto& entry : fs::recursive_directory_iterator(img_dir)) {
            if (entry.is_regular_file() && is_image_file(entry.path().string())) {
                filepaths.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory '" << img_dir << "': " << e.what() << std::endl;
    }
    
    return filepaths;
}

// 确保目录存在
void ensure_directory_exists(const std::string& dir_path) {
    try {
        if (!fs::exists(dir_path)) {
            fs::create_directories(dir_path);
            std::cout << "Created directory: " << dir_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to create directory '" << dir_path << "': " << e.what() << std::endl;
    }
}

// 获取文件的基本名（去除后缀）
std::string get_basename_without_suffix(const std::string& filename, const std::string& suffix) {
    std::string basename = fs::path(filename).stem().string();
    
    // 如果basename以suffix结尾，则去除
    if (!suffix.empty() && basename.length() > suffix.length()) {
        size_t pos = basename.rfind(suffix);
        if (pos != std::string::npos && pos == basename.length() - suffix.length()) {
            basename = basename.substr(0, pos);
        }
    }
    
    return basename;
}


// 保存结果到JSON文件（仿照Python格式）
void save_results_to_json(const std::string& output_dir,
                         const JsonParams& json_params,
                         const std::vector<SingleImageResult>& results,
                         const std::string& model_filename) {
    
    // 创建JSON对象
    json result_json;
    result_json["data_root"] = json_params.data_root;
    result_json["img_dir"] = json_params.data_root + '/' + json_params.img_dir_suffix;
    result_json["ann_dir"] = json_params.data_root + '/' + json_params.ann_dir_suffix;
    
    // 构建img_info数组
    json img_info_array = json::array();
    for (const auto& result : results) {
        json item;
        item["res"] = result.result_image_path;  // 结果图像路径
        item["filename"] = result.filename;      // 原始图像相对路径
        
        json ann_obj;
        ann_obj["seg_map"] = result.annotation_seg_map;  // 标注文件相对路径
        item["ann"] = ann_obj;
        
        img_info_array.push_back(item);
    }
    
    result_json["img_info"] = img_info_array;
    
    // 构建JSON文件名（与Python格式一致）
    std::string json_name;
    std::string dataset_path = output_dir;
    
    // 去除末尾斜杠
    if (!dataset_path.empty() && dataset_path.back() == '/') {
        dataset_path.pop_back();
    }
    
    // 提取数据集文件夹名
    std::string dataset_folder_name = fs::path(dataset_path).filename().string();
    if (dataset_folder_name.empty()) {
        dataset_folder_name = "dataset";
    }
    
    // 提取模型文件名
    std::string model_name = fs::path(model_filename).stem().string();
    
    // 构建完整JSON文件名
    json_name = model_name + "_" + dataset_folder_name + "_opencv_cpp_result.json";
    std::string json_path = output_dir + "/" + json_name;
    
    // 保存到文件
    std::ofstream ofs(json_path);
    if (ofs.is_open()) {
        ofs << std::setw(4) << result_json;  // 缩进4个空格，美化输出
        std::cout << "\n Result JSON saved to: " << json_path << std::endl;
        std::cout << "   Total records: " << results.size() << std::endl;
    } else {
        std::cerr << "\n Failed to save JSON to: " << json_path << std::endl;
    }
}



bool process_single_image_for_json(Segformer& segformer,
                                  const std::string& image_path,
                                  const std::string& img_base_dir,
                                  const JsonParams& json_params,
                                  const CmdParams& cmd_params,
                                  SingleImageResult& result_info) {
    
    // 读取图像
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
    if (image.empty()) {
        std::cerr << " Failed to read image: " << image_path << std::endl;
        return false;
    }
    
    // 处理图像

    cv::Mat result = segformer.process(image);
    if (result.empty()) {
        std::cerr << " Failed to process image: " << image_path << std::endl;
        return false;
    }
    
    // 获取相对于图像基目录的相对路径
    std::string relative_path;
    try {
        relative_path = fs::relative(image_path, img_base_dir).string();
    } catch (...) {
        relative_path = image_path;
    }
    
    // 提取基本文件名（不带后缀）
    std::string basename = get_basename_without_suffix(fs::path(image_path).filename().string(), 
                                                       json_params.img_suffix);
    
    // 构建各种路径
    result_info.filename = relative_path;
    
    // 结果图像路径（彩色）
    result_info.result_image_path = cmd_params.output_dir + "/images/" + basename + "_leftImg8bit.png";
    
    // 原始预测图路径（分类图）
    result_info.prediction_path = cmd_params.output_dir + "/result_cl/" + basename + ".png";
    
    // 标注文件相对路径

    std::string seg_map_filename = basename + json_params.seg_map_suffix;
    std::string seg_map_rel_path = fs::path(relative_path).parent_path().string();
    if (!seg_map_rel_path.empty()) {
        seg_map_rel_path += "/";
    }
    seg_map_rel_path += seg_map_filename;
    result_info.annotation_seg_map = seg_map_rel_path;
    
    std::cout << "seg_map_filename: " << seg_map_filename << std::endl;
    std::cout << "relative_path: " << relative_path << std::endl;
    std::cout << "image_path: " << fs::path(image_path).filename().string() << std::endl;
    std::cout << "basename: " << basename << std::endl;

    // 创建输出目录
    ensure_directory_exists(cmd_params.output_dir + "/images");
    ensure_directory_exists(cmd_params.output_dir + "/result_cl");
    
    // 保存结果图像
    if (json_params.save_result_images) {
        if (!cv::imwrite(result_info.result_image_path, result)) {
            std::cerr << "Warning: Failed to save result image: " 
                      << result_info.result_image_path << std::endl;
        }
    }
    
    
    return true;
}

// 处理Cityscapes数据集并生成JSON
void process_cityscapes_for_json(Segformer& segformer,
                                 const CmdParams& cmd_params,
                                 const JsonParams& json_params) {
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Processing Cityscapes Dataset       " << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 构建完整的数据集路径
    std::string full_data_root = cmd_params.dataset_path + "/" + json_params.data_root;
    std::string img_base_dir = full_data_root + "/" + json_params.img_dir_suffix;
    std::string ann_base_dir = full_data_root + "/" + json_params.ann_dir_suffix;
    
    std::cout << "Data root: " << full_data_root << std::endl;
    std::cout << "Image directory: " << img_base_dir << std::endl;
    std::cout << "Annotation directory: " << ann_base_dir << std::endl;
    
    // 检查目录是否存在
    if (!check_path_exists(img_base_dir)) {
        std::cerr << " Error: Image directory not found: " << img_base_dir << std::endl;
        return;
    }
    
    // 获取所有图像文件
    std::vector<std::string> image_files = get_image_files_recursive(img_base_dir);
    
    if (image_files.empty()) {
        std::cerr << " No image files found in: " << img_base_dir << std::endl;
        return;
    }
    
    std::cout << "\n Found " << image_files.size() << " images to process.\n" << std::endl;
    
    // 准备结果容器
    std::vector<SingleImageResult> all_results;
    int processed_count = 0;
    int error_count = 0;
    
    // 处理每张图像
    for (size_t i = 0; i < image_files.size(); ++i) {
        const auto& image_path = image_files[i];
        
        std::cout << "[" << std::setw(3) << (i+1) << "/" << std::setw(3) << image_files.size() << "] ";
        std::cout << "Processing: " << fs::path(image_path).filename().string() << std::endl;
        
        SingleImageResult result_info;
        
        if (process_single_image_for_json(segformer, image_path, img_base_dir, 
                                         json_params, cmd_params, result_info)) {
            all_results.push_back(result_info);
            processed_count++;
        } else {
            error_count++;
        }
    }
    
    // 保存JSON结果
    if (json_params.generate_json && !all_results.empty()) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "   Generating JSON Results File        " << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        save_results_to_json(cmd_params.output_dir, json_params, all_results, cmd_params.model_path);
    }
    
    // 打印统计信息
    std::cout << "\n========================================" << std::endl;
    std::cout << "           Processing Summary          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "✓ Successfully processed: " << processed_count << " images" << std::endl;
    if (error_count > 0) {
        std::cout << "✗ Failed to process: " << error_count << " images" << std::endl;
    }
    std::cout << " Average inference time: " 
              << std::fixed << std::setprecision(2) 
              << segformer.get_avg_inference_time() << " ms" << std::endl;
    std::cout << " Results saved to: " << cmd_params.output_dir << std::endl;
    
    // 显示一些示例结果
    if (!all_results.empty() && all_results.size() <= 5) {
        std::cout << "\nExample results:" << std::endl;
        for (const auto& result : all_results) {
            std::cout << "  • " << result.filename 
                      << " → " << fs::path(result.result_image_path).filename().string() << std::endl;
        }
    }
}


// 命令行解析函数
CmdParams parse_command_line(int argc, char** argv) {
    const char* keys = 
        "{help h usage ? |      | print help message}"
        "{dataset_path d | ./dataset | path to the image/dataset directory}"
        "{model_path m | ./models/segformer_float16.nb | path to the model file}"
        "{output_dir o | ./outputs | directory to save results}"
        "{core_id c | 0 | core ID for execution (0 or 1)}";

    cv::CommandLineParser parser(argc, argv, keys);
    
    if (parser.has("help")) {
        parser.printMessage();
        exit(0);
    }

    CmdParams config;
    config.dataset_path = parser.get<std::string>("dataset_path");
    config.model_path = parser.get<std::string>("model_path");
    config.output_dir = parser.get<std::string>("output_dir");
    config.core_id = parser.get<int>("core_id");
    config.batch_mode = true;

    // 验证参数
    if (!parser.check()) {
        parser.printErrors();
        exit(1);
    }

    // 检查路径是否存在
    struct stat info;
    if (stat(config.model_path.c_str(), &info) != 0) {
        std::cerr << "ERROR: Cannot find model file: " << config.model_path << std::endl;
        exit(1);
    }
    
    if (stat(config.dataset_path.c_str(), &info) != 0) {
        std::cerr << "ERROR: Cannot find dataset/image path: " << config.dataset_path << std::endl;
        exit(1);
    }

    // 检查输出目录，如果不存在则创建
    if (stat(config.output_dir.c_str(), &info) != 0) {
        std::cout << "Output directory does not exist, creating: " << config.output_dir << std::endl;
        try {
            std::filesystem::create_directories(config.output_dir);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "ERROR: Failed to create output directory: " << config.output_dir 
                    << ", error: " << e.what() << std::endl;
            exit(1);
        }
    }

    // 验证core_id的有效性
    if (config.core_id != 0 && config.core_id != 1) {
        std::cerr << "ERROR: core_id must be 0 or 1, got: " << config.core_id << std::endl;
        exit(1);
    }

    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "================================" << std::endl;
    std::cout << "      Segformer Inference       " << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        // 1. 解析命令行参数
        CmdParams config = parse_command_line(argc, argv);
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Dataset path: " << config.dataset_path << std::endl;
        std::cout << "  Model path: " << config.model_path << std::endl;
        std::cout << "  Output directory: " << config.output_dir << std::endl;
        std::cout << "  Core ID: " << config.core_id << std::endl;
        std::cout << "================================" << std::endl;
        
        // 2. 检查文件和目录是否存在
        if (!check_path_exists(config.model_path)) {
            std::cerr << "ERROR: Cannot find model file: " << config.model_path << std::endl;
            return -1;
        }
        
        if (!check_path_exists(config.dataset_path)) {
            std::cerr << "ERROR: Cannot find dataset directory: " << config.dataset_path << std::endl;
            return -1;
        }

        JsonParams json_config;
        
        // 3. 创建Segformer实例并进行初始化
        std::cout << "Initializing Segformer..." << std::endl;
        
        // 创建 Segformer 配置
        Segformer::SegformerConfig segformer_config;
        segformer_config.model_path = config.model_path;
        segformer_config.core_id = config.core_id;
        
        Segformer segformer(segformer_config);
        
        if (segformer.init(config.model_path) != 0) {
            std::cerr << "Segformer init failed." << std::endl;
            segformer.deinit();
            return -1;
        }
        
        std::cout << "Segformer initialized successfully." << std::endl;
        std::cout << "Model input size: " << segformer.get_input_size() << std::endl;
        std::cout << "================================" << std::endl;
        // 4. 处理数据集

        if (config.batch_mode) {
            // 批处理模式：处理整个Cityscapes数据集并生成JSON
            process_cityscapes_for_json(segformer, config, json_config);
        }

        // 5. 显示统计信息
        std::cout << "\n================================" << std::endl;
        std::cout << "          Statistics            " << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << "Total inferences: " << segformer.get_total_inferences() << std::endl;
        std::cout << "Average inference time: " << segformer.get_avg_inference_time() << " ms" << std::endl;
        
        // 6. 清理资源
        std::cout << "\nCleaning up resources..." << std::endl;
        segformer.deinit();
        
        std::cout << "\nDone!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}