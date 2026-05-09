#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "insightface.hpp"

namespace fs = std::filesystem;

struct EvalPair {
    std::string img1_path;
    std::string img2_path;
    bool is_same; 
};

struct EvalResult {
    float best_accuracy = 0.0f;
    float best_threshold = 0.0f;
    float auc = 0.0f;
    int total_pairs = 0;
    int valid_pairs = 0;
    int valid_positive_pairs = 0;
    int valid_negative_pairs = 0;
    
    struct EvalStats {
        int64_t total_images_processed = 0;  // 处理的总图片数
        int64_t total_pairs_verified = 0;  // 验证的总对数
        
        void reset() {
            total_images_processed = 0;
            total_pairs_verified = 0;
        }
        
        void update_extraction(double time_ms, int image_count) {
            total_images_processed += image_count;
        }
        
        void update_verification(double time_ms, int pair_count) {
            total_pairs_verified += pair_count;
        }
    } stats;
};

std::string to_four_digit(int num) {
    if (num < 10) {
        return "000" + std::to_string(num);
    } else if (num < 100) {
        return "00" + std::to_string(num);
    } else if (num < 1000) {
        return "0" + std::to_string(num);
    } else {
        return std::to_string(num);
    }
}

std::vector<EvalPair> read_lfw_pairs(const std::string& pairs_path) {
    std::vector<EvalPair> pairs;
    std::ifstream file(pairs_path);
    if (!file.is_open()) {
        std::cerr << " Error: Cannot open pairs file: " << pairs_path << std::endl;
        return pairs;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }

        EvalPair pair;
        if (tokens.size() == 3) {
            std::string name = tokens[0];
            int idx1 = std::stoi(tokens[1]);
            int idx2 = std::stoi(tokens[2]);
            
            // 使用4位数字格式
            pair.img1_path = name + "/" + name + "_" + to_four_digit(idx1) + ".jpg";
            pair.img2_path = name + "/" + name + "_" + to_four_digit(idx2) + ".jpg";
            pair.is_same = true;
            pairs.push_back(pair);
        } else if (tokens.size() == 4) {
            std::string name1 = tokens[0];
            int idx1 = std::stoi(tokens[1]);
            std::string name2 = tokens[2];
            int idx2 = std::stoi(tokens[3]);
            
            // 使用4位数字格式
            pair.img1_path = name1 + "/" + name1 + "_" + to_four_digit(idx1) + ".jpg";
            pair.img2_path = name2 + "/" + name2 + "_" + to_four_digit(idx2) + ".jpg";
            pair.is_same = false;
            pairs.push_back(pair);
        }
    }
    file.close();
    return pairs;
}

// 计算余弦相似度
float cosine_similarity(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size() || feat1.empty()) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < feat1.size(); ++i) {
        dot_product += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }
    
    if (norm1 < 1e-6 || norm2 < 1e-6) {
        return 0.0f;
    }
    
    return dot_product / (sqrt(norm1) * sqrt(norm2));
}

EvalResult evaluate_verification(const std::vector<float>& similarities, 
                                const std::vector<int>& labels,
                                EvalResult::EvalStats& stats) {
    EvalResult result;
    result.valid_pairs = static_cast<int>(similarities.size());
    
    
    int best_correct = 0;
    float best_threshold = 0.0f;
    
    for (int i = 0; i < 1000; ++i) {
        float threshold = i * 0.001f;
        int correct = 0;
        
        for (size_t j = 0; j < similarities.size(); ++j) {
            float sim = similarities[j];
            int label = labels[j];
            int pred = (sim >= threshold) ? 1 : 0;
            
            if (pred == label) {
                correct++;
            }
        }
        
        float accuracy = static_cast<float>(correct) / similarities.size();
        if (correct > best_correct || (correct == best_correct && accuracy > result.best_accuracy)) {
            best_correct = correct;
            result.best_accuracy = accuracy;
            result.best_threshold = threshold;
        }
    }
    
    result.valid_positive_pairs = 0;
    result.valid_negative_pairs = 0;
    for (int label : labels) {
        if (label == 1) result.valid_positive_pairs++;
        else result.valid_negative_pairs++;
    }
    
    float auc = 0.0f;
    for (float threshold = 0.0f; threshold <= 1.0f; threshold += 0.001f) {
        float tpr = 0.0f, fpr = 0.0f;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (size_t j = 0; j < similarities.size(); ++j) {
            float sim = similarities[j];
            int label = labels[j];
            int pred = (sim >= threshold) ? 1 : 0;
            
            if (label == 1 && pred == 1) tp++;
            else if (label == 1 && pred == 0) fn++;
            else if (label == 0 && pred == 1) fp++;
            else if (label == 0 && pred == 0) tn++;
        }
        
        if (tp + fn > 0) tpr = static_cast<float>(tp) / (tp + fn);
        if (fp + tn > 0) fpr = static_cast<float>(fp) / (fp + tn);
        
        if (threshold > 0.0f) {
            auc += tpr * 0.001f;
        }
    }
    
    
    result.auc = auc;
    result.stats = stats;
    
    return result;
}

int main(int argc, char* argv[]) {
    // 默认参数
    std::string model_dir = "models";
    std::string image_path = "";
    std::string lfw_data_dir = "./dataset/LFW";
    std::string lfw_pairs_file = "./dataset/lfw_pairs.txt";
    int core_id = 0;
    float score_thresh = 0.5f;
    float nms_thresh = 0.4f;
    bool eval_mode = false;  
    bool profile = true;     
    bool silent_mode = false; 
    std::string det_model_path = "models/det_500m_float16.nb"; 
    std::string rec_model_path = "models/w600k_mbf_float16.nb";
    
    
    // 参数解析
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--det-model" && i + 1 < argc) {
            det_model_path = argv[++i];
        } else if (arg == "--rec-model" && i + 1 < argc) {
            rec_model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--lfw-data" && i + 1 < argc) {
            lfw_data_dir = argv[++i];
        } else if (arg == "--lfw-pairs" && i + 1 < argc) {
            lfw_pairs_file = argv[++i];
        } else if (arg == "--eval") {
            eval_mode = true;  // 启用评估模式
            silent_mode = true;  // 评估模式下默认静默
        } else if (arg == "--core-id" && i + 1 < argc) {
            core_id = std::stoi(argv[++i]);
        } else if (arg == "--score-thresh" && i + 1 < argc) {
            score_thresh = std::stof(argv[++i]);
        } else if (arg == "--nms-thresh" && i + 1 < argc) {
            nms_thresh = std::stof(argv[++i]);
        } else if (arg == "--profile" && i + 1 < argc) {
            profile = (std::string(argv[++i]) == "true" || std::string(argv[++i]) == "1");
        } else if (arg == "--silent" && i + 1 < argc) {
            silent_mode = (std::string(argv[++i]) == "true" || std::string(argv[++i]) == "1");
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --det-model NAME   Detection model base name (default: models/det_500m_float16.nb)" << std::endl;
            std::cout << "  --rec-model NAME   Recognition model base name (default: models/w600k_mbf_float16.nb)" << std::endl;
            std::cout << "  --image PATH      Input image path (for single image mode)" << std::endl;
            std::cout << "  --eval            Enable LFW evaluation mode" << std::endl;
            std::cout << "  --lfw-data DIR    LFW dataset directory (default: ./dataset/LFW)" << std::endl;
            std::cout << "  --lfw-pairs FILE  LFW pairs.txt file (default: ./dataset/lfw_pairs.txt)" << std::endl;
            std::cout << "  --core-id ID      NPU core ID (0 or 1)" << std::endl;
            std::cout << "  --score-thresh F  Score threshold (default: 0.5)" << std::endl;
            std::cout << "  --nms-thresh F    NMS threshold (default: 0.4)" << std::endl;
            std::cout << "  --profile BOOL    Enable performance profiling (default: true)" << std::endl;
            std::cout << "  --silent BOOL     Enable silent mode (no progress output) (default: false)" << std::endl;
            std::cout << "  --help            Show this help" << std::endl;
            return 0;
        }
    }
    
    
    // 检查必要的文件
    if (!fs::exists(det_model_path)) {
        std::cerr << "Error: Detection model file not found: " << det_model_path << std::endl;
        return -1;
    }
    
    if (!fs::exists(rec_model_path)) {
        std::cerr << "Error: Recognition model file not found: " << rec_model_path << std::endl;
        return -1;
    }
    
    // 根据模式执行不同的逻辑
    if (eval_mode) {
        // ==================== LFW评估模式 ====================
        std::cout << "========================================" << std::endl;
        std::cout << "   LFW Dataset Evaluation Mode" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Detection model: " << det_model_path << std::endl;
        std::cout << "Recognition model: " << rec_model_path << std::endl;
        std::cout << "LFW data directory: " << lfw_data_dir << std::endl;
        std::cout << "LFW pairs file: " << lfw_pairs_file << std::endl;
        std::cout << "Score threshold: " << score_thresh << std::endl;
        std::cout << "Profiling: " << (profile ? "enabled" : "disabled") << std::endl;
        std::cout << "Silent mode: " << (silent_mode ? "enabled" : "disabled") << std::endl;
        std::cout << "========================================" << std::endl;
        
        // 检查LFW数据集
        if (!fs::exists(lfw_data_dir)) {
            std::cerr << "Error: LFW data directory not found: " << lfw_data_dir << std::endl;
            return -1;
        }
        
        if (!fs::exists(lfw_pairs_file)) {
            std::cerr << "Error: LFW pairs file not found: " << lfw_pairs_file << std::endl;
            return -1;
        }
        
        std::cout << "Reading LFW pairs file..." << std::endl;
        auto pairs = read_lfw_pairs(lfw_pairs_file);
        if (pairs.empty()) {
            std::cerr << "Error: No valid pairs found in " << lfw_pairs_file << std::endl;
            return -1;
        }
        std::cout << "Loaded " << pairs.size() << " evaluation pairs" << std::endl;
        
        int pos_pairs = std::count_if(pairs.begin(), pairs.end(), [](const EvalPair& p) { return p.is_same; });
        int neg_pairs = pairs.size() - pos_pairs;
        std::cout << "Positive pairs (same person): " << pos_pairs << std::endl;
        std::cout << "Negative pairs (different person): " << neg_pairs << std::endl;
        
        // 初始化检测器和识别器
        SCRFD::Config det_config;
        det_config.model_path = det_model_path;
        det_config.core_id = core_id;
        det_config.score_threshold = score_thresh;
        det_config.nms_threshold = nms_thresh;
        det_config.profile = profile;
        det_config.debug_level = 0; 
        
        SCRFD detector(det_config);
        if (detector.init(det_model_path) != 0) {
            std::cerr << "Error: Failed to initialize SCRFD detector" << std::endl;
            return -1;
        }
        
        FaceRecognizer::Config recog_config;
        recog_config.model_path = rec_model_path;
        recog_config.core_id = core_id;
        recog_config.profile = profile;
        
        FaceRecognizer recognizer(recog_config);
        if (recognizer.init(rec_model_path) != 0) {
            std::cerr << "Error: Failed to initialize Face Recognizer" << std::endl;
            detector.deinit();
            return -1;
        }

        TimeStamp det_ts,rec_ts;
        detector.enableProfile(&det_ts);
        recognizer.enableProfile(&rec_ts);


        
        std::cout << "Models initialized in " << std::fixed << std::setprecision(2) << std::endl;
        std::cout << "Detector and Recognizer initialized successfully." << std::endl;
        
        // 收集所有唯一的图片路径
        std::unordered_map<std::string, std::vector<float>> feature_cache;
        std::vector<std::string> all_images;
        for (const auto& pair : pairs) {
            all_images.push_back(pair.img1_path);
            all_images.push_back(pair.img2_path);
        }
        
        // 去重
        std::sort(all_images.begin(), all_images.end());
        all_images.erase(std::unique(all_images.begin(), all_images.end()), all_images.end());
        
        std::cout << "Total unique images: " << all_images.size() << std::endl;
        std::cout << "Starting feature extraction..." << std::endl;
        
        // 提取所有图片的特征
        int processed_count = 0;
        int failed_count = 0;
        
        for (size_t i = 0; i < all_images.size(); ++i) {
            const auto& img_rel_path = all_images[i];
            std::string img_full_path = lfw_data_dir + "/" + img_rel_path;
            
            if (!fs::exists(img_full_path)) {
                std::cerr << "Warning: Image not found: " << img_full_path << std::endl;
                failed_count++;
                continue;
            }
            
            det_ts.start();
            cv::Mat image = cv::imread(img_full_path, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
            det_ts.stop();
            det_ts.time_accumulation("imread_time");
            if (image.empty()) {
                std::cerr << "Warning: Failed to read image: " << img_full_path << std::endl;
                failed_count++;
                continue;
            }
            
            // 检测人脸
            auto detections = detector.detect(image);
            if (detections.empty()) {
                failed_count++;
                continue;
            }
            
            // 对齐第一个人脸
            cv::Mat aligned_face = align_face(image, detections[0].landmarks);
            if (aligned_face.empty()) {
                std::cerr << "Warning: Face alignment failed for: " << img_rel_path << std::endl;
                failed_count++;
                continue;
            }
            
            // 提取特征
            std::vector<float> feature = recognizer.extract_feature(aligned_face);
            if (feature.empty()) {
                std::cerr << "Warning: Feature extraction failed for: " << img_rel_path << std::endl;
                failed_count++;
                continue;
            }
            
            
            feature_cache[img_rel_path] = feature;
            processed_count++;
            
            // 显示进度
            if ((processed_count % 100 == 0) || (i == all_images.size() - 1)) {
                std::cout << "  Processed " << processed_count << " / " << all_images.size() 
                          << " images " << std::endl;
            }
        }
        
        std::cout << "Feature extraction completed in " << std::fixed << std::setprecision(2)  << std::endl;
        std::cout << "  Successfully processed: " << processed_count << " images" << std::endl;
        
        // 计算相似度
        std::cout << "Computing similarities..." << std::endl;
        
        std::vector<float> similarities;
        std::vector<int> labels;
        int valid_pairs = 0;
        
        for (size_t i = 0; i < pairs.size(); ++i) {
            const auto& pair = pairs[i];
            
            if (feature_cache.find(pair.img1_path) == feature_cache.end() ||
                feature_cache.find(pair.img2_path) == feature_cache.end()) {
                continue; // 跳过特征提取失败的图片对
            }
            
            const auto& feat1 = feature_cache[pair.img1_path];
            const auto& feat2 = feature_cache[pair.img2_path];
            
            float sim = cosine_similarity(feat1, feat2);
            similarities.push_back(sim);
            labels.push_back(pair.is_same ? 1 : 0);
            valid_pairs++;
        }
        
        std::cout << "Valid evaluation pairs: " << valid_pairs << " / " << pairs.size() << std::endl;
        
        // 评估性能
        EvalResult::EvalStats eval_stats;
        eval_stats.total_pairs_verified = valid_pairs;
        
        EvalResult result = evaluate_verification(similarities, labels, eval_stats);
        result.total_pairs = pairs.size();
        
        // 输出结果
        std::cout << "\n========================================" << std::endl;
        std::cout << "   LFW Evaluation Results" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total pairs: " << result.total_pairs << std::endl;
        std::cout << "Valid pairs: " << result.valid_pairs << std::endl;
        std::cout << "Valid positive pairs: " << result.valid_positive_pairs << std::endl;
        std::cout << "Valid negative pairs: " << result.valid_negative_pairs << std::endl;
        std::cout << "Best threshold: " << std::fixed << std::setprecision(6) << result.best_threshold << std::endl;
        std::cout << "Best accuracy: " << std::fixed << std::setprecision(4) << (result.best_accuracy * 100) << "%" << std::endl;
        
        // 输出模型性能统计
        if (profile) {
            std::cout << "\n" << std::string(50, '-') << std::endl;
            std::cout << "SCRFD Detector" << std::endl;

            std::cout << "average imread time: " << det_ts.time_map_lab["imread_time"] / detector.total_inferences_ << "ms" << std::endl;
            std::cout << "average preprocess time: " << det_ts.time_map_lab["preprocess_time"] / detector.total_inferences_ << "ms" << std::endl;
            std::cout << "average inference time: " <<  det_ts.time_map_lab["inference_time"] / detector.total_inferences_ << "ms" << std::endl;
            std::cout << "average postprocess time: " << det_ts.time_map_lab["postprocess_time"] / detector.total_inferences_ << "ms" << std::endl;
            std::cout << std::string(50, '-') << std::endl;
            
            std::cout << "\n" << std::string(50, '-') << std::endl;
            std::cout << "FaceRecognizer" << std::endl;
            std::cout << "average imread time: " << rec_ts.time_map_lab["imread_time"] / recognizer.total_inferences_ << "ms" << std::endl;
            std::cout << "average preprocess time: " << rec_ts.time_map_lab["preprocess_time"] / recognizer.total_inferences_ << "ms" << std::endl;
            std::cout << "average inference time: " <<  rec_ts.time_map_lab["inference_time"] / recognizer.total_inferences_ << "ms" << std::endl;
            std::cout << "average postprocess time: " << rec_ts.time_map_lab["postprocess_time"] / recognizer.total_inferences_ << "ms" << std::endl;

            std::cout << std::string(50, '-') << std::endl;
            
        }
        
        // 清理资源
        recognizer.deinit();
        detector.deinit();
        
    } else {
        // ==================== 单图检测模式 ====================
        // 确保在单图模式下至少需要指定图片
        if (image_path.empty()) {
            std::cerr << "Error: In single image mode, --image parameter is required" << std::endl;
            std::cerr << "  Use --eval for LFW evaluation mode" << std::endl;
            return -1;
        }
        
        if (!fs::exists(image_path)) {
            std::cerr << "Error: Image file not found: " << image_path << std::endl;
            return -1;
        }
        
        // 读取图片
        TimeStamp det_ts,rec_ts;
        
        det_ts.start();
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
        det_ts.stop();
        det_ts.time_accumulation("imread_time");

        if (image.empty()) {
            std::cerr << "Error: Failed to read image: " << image_path << std::endl;
            return -1;
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "   SCRFD Face Detection Test" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Detection model: " << det_model_path << std::endl;
        std::cout << "Image: " << image_path << " (" << image.cols << "x" << image.rows << ")" << std::endl;
        std::cout << "Core ID: " << core_id << std::endl;
        std::cout << "Score threshold: " << score_thresh << std::endl;
        std::cout << "NMS threshold: " << nms_thresh << std::endl;
        std::cout << "Profiling: " << (profile ? "enabled" : "disabled") << std::endl;
        std::cout << "========================================" << std::endl;
        
        // 配置检测器
        SCRFD::Config config;
        config.model_path = det_model_path;  
        config.core_id = core_id;
        config.score_threshold = score_thresh;
        config.nms_threshold = nms_thresh;
        config.profile = profile;
        config.debug_level = 0; 
        
        // 创建检测器
        SCRFD detector(config);
        detector.enableProfile(&det_ts);
        // 初始化
        
        std::cout << "Initializing SCRFD detector..." << std::endl;
        if (detector.init(det_model_path) != 0) {  
            std::cerr << "Error: Failed to initialize SCRFD detector" << std::endl;
            return -1;
        }
        
        std::cout << "Detector initialized successfully in " << std::fixed << std::setprecision(2) << std::endl;
        FaceRecognizer::Config recog_config;
        recog_config.model_path = rec_model_path;
        recog_config.core_id = core_id;
        recog_config.profile = profile;
        
        FaceRecognizer recognizer(recog_config);
        if (recognizer.init(rec_model_path) != 0) {
            std::cerr << "Error: Failed to initialize Face Recognizer" << std::endl;
            detector.deinit();
            return -1;
        }

        detector.enableProfile(&det_ts);
        recognizer.enableProfile(&rec_ts);


        // 执行检测
        std::cout << "Running face detection..." << std::endl;
        auto results = detector.detect(image);
        
        std::cout << "Detection completed." << std::endl;
        std::cout << "Found " << results.size() << " faces." << std::endl;
        // 对齐第一个人脸
        cv::Mat aligned_face = align_face(image, results[0].landmarks);
        if (aligned_face.empty()) {
            std::cerr << "Warning: Face alignment failed for " << std::endl;
        }
        
        // 提取特征
        std::vector<float> feature = recognizer.extract_feature(aligned_face);
        if (feature.empty()) {
            std::cerr << "Warning: Feature extraction failed " << std::endl;
        }
        
        
        // 打印结果
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& det = results[i];
            std::cout << "  Face " << i+1 << ":" << std::endl;
            std::cout << "    Score: " << std::fixed << std::setprecision(4) << det.score << std::endl;
            std::cout << "    BBox: [" << (int)det.bbox.x1 << ", " << (int)det.bbox.y1 
                      << ", " << (int)det.bbox.x2 << ", " << (int)det.bbox.y2 << "]" << std::endl;
            std::cout << "    Size: " << (int)det.bbox.width() << "x" << (int)det.bbox.height() 
                      << " (aspect: " << std::fixed << std::setprecision(2) 
                      << det.bbox.width()/det.bbox.height() << ")" << std::endl;
        }
        
        // 输出性能统计
        if (profile) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            std::cout << "SCRFD Detector" << std::endl;
            std::cout << "imread time: " << detector.m_ts->time_map_lab["imread_time"] << "ms" << std::endl;
            std::cout << "preprocess time: " << detector.m_ts->time_map_lab["preprocess_time"] << "ms" << std::endl;
            std::cout << "inference time: " << detector.m_ts->time_map_lab["inference_time"] << "ms" << std::endl;
            std::cout << "postprocess time: " << detector.m_ts->time_map_lab["postprocess_time"] << "ms" << std::endl;
            std::cout << std::string(50, '=') << std::endl;
            std::cout << "MBF Recognizer" << std::endl;
            std::cout << "imread time: " << recognizer.m_ts->time_map_lab["imread_time"] << "ms" << std::endl;
            std::cout << "preprocess time: " << recognizer.m_ts->time_map_lab["preprocess_time"] << "ms" << std::endl;
            std::cout << "inference time: " << recognizer.m_ts->time_map_lab["inference_time"] << "ms" << std::endl;
            std::cout << "postprocess time: " << recognizer.m_ts->time_map_lab["postprocess_time"] << "ms" << std::endl;
            std::cout << std::string(50, '=') << std::endl;
        }
        
        // 保存结果图片
        if (!results.empty()) {
            cv::Mat result_image = detector.draw(image, results);
            std::string output_path = "detection_result.jpg";
            if (cv::imwrite(output_path, result_image)) {
                std::cout << "Result saved to: " << output_path << std::endl;
            } else {
                std::cerr << "Failed to save result image" << std::endl;
            }
        }
        
        // 清理资源
        recognizer.deinit();
        detector.deinit();
        
        std::cout << "========================================" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    return 0;
}