// cifar100_validator.hpp
#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "clip.hpp"
#include "tokenizer/bert_tokenizer.hpp"
#include "tokenizer/tokenizer.hpp"

class CIFAR100_Validator {
private:
    bool is_chinese_;
    CLIP& clip_model_;
    std::shared_ptr<ta_runtime_context> nnrt_ctx_text_;
    std::shared_ptr<ta_runtime_context> nnrt_ctx_image_;
    
    bool use_coarse_labels_;  // 仅英文模式有效
    std::vector<std::string> fine_class_names_;    // 英文细粒度标签
    std::vector<std::string> coarse_class_names_;  // 英文粗粒度标签
    std::vector<std::string> chinese_labels_;      // 中文标签
    
    std::vector<std::function<std::string(const std::string&)>> chinese_templates;
    std::vector<std::function<std::string(const std::string&)>> english_templates;
    
    CLIPTokenizer* en_tokenizer_;  
    BertTokenizer* cn_tokenizer_;  
    
    const int IMAGE_SIZE = 224;
    const float MEAN_R = 0.48145466f;
    const float MEAN_G = 0.4578275f;
    const float MEAN_B = 0.40821073f;
    const float STD_R = 0.26862954f;
    const float STD_G = 0.26130258f;
    const float STD_B = 0.27577711f;
    const int CONTEXT_LENGTH = 52;
    
    void initialize_chinese_templates();
    void initialize_english_templates();
    std::vector<std::string> load_chinese_labels(const std::string& dataset_root) const;
    
    // 文本特征生成
    std::vector<int> tokenize_text(const std::string& text, size_t max_token_len);

    
    // 数据集加载
    bool read_cifar100_binary(const std::string& filename, 
                             std::vector<cv::Mat>& images, 
                             std::vector<int>& fine_labels,
                             std::vector<int>& coarse_labels);
    
    bool load_test_dataset_directory(const std::string& dataset_root, 
                                    std::vector<std::string>& image_paths, 
                                    std::vector<int>& labels);
    
    // 图片处理
    cv::Mat preprocess_image(const cv::Mat& img);
    std::vector<float> normalize_features(const std::vector<float>& features);
    void calculate_topk_accuracy(const std::vector<std::vector<float>>& image_features,
                                const std::vector<std::vector<float>>& text_features,
                                const std::vector<int>& true_labels,
                                int k, float& topk_accuracy);
    
    // 结果保存
    void save_detailed_results(const std::string& filename, 
                              float top1_acc, float top5_acc, 
                              int total, bool use_coarse, bool is_chinese);
    
public:
    CIFAR100_Validator(CLIP& clip,
                      const std::shared_ptr<ta_runtime_context>& ctx_text,
                      const std::shared_ptr<ta_runtime_context>& ctx_image,
                      bool is_chinese,
                      const std::string& vocab_path,
                      bool use_coarse_labels = false);
    
    ~CIFAR100_Validator();
    
    bool load_labels(const std::string& dataset_root);
    
    void validate_accuracy(const std::string& dataset_path, int max_samples = -1);
    
    size_t get_num_classes() const {
        if (is_chinese_) {
            return chinese_labels_.size();
        } else {
            return use_coarse_labels_ ? coarse_class_names_.size() : fine_class_names_.size();
        }
    }
    
    std::string get_label_type() const {
        if (is_chinese_) {
            return "100 Chinese fine labels";
        } else {
            return use_coarse_labels_ ? "20 English coarse labels" : "100 English fine labels";
        }
    }
};