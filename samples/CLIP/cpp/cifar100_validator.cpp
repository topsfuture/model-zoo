#include "clip.hpp"
#include "tokenizer/tokenizer.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>


class CIFAR100Validator {
private:
    CLIP& clip_model_;
    CLIPTokenizer tokenizer_;
    std::vector<std::string> fine_class_names_;
    std::vector<std::string> coarse_class_names_;
    std::vector<std::vector<int>> text_features_ids_;
    std::shared_ptr<ta_runtime_context> nnrt_ctx_text_;
    std::shared_ptr<ta_runtime_context> nnrt_ctx_image_;
    bool use_coarse_labels_;  // 新增：控制使用细粒度还是粗粒度标签
    
public:
    CIFAR100Validator(CLIP& clip, 
                     const std::shared_ptr<ta_runtime_context>& ctx_text,
                     const std::shared_ptr<ta_runtime_context>& ctx_image,
                     bool use_coarse_labels = false)  // 新增参数
        : clip_model_(clip), nnrt_ctx_text_(ctx_text), nnrt_ctx_image_(ctx_image),
          use_coarse_labels_(use_coarse_labels) {
        
        // CIFAR-100 细粒度类别名称（100个）
        fine_class_names_ = {
            "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle", 
            "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", 
            "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", 
            "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", 
            "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", 
            "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
            "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse",
            "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear",
            "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine", 
            "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
            "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
            "spider", "squirrel", "streetcar", "sunflower", "sweet pepper", "table",
            "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
            "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman",
            "worm"
        };
        
        // CIFAR-100 粗粒度类别名称（20个）
        coarse_class_names_ = {
            "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
            "household electrical devices", "household furniture", "insects", "large carnivores",
            "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
            "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
            "trees", "vehicles 1", "vehicles 2"
        };
        
        // 根据选择的标签类型生成文本特征
        size_t max_token_len = clip_model_.get_token_len();
        
        if (use_coarse_labels_) {
            // 使用粗粒度标签
            std::cout << "Using 20 coarse labels for validation" << std::endl;
            for (const auto& class_name : coarse_class_names_) {
                std::vector<int> token_ids;
                get_text_features(class_name, token_ids, max_token_len);
                text_features_ids_.push_back(token_ids);
            }
        } else {
            // 使用细粒度标签
            std::cout << "Using 100 fine labels for validation" << std::endl;
            for (const auto& class_name : fine_class_names_) {
                std::vector<int> token_ids;
                get_text_features(class_name, token_ids, max_token_len);
                text_features_ids_.push_back(token_ids);
            }
        }
    }
    
    void get_text_features(const std::string& text, std::vector<int>& ids, size_t max_token_len) {
        CLIPTokenizer tokenizer;
        ids = tokenizer.tokenize(text, nullptr, max_token_len, true);
    }
    
    // 读取CIFAR-100二进制文件
    bool read_cifar100_batch(const std::string& filename, 
                            std::vector<cv::Mat>& images, 
                            std::vector<int>& fine_labels,
                            std::vector<int>& coarse_labels) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open CIFAR-100 file: " << filename << std::endl;
            return false;
        }
        
        const int image_size = 32 * 32 * 3;
        const int record_size = 1 + 1 + image_size; // coarse + fine + pixels
        
        char buffer[record_size];
        
        while (file.read(buffer, record_size)) {
            // 读取标签
            uint8_t coarse_label = static_cast<uint8_t>(buffer[0]);
            uint8_t fine_label = static_cast<uint8_t>(buffer[1]);
            
            // 读取图像数据 (BGR格式保存到 Mat)
            cv::Mat image(32, 32, CV_8UC3);
            int pixel_index = 2;
            
            // CIFAR-100数据布局: 先是所有R通道，然后所有G通道，最后所有B通道
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < 32; ++y) {
                    for (int x = 0; x < 32; ++x) {
                        image.at<cv::Vec3b>(y, x)[2-c] = static_cast<uint8_t>(buffer[pixel_index++]);
                    }
                }
            }
            images.push_back(image);
            fine_labels.push_back(fine_label);
            coarse_labels.push_back(coarse_label);
        }
        
        file.close();
        return true;
    }
    
    // 验证精度
    void validate_accuracy(const std::string& cifar100_test_path, int max_samples = -1) {
        std::vector<cv::Mat> test_images;
        std::vector<int> fine_labels;
        std::vector<int> coarse_labels;
        
        // 创建全局计时器
        TimeStamp ts;
        clip_model_.enableProfile(&ts);
        
        // 读取测试数据
        if (!read_cifar100_batch(cifar100_test_path, test_images, fine_labels, coarse_labels)) {
            return;
        }
        
        if (max_samples > 0 && max_samples < test_images.size()) {
            test_images.resize(max_samples);
            fine_labels.resize(max_samples);
            coarse_labels.resize(max_samples);
        }
        
        std::cout << "Loaded " << test_images.size() << " test images" << std::endl;
        
        // 提取所有文本特征
        std::vector<std::vector<float>> text_features;
        for (size_t i = 0; i < text_features_ids_.size(); ++i) {
            std::string class_name = use_coarse_labels_ ? coarse_class_names_[i] : fine_class_names_[i];
            std::cout << "Extracting text feature for class: " << class_name << std::endl;
            std::vector<float> text_feature = clip_model_.encode_text(text_features_ids_[i], nnrt_ctx_text_);
            text_features.push_back(text_feature);
        }
        
        std::cout << "Extracted text features for " << text_features.size() << " classes" << std::endl;
        
        // 进行精度验证
        int correct_top1 = 0;
        int correct_top5 = 0;
        int total = test_images.size();
        
        for (size_t i = 0; i < test_images.size(); ++i) {
            if (i % 100 == 0) {
                std::cout << "Processing image " << i << "/" << test_images.size() << std::endl;
            }
            
            // 提取图像特征
            ts.start();
            std::vector<float> image_feature = clip_model_.encode_image_memory(test_images[i], nnrt_ctx_image_);
            ts.time_accumulation("encode_image_time");
            
            // 计算与所有文本特征的相似度
            std::vector<float> similarities;
            for (const auto& text_feature : text_features) {
                float similarity = 100.0f * std::inner_product(
                    image_feature.begin(), image_feature.end(),
                    text_feature.begin(), 0.0f);
                similarities.push_back(similarity);
            }
            
            // 获取top-1和top-5预测
            auto [top1_values, top1_indices] = clip_model_.topk(similarities, 1);
            auto [top5_values, top5_indices] = clip_model_.topk(similarities, 5);
            
            // 根据选择的标签类型确定真实标签
            int true_label = use_coarse_labels_ ? coarse_labels[i] : fine_labels[i];
            
            // 检查top-1准确率
            if (top1_indices[0] == true_label) {
                correct_top1++;
            }
            
            // 检查top-5准确率
            bool top5_correct = false;
            for (int idx : top5_indices) {
                if (idx == true_label) {
                    top5_correct = true;
                    break;
                }
            }
            if (top5_correct) {
                correct_top5++;
            }
            
            // 每100张图片打印一次进度
            if (i % 100 == 0) {
                std::cout << "Progress: " << i << "/" << total 
                         << " | Top-1: " << std::fixed << std::setprecision(2) 
                         << (100.0 * correct_top1 / (i + 1)) << "%"
                         << " | Top-5: " << (100.0 * correct_top5 / (i + 1)) << "%" << std::endl;
            }
        }
        
        // 计算最终准确率
        float top1_accuracy = 100.0f * correct_top1 / total;
        float top5_accuracy = 100.0f * correct_top5 / total;
        
        std::cout << "\n=== CIFAR-100 Validation Results ===" << std::endl;
        std::cout << "Label type: " << (use_coarse_labels_ ? "20 coarse labels" : "100 fine labels") << std::endl;
        std::cout << "Total images: " << total << std::endl;
        std::cout << "Top-1 Accuracy: " << std::fixed << std::setprecision(2) 
                 << top1_accuracy << "%" << std::endl;
        std::cout << "Top-5 Accuracy: " << top5_accuracy << "%" << std::endl;
        
        // 保存详细结果到文件
        std::string result_file = use_coarse_labels_ ? 
            "cifar100_coarse_validation_results.txt" : "cifar100_fine_validation_results.txt";
        save_detailed_results(result_file, top1_accuracy, top5_accuracy, total, use_coarse_labels_);
    }
    
private:
    void save_detailed_results(const std::string& filename, 
                              float top1_acc, float top5_acc, int total, bool use_coarse) {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "CIFAR-100 Validation Results\n";
            file << "============================\n";
            file << "Label type: " << (use_coarse ? "20 coarse labels" : "100 fine labels") << "\n";
            file << "Total test images: " << total << "\n";
            file << "Top-1 Accuracy: " << std::fixed << std::setprecision(2) 
                 << top1_acc << "%\n";
            file << "Top-5 Accuracy: " << top5_acc << "%\n";
            file << "Validation date: " << __DATE__ << " " << __TIME__ << "\n";
            
            // 保存类别名称
            file << "\nClass Names:\n";
            const auto& class_names = use_coarse ? coarse_class_names_ : fine_class_names_;
            for (size_t i = 0; i < class_names.size(); ++i) {
                file << i << ": " << class_names[i] << "\n";
            }
            
            file.close();
            std::cout << "Detailed results saved to: " << filename << std::endl;
        }
    }
};


int main(int argc, char* argv[]) {
    // 参数解析
    if (argc < 6) {
        std::cout << "Usage: " << argv[0] 
                 << " <image_model> <text_model> <text_projection_path> <cifar100_test_batch> [max_samples] [--coarse]" << std::endl;
        std::cout << "Example (fine labels): " << argv[0] 
                 << " ./models/clip.nb ./models/clip_text.nb ./text_projection.npy ./cifar100_test.bin 1000" << std::endl;
        std::cout << "Example (coarse labels): " << argv[0] 
                 << " ./models/clip.nb ./models/clip_text.nb ./text_projection.npy ./cifar100_test.bin 1000 --coarse" << std::endl;
        return -1;
    }

    /* 
    // 使用细粒度标签（100类）
    ./cifar100_validator ./models/clip_image_float16_f32o.nb ./models/clip_text_float16_f32o.nb ./text_projection_512_512.npy ./test.bin 1000
    
    // 使用粗粒度标签（20类）
    ./cifar100_validator ./models/clip_image_float16_f32o.nb ./models/clip_text_float16_f32o.nb ./text_projection_512_512.npy ./test.bin 1000 --coarse
    */
    
    std::string image_model = argv[1];
    std::string text_model = argv[2];
    std::string text_projection_path = argv[3];
    std::string cifar100_test_path = argv[4];
    int max_samples = (argc > 5) ? std::atoi(argv[5]) : -1;
    
    // 检查是否使用粗粒度标签
    bool use_coarse_labels = false;
    if (argc > 6 && std::string(argv[6]) == "--coarse") {
        use_coarse_labels = true;
    }
    
    // 创建context
    std::shared_ptr<ta_runtime_context> nnrt_ctx_text = std::make_shared<ta_runtime_context>(0);
    std::shared_ptr<ta_runtime_context> nnrt_ctx_image = std::make_shared<ta_runtime_context>(0);
    
    // 初始化CLIP模型
    CLIP clip;
    if (clip.init(image_model, text_model, text_projection_path, nnrt_ctx_text, nnrt_ctx_image) != 0) {
        std::cerr << "Model initialization failed" << std::endl;
        return -1;
    }
    
    // 创建验证器并运行验证
    CIFAR100Validator validator(clip, nnrt_ctx_text, nnrt_ctx_image, use_coarse_labels);
    validator.validate_accuracy(cifar100_test_path, max_samples);
    
    // 清理资源
    clip.deinit(nnrt_ctx_image, nnrt_ctx_text);
    
    return 0;
}