#include "cifar100_validator.hpp"

// 构造函数
CIFAR100_Validator::CIFAR100_Validator(CLIP& clip,
                                     const std::shared_ptr<ta_runtime_context>& ctx_text,
                                     const std::shared_ptr<ta_runtime_context>& ctx_image,
                                     bool is_chinese,
                                     const std::string& vocab_path,
                                     bool use_coarse_labels)
    : clip_model_(clip), nnrt_ctx_text_(ctx_text), nnrt_ctx_image_(ctx_image),
      is_chinese_(is_chinese), use_coarse_labels_(use_coarse_labels), 
      en_tokenizer_(nullptr), cn_tokenizer_(nullptr) {
    
    // 初始化中文模板
    if (is_chinese_) {
        initialize_chinese_templates();
        cn_tokenizer_ = new BertTokenizer(vocab_path);
        use_coarse_labels_ = false;  // 中文模式不支持粗粒度标签
    } else {
        initialize_english_templates();
        en_tokenizer_ = new CLIPTokenizer();

        // 初始化英文细粒度标签（100个）
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
        
        // 初始化英文粗粒度标签（20个）
        coarse_class_names_ = {
            "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
            "household electrical devices", "household furniture", "insects", "large carnivores",
            "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
            "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
            "trees", "vehicles 1", "vehicles 2"
        };
    }
}

// 析构函数
CIFAR100_Validator::~CIFAR100_Validator() {
    // 清理分词器
    if (en_tokenizer_) {
        delete en_tokenizer_;
    }
    if (cn_tokenizer_) {
        delete cn_tokenizer_;
    }
}



// 初始化中文模板
void CIFAR100_Validator::initialize_chinese_templates() {
    chinese_templates = {
        [](const std::string& c) { return c + "的照片"; },
        [](const std::string& c) { return "质量差的" + c + "的照片"; },
        [](const std::string& c) { return "许多" + c + "的照片"; },
        [](const std::string& c) { return c + "的雕塑"; },
        [](const std::string& c) { return "难以看到" + c + "的照片"; },
        [](const std::string& c) { return c + "的低分辨率照片"; },
        [](const std::string& c) { return c + "的渲染"; },
        [](const std::string& c) { return "涂鸦" + c; },
        [](const std::string& c) { return c + "的糟糕照片"; },
        [](const std::string& c) { return c + "的裁剪照片"; },
        [](const std::string& c) { return c + "的纹身"; },
        [](const std::string& c) { return c + "的刺绣照片"; },
        [](const std::string& c) { return "很难看到" + c + "的照片"; },
        [](const std::string& c) { return c + "的明亮照片"; },
        [](const std::string& c) { return "一张干净的" + c + "的照片"; },
        [](const std::string& c) { return "一张包含" + c + "的照片"; },
        [](const std::string& c) { return c + "的深色照片"; },
        [](const std::string& c) { return c + "的手绘画"; },
        [](const std::string& c) { return "我的" + c + "的照片"; },
        [](const std::string& c) { return "不自然的" + c + "的照片"; },
        [](const std::string& c) { return "一张酷的" + c + "的照片"; },
        [](const std::string& c) { return c + "的特写照片"; },
        [](const std::string& c) { return c + "的黑白照片"; },
        [](const std::string& c) { return "一幅" + c + "的画"; },
        [](const std::string& c) { return "一幅" + c + "的绘画"; },
        [](const std::string& c) { return "一张" + c + "的像素照片"; },
        [](const std::string& c) { return c + "的雕像"; },
        [](const std::string& c) { return "一张" + c + "的明亮照片"; },
        [](const std::string& c) { return c + "的裁剪照片"; },
        [](const std::string& c) { return "人造的" + c + "的照片"; },
        [](const std::string& c) { return "一张关于" + c + "的照片"; },
        [](const std::string& c) { return "损坏的" + c + "的jpeg照片"; },
        [](const std::string& c) { return c + "的模糊照片"; },
        [](const std::string& c) { return c + "的相片"; },
        [](const std::string& c) { return "一张" + c + "的好照片"; },
        [](const std::string& c) { return c + "的渲染照"; },
        [](const std::string& c) { return "视频游戏中的" + c; },
        [](const std::string& c) { return "一张" + c + "的照片"; },
        [](const std::string& c) { return c + "的涂鸦"; },
        [](const std::string& c) { return c + "的近距离照片"; },
        [](const std::string& c) { return c + "的折纸"; },
        [](const std::string& c) { return c + "在视频游戏中"; },
        [](const std::string& c) { return c + "的草图"; },
        [](const std::string& c) { return c + "的涂鸦照"; },
        [](const std::string& c) { return c + "的折纸形状"; },
        [](const std::string& c) { return "低分辨率的" + c + "的照片"; },
        [](const std::string& c) { return "玩具" + c; },
        [](const std::string& c) { return c + "的副本"; },
        [](const std::string& c) { return c + "的干净的照片"; },
        [](const std::string& c) { return "一张大" + c + "的照片"; },
        [](const std::string& c) { return c + "的重现"; },
        [](const std::string& c) { return "一张漂亮的" + c + "的照片"; },
        [](const std::string& c) { return "一张奇怪的" + c + "的照片"; },
        [](const std::string& c) { return "模糊的" + c + "的照片"; },
        [](const std::string& c) { return "卡通" + c; },
        [](const std::string& c) { return c + "的艺术作品"; },
        [](const std::string& c) { return c + "的素描"; },
        [](const std::string& c) { return "刺绣" + c; },
        [](const std::string& c) { return c + "的像素照"; },
        [](const std::string& c) { return c + "的拍照"; },
        [](const std::string& c) { return c + "的损坏的照片"; },
        [](const std::string& c) { return "高质量的" + c + "的照片"; },
        [](const std::string& c) { return "毛绒玩具" + c; },
        [](const std::string& c) { return "漂亮的" + c + "的照片"; },
        [](const std::string& c) { return "小" + c + "的照片"; },
        [](const std::string& c) { return "照片是奇怪的" + c; },
        [](const std::string& c) { return "漫画" + c; },
        [](const std::string& c) { return c + "的艺术照"; },
        [](const std::string& c) { return c + "的图形"; },
        [](const std::string& c) { return "大" + c + "的照片"; },
        [](const std::string& c) { return "黑白的" + c + "的照片"; },
        [](const std::string& c) { return c + "毛绒玩具"; },
        [](const std::string& c) { return "一张" + c + "的深色照片"; },
        [](const std::string& c) { return c + "的摄影图"; },
        [](const std::string& c) { return c + "的涂鸦照"; },
        [](const std::string& c) { return "玩具形状的" + c; },
        [](const std::string& c) { return "拍了" + c + "的照片"; },
        [](const std::string& c) { return "酷酷的" + c + "的照片"; },
        [](const std::string& c) { return "照片里的小" + c; },
        [](const std::string& c) { return c + "的刺青"; },
        [](const std::string& c) { return c + "的可爱的照片"; },
        [](const std::string& c) { return "一张" + c + "可爱的照片"; },
        [](const std::string& c) { return c + "可爱图片"; },
        [](const std::string& c) { return c + "酷炫图片"; },
        [](const std::string& c) { return "一张" + c + "的酷炫的照片"; },
        [](const std::string& c) { return "一张" + c + "的酷炫图片"; },
        [](const std::string& c) { return "这是" + c; },
        [](const std::string& c) { return c + "的好看照片"; },
        [](const std::string& c) { return "一张" + c + "的好看的图片"; },
        [](const std::string& c) { return c + "的好看图片"; },
        [](const std::string& c) { return c + "的照片。"; },
        [](const std::string& c) { return "质量差的" + c + "的照片。"; },
        [](const std::string& c) { return "许多" + c + "的照片。"; },
        [](const std::string& c) { return c + "的雕塑。"; },
        [](const std::string& c) { return "难以看到" + c + "的照片。"; },
        [](const std::string& c) { return c + "的低分辨率照片。"; },
        [](const std::string& c) { return c + "的渲染。"; },
        [](const std::string& c) { return "涂鸦" + c + "。"; },
        [](const std::string& c) { return c + "的糟糕照片。"; },
        [](const std::string& c) { return c + "的裁剪照片。"; },
        [](const std::string& c) { return c + "的纹身。"; },
        [](const std::string& c) { return c + "的刺绣照片。"; },
        [](const std::string& c) { return "很难看到" + c + "的照片。"; },
        [](const std::string& c) { return c + "的明亮照片。"; },
        [](const std::string& c) { return "一张干净的" + c + "的照片。"; },
        [](const std::string& c) { return "一张包含" + c + "的照片。"; },
        [](const std::string& c) { return c + "的深色照片。"; },
        [](const std::string& c) { return c + "的手绘画。"; },
        [](const std::string& c) { return "我的" + c + "的照片。"; },
        [](const std::string& c) { return "不自然的" + c + "的照片。"; },
        [](const std::string& c) { return "一张酷的" + c + "的照片。"; },
        [](const std::string& c) { return c + "的特写照片。"; },
        [](const std::string& c) { return c + "的黑白照片。"; },
        [](const std::string& c) { return "一幅" + c + "的画。"; },
        [](const std::string& c) { return "一幅" + c + "的绘画。"; },
        [](const std::string& c) { return "一张" + c + "的像素照片。"; },
        [](const std::string& c) { return c + "的雕像。"; },
        [](const std::string& c) { return "一张" + c + "的明亮照片。"; },
        [](const std::string& c) { return c + "的裁剪照片。"; },
        [](const std::string& c) { return "人造的" + c + "的照片。"; },
        [](const std::string& c) { return "一张关于" + c + "的照片。"; },
        [](const std::string& c) { return "损坏的" + c + "的jpeg照片。"; },
        [](const std::string& c) { return c + "的模糊照片。"; },
        [](const std::string& c) { return c + "的相片。"; },
        [](const std::string& c) { return "一张" + c + "的好照片。"; },
        [](const std::string& c) { return c + "的渲染照。"; },
        [](const std::string& c) { return "视频游戏中的" + c + "。"; },
        [](const std::string& c) { return "一张" + c + "的照片。"; },
        [](const std::string& c) { return c + "的涂鸦。"; },
        [](const std::string& c) { return c + "的近距离照片。"; },
        [](const std::string& c) { return c + "的折纸。"; },
        [](const std::string& c) { return c + "在视频游戏中。"; },
        [](const std::string& c) { return c + "的草图。"; },
        [](const std::string& c) { return c + "的涂鸦照。"; },
        [](const std::string& c) { return c + "的折纸形状。"; },
        [](const std::string& c) { return "低分辨率的" + c + "的照片。"; },
        [](const std::string& c) { return "玩具" + c + "。"; },
        [](const std::string& c) { return c + "的副本。"; },
        [](const std::string& c) { return c + "的干净的照片。"; },
        [](const std::string& c) { return "一张大" + c + "的照片。"; },
        [](const std::string& c) { return c + "的重现。"; },
        [](const std::string& c) { return "一张漂亮的" + c + "的照片。"; },
        [](const std::string& c) { return "一张奇怪的" + c + "的照片。"; },
        [](const std::string& c) { return "模糊的" + c + "的照片。"; },
        [](const std::string& c) { return "卡通" + c + "。"; },
        [](const std::string& c) { return c + "的艺术作品。"; },
        [](const std::string& c) { return c + "的素描。"; },
        [](const std::string& c) { return "刺绣" + c + "。"; },
        [](const std::string& c) { return c + "的像素照。"; },
        [](const std::string& c) { return c + "的拍照。"; },
        [](const std::string& c) { return c + "的损坏的照片。"; },
        [](const std::string& c) { return "高质量的" + c + "的照片。"; },
        [](const std::string& c) { return "毛绒玩具" + c + "。"; },
        [](const std::string& c) { return "漂亮的" + c + "的照片。"; },
        [](const std::string& c) { return "小" + c + "的照片。"; },
        [](const std::string& c) { return "照片是奇怪的" + c + "。"; },
        [](const std::string& c) { return "漫画" + c + "。"; },
        [](const std::string& c) { return c + "的艺术照。"; },
        [](const std::string& c) { return c + "的图形。"; },
        [](const std::string& c) { return "大" + c + "的照片。"; },
        [](const std::string& c) { return "黑白的" + c + "的照片。"; },
        [](const std::string& c) { return c + "毛绒玩具。"; },
        [](const std::string& c) { return "一张" + c + "的深色照片。"; },
        [](const std::string& c) { return c + "的摄影图。"; },
        [](const std::string& c) { return c + "的涂鸦照。"; },
        [](const std::string& c) { return "玩具形状的" + c + "。"; },
        [](const std::string& c) { return "拍了" + c + "的照片。"; },
        [](const std::string& c) { return "酷酷的" + c + "的照片。"; },
        [](const std::string& c) { return "照片里的小" + c + "。"; },
        [](const std::string& c) { return c + "的刺青。"; },
        [](const std::string& c) { return c + "的可爱的照片。"; },
        [](const std::string& c) { return "一张" + c + "可爱的照片。"; },
        [](const std::string& c) { return c + "可爱图片。"; },
        [](const std::string& c) { return c + "酷炫图片。"; },
        [](const std::string& c) { return "一张" + c + "的酷炫的照片。"; },
        [](const std::string& c) { return "一张" + c + "的酷炫图片。"; },
        [](const std::string& c) { return "这是" + c + "。"; },
        [](const std::string& c) { return c + "的好看照片。"; },
        [](const std::string& c) { return "一张" + c + "的好看的图片。"; },
        [](const std::string& c) { return c + "的好看图片。"; },
        [](const std::string& c) { return "一种叫" + c + "的花的照片"; },
        [](const std::string& c) { return "一种叫" + c + "的食物的照片"; },
        [](const std::string& c) { return c + "的卫星照片"; }
    };
}

// 初始化英文模板
void CIFAR100_Validator::initialize_english_templates() {
    // 英文模板相对简单，可以只用几个或一个
    english_templates = {
        // [](const std::string& c) { return "a photo of a " + c + "."; },
        [](const std::string& c) { return c; },
    };
}


// 加载中文标签
std::vector<std::string> CIFAR100_Validator::load_chinese_labels(const std::string& dataset_root) const {
    std::vector<std::string> labels;
    std::string label_file_path = dataset_root + "/label_cn.txt";
    std::ifstream file(label_file_path);
    
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开中文标签文件: " << label_file_path << std::endl;
        return labels;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    
    file.close();
    if (labels.size() != 100) {
        std::cout << "警告: 预期的中文标签数量是100，实际加载了 " << labels.size() << " 个" << std::endl;
    }
    
    return labels;
}



// 读取CIFAR-100二进制文件（英文模式）
bool CIFAR100_Validator::read_cifar100_binary(const std::string& filename, 
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

// 从目录结构加载测试图片（中文模式）
bool CIFAR100_Validator::load_test_dataset_directory(const std::string& dataset_root, 
                                                   std::vector<std::string>& image_paths, 
                                                   std::vector<int>& labels) {
    std::string test_dir = dataset_root + "/test/";
    
    // 检查目录是否存在
    struct stat info;
    if (stat(test_dir.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
        std::cerr << "错误: 测试目录不存在: " << test_dir << std::endl;
        return false;
    }
    
    // 遍历000-099目录
    for (int class_idx = 0; class_idx < 100; class_idx++) {
        char class_folder[5];
        snprintf(class_folder, sizeof(class_folder), "%03d", class_idx);
        
        std::string class_path = test_dir + class_folder;
        
        // 检查类别目录是否存在
        if (stat(class_path.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            std::cout << "警告: 类别目录不存在: " << class_path << std::endl;
            continue;
        }
        
        // 扫描目录中的图片文件
        std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"};
        
        for (const auto& entry : std::filesystem::directory_iterator(class_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::string file_ext = entry.path().extension().string();
                
                // 检查文件扩展名
                if (std::find(image_extensions.begin(), image_extensions.end(), file_ext) != image_extensions.end()) {
                    image_paths.push_back(file_path);
                    labels.push_back(class_idx);
                }
            }
        }
    }
    
    if (image_paths.empty()) {
        std::cerr << "错误: 没有找到任何测试图片" << std::endl;
        return false;
    }
    
    return true;
}


// 统一分词接口
std::vector<int> CIFAR100_Validator::tokenize_text(const std::string& text, size_t max_token_len) {
    if (is_chinese_) {
        // 使用中文分词器
        if (!cn_tokenizer_) {
            std::cerr << "错误: 中文分词器未初始化" << std::endl;
            return {};
        }
        return cn_tokenizer_->encode(text, max_token_len, true, true);
    } else {
        // 使用英文分词器
        if (!en_tokenizer_) {
            std::cerr << "错误: 英文分词器未初始化" << std::endl;
            return {};
        }
        return en_tokenizer_->tokenize(text, nullptr, max_token_len, true);
    }
}

// 图片预处理
cv::Mat CIFAR100_Validator::preprocess_image(const cv::Mat& img) {
    cv::Mat processed = img.clone();
    
    // 转换BGR到RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    
    // 调整大小（保持长宽比，最小边缩放到IMAGE_SIZE）
    int h = processed.rows;
    int w = processed.cols;
    
    int new_h, new_w;
    float scale = static_cast<float>(IMAGE_SIZE) / std::min(h, w);
    new_h = static_cast<int>(h * scale);
    new_w = static_cast<int>(w * scale);
    
    cv::resize(processed, processed, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);
    
    // 中心裁剪到IMAGE_SIZE x IMAGE_SIZE
    int start_h = (new_h - IMAGE_SIZE) / 2;
    int start_w = (new_w - IMAGE_SIZE) / 2;
    
    if (start_h < 0 || start_w < 0 || start_h + IMAGE_SIZE > new_h || start_w + IMAGE_SIZE > new_w) {
        // 如果图片太小，直接调整大小
        cv::resize(processed, processed, cv::Size(IMAGE_SIZE, IMAGE_SIZE), 0, 0, cv::INTER_CUBIC);
    } else {
        cv::Rect roi(start_w, start_h, IMAGE_SIZE, IMAGE_SIZE);
        processed = processed(roi).clone();
    }
    
    // 转换为float并归一化
    processed.convertTo(processed, CV_32FC3, 1.0 / 255.0);
    
    // 应用归一化参数
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    
    // 减去均值，除以标准差
    channels[0] = (channels[0] - MEAN_R) / STD_R;  // R
    channels[1] = (channels[1] - MEAN_G) / STD_G;  // G
    channels[2] = (channels[2] - MEAN_B) / STD_B;  // B
    
    cv::merge(channels, processed);
    
    return processed;
}

// 归一化特征向量
std::vector<float> CIFAR100_Validator::normalize_features(const std::vector<float>& features) {
    float norm = 0.0f;
    for (float val : features) {
        norm += val * val;
    }
    
    norm = std::sqrt(norm);
    
    if (norm < 1e-12) {
        return features;  // 避免除以零
    }
    
    std::vector<float> normalized(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        normalized[i] = features[i] / norm;
    }
    
    return normalized;
}

// 核心评估函数
void CIFAR100_Validator::validate_accuracy(const std::string& dataset_path, int max_samples) {
    if (is_chinese_) {
        if (!cn_tokenizer_) { std::cerr << "错误: 中文分词器未初始化" << std::endl; return; }
    } else {
        if (!en_tokenizer_) { std::cerr << "错误: 英文分词器未初始化" << std::endl; return; }
    }
    
    std::cout << "\n=================================================================" << std::endl;
    std::cout << (is_chinese_ ? "中文" : "英文") << "CLIP CIFAR-100 零样本分类评估" << std::endl;
    std::cout << "标签类型: " << get_label_type() << std::endl;
    std::cout << "数据集路径: " << dataset_path << std::endl;
    std::cout << "最大样本数: " << (max_samples > 0 ? std::to_string(max_samples) : "全部") << std::endl;
    std::cout << "=================================================================" << std::endl;
    
    // 创建全局计时器
    TimeStamp ts;
    clip_model_.enableProfile(&ts);
    
    // 步骤1: 加载标签
    if (is_chinese_) {
        std::cout << "\n步骤1: 加载中文标签..." << std::endl;
        chinese_labels_ = load_chinese_labels(dataset_path);
        if (chinese_labels_.empty()) {
            std::cerr << "错误: 无法加载中文标签" << std::endl;
            return;
        }
        std::cout << "加载了 " << chinese_labels_.size() << " 个中文类别标签" << std::endl;
    } else {
        std::cout << "\n步骤1: 使用" << (use_coarse_labels_ ? "20个粗粒度" : "100个细粒度") << "英文标签" << std::endl;
    }
    
    // 步骤2: 生成文本特征矩阵
    std::cout << "\n步骤2: 生成文本特征矩阵..." << std::endl;
    std::vector<std::vector<float>> text_features;
    size_t max_token_len = clip_model_.get_token_len();
    size_t num_classes = get_num_classes();
    
    for (size_t class_idx = 0; class_idx < num_classes; class_idx++) {
        std::string class_name;
        if (is_chinese_) {
            class_name = chinese_labels_[class_idx];
        } else {
            class_name = use_coarse_labels_ ? coarse_class_names_[class_idx] : fine_class_names_[class_idx];
        }
        
        std::vector<std::vector<float>> template_features; 
        const auto& templates = is_chinese_ ? chinese_templates : english_templates;
        std::vector<float> txt_feature;
        // 对每个模板生成文本特征
        for (const auto& template_func : templates) {
            std::string text = template_func(class_name);
            // 分词
            ts.start();
            std::vector<int> tokens = tokenize_text(text, max_token_len);
            ts.time_accumulation("tokenize_time");
            // 编码文本
            txt_feature = clip_model_.encode_text(tokens, nnrt_ctx_text_);
            // 归一化
            if (is_chinese_) {
                if (!txt_feature.empty()) {
                    ts.start();
                    std::vector<float> normalized_template_feat = normalize_features(txt_feature);
                    template_features.push_back(normalized_template_feat);
                    ts.time_accumulation("text_postprocess_time");
                }
            } else {
                text_features.push_back(txt_feature);
            }

        }
        if (is_chinese_) {
            ts.start();
            if (template_features.empty()) {
                std::cerr << "错误: 类别 '" << class_name << "' 没有生成有效的文本特征" << std::endl;
                continue;
            }
            
            // 计算模板特征的平均值
            std::vector<float> avg_feature(template_features[0].size(), 0.0f);
            for (const auto& feat : template_features) {
                for (size_t i = 0; i < feat.size(); i++) {
                    avg_feature[i] += feat[i];
                }
            }
            for (size_t i = 0; i < avg_feature.size(); i++) {
                avg_feature[i] /= template_features.size();
            }
            
            std::vector<float> normalized_feature = normalize_features(avg_feature);
            text_features.push_back(normalized_feature);
            ts.time_accumulation("text_postprocess_time");

        }

        if (class_idx < 3) {
            std::cout << "  类别 " << class_idx << " (" << class_name << "): 生成特征中 " << std::endl;
        } else if (class_idx == 3) {
            std::cout << "  ..." << std::endl;
        }
    }
    
    std::cout << "生成了 " << text_features.size() << " 个类别的文本特征" << std::endl;
    
    // 步骤3: 加载测试数据
    std::cout << "\n步骤3: 加载测试数据集..." << std::endl;
    std::vector<cv::Mat> test_images;
    std::vector<int> true_labels;
    std::vector<int> coarse_labels;  // 仅英文模式使用
    
    if (is_chinese_) {
        // 中文模式：从目录加载
        std::vector<std::string> image_paths;
        if (!load_test_dataset_directory(dataset_path, image_paths, true_labels)) {
            std::cerr << "错误: 加载测试数据集失败" << std::endl;
            return;
        }
        std::vector<cv::Mat> valid_images;
        std::vector<int> valid_labels;
        // 加载图片
        for (size_t i = 0; i < image_paths.size(); i++) {
            ts.start();
            cv::Mat img = cv::imread(image_paths[i], cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
            ts.time_accumulation("imread_time");
            if (img.empty()) {
                std::cerr << "警告: 无法加载图片: " << image_paths[i] << std::endl;
                continue;
            }
            valid_images.push_back(img);
            valid_labels.push_back(true_labels[i]);
            
            if (max_samples > 0 && valid_images.size() >= static_cast<size_t>(max_samples)) {
                break;
            }
        }
        
        // 调整标签向量
        test_images = std::move(valid_images);
        true_labels = std::move(valid_labels);
    } else {
        // en
        ts.start();
        if (!read_cifar100_binary(dataset_path, test_images, true_labels, coarse_labels)) {
            std::cerr << "错误: 加载CIFAR-100二进制文件失败" << std::endl;
            return;
        }
        ts.time_accumulation("imread_time");
    }
    if (max_samples > 0 && test_images.size() > static_cast<size_t>(max_samples)) {
        test_images.resize(max_samples);
        true_labels.resize(max_samples);
        coarse_labels.resize(max_samples);
    }
    if (use_coarse_labels_) {
        true_labels = coarse_labels;
    }
    std::cout << "加载了 " << test_images.size() << " 张测试图片" << std::endl;
    
    // 步骤4: 提取图片特征
    std::cout << "\n步骤4: 处理图片并提取特征..." << std::endl;
    std::vector<std::vector<float>> image_features;
    int processed_count = 0;
    int image_count = test_images.size();
    
    int correct_top1 = 0;
    int correct_top5 = 0;


    for (size_t i = 0; i < test_images.size(); i++) {
        if (i % 100 == 0) {
            std::cout << "  处理进度: " << i << "/" << image_count << std::endl;
        }
        // 提取特征
        std::vector<float> img_feature = clip_model_.encode_image_memory(test_images[i], nnrt_ctx_image_);
        if (img_feature.empty()) {
            std::cerr << "警告: 图片特征提取失败: 索引 " << i << std::endl;
            continue;
        }

        if (is_chinese_) {
            // 归一化
            ts.start();
            std::vector<float> normalized_features = normalize_features(img_feature);
            image_features.push_back(normalized_features);
            processed_count++;
            ts.time_accumulation("image_postprocess_time");
            if (image_features.empty()) {
                std::cerr << "错误: 没有成功提取任何图片特征" << std::endl;
                return;
            }
        } else {
            // 计算与所有文本特征的相似度
            processed_count++;
            std::vector<float> similarities;
            for (const auto& text_feature : text_features) {
                float similarity = 100.0f * std::inner_product(
                    img_feature.begin(), img_feature.end(),
                    text_feature.begin(), 0.0f);
                similarities.push_back(similarity);
            }
            
            // 获取top-1和top-5预测
            auto [top1_values, top1_indices] = clip_model_.topk(similarities, 1);
            auto [top5_values, top5_indices] = clip_model_.topk(similarities, 5);
            
            // 根据选择的标签类型确定真实标签
            int true_label = use_coarse_labels_ ? coarse_labels[i] : true_labels[i];
            
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
                std::cout << "Progress: " << i << "/" << image_count 
                         << " | Top-1: " << std::fixed << std::setprecision(2) 
                         << (100.0 * correct_top1 / (i + 1)) << "%"
                         << " | Top-5: " << (100.0 * correct_top5 / (i + 1)) << "%" << std::endl;
            }
            // continue;
        }

    }
    
    if (is_chinese_) {
    // 步骤5: 计算准确率
        std::cout << "\n步骤5: 计算准确率..." << std::endl;
        float top1_accuracy, top5_accuracy;
        calculate_topk_accuracy(image_features, text_features, true_labels, 1, top1_accuracy);
        calculate_topk_accuracy(image_features, text_features, true_labels, 5, top5_accuracy);
        
        // 步骤6: 输出结果
        std::cout << "\n=================================================================" << std::endl;
        std::cout << "评估结果" << std::endl;
        std::cout << "=================================================================" << std::endl;
        std::cout << "模型语言: " << (is_chinese_ ? "中文" : "英文") << std::endl;
        std::cout << "标签类型: " << get_label_type() << std::endl;
        std::cout << "测试图片数量: " << processed_count << std::endl;
        std::cout << "提示模板数量: " << (is_chinese_ ? chinese_templates.size() : english_templates.size()) << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Top-1 准确率: " << top1_accuracy * 100.0f << "%" << std::endl;
        std::cout << "Top-5 准确率: " << top5_accuracy * 100.0f << "%" << std::endl;
        std::cout << "=================================================================" << std::endl;
        
        // 保存结果到文件
        std::string result_file = is_chinese_ ? 
            "cifar100_cn_validation_results.txt" : 
            (use_coarse_labels_ ? "cifar100_en_coarse_results.txt" : "cifar100_en_fine_results.txt");
        
        save_detailed_results(result_file, top1_accuracy, top5_accuracy, 
                            processed_count, use_coarse_labels_, is_chinese_);

    } else {
       
        // 计算最终准确率
        float top1_accuracy = 100.0f * correct_top1 / processed_count;
        float top5_accuracy = 100.0f * correct_top5 / processed_count;
        
        std::cout << "\n=== CIFAR-100 Validation Results ===" << std::endl;
        std::cout << "Label type: " << (use_coarse_labels_ ? "20 coarse labels" : "100 fine labels") << std::endl;
        std::cout << "Total images: " << image_count << std::endl;
        std::cout << "Top-1 Accuracy: " << std::fixed << std::setprecision(2) 
                 << top1_accuracy << "%" << std::endl;
        std::cout << "Top-5 Accuracy: " << top5_accuracy << "%" << std::endl;
        
        // 保存详细结果到文件
        std::string result_file = use_coarse_labels_ ? 
            "cifar100_coarse_validation_results.txt" : "cifar100_fine_validation_results.txt";
        save_detailed_results(result_file, top1_accuracy, top5_accuracy, image_count, use_coarse_labels_, is_chinese_);

    }
 

    // 步骤7: 打印性能统计
    std::cout << "\n性能统计（平均时间）:" << std::endl;
    std::cout << "========================================" << std::endl;

    // 计算各种操作的基数
    // int num_classes = get_num_classes();
    int templates_count = is_chinese_ ? chinese_templates.size() : english_templates.size();
    int total_tokens = num_classes * templates_count;  // 模板数 × 类别数
    if (total_tokens == 0) total_tokens = 1;
    if (processed_count == 0) processed_count = 1;

    std::cout << "统计基数:" << std::endl;
    std::cout << "  图片处理次数: " << processed_count << std::endl;
    std::cout << "  文本处理次数: " << total_tokens << " (类别" << num_classes << " × 模板" << templates_count << ")" << std::endl;
    std::cout << "========================================" << std::endl;

    // 按标签类型打印平均时间
    if (ts.time_map_lab.find("tokenize_time") != ts.time_map_lab.end()) {
        std::cout << "  tokenize_time: " << ts.time_map_lab["tokenize_time"] / total_tokens << " ms" << std::endl;
    }
    if (ts.time_map_lab.find("encode_text_time") != ts.time_map_lab.end()) {
        std::cout << "  encode_text_time: " << ts.time_map_lab["encode_text_time"] / total_tokens << " ms" << std::endl;
    }
    if (ts.time_map_lab.find("text_postprocess_time") != ts.time_map_lab.end()) {
        std::cout << "  text_postprocess_time: " << ts.time_map_lab["text_postprocess_time"] / total_tokens << " ms" << std::endl;
    }
    if (ts.time_map_lab.find("imread_time") != ts.time_map_lab.end()) {
        std::cout << "  imread_time: " << ts.time_map_lab["imread_time"] / processed_count << " ms" << std::endl;
    }
    if (ts.time_map_lab.find("pre_time") != ts.time_map_lab.end()) {
        std::cout << "  pre_time: " << ts.time_map_lab["pre_time"] / processed_count << " ms" << std::endl;
    }
    if (ts.time_map_lab.find("encode_image_time") != ts.time_map_lab.end()) {
        std::cout << "  encode_image_time: " << ts.time_map_lab["encode_image_time"] / processed_count << " ms" << std::endl;
    }
    if (ts.time_map_lab.find("image_postprocess_time") != ts.time_map_lab.end()) {
        std::cout << "  image_postprocess_time: " << ts.time_map_lab["image_postprocess_time"] / processed_count << " ms" << std::endl;
    }



}

// 计算Top-K准确率
void CIFAR100_Validator::calculate_topk_accuracy(const std::vector<std::vector<float>>& image_features,
                                               const std::vector<std::vector<float>>& text_features,
                                               const std::vector<int>& true_labels,
                                               int k, float& topk_accuracy) {
    int correct = 0;
    int total = image_features.size();
    
    for (size_t i = 0; i < image_features.size(); i++) {
        // 计算与所有文本特征的相似度
        std::vector<std::pair<float, int>> similarities;
        
        for (size_t j = 0; j < text_features.size(); j++) {
            float similarity = 0.0f;
            
            // 计算点积（余弦相似度，因为特征已归一化）
            for (size_t dim = 0; dim < image_features[i].size(); dim++) {
                similarity += image_features[i][dim] * text_features[j][dim];
            }
            
            similarities.emplace_back(similarity, j);
        }
        
        // 按相似度降序排序
        std::sort(similarities.begin(), similarities.end(), 
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first > b.first;
                });
        
        // 检查真实标签是否在Top-K中
        bool found = false;
        for (int idx = 0; idx < k && idx < static_cast<int>(similarities.size()); idx++) {
            if (similarities[idx].second == true_labels[i]) {
                found = true;
                break;
            }
        }
        
        if (found) {
            correct++;
        }
    }
    
    topk_accuracy = static_cast<float>(correct) / total;
}

// 保存详细结果
void CIFAR100_Validator::save_detailed_results(const std::string& filename, 
                                             float top1_acc, float top5_acc, 
                                             int total, bool use_coarse, bool is_chinese) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "CIFAR-100 Validation Results\n";
        file << "============================\n";
        file << "Model language: " << (is_chinese ? "Chinese" : "English") << "\n";
        file << "Label type: " << (is_chinese ? "100 Chinese fine labels" : 
                                 (use_coarse ? "20 English coarse labels" : "100 English fine labels")) << "\n";
        file << "Total test images: " << total << "\n";
        file << std::fixed << std::setprecision(4);
        file << "Top-1 Accuracy: " << top1_acc << "\n";
        file << "Top-5 Accuracy: " << top5_acc << "\n";
        file << "Validation date: " << __DATE__ << " " << __TIME__ << "\n";
        
        // 保存类别名称
        file << "\nClass Names:\n";
        if (is_chinese) {
            for (size_t i = 0; i < chinese_labels_.size(); ++i) {
                file << i << ": " << chinese_labels_[i] << "\n";
            }
        } else if (use_coarse) {
            for (size_t i = 0; i < coarse_class_names_.size(); ++i) {
                file << i << ": " << coarse_class_names_[i] << "\n";
            }
        } else {
            for (size_t i = 0; i < fine_class_names_.size(); ++i) {
                file << i << ": " << fine_class_names_[i] << "\n";
            }
        }
        
        file.close();
        std::cout << "详细结果已保存到: " << filename << std::endl;
    }
}
