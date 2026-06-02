#include "clip.hpp"
#include "cifar100_validator.hpp"


// 命名空间
namespace fs = std::filesystem;

// 配置结构体

struct ModelConfig {
    std::string language;           // "cn" 或 "en"
    std::string vocab_path;         // 词汇表文件路径
    std::string image_model_path;   // 图像编码器模型路径
    std::string text_model_path;    // 文本编码器模型路径
    std::string text_projection_path; // 文本投影矩阵路径
    bool save_results;              // 保存结果到文件
    bool verbose;                   // 详细输出
    
    ModelConfig() : save_results(false), verbose(false) {}
};
struct InferConfig : public ModelConfig {
    std::string image_path;         // 单张图片路径
    std::string text_input;         // 单个文本输入
    std::string image_list_path;    // 图片列表文件
    std::string text_list_path;     // 文本列表文件
    
    InferConfig() : ModelConfig() {}
};

struct ValidateConfig : public ModelConfig {
    std::string dataset_path;       // 数据集路径
    int max_samples;                // 最大测试样本数
    
    ValidateConfig() : ModelConfig(), max_samples(-1) {}
};


// 函数声明
void print_help(const std::string& mode = "");
bool parse_infer_arguments(int argc, char* argv[], InferConfig& config);
bool parse_validate_arguments(int argc, char* argv[], ValidateConfig& config);
void run_infer_mode(const InferConfig& config, 
                   const std::shared_ptr<ta_runtime_context>& ctx_text,
                   const std::shared_ptr<ta_runtime_context>& ctx_image);
void run_validate_mode(const ValidateConfig& config,
                      const std::shared_ptr<ta_runtime_context>& ctx_text,
                      const std::shared_ptr<ta_runtime_context>& ctx_image);
std::vector<std::string> read_lines_from_file(const std::string& filepath);
bool is_chinese_mode(const std::string& lang);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "     Unified CLIP SOC Program" << std::endl;
    std::cout << "     Version: 1.0" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::string mode = argv[1];
    
    try {
        if (mode == "infer") {
            // 解析 infer 模式参数
            InferConfig config;
            if (!parse_infer_arguments(argc - 1, argv + 1, config)) {
                print_help("infer");
                return 1;
            }
            
            // 初始化TA Runtime上下文
            std::shared_ptr<ta_runtime_context> nnrt_ctx_text = 
                std::make_shared<ta_runtime_context>();
            std::shared_ptr<ta_runtime_context> nnrt_ctx_image = 
                std::make_shared<ta_runtime_context>();
            
            // 运行 infer 模式
            run_infer_mode(config, nnrt_ctx_text, nnrt_ctx_image);
            
        } else if (mode == "validate") {
            // 解析 validate 模式参数
            ValidateConfig config;
            if (!parse_validate_arguments(argc - 1, argv + 1, config)) {
                print_help("validate");
                return 1;
            }
            
            // 初始化TA Runtime上下文
            std::shared_ptr<ta_runtime_context> nnrt_ctx_text = 
                std::make_shared<ta_runtime_context>();
            std::shared_ptr<ta_runtime_context> nnrt_ctx_image = 
                std::make_shared<ta_runtime_context>();
            
            // 运行 validate 模式
            run_validate_mode(config, nnrt_ctx_text, nnrt_ctx_image);
            
        } else if (mode == "--help" || mode == "-h") {
            print_help();
            return 0;
        } else {
            std::cerr << "错误: 未知模式 '" << mode << "'" << std::endl;
            std::cerr << "可用模式: infer, validate" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "程序执行出错: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

bool parse_common_arguments(const std::vector<std::string>& args, 
                           size_t& i, 
                           ModelConfig& config) {
    const std::string& arg = args[i];
    
    if (arg == "--language" || arg == "-l") {
        if (i + 1 >= args.size()) return false;
        config.language = args[++i];
    } else if (arg == "--vocab" || arg == "-v") {
        if (i + 1 >= args.size()) return false;
        config.vocab_path = args[++i];
    } else if (arg == "--image_model" || arg == "-im") {
        if (i + 1 >= args.size()) return false;
        config.image_model_path = args[++i];
    } else if (arg == "--text_model" || arg == "-tm") {
        if (i + 1 >= args.size()) return false;
        config.text_model_path = args[++i];
    } else if (arg == "--text_projection" || arg == "-tp") {
        if (i + 1 >= args.size()) return false;
        config.text_projection_path = args[++i];
    } else if (arg == "--save") {
        config.save_results = true;
    } else if (arg == "--verbose") {
        config.verbose = true;
    } else {
        return false; // 不是公共参数
    }
    
    return true; // 成功解析公共参数
}
bool validate_model_config(const ModelConfig& config) {
    if (config.language.empty()) {
        std::cerr << "错误: 必须指定 --language" << std::endl;
        return false;
    }
    
    if (config.language == "cn" && config.vocab_path.empty()) {
        std::cerr << "错误: 中文模式必须提供 --vocab 参数" << std::endl;
        return false;
    }
    
    if (config.image_model_path.empty()) {
        std::cerr << "错误: 必须提供 --image_model 参数" << std::endl;
        return false;
    }
    
    if (config.text_model_path.empty()) {
        std::cerr << "错误: 必须提供 --text_model 参数" << std::endl;
        return false;
    }
    
    return true;
}
bool parse_infer_arguments(int argc, char* argv[], InferConfig& config) {
    std::vector<std::string> args(argv + 1, argv + argc);
    
    for (size_t i = 0; i < args.size(); i++) {
        if (parse_common_arguments(args, i, config)) {
            continue; 
        }
        
        // 处理 infer 特定的参数
        if (args[i] == "--image_path" || args[i] == "-i") { 
            if (i + 1 >= args.size()) return false;
            config.image_path = args[++i];
        } else if (args[i] == "--text" || args[i] == "-t") {
            if (i + 1 >= args.size()) return false;
            config.text_input = args[++i];
        } else if (args[i] == "--image_list" || args[i] == "-il") {
            if (i + 1 >= args.size()) return false;
            config.image_list_path = args[++i];
        } else if (args[i] == "--text_list" || args[i] == "-tl") {
            if (i + 1 >= args.size()) return false;
            config.text_list_path = args[++i];
        } else if (args[i] == "--help" || args[i] == "-h") {
            return false;
        } else {
            std::cerr << "错误: infer模式未知选项: " << args[i] << std::endl;
            return false;
        }
    }

    if (!validate_model_config(config)) {
        return false;
    }
    
    if (config.image_path.empty() && config.image_list_path.empty()) {
        std::cerr << "错误: infer模式需要提供 --image 或 --image_list" << std::endl;
        return false;
    }
    
    if (config.text_input.empty() && config.text_list_path.empty()) {
        std::cerr << "错误: infer模式需要提供 --text 或 --text_list" << std::endl;
        return false;
    }
    
    if (!config.image_path.empty() && !config.image_list_path.empty()) {
        std::cerr << "错误: --image 和 --image_list 不能同时使用" << std::endl;
        return false;
    }
    
    if (!config.text_input.empty() && !config.text_list_path.empty()) {
        std::cerr << "错误: --text 和 --text_list 不能同时使用" << std::endl;
        return false;
    }
    
    return true;
}

bool parse_validate_arguments(int argc, char* argv[], ValidateConfig& config) {
    std::vector<std::string> args(argv + 1, argv + argc);
    
    for (size_t i = 0; i < args.size(); i++) {
        if (parse_common_arguments(args, i, config)) {
            continue; 
        }
        
        // 处理 validate 特定的参数
        if (args[i] == "--dataset_path" || args[i] == "-d") {
            if (i + 1 >= args.size()) return false;
            config.dataset_path = args[++i];
        } else if (args[i] == "--max_samples" || args[i] == "-m") {
            if (i + 1 >= args.size()) return false;
            config.max_samples = std::stoi(args[++i]);
        } else if (args[i] == "--help" || args[i] == "-h") {
            return false;
        } else {
            std::cerr << "错误: validate模式未知选项: " << args[i] << std::endl;
            return false;
        }
    }
    
    if (!validate_model_config(config)) {
        return false;
    }
    
    if (config.dataset_path.empty()) {
        std::cerr << "错误: validate模式必须提供 --dataset_path 参数" << std::endl;
        return false;
    }
    
    return true;
}

// 打印帮助信息
void print_help(const std::string& mode) {
    if (mode.empty()) {
        std::cout << "用法: clip_soc <模式> [选项]" << std::endl;
        std::cout << std::endl;
        std::cout << "模式:" << std::endl;
        std::cout << "  infer       简单验证模式: 计算图片和文本的相似度" << std::endl;
        std::cout << "  validate    精度校验模式: 在CIFAR-100数据集上评估模型精度" << std::endl;
        std::cout << std::endl;
        std::cout << "查看具体模式帮助: clip_soc <模式> --help" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  # 中文单图单文推理" << std::endl;
        std::cout << "  ./clip_soc infer --language cn --vocab vocab.txt --image_model cn_img.nb --text_model cn_text.nb --image cat.jpg --text \"一只猫\"" << std::endl;
        std::cout << std::endl;
        std::cout << "  # 英文数据集精度评估" << std::endl;
        std::cout << "  ./clip_soc validate --language en --image_model en_img.nb --text_model en_text.nb --text_projection proj.bin --dataset_path test.bin" << std::endl;
        return;
    }
    
    if (mode == "infer") {
        std::cout << "用法: clip_soc infer [选项]" << std::endl;
        std::cout << std::endl;
        std::cout << "选项:" << std::endl;
        std::cout << "  模型参数:" << std::endl;
        std::cout << "    -l, --language <cn|en>       模型语言: 'cn' 中文, 'en' 英文 (必须)" << std::endl;
        std::cout << "    -v, --vocab <path>           词汇表文件路径 (中文模式必须)" << std::endl;
        std::cout << "    -im, --image_model <path>    图像编码器模型路径 (必须)" << std::endl;
        std::cout << "    -tm, --text_model <path>     文本编码器模型路径 (必须)" << std::endl;
        std::cout << "    -tp, --text_projection <path> 文本投影矩阵路径 (英文模式必须)" << std::endl;
        std::cout << std::endl;
        std::cout << "  输入参数 (必须提供一组):" << std::endl;
        std::cout << "    -i, --image_path <path>      图片目录路径" << std::endl;
        std::cout << "    -t, --text <string>          文本输入 (支持逗号分隔的多个文本，如: \"猫,狗,鸟\")" << std::endl;
        std::cout << std::endl;
        std::cout << "  其他选项:" << std::endl;
        std::cout << "    -h, --help                   显示此帮助信息" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  # 示例1: 中文模型 - 简单验证" << std::endl;
        std::cout << "  ./clip_soc infer \\" << std::endl;
        std::cout << "             --language cn --vocab ./models/vocab.txt \\" << std::endl;
        std::cout << "             --image_model ./models/clip_cn_image_float16.nb --text_model ./models/clip_cn_text_float16.nb \\" << std::endl;
        std::cout << "             --image_path \"datasets/test_images\"  --text \"流程图,人,车\"" << std::endl;
        std::cout << std::endl;
        std::cout << "  # 示例2: 英文模型 - 简单验证" << std::endl;
        std::cout << "  ./clip_soc infer \\" << std::endl;
        std::cout << "             --language en \\" << std::endl;
        std::cout << "             --image_model ./models/clip_image_float16.nb --text_model ./models/clip_text_float16.nb \\" << std::endl;
        std::cout << "             --text_projection ./models/text_projection_512_512.npy \\" << std::endl;
        std::cout << "             --image_path \"datasets/test_images\" --text \"a diagram,a person ,a Car\"" << std::endl;
    } else if (mode == "validate") {
        std::cout << "用法: clip_soc validate [选项]" << std::endl;
        std::cout << std::endl;
        std::cout << "选项:" << std::endl;
        std::cout << "  模型参数:" << std::endl;
        std::cout << "    -l, --language <cn|en>       模型语言: 'cn' 中文, 'en' 英文 (必须)" << std::endl;
        std::cout << "    -v, --vocab <path>           词汇表文件路径 (中文模式必须)" << std::endl;
        std::cout << "    -im, --image_model <path>    图像编码器模型路径 (必须)" << std::endl;
        std::cout << "    -tm, --text_model <path>     文本编码器模型路径 (必须)" << std::endl;
        std::cout << "    -tp, --text_projection <path> 文本投影矩阵路径 (英文模式必须)" << std::endl;
        std::cout << std::endl;
        std::cout << "  数据集参数:" << std::endl;
        std::cout << "    -d, --dataset_path <path>    数据集路径 (必须)" << std::endl;
        std::cout << "                                 中文模式: 包含test/和label_cn.txt的目录" << std::endl;
        std::cout << "                                 英文模式: CIFAR-100二进制文件路径" << std::endl;
        std::cout << "    -m, --max_samples <N>        最大测试样本数" << std::endl;
        std::cout << std::endl;
        std::cout << "  其他选项:" << std::endl;
        std::cout << "    -h, --help                   显示此帮助信息" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  # 示例3: 中文模型 - 数据集精度校验" << std::endl;
        std::cout << "  ./clip_soc validate \\" << std::endl;
        std::cout << "             --language cn --vocab ./models/vocab.txt \\" << std::endl;
        std::cout << "             --image_model ./models/clip_cn_image_float16.nb --text_model ./models/clip_cn_text_float16.nb \\" << std::endl;
        std::cout << "             --dataset_path ./datasets/cifar-100_cn \\" << std::endl;
        std::cout << "             --max_samples 10000" << std::endl;
        std::cout << std::endl;
        std::cout << "  # 示例4: 英文模型 - 数据集精度校验" << std::endl;
        std::cout << "  ./clip_soc validate \\" << std::endl;
        std::cout << "             --language en \\" << std::endl;
        std::cout << "             --image_model ./models/clip_image_float16.nb --text_model ./models/clip_text_float16.nb \\" << std::endl;
        std::cout << "             --text_projection ./models/text_projection_512_512.npy \\" << std::endl;
        std::cout << "             --dataset_path ./datasets/cifar-100_en/test.bin \\" << std::endl;
        std::cout << "             --max_samples 10000" << std::endl;
    }
}

// 判断是否为中文模式
bool is_chinese_mode(const std::string& lang) {
    return (lang == "cn" || lang == "zh" || lang == "chinese");
}

// 从文件读取行
std::vector<std::string> read_lines_from_file(const std::string& filepath) {
    std::vector<std::string> lines;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filepath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    file.close();
    return lines;
}


std::vector<std::string> get_image_paths(const std::string& input_path) {
    std::vector<std::string> image_paths;
    
    // 检查路径是否存在
    if (!fs::exists(input_path)) {
        std::cerr << "错误: 路径不存在: " << input_path << std::endl;
        return image_paths;
    }
    if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // 只处理常见的图片格式
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
                    ext == ".bmp" || ext == ".JPG" || ext == ".JPEG") {
                    image_paths.push_back(entry.path().string());
                }
            }
        }
        std::cout << "   找到图片: " << image_paths.size() << " 张" << std::endl;
    } 
    else if (fs::is_regular_file(input_path)) {
        std::string ext = fs::path(input_path).extension().string();
        if (ext == ".txt") {
            // 文本文件，按行读取
            image_paths = read_lines_from_file(input_path);
            std::cout << "   图片列表文件: " << image_paths.size() << " 张图片" << std::endl;
        } else {
            // 单个图片文件
            image_paths.push_back(input_path);
            std::cout << "   单张图片: " << fs::path(input_path).filename().string() << std::endl;
        }
    }
    
    return image_paths;
}
// 运行简单验证模式
void run_infer_mode(const InferConfig& config, 
                   const std::shared_ptr<ta_runtime_context>& ctx_text,
                   const std::shared_ptr<ta_runtime_context>& ctx_image) {
    
    bool is_chinese = is_chinese_mode(config.language);
    
    std::cout << "\n[模式] 简单验证 (推理)" << std::endl;
    std::cout << "[语言] " << (is_chinese ? "中文" : "英文") << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 1. 初始化CLIP模型
    std::cout << "1. 初始化CLIP模型..." << std::endl;
    CLIP clip;
    int status = clip.init(config.image_model_path, 
                          config.text_model_path,
                          ctx_text,
                          ctx_image,
                          config.text_projection_path,
                          is_chinese);
    
    if (status != 0) {
        clip.deinit(ctx_image, ctx_text);
        throw std::runtime_error("CLIP模型初始化失败");
    }
    std::cout << "   CLIP模型初始化成功" << std::endl;
    
    // 2. 加载图片列表
    std::vector<std::string> image_paths;
    if (!config.image_path.empty()) {
        image_paths = get_image_paths(config.image_path);
        if (image_paths.empty()) {
            throw std::runtime_error("没有找到任何图片");
        }
    } else if (!config.image_list_path.empty()) {
        image_paths = read_lines_from_file(config.image_list_path);
        std::cout << "   图片列表文件: " << image_paths.size() << " 张图片" << std::endl;
    }
    
    // 3. 加载文本列表
    std::vector<std::string> texts;
    if (!config.text_input.empty()) {
        // 新增：检查是否为逗号分隔的列表
        if (config.text_input.find(',') != std::string::npos) {
            // 按逗号分割字符串
            std::stringstream ss(config.text_input);
            std::string item;
            while (std::getline(ss, item, ',')) {
                // 去除首尾空格
                item.erase(0, item.find_first_not_of(" \t"));
                item.erase(item.find_last_not_of(" \t") + 1);
                if (!item.empty()) {
                    texts.push_back(item);
                }
            }
            std::cout << "   文本列表（命令行）: " << texts.size() << " 个文本" << std::endl;
        } else {
            // 单个文本
            texts.push_back(config.text_input);
            std::cout << "   单个文本: " << config.text_input << std::endl;
        }
    } else if (!config.text_list_path.empty()) {
        texts = read_lines_from_file(config.text_list_path);
        std::cout << "   文本列表（文件）: " << texts.size() << " 个文本" << std::endl;
    }

    // 4. 创建计时器
    TimeStamp ts;
    clip.enableProfile(&ts);
    
    // 5. 准备 token
    std::vector<std::vector<int>> tokenized_texts;
    std::vector<std::string> processed_texts;
    
    for (size_t i = 0; i < texts.size(); i++) {
        std::string text = texts[i];
        
        // 分词
        std::vector<int> tokens;
        if (is_chinese) {
            BertTokenizer tokenizer(config.vocab_path);
            tokens = tokenizer.encode(text, clip.get_token_len(), true, true);
        } else {
            CLIPTokenizer tokenizer;
            tokens = tokenizer.tokenize(text, nullptr, clip.get_token_len(), true);
        }
        
        if (tokens.empty()) {
            std::cerr << "      警告: 文本分词失败，跳过: " << text << std::endl;
            continue;
        }
        
        tokenized_texts.push_back(tokens);
        processed_texts.push_back(text);
    }
    
    if (tokenized_texts.empty()) {
        throw std::runtime_error("没有成功处理任何文本");
    }
    
    // 6. 调用CLIP的infer方法
    std::cout << "\n7. 开始双向检索..." << std::endl;
    int infer_status = clip.infer(image_paths, tokenized_texts, processed_texts, ctx_text, ctx_image);
    
    if (infer_status != 0) {
        std::cerr << "错误: CLIP infer 执行失败" << std::endl;
        return;
    }
    
    // 7. 输出性能统计
    std::cout << "\n========================================" << std::endl;
    std::cout << "推理完成!" << std::endl;
    clip.deinit(ctx_image, ctx_text);

}

// 运行精度校验模式
void run_validate_mode(const ValidateConfig& config,
                      const std::shared_ptr<ta_runtime_context>& ctx_text,
                      const std::shared_ptr<ta_runtime_context>& ctx_image) {
    
    bool is_chinese = is_chinese_mode(config.language);
    
    std::cout << "\n[模式] 数据集精度校验" << std::endl;
    std::cout << "[语言] " << (is_chinese ? "中文" : "英文") << std::endl;
    // if (!is_chinese && config.use_coarse_labels) {
    //     std::cout << "[标签] 20个粗粒度类别" << std::endl;
    // }
    std::cout << "========================================" << std::endl;
    
    // 1. 初始化CLIP模型
    std::cout << "1. 初始化CLIP模型..." << std::endl;
    CLIP clip;
    int status = clip.init(config.image_model_path, 
                          config.text_model_path,
                          ctx_text,
                          ctx_image,
                          config.text_projection_path,
                          is_chinese);
    
    if (status != 0) {
        clip.deinit(ctx_image, ctx_text);
        throw std::runtime_error("CLIP模型初始化失败");
    }
    std::cout << "   CLIP模型初始化成功" << std::endl;
    
    // 2. 创建并运行验证器
    std::cout << "\n2. 初始化验证器..." << std::endl;
    CIFAR100_Validator validator(clip, ctx_text, ctx_image, 
                                is_chinese, config.vocab_path
                                );
    
    std::cout << "\n3. 开始评估..." << std::endl;
    validator.validate_accuracy(config.dataset_path, config.max_samples);

    std::cout << "\n========================================" << std::endl;
    std::cout << "评估完成!" << std::endl;
    clip.deinit(ctx_image, ctx_text);


}