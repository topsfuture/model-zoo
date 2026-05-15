#ifndef RETINAFACE_HPP
#define RETINAFACE_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ta-runtime-api.h>
#include <vector>
#include <string>
#include <map>
#include <chrono> // TimeStamp 需要用到

// ==========================================
// 数据结构定义
// ==========================================

// 命令行参数结构体 (修复 parse_arguments 缺少返回值定义的问题)
struct CmdParams {
    std::string input;
    std::string model;
    std::string output;
    float conf_thresh;
    float nms_thresh;
    bool batch_mode;
    bool save_result;
};

// 检测框结构体
typedef struct facebox_s {
    float x1, y1, x2, y2;
    float score;
    float landmarks[10];  // 5个关键点
    int class_id;        // 类别 ID
} facebox;

// 预处理参数
typedef struct PreprocessParams {
    float ratio;
    int top;
    int bottom;
    int left;
    int right;
    cv::Size src_size;
} PreprocessParams;

// 先验框等配置参数
struct RetinaFaceConfig {
    std::vector<std::vector<int>> min_sizes = {{16, 32}, {64, 128}, {256, 512}};
    std::vector<int> steps = {8, 16, 32};
    std::vector<float> variance = {0.1f, 0.2f};
    int image_size = 640;
    bool clip = false;
};

// ==========================================
// 性能分析工具类
// ==========================================

class TimeStamp {
private:
    std::chrono::system_clock::time_point start_time, end_time;

public:
    std::map<std::string, float> time_map_lab;

    TimeStamp() {
        start();
        time_map_lab["imread_time"] = 0.0f;
        time_map_lab["pre_time"] = 0.0f;
        time_map_lab["infer_time"] = 0.0f;
        time_map_lab["post_time"] = 0.0f;
        time_map_lab["hardware_time"] = 0.0f;
    }

    void start() {
        stop();
        this->start_time = this->end_time;
    }

    void stop() {
#ifdef _MSC_VER
        this->end_time = std::chrono::system_clock::now();
#else
        this->end_time = std::chrono::high_resolution_clock::now();
#endif
    }

    float cost() {
        if (this->end_time <= this->start_time) {
            this->stop();
        }
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        return static_cast<float>(ms) / 1000.f;
    }

    void time_accumulation(std::string label) {
        if (this->end_time <= this->start_time) {
            this->stop();
        }
        auto gap = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        auto it = time_map_lab.find(label);
        if (it == time_map_lab.end()) return; 
        time_map_lab[label] += gap / 1000.f;
    }
};

// ==========================================
// RetinaFace 推理类
// ==========================================

class RetinaFace {
public:
    RetinaFace();
    ~RetinaFace();

    TimeStamp *ts_ = nullptr;

    // 初始化和释放
    bool init(const std::string& model_path);
    void deinit();

    // 单图推理并保存结果
    bool detect_and_save(const cv::Mat& image,
                         const std::string& output_path,
                         std::vector<facebox>& objects,
                         float conf_thresh = 0.25f,
                         float nms_thresh = 0.45f);

    struct BatchResult {
        std::string image_name;
        std::vector<facebox> objects;
        float inference_time;
        float postprocess_time;
    };

    // 性能分析
    void enableProfile(TimeStamp *ts);

    // 绘制结果 (从 private 移到 public, 方便 main 函数调用)
    void draw_objects(const cv::Mat& bgr, const std::vector<facebox>& objects, 
                      const std::string& output_name);

private:
    // 模型上下文
    struct ModelContext {
        ta_runtime_context nnrt_context;
        taconn_input_t* input_tensors;
        taconn_buffer_t* output_buffer;
        int input_num;
        int output_num;
        std::vector<taconn_inout_attr_t> ins_attr;
        std::vector<taconn_inout_attr_t> outs_attr;
    };
    
    ModelContext ctx_;
    bool initialized_;
    
    // 模型输入尺寸
    int input_height_;
    int input_width_;

    // --- 核心流转函数 ---
    void preprocess_image(const cv::Mat& src, cv::Mat& dst, PreprocessParams& params);
    
    std::vector<facebox> postprocess(const std::vector<float>& loc_data, 
                                     const std::vector<float>& conf_data,
                                     const std::vector<float>& landm_data, 
                                     float score_thresh, 
                                     float nms_thresh, 
                                     float resize_ratio);

    // --- 数据转换与工具函数 ---
    std::vector<float> get_real_float_data(void* raw_data, const taconn_inout_attr_t& attr);

    size_t get_element_num(const taconn_inout_attr_t& input_attr);
    
    size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count);
    
    #if 0
    void normalize_and_quantize(uint8_t* src, void* dst, size_t num_elements, 
                                const taconn_inout_attr_t& attr);
    #else
    void normalize_and_quantize(const std::vector<cv::Mat>& channels, void* dst, 
        const taconn_inout_attr_t& attr);
    #endif
};

// ==========================================
// 全局工具函数
// ==========================================

bool file_exists(const std::string& path);
CmdParams parse_arguments(int argc, char** argv);

#endif