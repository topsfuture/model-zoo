// insightface.hpp
#ifndef INSIGHTFACE_HPP
#define INSIGHTFACE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <map>
#include "ta-runtime-api.h"

// 基本数据结构定义
struct BBox {
    float x1, y1, x2, y2, score;
    
    BBox() : x1(0), y1(0), x2(0), y2(0), score(0) {}
    BBox(float _x1, float _y1, float _x2, float _y2, float _score) 
        : x1(_x1), y1(_y1), x2(_x2), y2(_y2), score(_score) {}
    
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
};

struct Landmarks {
    std::array<cv::Point2f, 5> points;
    
    Landmarks() {
        for (auto& p : points) p = cv::Point2f(0, 0);
    }
};

struct DetectionResult {
    BBox bbox;
    Landmarks landmarks;
    float score;
    
    DetectionResult() : score(0) {}
};


class TimeStamp {
private:
    std::chrono::system_clock::time_point start_time, end_time;

public:
    TimeStamp() {
        start();
        time_map_lab["imread_time"] = 0.0f;
        time_map_lab["preprocess_time"] = 0.0f;
        time_map_lab["inference_time"] = 0.0f;
        time_map_lab["postprocess_time"] = 0.0f;
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

    std::map<std::string, float> time_map_lab;
    
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
        if (it == time_map_lab.end())
            return; 
        time_map_lab[label] += gap/1000.f;
    }
};


class Runtime {
public:
    struct Config {
        int core_id = 0;
        int memory_alignment = 256;
    };
    
    Runtime();
    ~Runtime();
    
    int init_ctx();
    int init_ctx(const Config& config);
    int load_model(const std::string& model_path);
    int query_io_count(int& input_num, int& output_num);
    int get_input_attribute(int index, taconn_inout_attr_t& attr);
    int get_output_attribute(int index, taconn_inout_attr_t& attr);
    int allocate_input_tensor(int index, size_t size, taconn_input_t& tensor);
    int allocate_output_buffer(int index, size_t size, taconn_buffer_t& buffer);
    int set_input_tensors(int count, taconn_input_t* tensors);
    int set_output_buffers(int count, taconn_buffer_t* buffers);
    int run_inference();
    void cleanup();
    
    std::shared_ptr<ta_runtime_context> get_context() { return nnrt_ctx_; }
    
private:
    Config config_;
    std::shared_ptr<ta_runtime_context> nnrt_ctx_ = nullptr;
    bool rt_flag = false;
};

class QuantizationUtils {
public:
    // 反量化相关函数
    static float dequantize_float32(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_float16(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_uint8(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_int8(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_int32(void* data, size_t idx, int32_t zp, float scale);
    
    // 量化相关函数
    static uint16_t f32_to_f16(float value);
    static void quantize(float* src, void* dst, size_t num_elements, const taconn_inout_attr_t& attr);
    
    // 反量化函数指针类型
    using DequantizeFunc = float(*)(void*, size_t, int32_t, float);
    
    // 获取反量化函数
    static DequantizeFunc get_dequantize_func(uint32_t data_format);
};

struct ModelInfo {
    int model_input_width = 0;
    int model_input_height = 0;
    int model_output_width = 0;
    int model_output_height = 0;
    int input_channel = 0;
    std::string model_name = "unknown";
    
    int input_num = 0;
    int output_num = 0;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<taconn_inout_attr_t> input_attributes;
    std::vector<taconn_inout_attr_t> output_attributes;
    
    std::vector<taconn_input_t> input_tensors;
    std::vector<taconn_buffer_t> output_buffers;
};

class ModelManager {
public:
    ModelManager(std::shared_ptr<Runtime> runtime);
    ~ModelManager();
    
    int init_runtime(const std::string& model_path, const Runtime::Config& runtime_config);
    int quantize_input_tensor(int input_index, const cv::Mat& input);
    int run_network();
    cv::Mat get_output_data(int output_index);
    void destroy_iomem();
    
    void set_runtime(std::shared_ptr<Runtime> runtime);
    
    taconn_input_t* get_input_tensor(int index);
    taconn_buffer_t* get_output_buffer(int index);
    size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count);
    size_t get_element_num(const taconn_inout_attr_t& attr);
    
    const ModelInfo& get_model_info() const { return model_info_; }
    
private:
    std::shared_ptr<Runtime> runtime_;
    ModelInfo model_info_;
    std::vector<taconn_buffer_t> input_buffers_;
    bool iomem_allocated_ = false;
    bool model_loaded_ = false;
};



class SCRFD {
public:
    struct Config {
        std::string model_path;
        int core_id = 0;
        int memory_alignment = 256;
        cv::Size target_size = cv::Size(640, 640);
        float score_threshold = 0.5f;
        float nms_threshold = 0.4f;
        
        cv::Scalar mean = cv::Scalar(127.5, 127.5, 127.5);
        float scale = 1.0f / 128.0f;
        bool swap_rb = true;
        
        bool profile = true;
        int debug_level = 0;
    };
    
    SCRFD();
    explicit SCRFD(const Config& config);
    ~SCRFD();
    
    int init(const std::string& model_path);
    void deinit();
    
    std::vector<DetectionResult> detect(const cv::Mat& image);
    
    cv::Mat draw(const cv::Mat& image, const std::vector<DetectionResult>& results);
    
    bool is_initialized() const { return initialized_; }
    const Config& get_config() const { return config_; }
    
    TimeStamp* m_ts;
    void enableProfile(TimeStamp *ts){
        m_ts = ts;
    };
    int total_inferences_ = 0;

private:
    bool preprocess(const cv::Mat& image, float& det_scale);
    bool run_model();
    std::vector<DetectionResult> postprocess(const cv::Mat& image, float det_scale);
    
    std::vector<cv::Point2f> generate_anchors(int feat_height, int feat_width, int stride);
    BBox decode_bbox(const cv::Point2f& anchor, const float* bbox_delta, float stride);
    Landmarks decode_landmarks(const cv::Point2f& anchor, const float* kps_delta, float stride);
    float calculate_iou(const BBox& bbox1, const BBox& bbox2);
    std::vector<int> nms(const std::vector<DetectionResult>& detections, float iou_threshold);
    
    std::shared_ptr<Runtime> runtime_;
    std::shared_ptr<ModelManager> model_manager_;
    Config config_;
    bool initialized_ = false;
    


};

cv::Mat align_face(const cv::Mat& image, const Landmarks& landmarks);

class FaceRecognizer {
public:
    struct Config {
        std::string model_path;
        int core_id = 0;
        cv::Size input_size = cv::Size(112, 112);
        cv::Scalar mean = cv::Scalar(127.5, 127.5, 127.5);
        float scale = 1.0f / 128.0f;
        bool swap_rb = true;
        bool profile = true;
    };

    FaceRecognizer();
    explicit FaceRecognizer(const Config& config);
    ~FaceRecognizer();

    int init(const std::string& model_path);
    void deinit();

    std::vector<float> extract_feature(const cv::Mat& aligned_face);

    bool is_initialized() const { return initialized_; }
    const Config& get_config() const { return config_; }
    
    void enableProfile(TimeStamp *ts){
        m_ts = ts;
    };
    TimeStamp* m_ts;

    int total_inferences_ = 0;

private:
    bool preprocess(const cv::Mat& aligned_face);
    bool run_model();
    std::vector<float> postprocess();

    std::shared_ptr<Runtime> runtime_;
    std::shared_ptr<ModelManager> model_manager_;
    Config config_;
    bool initialized_ = false;
    


};

#endif // INSIGHTFACE_HPP