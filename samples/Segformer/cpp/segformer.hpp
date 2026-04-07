#pragma once
#include <nlohmann/json.hpp>  
#include <filesystem>
#include <algorithm>
#include <cctype>

#include <chrono>
#include <string>
#include <cstdio>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <dirent.h>
#include <iomanip>
#include <map>

#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ta-runtime-api.h>
#include <unordered_map> 
#include <unistd.h>

typedef float (*DequantizeFunc)(void* data, size_t idx, int32_t zp, float scale);

#define CORE_0 0
#define CORE_1 1

size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count); 
size_t get_element_num(const taconn_inout_attr_t input_attr);
void mat_to_tensor(const cv::Mat& mat, uint8_t* tensor);

std::vector<std::string> get_subdirectories(const std::string& directory);
bool is_image_file(const std::string& filename);


class TimeStamp {
private:
    std::chrono::system_clock::time_point start_time, end_time;

public:
    TimeStamp() {
        start();
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
    
    // 公共成员函数声明
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
    
    // 获取上下文的公共方法
    std::shared_ptr<ta_runtime_context> get_context() const { return nnrt_ctx_; }
    float get_last_inference_time() const { return last_inference_ms_; }
private:

    mutable std::chrono::high_resolution_clock::time_point inference_start_;
    mutable float last_inference_ms_ = 0.0f;
    std::shared_ptr<ta_runtime_context> nnrt_ctx_;  
    Config config_;
    bool rt_flag = false;
}; 


// 量化工具类
class QuantizationUtils {
public:
    // 反量化函数指针类型
    typedef float (*DequantizeFunc)(void* data, size_t idx, int32_t zp, float scale);

    // 反量化函数
    static float dequantize_float32(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_float16(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_uint8(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_int8(void* data, size_t idx, int32_t zp, float scale);
    static float dequantize_int32(void* data, size_t idx, int32_t zp, float scale);
    static DequantizeFunc get_dequantize_func(uint32_t data_format);

    // 量化函数
    static void quantize(float* src, void* dst, size_t num_elements, const taconn_inout_attr_t& attr);

    // float32到float16转换
    static uint16_t f32_to_f16(float value);
};

using QU = QuantizationUtils;







class ModelManager {
public:
    struct ModelInfo {
        int input_num = 0;
        int output_num = 0;
        int model_input_height = 512;
        int model_input_width = 1024;
        std::string model_name;
        
        std::vector<taconn_input_t> input_tensors;
        std::vector<taconn_buffer_t> output_buffers;
        std::vector<taconn_inout_attr_t> input_attributes;
        std::vector<taconn_inout_attr_t> output_attributes;
    };
    
    ModelManager(std::shared_ptr<Runtime> runtime = nullptr);
    ~ModelManager();
    
    void set_runtime(std::shared_ptr<Runtime> runtime);
    int init_runtime(const std::string& model_path, const Runtime::Config& runtime_config = {});
    void destory_iomem();
    
    const ModelInfo& get_model_info() const { return model_info_; }
    taconn_input_t* get_input_tensor(int index);
    taconn_buffer_t* get_output_buffer(int index);

    int run_network();
    int quantize_input_tensor(int index, const cv::Mat& preprocessed_data);
    cv::Mat get_output_data(int index);    

    bool is_model_loaded() const { return model_loaded_; }
    static size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count);
    static size_t get_element_num(const taconn_inout_attr_t& attr);
    
private:
    std::shared_ptr<Runtime> runtime_;
    ModelInfo model_info_;
    bool model_loaded_ = false;


};

class Segformer {
public:
    struct SegformerConfig {
        std::string model_path;
        int core_id = 0;
        int memory_alignment = 256;
        cv::Size target_size = cv::Size(1024, 512);
        
        bool normalize = true;

        cv::Scalar mean = cv::Scalar(123.675, 116.28, 103.53);
        cv::Scalar stddev = cv::Scalar(58.395, 57.12, 57.375);
        int num_classes = 19;
    };
    
    Segformer();
    explicit Segformer(const SegformerConfig& config);
    ~Segformer();
    
    int init(const std::string& model_path);
    void deinit();  
    cv::Mat process(const cv::Mat& image);
    
    bool is_initialized() const { return init_flag; }
    cv::Size get_input_size() const;
    const SegformerConfig& get_config() const { return config_; }
    
    int get_total_inferences() const { return total_inferences_; }
    float get_avg_inference_time() const;
    
private:
    cv::Mat preprocess(const cv::Mat& image);
    bool prepare_model_input(const cv::Mat& preprocessed);
    cv::Mat postprocess_results();
    void init_color_palette();
    
    std::shared_ptr<Runtime> runtime_;
    std::shared_ptr<ModelManager> model_manager_;
    SegformerConfig config_;
    
    bool init_flag = false;
    int total_inferences_ = 0;
    std::vector<float> inference_times_;
    std::vector<cv::Vec3b> color_palette_;
};