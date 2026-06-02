
#pragma once

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


enum DataType {
    kFloat32,   // 32位浮点数 (float)
    kFloat16,   // 16位浮点数 (__fp16/half)
    kUInt8,     // 无符号8位整型 (uint8_t)
    kInt8,      // 有符号8位整型 (int8_t)
    kInt16,     // 16位整型 (int16_t)
    kInt32,     // 32位整型 (int32_t)
    kBFloat16   // 谷歌大脑的16位浮点格式
};


size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count); 
size_t get_element_num(const taconn_inout_attr_t input_attr);
void mat_to_tensor(const cv::Mat& mat, uint8_t* tensor);


class TimeStamp
{
private:
    std::chrono::system_clock::time_point start_time, end_time;

public:
    TimeStamp()
    {
        start();
        time_map_lab["tokenize_time"] = 0.0f;
        time_map_lab["imread_time"] = 0.0f;
        time_map_lab["pre_time"] = 0.0f;
        time_map_lab["encode_text_time"] = 0.0f;
        time_map_lab["encode_image_time"] = 0.0f;
        time_map_lab["text_postprocess_time"] = 0.0f;
        time_map_lab["image_postprocess_time"] = 0.0f;
    }

    void start()
    {
        stop();
        this->start_time = this->end_time;
    }

    void stop()
    {
#ifdef _MSC_VER
        this->end_time = std::chrono::system_clock::now();
#else
        this->end_time = std::chrono::high_resolution_clock::now();
#endif
    }

    std::map<std::string, float> time_map_lab;
    float cost()
    {
        if (this->end_time <= this->start_time)
        {
            this->stop();
        }

        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        return static_cast<float>(ms) / 1000.f;
    }
    void time_accumulation(std::string label){
        if (this->end_time <= this->start_time)
        {
            this->stop();
        }

        auto gap = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        auto it = time_map_lab.find(label);
        if (it == time_map_lab.end())
            return; 
        time_map_lab[label] += gap/1000.f;
    }
};






class CLIP{
public:

    int init(const std::string& image_model, const std::string& text_model,
                const std::shared_ptr<ta_runtime_context>& nnrt_context_text,
                const std::shared_ptr<ta_runtime_context>& nnrt_context_images,
                const std::string& text_projection_path = "",
                bool is_chinese = false
                );
    void deinit(std::shared_ptr<ta_runtime_context> nnrt_context_images,
              std::shared_ptr<ta_runtime_context> nnrt_context_text); 
    void preprocess(const cv::Mat& image);
    std::vector<float> encode_image(const std::string image_path , const std::shared_ptr<ta_runtime_context>& nnrt_context_images);
    std::vector<float> encode_text(const std::vector<int>& text, const std::shared_ptr<ta_runtime_context>& nnrt_context_text);
    std::vector<float> calculate_similarity(const std::vector<float>& image_features,
                                        const std::vector<std::vector<float>>& text_features);
    std::pair<std::vector<float>, std::vector<int>> topk(const std::vector<float>& x, int k);
    size_t get_token_len() const;

    void quantize(float* src, void* dst, size_t num_elements, 
                                   const taconn_inout_attr_t& attr);
    int infer(const std::vector<std::string>& image_paths, const std::vector<std::vector<int>> tokenlized_text,
           const std::vector<std::string>& text_inputs,
           const std::shared_ptr<ta_runtime_context>& nnrt_context_text,
           const std::shared_ptr<ta_runtime_context>& nnrt_context_images);

    void enableProfile(TimeStamp *ts);
    std::vector<float> encode_image_memory(const cv::Mat& image, 
                                      const std::shared_ptr<ta_runtime_context>& nnrt_context_images);
    std::string model_name;
    int top_k = 5;

    TimeStamp *ts_ = NULL;


private:
    bool is_chinese_;

    std::vector<float> preprocess_cpu_letterbox(const cv::Mat& image);
    cv::Mat mobile_clip_preprocess(const cv::Mat& image);
    
    std::tuple<cv::Mat, std::pair<float, float>, std::pair<float, float>> letterbox(const cv::Mat& im, const cv::Size& new_shape, 
        const cv::Scalar& color = cv::Scalar(114, 114, 114), bool auto_pad = false, bool scaleFill = false, 
        bool scaleup = true, int stride = 32);

    void normalize(std::vector<float>& features);
    std::vector<float> softmax(const std::vector<float>& x);
    
    struct ImageModelInfo{
        std::vector<taconn_input_t> input_tensors;
        std::vector<taconn_buffer_t> output_buffers;
        int input_num;
        int output_num;
        std::vector<taconn_inout_attr_t> ins_attr;
        std::vector<taconn_inout_attr_t> outs_attr;
        int model_input_h = 224;
        int model_input_w = 224;
        std::string model_name;
    } image_model_info;

    struct TextModelInfo{
        std::vector<taconn_input_t> input_tensors;
        std::vector<taconn_buffer_t> output_buffers;

        int input_num;
        int output_num;
        std::vector<taconn_inout_attr_t> ins_attr;
        std::vector<taconn_inout_attr_t> outs_attr;
        int batch_size = 1;
        int token_len = 52;
        
        std::string model_name;
    } text_model_info;

    // 使用 OpenCV Mat 
    cv::Mat text_projection;
    size_t embed_dim = 512; 
    size_t hidden_dim = 512;



};