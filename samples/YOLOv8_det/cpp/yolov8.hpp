
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
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ta-runtime-api.h>
#include <unordered_map> 

#include <unistd.h>


typedef float (*DequantizeFunc)(void* data, size_t idx, int32_t zp, float scale);


extern const char* CLASS_NAMES[];



enum DataType {
    kFloat32,   // 32位浮点数 (float)
    kFloat16,   // 16位浮点数 (__fp16/half)
    kUInt8,     // 无符号8位整型 (uint8_t)
    kInt8,      // 有符号8位整型 (int8_t)
    kInt16,     // 16位整型 (int16_t)
    kInt32,     // 32位整型 (int32_t)
    kBFloat16   // 谷歌大脑的16位浮点格式
};

class TimeStamp
{
private:
    std::chrono::system_clock::time_point start_time, end_time;

public:
    TimeStamp()
    {
        start();
        time_map_lab["imread_time"] = 0.0f;
        time_map_lab["pre_time"] = 0.0f;
        time_map_lab["infer_time"] = 0.0f;
        time_map_lab["post_time"] = 0.0f;
        // time_map_lab["hardware_time"] = 0.0f;
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

typedef struct PreprocessParams {
    float ratio;     
    int top;         
    int bottom;      
    int left;        
    int right;       
    cv::Size src_size; 
} PreprocessParams;

struct YoloV8Box {
  float x, y, width, height;
  int class_id;
  float score;
};

using YoloV8BoxVec = std::vector<YoloV8Box>;

class YOLOV8 {
    
    float conf_thresh;
    float nms_thresh;
    std::shared_ptr<ta_runtime_context> nnrt_context;
    taconn_input_t* input_tensors;
    taconn_buffer_t* output_buffer;
    int input_num;
    int output_num;
    DataType output_dtype = DataType::kFloat32;  // 输出数据类型
    std::vector<taconn_inout_attr_t> ins_attr;
    std::vector<taconn_inout_attr_t> outs_attr;
    
  // model info 
  
    int model_input_h = 640;
    int model_input_w = 640;

    std::string model_name;
    bool initialized_ = false;
    // for profiling
    TimeStamp *ts_ = NULL;
    
    std::vector<PreprocessParams> pre_params;
  private:

    int preprocess_image(const cv::Mat& src, cv::Mat& dst,PreprocessParams& params);
    void post_process(const cv::Mat& image, const PreprocessParams& pre_params,
                    YoloV8BoxVec& objects);
    size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count) ;
    int pre_process(cv::Mat &images);
    size_t get_element_num(const taconn_inout_attr_t input_attr);
    void mat_to_tensor(const cv::Mat& mat, uint8_t* tensor);
    void normalize_and_quantize(uint8_t* src, void* dst, size_t num_elements, 
                                    const taconn_inout_attr_t& attr);
    float g_sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

    void generate_proposals_yolov8_generic(
            int stride, void* feat, DataType dtype, int32_t zp, float scale,
            float prob_threshold, YoloV8BoxVec& objects,
            int letterbox_cols, int letterbox_rows, int cls_num);

  public:
    std::vector<taconn_profile_t> hardware_time;
    YOLOV8(std::shared_ptr<ta_runtime_context> context,std::string model_file,
            float conf_thresh_i,float nms_thresh_i);
    ~YOLOV8();
    void enableProfile(TimeStamp *ts);
    void draw_objects(const cv::Mat& bgr, const YoloV8BoxVec& objects, 
                    const std::string& output_name, double fontScale = 0.5, int thickness = 1);
    int Init();
    size_t element_size(DataType dtype);
    int Detect(std::vector<cv::Mat>& input_images, std::vector<YoloV8BoxVec>& box);
    int deInit();
};



