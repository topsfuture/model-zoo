#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ta-runtime-api.h>
#include <vector>
#include <string>
#include <map>


// 枚举定义
enum class TensorLayout {
    HWC, CHW, WHC, HCW, CWH, WCH
};

// 结构体定义
typedef struct box_rect_s {
    float left;
    float top;
    float height;
    float width;
} bbox;

typedef struct object_s {
    bbox box;
    int class_id;
    float prob;
} Object;

typedef struct PreprocessParams {
    float ratio;
    int top;
    int bottom;
    int left;
    int right;
    cv::Size src_size;
} PreprocessParams;

typedef struct CmdParams {
    float conf_thresh;
    float nms_thresh;
    std::string input;
    std::string model;
    std::string output;
    bool batch_mode;
    bool save_result;  // 批量推理时是否保存检测结果图片
} CmdParams;


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
        time_map_lab["hardware_time"] = 0.0f;
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


// YOLO11检测器类
class YOLO11Detector {
public:
    YOLO11Detector();
    ~YOLO11Detector();
    TimeStamp *ts_ = NULL;
    // 初始化和释放
    bool init(const std::string& model_path);
    void deinit();

    // 单图推理并保存结果
    bool detect_and_save(const cv::Mat& image,
                        const std::string& output_path,
                        std::vector<Object>& objects,
                        float conf_thresh = 0.25f,
                        float nms_thresh = 0.45f);

    struct BatchResult {
        std::string image_name;
        std::vector<Object> objects;
        float inference_time;
        float postprocess_time;
    };

    // 性能分析
    void enableProfile(TimeStamp *ts);

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

    // 内部辅助函数
    void preprocess_image(const cv::Mat& src, cv::Mat& dst, PreprocessParams& params);
    bool inference();
    void post_process(const cv::Mat& image, 
                     const PreprocessParams& pre_params,
                     float conf_thresh, 
                     float nms_thresh,
                     std::vector<Object>& objects,
                     const std::string& output_image_path = "");

    // 反量化相关 - 函数重载
    float dequantize(float value, int32_t zp, float scale);
    float dequantize(uint16_t value, int32_t zp, float scale);
    float dequantize(uint8_t value, int32_t zp, float scale);
    float dequantize(int8_t value, int32_t zp, float scale);
    
    float dequantize_value(void* data, size_t idx, uint32_t data_format, int32_t zp, float scale);
    
    // Softmax计算
    float softmax_with_stride(void* src, float* dst, int length, 
                             int stride, int32_t zp, float scale, uint32_t data_format);
    
    // Proposal生成
    void generate_proposals(int stride, void* feat, uint32_t data_format, 
                           int32_t zp, float scale,
                           float prob_threshold, std::vector<Object>& objects,
                           int letterbox_cols, int letterbox_rows);

    // NMS相关
    void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& objects);
    float intersection_area(const Object& a, const Object& b);
    void nms_sorted_bboxes(const std::vector<Object>& objects, 
                          std::vector<int>& picked, float nms_threshold);

    // 坐标转换
    void inverse_coordinates(bbox& box, const PreprocessParams& params);

    // 绘制结果
    void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, 
                     const std::string& output_name);

    // 工具函数
    size_t get_element_num(const taconn_inout_attr_t& input_attr);
    size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count);
    size_t element_size(uint32_t data_format);
    void normalize_and_quantize(uint8_t* src, void* dst, size_t num_elements, 
                               const taconn_inout_attr_t& attr);
    uint16_t float32_to_float16(float value);
    void mat_to_tensor(const cv::Mat& mat, uint8_t* tensor);
    void print_taconn_inout_attr(const taconn_inout_attr_t& attr);
};

// 全局工具函数
bool file_exists(const std::string& path);
CmdParams parse_arguments(int argc, char** argv);

#endif // YOLO_DETECTOR_HPP
