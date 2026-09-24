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
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <float.h>
#include <unistd.h>

// ==================== Constants ====================
static const int NUM_CLASSES = 80;
static const int NUM_MASKS = 32;
static const int MAX_DET = 300;
static const float MASK_THRES = 0.5f;
static const int MAX_STRIDE = 32;

// COCO class names
static const char* CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// COCO colors (BGR format for OpenCV)
static const std::vector<cv::Scalar> COCO_COLORS = {
    {128, 56, 0, 255}, {128, 226, 255, 0}, {128, 0, 94, 255}, {128, 0, 37, 255}, {128, 0, 255, 94},
    {128, 255, 226, 0}, {128, 0, 18, 255}, {128, 255, 151, 0}, {128, 170, 0, 255}, {128, 0, 255, 56},
    {128, 255, 0, 75}, {128, 0, 75, 255}, {128, 0, 255, 169}, {128, 255, 0, 207}, {128, 75, 255, 0},
    {128, 207, 0, 255}, {128, 37, 0, 255}, {128, 0, 207, 255}, {128, 94, 0, 255}, {128, 0, 255, 113},
    {128, 255, 18, 0}, {128, 255, 0, 56}, {128, 18, 0, 255}, {128, 0, 255, 226}, {128, 170, 255, 0},
    {128, 255, 0, 245}, {128, 151, 255, 0}, {128, 132, 255, 0}, {128, 75, 0, 255}, {128, 151, 0, 255},
    {128, 0, 151, 255}, {128, 132, 0, 255}, {128, 0, 255, 245}, {128, 255, 132, 0}, {128, 226, 0, 255},
    {128, 255, 37, 0}, {128, 207, 255, 0}, {128, 0, 255, 207}, {128, 94, 255, 0}, {128, 0, 226, 255},
    {128, 56, 255, 0}, {128, 255, 94, 0}, {128, 255, 113, 0}, {128, 0, 132, 255}, {128, 255, 0, 132},
    {128, 255, 170, 0}, {128, 255, 0, 188}, {128, 113, 255, 0}, {128, 245, 0, 255}, {128, 113, 0, 255},
    {128, 255, 188, 0}, {128, 0, 113, 255}, {128, 255, 0, 0}, {128, 0, 56, 255}, {128, 255, 0, 113},
    {128, 0, 255, 188}, {128, 255, 0, 94}, {128, 255, 0, 18}, {128, 18, 255, 0}, {128, 0, 255, 132},
    {128, 0, 188, 255}, {128, 0, 245, 255}, {128, 0, 169, 255}, {128, 37, 255, 0}, {128, 255, 0, 151},
    {128, 188, 0, 255}, {128, 0, 255, 37}, {128, 0, 255, 0}, {128, 255, 0, 170}, {128, 255, 0, 37},
    {128, 255, 75, 0}, {128, 0, 0, 255}, {128, 255, 207, 0}, {128, 255, 0, 226}, {128, 255, 245, 0},
    {128, 188, 255, 0}, {128, 0, 255, 18}, {128, 0, 255, 75}, {128, 0, 255, 151}, {128, 255, 56, 0},
    {128, 245, 255, 0}
};

// ==================== Data Structures ====================
struct bbox {
    float left, top, width, height;
};

struct Object {
    bbox box;
    int class_id;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_coef;
};

struct PreprocessParams {
    float ratio;
    int top, bottom, left, right;
    cv::Size src_size;
};

// ==================== TimeStamp Class ====================
class TimeStamp {
private:
    std::chrono::system_clock::time_point start_time, end_time;
public:
    TimeStamp() {
        start();
        time_map_lab["imread_time"] = 0.0f;
        time_map_lab["pre_time"] = 0.0f;
        time_map_lab["infer_time"] = 0.0f;
        time_map_lab["post_time"] = 0.0f;
    }
    void start() {
        stop();
        this->start_time = this->end_time;
    }
    void stop() {
        this->end_time = std::chrono::high_resolution_clock::now();
    }
    std::map<std::string, float> time_map_lab;
    float cost() {
        if (this->end_time <= this->start_time) this->stop();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        return static_cast<float>(ms) / 1000.f;
    }
    void time_accumulation(std::string label) {
        if (this->end_time <= this->start_time) this->stop();
        auto gap = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        auto it = time_map_lab.find(label);
        if (it == time_map_lab.end()) return;
        time_map_lab[label] += gap / 1000.f;
    }
};

// ==================== YOLO11SegDetector Class ====================
class YOLO11SegDetector {
public:
    YOLO11SegDetector();
    ~YOLO11SegDetector();

    bool init(const std::string& model_path);
    void deinit();

    bool detect_and_save(const cv::Mat& image,
                         const std::string& output_path,
                         std::vector<Object>& objects,
                         float conf_thresh,
                         float nms_thresh,
                         int max_det = -1);

    void enableProfile(TimeStamp* ts) { ts_ = ts; }
    static bool file_exists(const std::string& path);
    static void print_taconn_inout_attr(const taconn_inout_attr_t& attr);

private:
    taconn_input_t* input_tensors_ = nullptr;
    taconn_buffer_t* output_buffer_ = nullptr;
    taconn_input_output_num_t num_;
    int input_num_ = 0;
    int output_num_ = 0;
    std::vector<taconn_inout_attr_t> ins_attr_;
    std::vector<taconn_inout_attr_t> outs_attr_;
    int input_height_ = 640;
    int input_width_ = 640;
    bool initialized_ = false;
    TimeStamp* ts_ = nullptr;
    ta_runtime_context ctx_ = 0;

    void preprocess_image(const cv::Mat& src, cv::Mat& dst, PreprocessParams& params);
    bool inference();
    void post_process(const cv::Mat& image,
                      const PreprocessParams& pre_params,
                      float conf_thresh, float nms_thresh,
                      std::vector<Object>& objects,
                      const std::string& output_image_path,
                      int max_det = -1);

    size_t get_element_num(const taconn_inout_attr_t& attr);
    size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count);
    size_t element_size(uint32_t data_format);

    float dequantize(float value, int32_t zp, float scale);
    float dequantize(uint16_t value, int32_t zp, float scale);
    float dequantize(uint8_t value, int32_t zp, float scale);
    float dequantize(int8_t value, int32_t zp, float scale);
    float dequantize_value(void* data, size_t idx, uint32_t data_format, int32_t zp, float scale);

    uint16_t float32_to_float16(float value);
    float float16_to_float32(uint16_t value);

    void generate_proposals(void* feat, uint32_t data_format,
                            int32_t zp, float scale,
                            float conf_thresh, std::vector<Object>& objects,
                            int letterbox_cols, int letterbox_rows);

    void qsort_descent_inplace(std::vector<Object>& objects);
    void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
    float intersection_area(const Object& a, const Object& b);
    void nms_sorted_bboxes(const std::vector<Object>& objects,
                           std::vector<int>& picked, float nms_thresh);

    void inverse_coordinates(bbox& box, const PreprocessParams& params);
    void inverse_coordinates_and_mask(Object& obj, const PreprocessParams& params);
    void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects,
                      const std::string& output_name);

    void mat_to_tensor(const cv::Mat& mat, uint8_t* tensor);
    void normalize_and_quantize(uint8_t* src, void* dst, size_t num_elements,
                                const taconn_inout_attr_t& attr);

    void generate_masks(const float* protos, int proto_h, int proto_w,
                        const std::vector<Object>& objects,
                        const PreprocessParams& params,
                        std::vector<Object>& output_objects);
};

// Utility
bool file_exists(const std::string& path);
