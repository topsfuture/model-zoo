#include "yolo11.hpp"
#include "timer.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <unistd.h>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ta-runtime-api.h>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include <cfloat>
#include <cmath>

// COCO类别名称
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

// COCO颜色
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

// ==================== YOLO11Detector 类实现 ====================

YOLO11Detector::YOLO11Detector() : initialized_(false), input_height_(640), input_width_(640) {
    ctx_.nnrt_context = 0;
    ctx_.input_tensors = nullptr;
    ctx_.output_buffer = nullptr;
    ctx_.input_num = 0;
    ctx_.output_num = 0;
}

YOLO11Detector::~YOLO11Detector() {
    if (initialized_) {
        deinit();
    }
}

bool YOLO11Detector::init(const std::string& model_path) {
    if (initialized_) {
        std::cerr << "Detector already initialized" << std::endl;
        return false;
    }

    // 声明变量（放在 goto 之前避免跨越初始化）
    int status;
    taconn_input_output_num_t num = {0};

    // 1. 初始化 TA Runtime
    status = ta_runtime_init();
    if (status != 0) {
        std::cerr << "Failed to initialize TACO runtime: " << status << std::endl;
        goto CLEANUP;
    }

    // 2. 设置网络上下文
    ctx_.nnrt_context = 0;

    // 3. 加载模型
    status = ta_runtime_load_model_from_file(&ctx_.nnrt_context, model_path.c_str(), 0);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    // 4. 查询输入输出数量
    status = ta_runtime_query(&ctx_.nnrt_context, TACONN_QUERY_IN_OUT_NUM, &num);
    std::cout << "Input num: " << num.input_num << ", Output num: " << num.output_num << std::endl;
    ctx_.input_num = num.input_num;
    ctx_.output_num = num.output_num;

    // 5. 查询输入属性
    for (int i = 0; i < ctx_.input_num; i++) {
        taconn_inout_attr_t input_attr = {0};
        input_attr.index = i;
        status = ta_runtime_query(&ctx_.nnrt_context, TACONN_QUERY_INPUT_ATTR, &input_attr);
        print_taconn_inout_attr(input_attr);
        ctx_.ins_attr.push_back(input_attr);
    }
    
    //只对yolo11模型有效
    if (ctx_.input_num > 0 && ctx_.ins_attr[0].dim_count >= 3) {
        input_height_ = ctx_.ins_attr[0].dim_size[1];
        input_width_ = ctx_.ins_attr[0].dim_size[0];
    }

    
    // 6. 查询输出属性
    for (int i = 0; i < ctx_.output_num; i++) {
        taconn_inout_attr_t output_attr = {0};
        output_attr.index = i;
        status = ta_runtime_query(&ctx_.nnrt_context, TACONN_QUERY_OUTPUT_ATTR, &output_attr);
        print_taconn_inout_attr(output_attr);
        ctx_.outs_attr.push_back(output_attr);
    }

    // 7. 设置输入
    ctx_.input_tensors = (taconn_input_t*)malloc(sizeof(taconn_input_t) * ctx_.input_num);
    if (!ctx_.input_tensors) {
        std::cerr << "Allocate input_tensors failed" << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < ctx_.input_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ctx_.ins_attr[i].data_format);
        size_t input_buffer_size = calculate_buffer_size(data_type, get_element_num(ctx_.ins_attr[i]));

        ctx_.input_tensors[i].index = i;
        ctx_.input_tensors[i].size = input_buffer_size;
        ctx_.input_tensors[i].data = nullptr;
        
        if (posix_memalign((void**)&ctx_.input_tensors[i].data, 256, input_buffer_size) != 0) {
            std::cerr << "Failed to allocate input buffer" << std::endl;
            goto CLEANUP;
        }
        memset(ctx_.input_tensors[i].data, 0, input_buffer_size);
    }

    status = ta_runtime_set_input_cva(&ctx_.nnrt_context, ctx_.input_num, ctx_.input_tensors);
    if (status != 0) {
        std::cerr << "Set input failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    // 8. 设置输出
    ctx_.output_buffer = (taconn_buffer_t*)malloc(sizeof(taconn_buffer_t) * ctx_.output_num);
    if (!ctx_.output_buffer) {
        std::cerr << "Allocate output_buffer failed" << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < ctx_.output_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ctx_.outs_attr[i].data_format);
        size_t output_buffer_size = calculate_buffer_size(data_type, get_element_num(ctx_.outs_attr[i]));
        
        status = ta_runtime_create_buffer(&ctx_.nnrt_context, output_buffer_size, &ctx_.output_buffer[i]);
        if (status != 0) {
            std::cerr << "Create output buffer " << i << " failed: 0x" << std::hex << status << std::dec << std::endl;
            goto CLEANUP;
        }
    }

    status = ta_runtime_set_output(&ctx_.nnrt_context, ctx_.output_num, ctx_.output_buffer);
    if (status != 0) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    initialized_ = true;
    std::cout << "Model initialized successfully" << std::endl;
    return true;
CLEANUP:
    if (ctx_.input_tensors) {
        for (int i = 0; i < ctx_.input_num; i++) {
            if (ctx_.input_tensors[i].data) {
                free(ctx_.input_tensors[i].data);
            }
        }
        free(ctx_.input_tensors);
        ctx_.input_tensors = nullptr;
    }
    if (ctx_.output_buffer) {
        for (int i = 0; i < ctx_.output_num; i++) {
            ta_runtime_destroy_buffer(&ctx_.nnrt_context, &ctx_.output_buffer[i]);
        }
        free(ctx_.output_buffer);
        ctx_.output_buffer = nullptr;
    }
    if (ctx_.nnrt_context) {
        ta_runtime_destroy_context(&ctx_.nnrt_context);
        ctx_.nnrt_context = 0;
    }
    ta_runtime_deinit();
    initialized_ = false;
    std::cout << "Model deinitialized" << std::endl;
    return false;
}

void YOLO11Detector::deinit() {
    if (!initialized_) {
        return;
    }

    // 释放输出buffer
    for (int i = 0; i < ctx_.output_num; i++) {
        ta_runtime_destroy_buffer(&ctx_.nnrt_context, &ctx_.output_buffer[i]);
    }

    // 释放输入buffer
    if (ctx_.input_tensors) {
        for (int i = 0; i < ctx_.input_num; i++) {
            if (ctx_.input_tensors[i].data) {
                free(ctx_.input_tensors[i].data);
            }
        }
        free(ctx_.input_tensors);
        ctx_.input_tensors = nullptr;
    }

    // 释放输出数组
    if (ctx_.output_buffer) {
        free(ctx_.output_buffer);
        ctx_.output_buffer = nullptr;
    }

    // 销毁上下文
    ta_runtime_destroy_context(&ctx_.nnrt_context);
    
    // 释放runtime
    ta_runtime_deinit();
    
    initialized_ = false;
    std::cout << "Model deinitialized" << std::endl;
}

bool YOLO11Detector::detect_and_save(const cv::Mat& image,
                                    const std::string& output_path,
                                    std::vector<Object>& objects,
                                    float conf_thresh,
                                    float nms_thresh) {
    if (!initialized_) {
        std::cerr << "Detector not initialized" << std::endl;
        return false;
    }

    // 预处理
    if (ts_) ts_->start();
    PreprocessParams pre_params;
    cv::Mat processed_image;
    preprocess_image(image, processed_image, pre_params);
    if (ts_) ts_->time_accumulation("pre_time");

    // 推理
    if (ts_) ts_->start();
    if (!inference()) {
        std::cerr << "Inference failed" << std::endl;
        return false;
    }
    if (ts_) ts_->time_accumulation("infer_time");

    // 后处理
    if (ts_) ts_->start();
    post_process(image, pre_params, conf_thresh, nms_thresh, objects, output_path);
    if (ts_) ts_->time_accumulation("post_time");

    return true;
}

// ==================== 私有方法实现 ====================

void YOLO11Detector::preprocess_image(const cv::Mat& src, cv::Mat& dst, PreprocessParams& params) {
    params.src_size = src.size();
    params.ratio = std::min(static_cast<float>(input_height_) / src.rows,
                           static_cast<float>(input_width_) / src.cols);
    
    int new_w = static_cast<int>(src.cols * params.ratio);
    int new_h = static_cast<int>(src.rows * params.ratio);
    
    cv::resize(src, dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    params.top = (input_height_ - new_h) / 2;
    params.bottom = input_height_ - new_h - params.top;
    params.left = (input_width_ - new_w) / 2;
    params.right = input_width_ - new_w - params.left;
    
    cv::copyMakeBorder(dst, dst, params.top, params.bottom, 
                      params.left, params.right, cv::BORDER_CONSTANT, 
                      cv::Scalar(114, 114, 114));
    
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    
    for (int i = 0; i < ctx_.input_num; i++) {
        // 检查是否有预处理节点
        if (ctx_.ins_attr[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE &&
            ctx_.ins_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            memcpy(ctx_.input_tensors[i].data, dst.data, ctx_.input_tensors[i].size);
        }
        else {
            // 转换为CHW格式
            uint8_t* input_tensor_uint8 = (uint8_t*)malloc(sizeof(uint8_t) * input_height_ * input_width_ * 3);
            mat_to_tensor(dst, input_tensor_uint8);
            
            size_t num_elements = get_element_num(ctx_.ins_attr[i]);
            normalize_and_quantize(input_tensor_uint8, ctx_.input_tensors[i].data, 
                                  num_elements, ctx_.ins_attr[i]);
            free(input_tensor_uint8);
        }
    }
}

bool YOLO11Detector::inference() {
    // 执行推理
    int status = ta_runtime_run_network(&ctx_.nnrt_context);
    if (status != 0) {
        std::cerr << "Run network failed: 0x" << std::hex << status << std::dec << std::endl;
        return false;
    }
    
    status = ta_runtime_invalidate_buffer(&ctx_.nnrt_context, ctx_.output_buffer);
    if (status != 0) {
        std::cerr << "Invalidate output buffer failed: 0x" << std::hex << status << std::dec << std::endl;
        return false;
    }
    
    return true;
}

void YOLO11Detector::post_process(const cv::Mat& image, 
                                 const PreprocessParams& pre_params,
                                 float conf_thresh, 
                                 float nms_thresh,
                                 std::vector<Object>& objects,
                                 const std::string& output_image_path) {
    std::vector<Object> proposals;
    int stride[3] = {8, 16, 32};

    // 生成proposals
    for (int i = 0; i < ctx_.output_num; ++i) {
        void* output_data = ctx_.output_buffer[i].data;
        
        uint32_t data_format = ctx_.outs_attr[i].data_format;
        
        
        // 获取量化参数
        int32_t zp = 0;
        float scale = 1.0f;
        if (ctx_.outs_attr[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
            zp = ctx_.outs_attr[i].quant_data.affine.tf_zero_point;
            scale = ctx_.outs_attr[i].quant_data.affine.tf_scale;
        }
        
        
        generate_proposals(stride[i], output_data, data_format, zp, scale,
                         conf_thresh, proposals, input_width_, input_height_);
    }

    // 排序和NMS
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh);

    // 提取最终检测结果并转换坐标
    objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); i++) {
        objects[i] = proposals[picked[i]];
        inverse_coordinates(objects[i].box, pre_params);
    }

    if (!output_image_path.empty()) {
        draw_objects(image, objects, output_image_path);
    }
}

// 反量化函数重载实现
float YOLO11Detector::dequantize(float value, int32_t, float) {
    return value;
}

float YOLO11Detector::dequantize(uint16_t value, int32_t, float) {
    // float16 转 float32
    uint32_t sign = (value >> 15) & 0x1;
    uint32_t exponent = (value >> 10) & 0x1F;
    uint32_t mantissa = value & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t f = (sign << 31);
            return *reinterpret_cast<float*>(&f);
        }
        const uint32_t exp_offset = 103;
        uint32_t f = (sign << 31) | (exp_offset << 23) | (mantissa << 13);
        return *reinterpret_cast<float*>(&f);
    } 
    else if (exponent == 31) {
        uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&f);
    }
    
    exponent += (127 - 15);
    uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *reinterpret_cast<float*>(&f);
}

float YOLO11Detector::dequantize(uint8_t value, int32_t zp, float scale) {
    return ((float)value - (float)zp) * scale;
}

float YOLO11Detector::dequantize(int8_t value, int32_t zp, float scale) {
    return ((float)value - (float)zp) * scale;
}

// 通用反量化函数
float YOLO11Detector::dequantize_value(void* data, size_t idx, uint32_t data_format, 
                                      int32_t zp, float scale) {

     
    switch (data_format) {
        case taconn_data_format_e::TACONN_DATA_FORMAT_FP32:
            return dequantize(static_cast<float*>(data)[idx], zp, scale);
        case taconn_data_format_e::TACONN_DATA_FORMAT_FP16:
            return dequantize(static_cast<uint16_t*>(data)[idx], zp, scale);
        case taconn_data_format_e::TACONN_DATA_FORMAT_UINT8:
            return dequantize(static_cast<uint8_t*>(data)[idx], zp, scale);
        case taconn_data_format_e::TACONN_DATA_FORMAT_INT8:
            return dequantize(static_cast<int8_t*>(data)[idx], zp, scale);
        default:
            std::cerr << "WARNING: Unknown data_format: 0x" 
                     << std::hex << data_format << std::dec << std::endl;
            return 0.0f;
    }
}

float YOLO11Detector::softmax_with_stride(void* src, float* dst, int length, 
                                         int stride, int32_t zp, float scale, uint32_t data_format) {
    // 找出最大值
    float max_val = -FLT_MAX;
    for (int i = 0; i < length; i++) {
        float val = dequantize_value(src, i * stride, data_format, zp, scale);
        if (val > max_val) max_val = val;
    }

    // 计算指数和
    float sum = 0.f;
    for (int i = 0; i < length; i++) {
        float val = dequantize_value(src, i * stride, data_format, zp, scale);
        float exp_val = expf(val - max_val);
        dst[i] = exp_val;
        sum += exp_val;
    }

    // 归一化并计算加权和
    float weighted_sum = 0.f;
    for (int i = 0; i < length; i++) {
        dst[i] /= sum;
        weighted_sum += i * dst[i];
    }

    return weighted_sum;
}

void YOLO11Detector::generate_proposals(int stride, void* feat, uint32_t data_format, 
                                       int32_t zp, float scale,
                                       float prob_threshold, std::vector<Object>& objects,
                                       int letterbox_cols, int letterbox_rows) {
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int reg_max = 16;
    int cls_num = 80;
    
    int channel_size = feat_h * feat_w;
    std::vector<float> dis_after_sm(4 * reg_max, 0.f);

    for (int h = 0; h < feat_h; h++) {
        for (int w = 0; w < feat_w; w++) {
            int spatial_offset = h * feat_w + w;
            
            // 处理分类得分
            int class_index = 0;
            float class_score = -FLT_MAX;
            
            size_t cls_start = 4 * reg_max * channel_size;
            
            for (int s = 0; s < cls_num; s++) {
                size_t offset = cls_start + s * channel_size + spatial_offset;
                float score = dequantize_value(feat, offset, data_format, zp, scale);
                
                if (score > class_score) {
                    class_index = s;
                    class_score = score;
                }
            }
            
            float box_prob = 1.0f / (1.0f + expf(-class_score));
            if (box_prob < prob_threshold) continue;
            
            // DFL处理
            float pred_ltrb[4];
            for (int c = 0; c < 4; c++) {
                size_t box_start = c * reg_max * channel_size;
                size_t offset = box_start + spatial_offset;
                
                char* feat_ptr = static_cast<char*>(feat) + offset * element_size(data_format);
                pred_ltrb[c] = softmax_with_stride(feat_ptr, dis_after_sm.data() + c * reg_max,
                                                  reg_max, channel_size, zp, scale, data_format) * stride;
            }
            
            float pb_cx = (w + 0.5f) * stride;
            float pb_cy = (h + 0.5f) * stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            x0 = std::max(0.0f, std::min(x0, static_cast<float>(letterbox_cols - 1)));
            y0 = std::max(0.0f, std::min(y0, static_cast<float>(letterbox_rows - 1)));
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(letterbox_cols - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(letterbox_rows - 1)));

            Object obj;
            obj.box.left = x0;
            obj.box.top = y0;
            obj.box.width = x1 - x0;
            obj.box.height = y1 - y0;
            obj.class_id = class_index;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

void YOLO11Detector::qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;

        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

void YOLO11Detector::qsort_descent_inplace(std::vector<Object>& objects) {
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

float YOLO11Detector::intersection_area(const Object& a, const Object& b) {
    float inter_left = std::max(a.box.left, b.box.left);
    float inter_top = std::max(a.box.top, b.box.top);
    float inter_right = std::min(a.box.left + a.box.width, b.box.left + b.box.width);
    float inter_bottom = std::min(a.box.top + a.box.height, b.box.top + b.box.height);

    float w = std::max(0.f, inter_right - inter_left);
    float h = std::max(0.f, inter_bottom - inter_top);

    return w * h;
}

void YOLO11Detector::nms_sorted_bboxes(const std::vector<Object>& objects, 
                                      std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = objects.size();
    if (n == 0) return;

    // 计算所有框的面积
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].box.width * objects[i].box.height;
    }

    // 按类别分组 (class-aware NMS)
    std::unordered_map<int, std::vector<int>> class_map;
    for (int i = 0; i < n; i++) {
        int cls = objects[i].class_id;
        class_map[cls].push_back(i);
    }

    // 对每个类别单独进行NMS
    for (auto& pair : class_map) {
        auto& indices = pair.second;
        std::vector<int> class_picked;

        // 对该类别内的框进行NMS
        for (size_t i = 0; i < indices.size(); i++) {
            int idx_i = indices[i];
            const Object& a = objects[idx_i];

            int keep = 1;
            for (size_t j = 0; j < class_picked.size(); j++) {
                int idx_j = class_picked[j];
                const Object& b = objects[idx_j];

                float inter_area = intersection_area(a, b);
                float union_area = areas[idx_i] + areas[idx_j] - inter_area;
                if (inter_area / union_area > nms_threshold) {
                    keep = 0;
                    break;
                }
            }

            if (keep) {
                class_picked.push_back(idx_i);
            }
        }

        // 将该类别保留的框加入最终结果
        picked.insert(picked.end(), class_picked.begin(), class_picked.end());
    }

    // 按置信度排序最终结果
    // std::sort(picked.begin(), picked.end(), [&](int i, int j) {
    //     return objects[i].prob > objects[j].prob;
    // });
}

void YOLO11Detector::inverse_coordinates(bbox& box, const PreprocessParams& params) {
    // 去除填充偏移
    box.left -= params.left;
    box.top -= params.top;

    // 缩放回原始尺寸
    const float inv_ratio = 1.0f / params.ratio;
    box.left *= inv_ratio;
    box.top *= inv_ratio;
    box.width *= inv_ratio;
    box.height *= inv_ratio;

    // 计算右下角坐标
    const float x1 = box.left + box.width;
    const float y1 = box.top + box.height;

    // 坐标裁剪
    box.left = std::max(0.0f, std::min(box.left, static_cast<float>(params.src_size.width)));
    box.top = std::max(0.0f, std::min(box.top, static_cast<float>(params.src_size.height)));
    const float clamped_x1 = std::max(0.0f, std::min(x1, static_cast<float>(params.src_size.width)));
    const float clamped_y1 = std::max(0.0f, std::min(y1, static_cast<float>(params.src_size.height)));

    // 重新计算宽高
    box.width = std::max(0.0f, clamped_x1 - box.left);
    box.height = std::max(0.0f, clamped_y1 - box.top);
}

void YOLO11Detector::draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, 
                                 const std::string& output_name) {
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];


        cv::rectangle(image, 
                     cv::Point((int)obj.box.left, (int)obj.box.top), 
                     cv::Point((int)(obj.box.left + obj.box.width), (int)(obj.box.top + obj.box.height)),
                     COCO_COLORS[obj.class_id % COCO_COLORS.size()], 2);

        

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.class_id], obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.box.left;
        int y = obj.box.top - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                     cv::Scalar(0, 0, 0), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    cv::imwrite(output_name, image);
}

// ==================== 工具函数实现 ====================

size_t YOLO11Detector::get_element_num(const taconn_inout_attr_t& input_attr) {
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i) {
        size *= input_attr.dim_size[i];
    }
    return size;
}

size_t YOLO11Detector::calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
    switch(format) {
        case TACONN_DATA_FORMAT_FP32:
        case TACONN_DATA_FORMAT_INT32:
        case TACONN_DATA_FORMAT_UINT32:
            return sizeof(uint32_t) * element_count;
        
        case TACONN_DATA_FORMAT_FP16:
        case TACONN_DATA_FORMAT_BFP16:
        case TACONN_DATA_FORMAT_INT16:
        case TACONN_DATA_FORMAT_UINT16:
            return sizeof(uint16_t) * element_count;
        
        case TACONN_DATA_FORMAT_UINT8:
        case TACONN_DATA_FORMAT_INT8:
        case TACONN_DATA_FORMAT_CHAR:
        case TACONN_DATA_FORMAT_BOOL8:
            return sizeof(uint8_t) * element_count;
        
        case TACONN_DATA_FORMAT_FP64:
        case TACONN_DATA_FORMAT_INT64:
        case TACONN_DATA_FORMAT_UINT64:
            return sizeof(uint64_t) * element_count;
        
        case TACONN_DATA_FORMAT_INT4:
        case TACONN_DATA_FORMAT_UINT4:
            return (element_count + 1) / 2;
        
        default:
            std::cerr << "Unsupported data format: " << format << std::endl;
            return 0;
    }
}

size_t YOLO11Detector::element_size(uint32_t data_format) {
    switch (data_format) {
        case taconn_data_format_e::TACONN_DATA_FORMAT_FP32: return sizeof(float);
        case taconn_data_format_e::TACONN_DATA_FORMAT_FP16: return sizeof(uint16_t);
        case taconn_data_format_e::TACONN_DATA_FORMAT_UINT8: return sizeof(uint8_t);
        case taconn_data_format_e::TACONN_DATA_FORMAT_INT8: return sizeof(int8_t);
        default: 
            std::cerr << "Unsupported data format: " << data_format << std::endl;
            return 1;
    }
}

uint16_t YOLO11Detector::float32_to_float16(float value) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (bits >> 31) & 0x1;
    int exponent = (bits >> 23) & 0xFF;
    uint32_t fraction = bits & 0x7FFFFF;
    
    // 处理特殊情况
    if (exponent == 0 && fraction == 0) {
        return sign << 15;
    }
    if (exponent == 0xFF) {
        if (fraction == 0) {
            return (sign << 15) | 0x7C00;
        } else {
            return (sign << 15) | 0x7E00;
        }
    }
    
    exponent -= 127;
    
    if (exponent < -14) {
        fraction = (0x800000 + fraction) >> (13 - exponent - 14);
        fraction |= (fraction >> 13) & 1;
        return (sign << 15) | fraction;
    }
    
    if (exponent > 15) {
        return (sign << 15) | 0x7C00;
    }
    
    exponent += 15;
    fraction >>= 13;
    
    if (fraction & 0x1000) {
        fraction += 1;
        if (fraction & 0x8000) {
            fraction >>= 1;
            exponent += 1;
        }
    }
    
    return (sign << 15) | (exponent << 10) | (fraction & 0x3FF);
}

void YOLO11Detector::normalize_and_quantize(uint8_t* src, void* dst, size_t num_elements, 
                                           const taconn_inout_attr_t& attr) {
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE) {
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
            float* dst_float = static_cast<float*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                dst_float[i] = src[i] / 255.0f;
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
            uint16_t* dst_fp16 = static_cast<uint16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i] / 255.0f;
                dst_fp16[i] = float32_to_float16(normalized);
            }
        }
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            std::cout << "Pre-processing node detected" << std::endl;
            memcpy(dst, src, num_elements);
        }
    } 
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i] / 255.0f;
                int32_t quantized = lrintf(normalized / scale) + zero_point;
                dst_int8[i] = static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            memcpy(dst, src, num_elements * sizeof(uint8_t));
        }
    } 
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_DFP) {
        int fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i] / 255.0f;
                int32_t scaled = lrintf(normalized * (1 << fixed_point_pos));
                dst_int8[i] = static_cast<int8_t>(std::max(-128, std::min(127, scaled)));
            }
        }
    }
}

void YOLO11Detector::mat_to_tensor(const cv::Mat& mat, uint8_t* tensor) {
    
    int total_pixels = mat.rows * mat.cols;
    int channels = mat.channels();
    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);
    // 按 CHW 顺序拷贝每个通道
    for (int c = 0; c < channels; ++c) {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(uint8_t));
    }
}


// ==================== 全局工具函数 ====================

bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

CmdParams parse_arguments(int argc, char** argv) {
    const char* keys = 
        "{help h usage ? |      | print help message   }"
        "{input i      | test.jpg  | input image file or directory for batch mode    }"
        "{model m      | yolo11s_int8.nb  | model file           }"
        "{output o | output.jpg | output image file or json file for batch mode    }"
        "{conf_thresh | 0.25 | confidence threshold for filter boxes}"
        "{nms_thresh | 0.45 | iou threshold for nms}"
        "{batch b | false | batch mode for dataset validation}"
        "{save_result | false | save detection result images in batch mode}";
        
    cv::CommandLineParser parser(argc, argv, keys);
    
    if (parser.get<bool>("help")) {
        parser.printMessage();
        exit(0);
    }
    
    CmdParams config;
    config.input = parser.get<std::string>("input");
    config.model = parser.get<std::string>("model");
    config.output = parser.get<std::string>("output");
    config.conf_thresh = parser.get<float>("conf_thresh");
    config.nms_thresh = parser.get<float>("nms_thresh");
    config.batch_mode = parser.get<bool>("batch");
    config.save_result = parser.get<bool>("save_result");

    return config;
}

void YOLO11Detector::print_taconn_inout_attr(const taconn_inout_attr_t& attr) {
    std::cout << "--------------------------------------------------------------" << "\n";
    std::cout << "    Tensor Attribute index:      |  "<< attr.index << "\n";
    std::cout << "    dim_count:                   |  " << attr.dim_count << "\n";
    
    std::cout << "    dim_size:                    |  [";
    for (unsigned int i = 0; i < attr.dim_count; ++i) {
        std::cout << attr.dim_size[i];
        if (i != attr.dim_count - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "    data_format:                 |  " << attr.data_format << "\n";
    std::cout << "    quant_format:                |  " << attr.quant_format << "\n";
    
    // dfp格式
    {
        std::cout << "    quant_data (dfp):            \n";
        std::cout << "        fixed_point_pos:         |  " << attr.quant_data.dfp.fixed_point_pos << "\n";
    }
    { // affine格式
        std::cout << "    quant_data (affine):\n";
        std::cout << "        tf_scale:                |  " << std::fixed << std::setprecision(6) 
                  << attr.quant_data.affine.tf_scale << "\n";
        std::cout << "        tf_zero_point:           |  " << attr.quant_data.affine.tf_zero_point << "\n";
    }
    
    std::cout << "    name:                        |  " << attr.name << "\n";
    std::cout << "--------------------------------------------------------------" << "\n";

}

void YOLO11Detector::enableProfile(TimeStamp *ts) {
    ts_ = ts;
  }