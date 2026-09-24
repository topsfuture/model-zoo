#include "yolo11_seg.hpp"
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

// ==================== Constructor / Destructor ====================

YOLO11SegDetector::YOLO11SegDetector() : initialized_(false), input_height_(640), input_width_(640)
{
    ctx_ = 0;
}

YOLO11SegDetector::~YOLO11SegDetector()
{
    if (initialized_)
    {
        deinit();
    }
}

// ==================== Init / Deinit ====================

bool YOLO11SegDetector::init(const std::string &model_path)
{
    if (initialized_)
    {
        std::cerr << "Detector already initialized" << std::endl;
        return false;
    }

    int status;
    taconn_input_output_num_t num = {0};

    status = ta_runtime_init();
    if (status != 0)
    {
        std::cerr << "Failed to initialize TACO runtime: " << status << std::endl;
        goto CLEANUP;
    }

    ctx_ = 0;

    status = ta_runtime_load_model_from_file(&ctx_, model_path.c_str(), 0);
    if (status != 0)
    {
        std::cerr << "Load model from file failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    status = ta_runtime_query(&ctx_, TACONN_QUERY_IN_OUT_NUM, &num);
    std::cout << "Input num: " << num.input_num << ", Output num: " << num.output_num << std::endl;
    input_num_ = num.input_num;
    output_num_ = num.output_num;

    for (int i = 0; i < input_num_; i++)
    {
        taconn_inout_attr_t input_attr = {0};
        input_attr.index = i;
        status = ta_runtime_query(&ctx_, TACONN_QUERY_INPUT_ATTR, &input_attr);
        print_taconn_inout_attr(input_attr);
        ins_attr_.push_back(input_attr);
    }

    if (input_num_ > 0 && ins_attr_[0].dim_count >= 3)
    {
        // Detect preprocess node: UINT8 + QNT_TYPE_NONE
        bool has_preprocess_node = (ins_attr_[0].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE &&
                                    ins_attr_[0].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8);

        if (has_preprocess_node)
        {
            // dims: [3, 640, 640, 1] → H=dim[1], W=dim[2]
            input_height_ = ins_attr_[0].dim_size[1];
            input_width_ = ins_attr_[0].dim_size[2];
        }
        else
        {
            // dims: [640, 640, 3, 1] → H=dim[0], W=dim[1]
            input_height_ = ins_attr_[0].dim_size[0];
            input_width_ = ins_attr_[0].dim_size[1];
        }
    }

    for (int i = 0; i < output_num_; i++)
    {
        taconn_inout_attr_t output_attr = {0};
        output_attr.index = i;
        status = ta_runtime_query(&ctx_, TACONN_QUERY_OUTPUT_ATTR, &output_attr);
        print_taconn_inout_attr(output_attr);
        outs_attr_.push_back(output_attr);
    }

    // Allocate input buffers
    input_tensors_ = (taconn_input_t *)malloc(sizeof(taconn_input_t) * input_num_);
    if (!input_tensors_)
    {
        std::cerr << "Allocate input_tensors failed" << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < input_num_; i++)
    {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ins_attr_[i].data_format);
        size_t input_buffer_size = calculate_buffer_size(data_type, get_element_num(ins_attr_[i]));

        input_tensors_[i].index = i;
        input_tensors_[i].size = input_buffer_size;
        input_tensors_[i].data = nullptr;

        if (posix_memalign((void **)&input_tensors_[i].data, 256, input_buffer_size) != 0)
        {
            std::cerr << "Failed to allocate input buffer" << std::endl;
            goto CLEANUP;
        }
        memset(input_tensors_[i].data, 0, input_buffer_size);
    }

    status = ta_runtime_set_input_cva(&ctx_, input_num_, input_tensors_);
    if (status != 0)
    {
        std::cerr << "Set input failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    // Allocate output buffers
    output_buffer_ = (taconn_buffer_t *)malloc(sizeof(taconn_buffer_t) * output_num_);
    if (!output_buffer_)
    {
        std::cerr << "Allocate output_buffer failed" << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < output_num_; i++)
    {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(outs_attr_[i].data_format);
        size_t output_buffer_size = calculate_buffer_size(data_type, get_element_num(outs_attr_[i]));

        status = ta_runtime_create_buffer(&ctx_, output_buffer_size, &output_buffer_[i]);
        if (status != 0)
        {
            std::cerr << "Create output buffer " << i << " failed: 0x" << std::hex << status << std::dec << std::endl;
            goto CLEANUP;
        }
    }

    status = ta_runtime_set_output(&ctx_, output_num_, output_buffer_);
    if (status != 0)
    {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    initialized_ = true;
    std::cout << "Model initialized successfully" << std::endl;
    return true;

CLEANUP:
    if (input_tensors_)
    {
        for (int i = 0; i < input_num_; i++)
        {
            if (input_tensors_[i].data)
                free(input_tensors_[i].data);
        }
        free(input_tensors_);
        input_tensors_ = nullptr;
    }
    if (output_buffer_)
    {
        for (int i = 0; i < output_num_; i++)
        {
            ta_runtime_destroy_buffer(&ctx_, &output_buffer_[i]);
        }
        free(output_buffer_);
        output_buffer_ = nullptr;
    }
    if (ctx_)
    {
        ta_runtime_destroy_context(&ctx_);
        ctx_ = 0;
    }
    ta_runtime_deinit();
    initialized_ = false;
    return false;
}

void YOLO11SegDetector::deinit()
{
    if (!initialized_)
        return;

    for (int i = 0; i < output_num_; i++)
    {
        ta_runtime_destroy_buffer(&ctx_, &output_buffer_[i]);
    }

    if (input_tensors_)
    {
        for (int i = 0; i < input_num_; i++)
        {
            if (input_tensors_[i].data)
                free(input_tensors_[i].data);
        }
        free(input_tensors_);
        input_tensors_ = nullptr;
    }

    if (output_buffer_)
    {
        free(output_buffer_);
        output_buffer_ = nullptr;
    }

    ta_runtime_destroy_context(&ctx_);
    ta_runtime_deinit();

    initialized_ = false;
    std::cout << "Model deinitialized" << std::endl;
}

// ==================== Detect ====================

bool YOLO11SegDetector::detect_and_save(const cv::Mat &image,
                                        const std::string &output_path,
                                        std::vector<Object> &objects,
                                        float conf_thresh,
                                        float nms_thresh,
                                        int max_det)
{
    if (!initialized_)
    {
        std::cerr << "Detector not initialized" << std::endl;
        return false;
    }

    if (ts_)
        ts_->start();
    PreprocessParams pre_params;
    cv::Mat processed_image;
    preprocess_image(image, processed_image, pre_params);
    if (ts_)
        ts_->time_accumulation("pre_time");

    if (ts_)
        ts_->start();
    if (!inference())
    {
        std::cerr << "Inference failed" << std::endl;
        return false;
    }
    if (ts_)
        ts_->time_accumulation("infer_time");

    if (ts_)
        ts_->start();
    post_process(image, pre_params, conf_thresh, nms_thresh, objects, output_path, max_det);
    if (ts_)
        ts_->time_accumulation("post_time");

    return true;
}

// ==================== Preprocess ====================

void YOLO11SegDetector::preprocess_image(const cv::Mat &src, cv::Mat &dst, PreprocessParams &params)
{
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

    for (int i = 0; i < input_num_; i++)
    {

        if (ins_attr_[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE &&
            ins_attr_[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8)
        {
            // NB model has preprocess node: input is raw UINT8 HWC pixels, no normalization
            memcpy(input_tensors_[i].data, dst.data, input_tensors_[i].size);
        }
        else
        {
            // No preprocess node: C++ does HWC→CHW + /255 normalization + type conversion
            uint8_t *input_tensor_uint8 = (uint8_t *)malloc(input_tensors_[i].size);
            mat_to_tensor(dst, input_tensor_uint8);

            size_t num_elements = get_element_num(ins_attr_[i]);
            normalize_and_quantize(input_tensor_uint8, input_tensors_[i].data,
                                   num_elements, ins_attr_[i]);
            free(input_tensor_uint8);
        }
    }
}

// ==================== Inference ====================

bool YOLO11SegDetector::inference()
{
    int status = ta_runtime_run_network(&ctx_);
    if (status != 0)
    {
        std::cerr << "Run network failed: 0x" << std::hex << status << std::dec << std::endl;
        return false;
    }

    status = ta_runtime_invalidate_buffer(&ctx_, output_buffer_);
    if (status != 0)
    {
        std::cerr << "Invalidate output buffer failed: 0x" << std::hex << status << std::dec << std::endl;
        return false;
    }

    return true;
}

// ==================== Post Process ====================

void YOLO11SegDetector::post_process(const cv::Mat &image,
                                     const PreprocessParams &pre_params,
                                     float conf_thresh,
                                     float nms_thresh,
                                     std::vector<Object> &objects,
                                     const std::string &output_image_path,
                                     int max_det)
{
    // output0: (1, 116, 8400) - detection output
    // output1: (1, 32, 160, 160) - mask prototypes

    std::vector<Object> proposals;

    void *output_data = output_buffer_[0].data;
    uint32_t data_format = outs_attr_[0].data_format;

    int32_t zp = 0;
    float scale = 1.0f;
    if (outs_attr_[0].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC)
    {
        zp = outs_attr_[0].quant_data.affine.tf_zero_point;
        scale = outs_attr_[0].quant_data.affine.tf_scale;
    }

    generate_proposals(output_data, data_format, zp, scale,
                       conf_thresh, proposals, input_width_, input_height_);

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thresh);

    objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); i++)
    {
        objects[i] = proposals[picked[i]];
    }

    // Re-sort by prob descending after NMS (class-wise NMS doesn't preserve global order)
    qsort_descent_inplace(objects);

    // Generate masks BEFORE inverse_coordinates (protos are in letterbox space)
    if (output_num_ >= 2 && !objects.empty())
    {
        // CRITICAL: Dequantize protos data
        uint32_t proto_format = outs_attr_[1].data_format;
        int32_t proto_zp = 0;
        float proto_scale = 1.0f;
        if (outs_attr_[1].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC)
        {
            proto_zp = outs_attr_[1].quant_data.affine.tf_zero_point;
            proto_scale = outs_attr_[1].quant_data.affine.tf_scale;
        }

        size_t proto_elem_count = 1 * 32 * 160 * 160;
        std::vector<float> protos_float(proto_elem_count);

        // Dequantize based on data format
        for (size_t i = 0; i < proto_elem_count; i++)
        {
            protos_float[i] = dequantize_value(output_buffer_[1].data, i, proto_format, proto_zp, proto_scale);
        }

        std::vector<Object> objects_with_masks(objects.size());
        generate_masks(protos_float.data(), 160, 160, objects, pre_params, objects_with_masks);
        objects = objects_with_masks;
    }

    // Now inverse coordinates: bbox + mask to original image space
    for (size_t i = 0; i < objects.size(); i++)
    {
        inverse_coordinates_and_mask(objects[i], pre_params);
    }

    // Apply max_det limit
    if (max_det > 0 && (int)objects.size() > max_det)
    {
        objects.resize(max_det);
    }

    if (!output_image_path.empty())
    {
        draw_objects(image, objects, output_image_path);
    }
}

// ==================== Dequantize ====================

float YOLO11SegDetector::dequantize(float value, int32_t, float)
{
    return value;
}

float YOLO11SegDetector::dequantize(uint16_t value, int32_t, float)
{
    return float16_to_float32(value);
}

float YOLO11SegDetector::dequantize(uint8_t value, int32_t zp, float scale)
{
    return ((float)value - (float)zp) * scale;
}

float YOLO11SegDetector::dequantize(int8_t value, int32_t zp, float scale)
{
    return ((float)value - (float)zp) * scale;
}

float YOLO11SegDetector::dequantize_value(void *data, size_t idx, uint32_t data_format,
                                          int32_t zp, float scale)
{
    switch (data_format)
    {
    case taconn_data_format_e::TACONN_DATA_FORMAT_FP32:
        return dequantize(static_cast<float *>(data)[idx], zp, scale);
    case taconn_data_format_e::TACONN_DATA_FORMAT_FP16:
        return dequantize(static_cast<uint16_t *>(data)[idx], zp, scale);
    case taconn_data_format_e::TACONN_DATA_FORMAT_UINT8:
        return dequantize(static_cast<uint8_t *>(data)[idx], zp, scale);
    case taconn_data_format_e::TACONN_DATA_FORMAT_INT8:
        return dequantize(static_cast<int8_t *>(data)[idx], zp, scale);
    default:
        std::cerr << "WARNING: Unknown data_format: 0x"
                  << std::hex << data_format << std::dec << std::endl;
        return 0.0f;
    }
}

// ==================== Generate Proposals (no DFL, 4+80+32=116) ====================

void YOLO11SegDetector::generate_proposals(void *feat, uint32_t data_format,
                                           int32_t zp, float scale,
                                           float prob_threshold, std::vector<Object> &objects,
                                           int letterbox_cols, int letterbox_rows)
{
    const int num_anchors = 8400;
    const int channel_size = num_anchors;

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        // CHW: [116, 8400, 1] → offset = c * 8400 + anchor_idx
        float bbox[4];
        for (int c = 0; c < 4; c++)
            bbox[c] = dequantize_value(feat, c * channel_size + anchor_idx, data_format, zp, scale);

        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int s = 0; s < NUM_CLASSES; s++)
        {
            float score = dequantize_value(feat, (4 + s) * channel_size + anchor_idx, data_format, zp, scale);
            if (score > class_score)
            {
                class_index = s;
                class_score = score;
            }
        }

        float box_prob = class_score;
        if (box_prob < prob_threshold)
            continue;

        std::vector<float> mask_coef(NUM_MASKS);
        for (int m = 0; m < NUM_MASKS; m++)
            mask_coef[m] = dequantize_value(feat, (4 + NUM_CLASSES + m) * channel_size + anchor_idx, data_format, zp, scale);

        float cx = bbox[0];
        float cy = bbox[1];
        float w = bbox[2];
        float h = bbox[3];

        float x0 = cx - w / 2.0f;
        float y0 = cy - h / 2.0f;
        float x1 = cx + w / 2.0f;
        float y1 = cy + h / 2.0f;

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
        obj.mask_coef = mask_coef;

        objects.push_back(obj);
    }
}

// ==================== NMS ====================

void YOLO11SegDetector::qsort_descent_inplace(std::vector<Object> &objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;
        while (objects[j].prob < p)
            j--;
        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(objects, left, j);
    if (i < right)
        qsort_descent_inplace(objects, i, right);
}

void YOLO11SegDetector::qsort_descent_inplace(std::vector<Object> &objects)
{
    if (objects.empty())
        return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

float YOLO11SegDetector::intersection_area(const Object &a, const Object &b)
{
    float inter_left = std::max(a.box.left, b.box.left);
    float inter_top = std::max(a.box.top, b.box.top);
    float inter_right = std::min(a.box.left + a.box.width, b.box.left + b.box.width);
    float inter_bottom = std::min(a.box.top + a.box.height, b.box.top + b.box.height);

    float w = std::max(0.f, inter_right - inter_left);
    float h = std::max(0.f, inter_bottom - inter_top);
    return w * h;
}

void YOLO11SegDetector::nms_sorted_bboxes(const std::vector<Object> &objects,
                                          std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();
    if (n == 0)
        return;

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].box.width * objects[i].box.height;
    }

    std::unordered_map<int, std::vector<int>> class_map;
    for (int i = 0; i < n; i++)
    {
        class_map[objects[i].class_id].push_back(i);
    }

    for (auto &pair : class_map)
    {
        auto &indices = pair.second;
        std::vector<int> class_picked;

        for (size_t i = 0; i < indices.size(); i++)
        {
            int idx_i = indices[i];
            const Object &a = objects[idx_i];

            int keep = 1;
            for (size_t j = 0; j < class_picked.size(); j++)
            {
                int idx_j = class_picked[j];
                const Object &b = objects[idx_j];

                float inter_area = intersection_area(a, b);
                float union_area = areas[idx_i] + areas[idx_j] - inter_area;
                if (inter_area / union_area > nms_threshold)
                {
                    keep = 0;
                    break;
                }
            }

            if (keep)
            {
                class_picked.push_back(idx_i);
            }
        }

        picked.insert(picked.end(), class_picked.begin(), class_picked.end());
    }
}

// ==================== Coordinate Transform ====================

void YOLO11SegDetector::inverse_coordinates_and_mask(Object &obj, const PreprocessParams &params)
{
    // Inverse bbox
    inverse_coordinates(obj.box, params);

    // Inverse mask: letterbox canvas (640x640) -> original image canvas
    if (!obj.mask.empty())
    {
        // Step 1: Remove letterbox padding
        int pad_left = params.left;
        int pad_top = params.top;
        int content_w = input_width_ - params.left - params.right;
        int content_h = input_height_ - params.top - params.bottom;

        cv::Rect content_rect(pad_left, pad_top, content_w, content_h);
        content_rect &= cv::Rect(0, 0, obj.mask.cols, obj.mask.rows);
        cv::Mat mask_content = obj.mask(content_rect);

        // Step 2: Resize to original image size
        cv::Mat mask_orig;
        cv::resize(mask_content, mask_orig, params.src_size, 0, 0, cv::INTER_LINEAR);

        // Step 3: Threshold again after resize interpolation
        cv::threshold(mask_orig, mask_orig, MASK_THRES, 1.0f, cv::THRESH_BINARY);

        obj.mask = mask_orig;
    }
}

void YOLO11SegDetector::inverse_coordinates(bbox &box, const PreprocessParams &params)
{
    box.left -= params.left;
    box.top -= params.top;

    const float inv_ratio = 1.0f / params.ratio;
    box.left *= inv_ratio;
    box.top *= inv_ratio;
    box.width *= inv_ratio;
    box.height *= inv_ratio;

    const float x1 = box.left + box.width;
    const float y1 = box.top + box.height;

    box.left = std::max(0.0f, std::min(box.left, static_cast<float>(params.src_size.width)));
    box.top = std::max(0.0f, std::min(box.top, static_cast<float>(params.src_size.height)));
    const float clamped_x1 = std::max(0.0f, std::min(x1, static_cast<float>(params.src_size.width)));
    const float clamped_y1 = std::max(0.0f, std::min(y1, static_cast<float>(params.src_size.height)));

    box.width = std::max(0.0f, clamped_x1 - box.left);
    box.height = std::max(0.0f, clamped_y1 - box.top);
}

// ==================== Draw ====================

void YOLO11SegDetector::draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects,
                                     const std::string &output_name)
{
    cv::Mat image = bgr.clone();

    // Step 1: Build union mask (any pixel belonging to ANY object)
    cv::Mat any_mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
    for (size_t i = 0; i < objects.size(); i++)
    {
        if (objects[i].mask.empty())
            continue;
        cv::Mat m8;
        objects[i].mask.convertTo(m8, CV_8U, 255.0);
        any_mask |= m8;
    }

    // Step 2: Darken everything OUTSIDE the union mask, keep inside (objects) as-is
    float dim = 0.3f; // outside pixels dimmed to 30% brightness
    for (int y = 0; y < image.rows; y++)
    {
        const uint8_t *mask_row = any_mask.ptr<uint8_t>(y);
        cv::Vec3b *img_row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; x++)
        {
            if (mask_row[x] == 0)
            {
                img_row[x][0] = (uint8_t)(img_row[x][0] * dim);
                img_row[x][1] = (uint8_t)(img_row[x][1] * dim);
                img_row[x][2] = (uint8_t)(img_row[x][2] * dim);
            }
        }
    }

    // Draw bounding boxes and labels on top
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object &obj = objects[i];
        cv::Scalar color = COCO_COLORS[obj.class_id % COCO_COLORS.size()];

        cv::rectangle(image,
                      cv::Point((int)obj.box.left, (int)obj.box.top),
                      cv::Point((int)(obj.box.left + obj.box.width), (int)(obj.box.top + obj.box.height)),
                      color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.class_id], obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.box.left;
        int y = (int)obj.box.top - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(0, 0, 0), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    cv::imwrite(output_name, image);
}

// ==================== Utility Functions ====================

size_t YOLO11SegDetector::get_element_num(const taconn_inout_attr_t &input_attr)
{
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i)
    {
        size *= input_attr.dim_size[i];
    }
    return size;
}

size_t YOLO11SegDetector::calculate_buffer_size(taconn_data_format_t format, size_t element_count)
{
    switch (format)
    {
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

size_t YOLO11SegDetector::element_size(uint32_t data_format)
{
    switch (data_format)
    {
    case taconn_data_format_e::TACONN_DATA_FORMAT_FP32:
        return sizeof(float);
    case taconn_data_format_e::TACONN_DATA_FORMAT_FP16:
        return sizeof(uint16_t);
    case taconn_data_format_e::TACONN_DATA_FORMAT_UINT8:
        return sizeof(uint8_t);
    case taconn_data_format_e::TACONN_DATA_FORMAT_INT8:
        return sizeof(int8_t);
    default:
        std::cerr << "Unsupported data format: " << data_format << std::endl;
        return 1;
    }
}

// ==================== Float16 Conversion ====================

uint16_t YOLO11SegDetector::float32_to_float16(float value)
{
    uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
    uint16_t sign = (bits >> 31) & 0x1;
    int exponent = (bits >> 23) & 0xFF;
    uint32_t fraction = bits & 0x7FFFFF;

    if (exponent == 0 && fraction == 0)
    {
        return sign << 15;
    }
    if (exponent == 0xFF)
    {
        if (fraction == 0)
        {
            return (sign << 15) | 0x7C00;
        }
        else
        {
            return (sign << 15) | 0x7E00;
        }
    }

    exponent -= 127;

    if (exponent < -14)
    {
        fraction = (0x800000 + fraction) >> (13 - exponent - 14);
        fraction |= (fraction >> 13) & 1;
        return (sign << 15) | fraction;
    }

    if (exponent > 15)
    {
        return (sign << 15) | 0x7C00;
    }

    exponent += 15;
    fraction >>= 13;

    if (fraction & 0x1000)
    {
        fraction += 1;
        if (fraction & 0x8000)
        {
            fraction >>= 1;
            exponent += 1;
        }
    }

    return (sign << 15) | (exponent << 10) | (fraction & 0x3FF);
}

float YOLO11SegDetector::float16_to_float32(uint16_t value)
{
    uint16_t sign = (value >> 15) & 0x1;
    uint16_t exponent = (value >> 10) & 0x1F;
    uint16_t fraction = value & 0x3FF;

    if (exponent == 0 && fraction == 0)
    {
        uint32_t f = (sign << 31);
        return *reinterpret_cast<float *>(&f);
    }

    if (exponent == 31)
    {
        uint32_t f = (sign << 31) | 0x7F800000 | (fraction << 13);
        return *reinterpret_cast<float *>(&f);
    }

    exponent += (127 - 15);
    uint32_t f = (sign << 31) | (exponent << 23) | (fraction << 13);
    return *reinterpret_cast<float *>(&f);
}

// ==================== Normalize and Quantize ====================

void YOLO11SegDetector::normalize_and_quantize(uint8_t *src, void *dst, size_t num_elements,
                                               const taconn_inout_attr_t &attr)
{
    // Called when input is FP16/FP32 (no preprocess node in NB model).
    // C++ handles /255 normalization + type conversion.
    // When NB has preprocess node, input is UINT8 and this function is NOT called
    // (handled in preprocess_image branch: raw memcpy/HWC→CHW only).

    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE)
    {
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32)
        {
            float *dst_float = static_cast<float *>(dst);
            for (size_t i = 0; i < num_elements; i++)
            {
                dst_float[i] = src[i] / 255.0f;
            }
        }
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16)
        {
            uint16_t *dst_fp16 = static_cast<uint16_t *>(dst);
            for (size_t i = 0; i < num_elements; i++)
            {
                float normalized = src[i] / 255.0f;
                dst_fp16[i] = float32_to_float16(normalized);
            }
        }
        else
        {
            std::cerr << "Unsupported data format for non-quantized input: "
                      << attr.data_format << std::endl;
        }
    }
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC)
    {
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;

        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8)
        {
            int8_t *dst_int8 = static_cast<int8_t *>(dst);
            for (size_t i = 0; i < num_elements; i++)
            {
                float normalized = src[i] / 255.0f;
                int32_t quantized = lrintf(normalized / scale) + zero_point;
                dst_int8[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
            }
        }
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8)
        {
            uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
            memcpy(dst_uint8, src, num_elements * sizeof(uint8_t));
        }
    }
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_DFP)
    {
        int fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;

        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8)
        {
            int8_t *dst_int8 = static_cast<int8_t *>(dst);
            for (size_t i = 0; i < num_elements; i++)
            {
                float normalized = src[i] / 255.0f;
                int32_t scaled = lrintf(normalized * (1 << fixed_point_pos));
                dst_int8[i] = static_cast<int8_t>(std::clamp(scaled, -128, 127));
            }
        }
    }
}

// ==================== Mat to Tensor (CHW) ====================

void YOLO11SegDetector::mat_to_tensor(const cv::Mat &mat, uint8_t *tensor)
{
    int total_pixels = mat.rows * mat.cols;
    int channels = mat.channels();
    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);
    for (int c = 0; c < channels; ++c)
    {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(uint8_t));
    }
}

// ==================== Global Utility ====================

bool file_exists(const std::string &path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool YOLO11SegDetector::file_exists(const std::string &path)
{
    return ::file_exists(path);
}

// ==================== Print Tensor Attributes ====================

// ==================== Print Tensor Attributes ====================

void YOLO11SegDetector::print_taconn_inout_attr(const taconn_inout_attr_t &attr)
{
    std::cout << "====================================================" << "\n";
    std::cout << "  index: " << attr.index << "\n";
    std::cout << "  name: " << attr.name << "\n";

    std::string qnt_str = "None";
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_DFP)
    {
        qnt_str = "DFP";
    }
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC)
    {
        qnt_str = "Affine";
        std::cout << "  tf_scale: " << attr.quant_data.affine.tf_scale << "\n";
        std::cout << "  tf_zero_point: " << attr.quant_data.affine.tf_zero_point << "\n";
    }
    std::cout << "  quant_format: " << qnt_str << "\n";

    std::string fmt = "Unknown";
    switch (attr.data_format)
    {
    case taconn_data_format_e::TACONN_DATA_FORMAT_FP32:
        fmt = "FP32";
        break;
    case taconn_data_format_e::TACONN_DATA_FORMAT_FP16:
        fmt = "FP16";
        break;
    case taconn_data_format_e::TACONN_DATA_FORMAT_INT8:
        fmt = "INT8";
        break;
    case taconn_data_format_e::TACONN_DATA_FORMAT_UINT8:
        fmt = "UINT8";
        break;
    case taconn_data_format_e::TACONN_DATA_FORMAT_INT16:
        fmt = "INT16";
        break;
    case taconn_data_format_e::TACONN_DATA_FORMAT_BFP16:
        fmt = "BFP16";
        break;
    default:
        break;
    }
    std::cout << "  data_format: " << fmt << "\n";
    std::cout << "  dim_count: " << attr.dim_count << "\n";
    std::cout << "  dims: [";
    for (unsigned int i = 0; i < attr.dim_count; i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << attr.dim_size[i];
    }
    std::cout << "]\n";
    std::cout << "--------------------------------------------------------------" << "\n";
}

// ==================== Mask Generation ====================

void YOLO11SegDetector::generate_masks(const float *protos, int proto_h, int proto_w,
                                       const std::vector<Object> &objects,
                                       const PreprocessParams &params,
                                       std::vector<Object> &output_objects)
{
    if (objects.empty() || !protos)
        return;
    int proto_size = proto_h * proto_w;
    int lb_cols = input_width_;  // letterbox cols (640)
    int lb_rows = input_height_; // letterbox rows (640)

    for (size_t det_idx = 0; det_idx < objects.size(); det_idx++)
    {
        const Object &obj = objects[det_idx];
        output_objects[det_idx] = obj;
        if (obj.mask_coef.empty())
            continue;

        // 1. Use raw mask_coef directly (ultralytics: sigmoid(protos @ mc_raw))
        //    Do NOT sigmoid mc_raw — that compresses range and causes double-sigmoid saturation

        // 2. protos @ mc_raw -> sigmoid -> full mask (160x160)
        //    Protos are in letterbox space (640x640 mapped to 160x160)
        cv::Mat full_mask(proto_h, proto_w, CV_32F);
        for (int y = 0; y < proto_h; y++)
        {
            for (int x = 0; x < proto_w; x++)
            {
                int si = y * proto_w + x;
                float val = 0.0f;
                for (int m = 0; m < NUM_MASKS; m++)
                    val += protos[m * proto_size + si] * obj.mask_coef[m]; // CHW layout: [m*H*W + y*W + x]
                full_mask.at<float>(y, x) = 1.0f / (1.0f + expf(-val));
            }
        }

        // Check protos stats (HWC layout)
        float proto_min = protos[0], proto_max = protos[0], proto_sum = 0;
        for (int pi = 0; pi < proto_size * NUM_MASKS; pi++)
        {
            if (protos[pi] < proto_min)
                proto_min = protos[pi];
            if (protos[pi] > proto_max)
                proto_max = protos[pi];
            proto_sum += protos[pi];
        }

        // Check full_mask stats
        float mask_min = full_mask.at<float>(0, 0);
        float mask_max = full_mask.at<float>(0, 0);
        float mask_sum = 0;
        for (int my = 0; my < proto_h; my++)
        {
            for (int mx = 0; mx < proto_w; mx++)
            {
                float v = full_mask.at<float>(my, mx);
                if (v < mask_min)
                    mask_min = v;
                if (v > mask_max)
                    mask_max = v;
                mask_sum += v;
            }
        }

        // 3. Bbox is in LETTERBOX coordinates (640x640)
        //    Map letterbox -> proto: scale = 160/640 = 0.25
        float scale_x = (float)proto_w / lb_cols;
        float scale_y = (float)proto_h / lb_rows;

        int bx = (int)obj.box.left;
        int by = (int)obj.box.top;
        int box_w = (int)obj.box.width;
        int box_h = (int)obj.box.height;
        if (box_w <= 0 || box_h <= 0)
            continue;

        int px = (int)(bx * scale_x);
        int py = (int)(by * scale_y);
        int pw = (int)(box_w * scale_x);
        int ph = (int)(box_h * scale_y);

        px = std::max(0, std::min(px, proto_w - 1));
        py = std::max(0, std::min(py, proto_h - 1));
        pw = std::max(1, std::min(pw, proto_w - px));
        ph = std::max(1, std::min(ph, proto_h - py));

        cv::Mat mask_crop = full_mask(cv::Rect(px, py, pw, ph));

        // 4. Resize to bbox size (letterbox coords)
        cv::Mat mask_resized;
        cv::resize(mask_crop, mask_resized, cv::Size(box_w, box_h), 0, 0, cv::INTER_LINEAR);

        // 5. Binary threshold
        cv::Mat mask_binary;
        cv::threshold(mask_resized, mask_binary, MASK_THRES, 1.0f, cv::THRESH_BINARY);

        // 6. Store mask in letterbox-sized canvas (640x640)
        //    inverse_coordinates_and_mask will transform to original image later
        output_objects[det_idx].mask = cv::Mat::zeros(lb_rows, lb_cols, CV_32F);
        bx = std::max(0, std::min(bx, lb_cols - box_w));
        by = std::max(0, std::min(by, lb_rows - box_h));

        cv::Mat roi = output_objects[det_idx].mask(cv::Rect(bx, by, box_w, box_h));
        mask_binary.copyTo(roi);
    }
}
