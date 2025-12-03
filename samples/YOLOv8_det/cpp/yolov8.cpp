#include "yolov8.hpp"

const char* CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

YOLOV8::YOLOV8(std::shared_ptr<ta_runtime_context> context,std::string model_file,float conf_thresh_i ,float nms_thresh_i ):nnrt_context(context),model_name(model_file),
        conf_thresh(conf_thresh_i),nms_thresh(nms_thresh_i){
    
    input_tensors = nullptr;
    output_buffer = nullptr;
    input_num = 0;
    output_num = 0;
    initialized_ = false;
}

YOLOV8::~YOLOV8() {
    if (initialized_) {
        deInit();
    }
}
void print_taconn_inout_attr(const taconn_inout_attr_t& attr) {
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

size_t YOLOV8::get_element_num(const taconn_inout_attr_t input_attr) {
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i) {
        size *= input_attr.dim_size[i];
    }
    return size;
}

void YOLOV8::mat_to_tensor(const cv::Mat& mat, uint8_t* tensor) {
    /* work as preprocess_node_layer */
    int channels = mat.channels();
    int total_pixels = mat.rows * mat.cols;

    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);

    // 按 CHW 顺序拷贝每个通道
    for (int c = 0; c < channels; ++c) {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(uint8_t));
    }
            
}
// 具体转换函数实现
float dequantize_float32(void* data, size_t idx, int32_t, float) {
    return static_cast<float*>(data)[idx];
}
float dequantize_float16(void* data, size_t idx, int32_t, float) {
    uint16_t h =  static_cast<uint16_t*>(data)[idx];
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t f = (sign << 31); // ±0.0
            return *reinterpret_cast<float*>(&f); // sign? -0.0f : 0.0f
        }
        // 非正规数直接计算: sign * 2^{-24} * mantissa
        const uint32_t exp_offset = 103;  // 127 - 24
        uint32_t f = (sign << 31) | (exp_offset << 23) | (mantissa << 13);
        return *reinterpret_cast<float*>(&f);
    } 
    else if (exponent == 31) {
        uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&f);
    }
    
    // 正规数
    exponent += (127 - 15);  // 偏置调整
    uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *reinterpret_cast<float*>(&f);
}

float dequantize_uint8(void* data, size_t idx, int32_t zp, float scale) {
    uint8_t val = static_cast<uint8_t*>(data)[idx];
    return ((float)val - (float)zp) * scale;
}

float dequantize_int8(void* data, size_t idx, int32_t zp, float scale) {
    int8_t val = static_cast<int8_t*>(data)[idx];
    return ((float)val - (float)zp) * scale;
}

DequantizeFunc dequantize_funcs[] = {
    dequantize_float32,   // kFloat32
    dequantize_float16,   // kFloat16
    dequantize_uint8,     // kUInt8
    dequantize_int8,      // kInt8
};

int YOLOV8::preprocess_image(const cv::Mat& src, cv::Mat& dst,PreprocessParams& params){
    params.src_size = src.size();
    params.ratio = std::min(static_cast<float>(model_input_h) / src.rows,
                           static_cast<float>(model_input_w) / src.cols);
    
    int new_w = static_cast<int>(src.cols * params.ratio);
    int new_h = static_cast<int>(src.rows * params.ratio);
    
    cv::resize(src, dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    params.top = (model_input_h - new_h) / 2;
    params.bottom = model_input_h - new_h - params.top;
    params.left = (model_input_w - new_w) / 2;
    params.right = model_input_w - new_w - params.left;
    
    cv::copyMakeBorder(dst, dst, params.top, params.bottom, 
                      params.left, params.right, cv::BORDER_CONSTANT, 
                      cv::Scalar(114, 114, 114));
    
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    return 0;
    
}
static uint16_t f32_to_f16(float value) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint16_t h = ((x >> 16) & 0x8000) | 
                ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
                ((x >> 13) & 0x03ff);
    return h;
}
void YOLOV8::normalize_and_quantize(uint8_t* src, void* dst, size_t num_elements, 
                                   const taconn_inout_attr_t& attr) {
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE) {
        // 无量化处理 - 转换为 float32 或 float16
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
            // 转换为 float32 并标准化
            float* dst_float = static_cast<float*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                dst_float[i] = src[i] / 255.0f;
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
            // 转换为 float16 并标准化
            uint16_t* dst_fp16 = static_cast<uint16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i] / 255.0f;
                dst_fp16[i] = f32_to_f16(normalized);
            }
        }
        else {
            std::cerr<<  "Unsupported data format for non-quantized input: " <<
                    attr.data_format << std::endl;
        }
    } 

    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        // 对称量化 (int8, uint8)
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i] / 255.0f;
                int32_t quantized = lrintf(normalized / scale) + zero_point;
                dst_int8[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            uint8_t* dst_uint8 = static_cast<uint8_t*>(dst);
            // 直接复制 uint8 数据
            memcpy(dst_uint8, src, num_elements * sizeof(uint8_t));
        }
    } 
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_DFP) {
        // DFP量化格式处理
        int fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i] / 255.0f;
                int32_t scaled = lrintf(normalized * (1 << fixed_point_pos));
                dst_int8[i] = static_cast<int8_t>(std::clamp(scaled, -128, 127));
            }
        }
    }
}
int YOLOV8::pre_process(cv::Mat &images){
    
    for(int i = 0; i < input_num; i++){
        cv::Mat dst;
        PreprocessParams params;
        ts_->start();
        preprocess_image(images, dst,params);
        ts_->time_accumulation("pre_time");
        pre_params.push_back(params);
        if (ins_attr[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE &&
            ins_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            memcpy(input_tensors[i].data, dst.data, input_tensors[i].size);  
        }
        else{    // prepare input layout to  (N)CHW
            uint8_t* input_tensor_uint8 = nullptr;
            input_tensor_uint8 = (uint8_t*)malloc(sizeof(uint8_t) * model_input_h * model_input_w * 3);
            if(input_tensor_uint8 == NULL) return -1;
            mat_to_tensor(dst, input_tensor_uint8);
            
            size_t num_elements = get_element_num(ins_attr[i]);
            normalize_and_quantize(
                input_tensor_uint8, 
                input_tensors[i].data, 
                num_elements, 
                ins_attr[i]
            );
            free(input_tensor_uint8);
        }
    }
    return 0;
}

size_t YOLOV8::element_size(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32: return sizeof(float);
        case DataType::kFloat16: return sizeof(uint16_t);
        case DataType::kUInt8:   return sizeof(uint8_t);
        case DataType::kInt8:    return sizeof(int8_t);
        default: 
            std::cerr << "Unsupported data format: " << dtype << std::endl;
            return 1;
    }
}

float softmax_with_stride_generic(void* src, float* dst, int length, 
                                        int stride, int32_t zp, float scale,
                                        DataType dtype) 
{
    // 获取对应的反量化函数
    DequantizeFunc dequant = dequantize_funcs[static_cast<int>(dtype)];
    
    // 找出最大值
    float max_val = -FLT_MAX;
    for (int i = 0; i < length; i++) {
        float val = dequant(src, i * stride, zp, scale);
        if (val > max_val) max_val = val;
    }

    // 计算指数和
    float sum = 0.f;
    for (int i = 0; i < length; i++) {
        float val = dequant(src, i * stride, zp, scale);
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
void YOLOV8::generate_proposals_yolov8_generic(
    int stride, void* feat, DataType dtype, int32_t zp, float scale,
    float prob_threshold, YoloV8BoxVec& objects,
    int letterbox_cols, int letterbox_rows, int cls_num = 80)
{
    /* clip at last conv layer, out tensor shape (1,3*85,h,w) */
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int reg_max = 16;
    
    int channel_size = feat_h * feat_w;
    std::vector<float> dis_after_sm(4 * reg_max, 0.f);
    DequantizeFunc dequant = dequantize_funcs[static_cast<int>(dtype)];

    for (int h = 0; h < feat_h; h++) {
        for (int w = 0; w < feat_w; w++) {
            int spatial_offset = h * feat_w + w;
            
            // 处理分类得分
            int class_index = 0;
            float class_score = -FLT_MAX;
            
            size_t cls_start = 4 * reg_max * channel_size;
            
            for (int s = 0; s < cls_num; s++) {
                size_t offset = cls_start + s * channel_size + spatial_offset;
                float score = dequant(feat, offset, zp, scale);
                
                if (score > class_score) {
                    class_index = s;
                    class_score = score;
                }
            }
            
            float box_prob = 1.0f / (1.0f + expf(-class_score));
            if (box_prob < prob_threshold) continue;
            
            // dfl
            float pred_ltrb[4];
            for (int c = 0; c < 4; c++) {
                size_t box_start = c * reg_max * channel_size;
                size_t offset = box_start + spatial_offset;
                
                pred_ltrb[c] = softmax_with_stride_generic(
                    static_cast<char*>(feat) + offset * element_size(dtype),
                    dis_after_sm.data() + c * reg_max,
                    reg_max,
                    channel_size,
                    zp,
                    scale,
                    dtype
                ) * stride;
            }
            
            float pb_cx = (w + 0.5f) * stride;
            float pb_cy = (h + 0.5f) * stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            x0 = std::clamp(x0, 0.0f, static_cast<float>(letterbox_cols - 1));
            y0 = std::clamp(y0, 0.0f, static_cast<float>(letterbox_rows - 1));
            x1 = std::clamp(x1, 0.0f, static_cast<float>(letterbox_cols - 1));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(letterbox_rows - 1));

                // 创建Object并添加到结果
            YoloV8Box obj;
            obj.x = x0;
            obj.y = y0;
            obj.width = x1 - x0;
            obj.height = y1 - y0;
            obj.class_id = class_index;
            obj.score = box_prob;
            
            objects.push_back(obj);
            
        }
    }
    return;
}

void g_qsort_descent_inplace(YoloV8BoxVec& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].score;

    while (i <= j)
    {
        while (faceobjects[i].score > p)
            i++;

        while (faceobjects[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) g_qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) g_qsort_descent_inplace(faceobjects, i, right);
        }
    }
}
void g_qsort_descent_inplace(YoloV8BoxVec& faceobjects)
{
    if (faceobjects.empty())
        return;

    g_qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}
DataType convert_to_data_type(uint32_t data_format) {
  switch (data_format) {
      case taconn_data_format_e::TACONN_DATA_FORMAT_FP32: return DataType::kFloat32;
      case taconn_data_format_e::TACONN_DATA_FORMAT_FP16: return DataType::kFloat16;
      case taconn_data_format_e::TACONN_DATA_FORMAT_UINT8:   return DataType::kUInt8;
      case taconn_data_format_e::TACONN_DATA_FORMAT_INT8:    return DataType::kInt8;
      // 添加其他数据类型的转换
      default: 
          std::cerr<<  "Unsupported data format: "<< data_format << std::endl;
          return DataType::kFloat32; // 默认返回float32
  }
}
void YOLOV8::draw_objects(const cv::Mat& bgr, const YoloV8BoxVec& objects, const std::string& output_name, double fontScale, int thickness)
{
    static const std::vector<cv::Scalar> COCO_COLORS = {
        {128, 56, 0, 255}, {128, 226, 255, 0}, {128, 0, 94, 255}, {128, 0, 37, 255}, {128, 0, 255, 94}, {128, 255, 226, 0}, {128, 0, 18, 255}, {128, 255, 151, 0}, {128, 170, 0, 255}, {128, 0, 255, 56}, {128, 255, 0, 75}, {128, 0, 75, 255}, {128, 0, 255, 169}, {128, 255, 0, 207}, {128, 75, 255, 0}, {128, 207, 0, 255}, {128, 37, 0, 255}, {128, 0, 207, 255}, {128, 94, 0, 255}, {128, 0, 255, 113}, {128, 255, 18, 0}, {128, 255, 0, 56}, {128, 18, 0, 255}, {128, 0, 255, 226}, {128, 170, 255, 0}, {128, 255, 0, 245}, {128, 151, 255, 0}, {128, 132, 255, 0}, {128, 75, 0, 255}, {128, 151, 0, 255}, {128, 0, 151, 255}, {128, 132, 0, 255}, {128, 0, 255, 245}, {128, 255, 132, 0}, {128, 226, 0, 255}, {128, 255, 37, 0}, {128, 207, 255, 0}, {128, 0, 255, 207}, {128, 94, 255, 0}, {128, 0, 226, 255}, {128, 56, 255, 0}, {128, 255, 94, 0}, {128, 255, 113, 0}, {128, 0, 132, 255}, {128, 255, 0, 132}, {128, 255, 170, 0}, {128, 255, 0, 188}, {128, 113, 255, 0}, {128, 245, 0, 255}, {128, 113, 0, 255}, {128, 255, 188, 0}, {128, 0, 113, 255}, {128, 255, 0, 0}, {128, 0, 56, 255}, {128, 255, 0, 113}, {128, 0, 255, 188}, {128, 255, 0, 94}, {128, 255, 0, 18}, {128, 18, 255, 0}, {128, 0, 255, 132}, {128, 0, 188, 255}, {128, 0, 245, 255}, {128, 0, 169, 255}, {128, 37, 255, 0}, {128, 255, 0, 151}, {128, 188, 0, 255}, {128, 0, 255, 37}, {128, 0, 255, 0}, {128, 255, 0, 170}, {128, 255, 0, 37}, {128, 255, 75, 0}, {128, 0, 0, 255}, {128, 255, 207, 0}, {128, 255, 0, 226}, {128, 255, 245, 0}, {128, 188, 255, 0}, {128, 0, 255, 18}, {128, 0, 255, 75}, {128, 0, 255, 151}, {128, 255, 56, 0}, {128, 245, 255, 0}};
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const YoloV8Box& obj = objects[i];
        if(obj.class_id >= 80)
            continue;
        fprintf(stdout, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.class_id, obj.score * 100, obj.x,
                obj.y, obj.x + obj.width, obj.y + obj.height, CLASS_NAMES[obj.class_id]);

        cv::rectangle(image, 
              cv::Point((int)obj.x, (int)obj.y), 
              cv::Point((int)(obj.x + obj.width), (int)(obj.y + obj.height)),
              COCO_COLORS[obj.class_id % COCO_COLORS.size()], 
              thickness);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.class_id], obj.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);

        int x = obj.x;
        int y = obj.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(0, 0, 0), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                    cv::Scalar(255, 255, 255), thickness);
    }

    cv::imwrite(std::string(output_name), image);
}
inline float g_intersection_area(const YoloV8Box& a, const YoloV8Box& b)
{
    // cv::Rect_<float> inter = a.rect & b.rect;
    // return inter.area();
    float inter_left = std::max(a.x, b.x);
    float inter_top = std::max(a.y, b.y);
    float inter_right = std::min(a.x + a.width, b.x + b.width);
    float inter_bottom = std::min(a.y + a.height, b.y + b.height);

    
    float w = std::max(0.f, inter_right - inter_left); 
    float h = std::max(0.f, inter_bottom - inter_top); 

   
    return w * h;
}
void g_nms_sorted_bboxes_class_aware(const YoloV8BoxVec& objects, 
                                     std::vector<int>& picked, 
                                     float nms_threshold)
{   /*class aware nms*/
    picked.clear();
    const int n = objects.size();
    if (n == 0) return;

    // 1. 计算所有边界框的面积
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].width * objects[i].height;
    }

    // 2. 按类别分组
    std::unordered_map<int, std::vector<int>> class_indices;
    for (int i = 0; i < n; i++)
    {
        class_indices[objects[i].class_id].push_back(i);
    }

    // 3. 对每个类别单独应用NMS
    for (auto& pair : class_indices)
    {
        const int cls = pair.first;
        std::vector<int>& indices = pair.second;
        std::vector<int> class_picked;

        // 对当前类别的对象应用NMS
        for (int i = 0; i < indices.size(); i++)
        {
            const int idx_i = indices[i];
            const YoloV8Box& a = objects[idx_i];
            
            int keep = 1;
            for (int j = 0; j < class_picked.size(); j++)
            {
                const int idx_j = class_picked[j];
                const YoloV8Box& b = objects[idx_j];
                
                // 计算IoU
                float inter_area = g_intersection_area(a, b);
                float union_area = areas[idx_i] + areas[idx_j] - inter_area;
                float iou = inter_area / union_area;
                
                if (iou > nms_threshold)
                {
                    keep = 0;
                    break;
                }
            }
            
            if (keep)
                class_picked.push_back(idx_i);
        }
        
        // 4. 合并结果
        picked.insert(picked.end(), class_picked.begin(), class_picked.end());
    }

    std::sort(picked.begin(), picked.end(), [&](int i, int j) {
        return objects[i].score > objects[j].score;
    });
}
void inverse_coordinates(YoloV8Box& box, const PreprocessParams& params) {
    // Step 1: 去除填充偏移（浮点计算）
    box.x -= params.left;
    box.y -= params.top;

    // Step 2: 缩放回原始尺寸（保留浮点精度）
    const float inv_ratio = 1.0f / params.ratio;
    box.x *= inv_ratio;
    box.y *= inv_ratio;
    box.width *= inv_ratio;
    box.height *= inv_ratio;

    // Step 3: 计算右下角坐标
    const float x1 = box.x + box.width;
    const float y1 = box.y + box.height;

    // Step 4: 坐标裁剪（基于原始图像尺寸）
    box.x = std::clamp(box.x, 0.0f, static_cast<float>(params.src_size.width));
    box.y = std::clamp(box.y, 0.0f, static_cast<float>(params.src_size.height));
    const float clamped_x1 = std::clamp(x1, 0.0f, static_cast<float>(params.src_size.width));
    const float clamped_y1 = std::clamp(y1, 0.0f, static_cast<float>(params.src_size.height));

    // Step 5: 重新计算宽高（确保非负）
    box.width = std::max(box.width, clamped_x1 - box.x);
    box.height = std::max(box.height, clamped_y1 - box.y);
}
void YOLOV8::post_process(const cv::Mat& image, 
                  const PreprocessParams& pre_params,
                  YoloV8BoxVec& objects) 
{
    YoloV8BoxVec proposals;


    for (int i = 0; i < output_num; ++i) {
        // 使用现有的output_buffer成员
        void* output_data = output_buffer[i].data;
        DataType dtype = convert_to_data_type(outs_attr[i].data_format);// 从属性获取数据类型
        // 获取量化参数
        int32_t zp = 0;
        float scale = 1.0f;
        if (outs_attr[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
            zp = outs_attr[i].quant_data.affine.tf_zero_point;
            scale = outs_attr[i].quant_data.affine.tf_scale;
        }
        
        int stride = (1 << i) * 8;  // 根据层级计算stride

        generate_proposals_yolov8_generic(
            stride,
            output_data,
            dtype,
            zp,
            scale,
            conf_thresh,
            proposals,
            model_input_w,
            model_input_h
        );

    }

    // 排序和NMS
    g_qsort_descent_inplace(proposals);
    std::vector<int> picked;
    // g_nms_sorted_bboxes(proposals, picked, NMS_THRESHOLD);
    g_nms_sorted_bboxes_class_aware(proposals, picked, nms_thresh);

    // 提取最终检测结果
    objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); i++) {
        objects[i] = proposals[picked[i]];
        inverse_coordinates(objects[i], pre_params);
    }

}
size_t YOLOV8::calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
    switch(format) {
        // 32位类型 (4字节)
        case TACONN_DATA_FORMAT_FP32:
        case TACONN_DATA_FORMAT_INT32:
        case TACONN_DATA_FORMAT_UINT32:
            return sizeof(uint32_t) * element_count;
        
        // 16位类型 (2字节)
        case TACONN_DATA_FORMAT_FP16:
        case TACONN_DATA_FORMAT_BFP16:
        case TACONN_DATA_FORMAT_INT16:
        case TACONN_DATA_FORMAT_UINT16:
            return sizeof(uint16_t) * element_count;
        
        // 8位类型 (1字节)
        case TACONN_DATA_FORMAT_UINT8:
        case TACONN_DATA_FORMAT_INT8:
        case TACONN_DATA_FORMAT_CHAR:
        case TACONN_DATA_FORMAT_BOOL8:
            return sizeof(uint8_t) * element_count;
        
        // 64位类型 (8字节)
        case TACONN_DATA_FORMAT_FP64:
        case TACONN_DATA_FORMAT_INT64:
        case TACONN_DATA_FORMAT_UINT64:
            return sizeof(uint64_t) * element_count;
        
        // 4位特殊类型
        case TACONN_DATA_FORMAT_INT4:
        case TACONN_DATA_FORMAT_UINT4:
            return (element_count + 1) / 2; // 向上取整
        
        default:
            std::cerr<<  "Unsupported data format: " << format<<std::endl;
            return 0;
    }
}
int YOLOV8::Detect(std::vector<cv::Mat>& input_images, std::vector<YoloV8BoxVec>& box){
    for(int i = 0; i < input_images.size(); i++){
        
        pre_process(input_images[i]);
        
        ts_->start();
        int status = ta_runtime_run_network(nnrt_context.get());
        if (status != 0) {
            std::cerr << "Run network failed." << std::endl;
            return status;
        }
        // 3. invalidate_output
        status = ta_runtime_invalidate_buffer(nnrt_context.get(),output_buffer);
        if (status != 0) {
            std::cerr << "Invalidate output buffer failed." << std::endl;
            return status;
        }
        ts_->time_accumulation("infer_time");
        taconn_profile_t single_haredware_time;
        status = ta_runtime_query(nnrt_context.get(),TACONN_QUERY_PROFILING,&single_haredware_time);
        if(status != 0){
            std::cerr << "Query hardware time failed: " << status << std::endl;
        }
        hardware_time.push_back(single_haredware_time);
        // ts_->time_map_lab["hardware_time"] += single_haredware_time.inference_time /1000.0f;
        YoloV8BoxVec detected_objects;
        
        ts_->start();
        post_process(input_images[i], pre_params[i],detected_objects);
        ts_->time_accumulation("post_time");

        box.push_back(detected_objects);
        if (i % 10 == 0 || i == input_images.size()) {
            std::cout << "Processed " << i << "/" << input_images.size() 
                 << " images (" << detected_objects.size() << " objects detected)" << std::endl;
        }
    
    }
    
    return 0;
}
int YOLOV8::Init()
{ 
    if (initialized_) {
        std::cerr << "Detector already initialized" << std::endl;
        return -1;
    }
    // 1. init taruntime 
  
    taconn_input_output_num_t num = {};
   
    int status = ta_runtime_init();
    if (status != 0) {
        std::cerr << "Failed to initialize TA runtime: " << status << std::endl;
        goto CLEANUP;
    }

    // 2. load model
    status  = ta_runtime_load_model_from_file(nnrt_context.get(), model_name.c_str(), 0);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status<< std::dec<< std::endl;
        goto CLEANUP;
    }
    
    status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_IN_OUT_NUM, &num);
    std::cout << "Input num is : " << num.input_num << ", Output num is : " << num.output_num << std::endl;
    input_num = num.input_num;
    output_num = num.output_num;  // cut model

    
    for (int i=0; i<input_num; i++)
    {
        taconn_inout_attr_t input_attr = {0};
        input_attr.index = i;
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_INPUT_ATTR, &input_attr);
        print_taconn_inout_attr(input_attr);
        ins_attr.push_back(input_attr);


    }
    //仅适用于当前单输入情况
    model_input_h = ins_attr[0].dim_size[0];
    model_input_w = ins_attr[0].dim_size[1];

    
    for (int i=0; i<output_num; i++)
    {
        taconn_inout_attr_t output_attr = {0};
        output_attr.index = i; 
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_OUTPUT_ATTR, &output_attr);
        print_taconn_inout_attr(output_attr);
        outs_attr.push_back(output_attr);
    }
     

    // 3. set input
    // const size_t input_size = 640 * 640 * 3; // 640x640 RGB image
    input_tensors = (taconn_input_t*)malloc(sizeof(taconn_input_t) * input_num);
    if (!input_tensors) {
        std::cerr << "Allocate input_tensors failed" << std::endl;
        goto CLEANUP;
    }

    for (int i=0; i<input_num; i++){
         
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ins_attr[i].data_format);
        size_t input_buffer_size = calculate_buffer_size(data_type, get_element_num(ins_attr[i]));
      
        input_tensors[i].index = i;
        input_tensors[i].size = input_buffer_size;  // size in bytes
        input_tensors[i].data = nullptr;  // will be set later
        if (posix_memalign((void**)&input_tensors[i].data, 256, input_buffer_size)!=0){
            std::cerr << "Failed to allocate input buffer" << std::endl;
            goto CLEANUP;
        }
        memset(input_tensors[i].data, 0, input_buffer_size);

    }
    status = ta_runtime_set_input_cva(nnrt_context.get(), input_num, input_tensors);
    if (status != 0) {
        std::cerr << "Set input failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    // 4. set output
    output_buffer = (taconn_buffer_t*)malloc(sizeof(taconn_buffer_t) * output_num);
    if (!output_buffer) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }
    
    
    for (int i = 0; i < output_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(outs_attr[i].data_format);
        size_t output_buffer_size = calculate_buffer_size(data_type, get_element_num(outs_attr[i]));
        status = ta_runtime_create_buffer(nnrt_context.get(), output_buffer_size, &output_buffer[i]);
        if (status != 0) {
            std::cerr << "Create output buffer "<< i << "failed: 0x%x" << std::hex << status << std::dec << std::endl;
            goto CLEANUP;
        }
    }
    
    status = ta_runtime_set_output(nnrt_context.get(), output_num, output_buffer);
    if (status != 0) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }
    initialized_ = true;
    return status;

CLEANUP:
    if (input_tensors) {
        for (int i = 0; i < input_num; i++) {
            if (input_tensors[i].data) {
                free(input_tensors[i].data);
            }
        }
        free(input_tensors);
        input_tensors = nullptr;
    }
    if (output_buffer) {
        for (int i = 0; i < output_num; i++) {
            ta_runtime_destroy_buffer(nnrt_context.get(), &output_buffer[i]);
        }
        free(output_buffer);
        output_buffer = nullptr;
    }
    if (nnrt_context.get()) {
        ta_runtime_destroy_context(nnrt_context.get());
        
    }
    ta_runtime_deinit();
    initialized_ = false;
    std::cout << "Model deinitialized" << std::endl;
    return status;

}   
int YOLOV8::deInit()
{
    if (!initialized_) {
        return -1;
    }
    // free output buffer
    for (int i = 0; i < output_num; i++) {
        ta_runtime_destroy_buffer(nnrt_context.get(), &output_buffer[i]);
    }

    // free input buffer
    if (input_tensors) {
        for(int i = 0; i < input_num; i++){
            if(input_tensors[i].data)
                free(input_tensors[i].data);
        }
        free(input_tensors);
        input_tensors = nullptr;
    }

    // free output ctx array
    if (output_buffer) {
        free(output_buffer);
        output_buffer = nullptr;
    }

    // free ctx
    if(nnrt_context.get())
        ta_runtime_destroy_context(nnrt_context.get());
    // free runtime
    ta_runtime_deinit();
    initialized_ = false;
    return 0;
}
void YOLOV8::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}