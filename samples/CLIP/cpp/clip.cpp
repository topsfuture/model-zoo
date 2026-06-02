#include "clip.hpp"

// #include <vector>
// #include <memory>
// #include <stdexcept>


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

static uint16_t f32_to_f16(float value) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint16_t h = ((x >> 16) & 0x8000) | 
                ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
                ((x >> 13) & 0x03ff);
    return h;
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

/* support func for CLIP */

void CLIP::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}


size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
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

size_t get_element_num(const taconn_inout_attr_t input_attr) {
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i) {
        size *= input_attr.dim_size[i];
    }
    return size;
}


int ta_init_model(const std::shared_ptr<ta_runtime_context>& nnrt_context_image,
                    const std::string& image_model,
                    std::vector<taconn_input_t>& input_tensors,     
                    std::vector<taconn_buffer_t>& output_buffers,   
                    std::vector<taconn_inout_attr_t>& ins_attr,
                    std::vector<taconn_inout_attr_t>& outs_attr,
                    int& input_num,
                    int& output_num,
                    int& model_input_h,
                    int& model_input_w,
                    int core_id) {
    
    if (!nnrt_context_image || image_model.empty()) {
        std::cerr << "Invalid parameters" << std::endl;
        return -1;
    }

    // 1. load model
    int status = ta_runtime_load_model_from_file(nnrt_context_image.get(), image_model.c_str(), core_id);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status << std::dec << std::endl;
        return status;
    }

    // 2. query input/output num
    taconn_input_output_num_t num = {};
    status = ta_runtime_query(nnrt_context_image.get(), TACONN_QUERY_IN_OUT_NUM, &num);
    if (status != 0) {
        std::cerr << "Query input/output number failed: 0x" << std::hex << status << std::dec << std::endl;
        return status;
    }
    
    input_num = num.input_num;
    output_num = num.output_num;
    std::cout << "Input num: " << input_num << ", Output num: " << output_num << std::endl;

    if (input_num <= 0 || output_num <= 0) {
        std::cerr << "Invalid input/output number" << std::endl;
        return -1;
    }

    // 3. 重置 vector 大小
    input_tensors.resize(input_num);
    output_buffers.resize(output_num);

    // 4. query input attributes
    std::vector<taconn_inout_attr_t> input_attr(input_num);
    for (int i = 0; i < input_num; i++) {
        input_attr[i].index = i;
        status = ta_runtime_query(nnrt_context_image.get(), TACONN_QUERY_INPUT_ATTR, &input_attr[i]);
        if (status != 0) {
            std::cerr << "Query input attribute " << i << " failed: 0x" << std::hex << status << std::dec << std::endl;
            return status;
        }
        ins_attr.push_back(input_attr[i]);
        print_taconn_inout_attr(input_attr[i]);
    }

    // 5. set model input size
    std::cout << "Model input dimensions: " << input_attr[0].dim_count << "D" << std::endl;
    if (input_attr[0].dim_count >= 2) {
        model_input_h = input_attr[0].dim_size[1];
        model_input_w = input_attr[0].dim_size[0];
        std::cout << "Model input size (w x h): " << model_input_w << " x " << model_input_h << std::endl;
    } 
    else {
        std::cerr << "Invalid input dimensions" << std::endl;
        return -1;
    }

    // 6. query output attributes
    std::vector<taconn_inout_attr_t> output_attr(output_num);
    for (int i = 0; i < output_num; i++) {
        output_attr[i].index = i;
        status = ta_runtime_query(nnrt_context_image.get(), TACONN_QUERY_OUTPUT_ATTR, &output_attr[i]);
        if (status != 0) {
            std::cerr << "Query output attribute " << i << " failed: 0x" << std::hex << status << std::dec << std::endl;
            return status;
        }
        outs_attr.push_back(output_attr[i]);
        print_taconn_inout_attr(output_attr[i]);
    }

    // 7. allocate memory for each input tensor
    for (int i = 0; i < input_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(input_attr[i].data_format);
        size_t input_buffer_size = calculate_buffer_size(data_type, get_element_num(input_attr[i]));
        
        input_tensors[i].index = i;
        input_tensors[i].size = input_buffer_size;
        
        if (posix_memalign((void**)&input_tensors[i].data, 256, input_buffer_size) != 0) {
            std::cerr << "Failed to allocate input buffer for tensor " << i << std::endl;
            // 清理已分配的内存
            for (int j = 0; j < i; j++) {
                free(input_tensors[j].data);
            }
            return -1;
        }
        memset(input_tensors[i].data, 0, input_buffer_size);
    }

    // 8. set input
    status = ta_runtime_set_input_cva(nnrt_context_image.get(), input_num, input_tensors.data());
    if (status != 0) {
        std::cerr << "Set input failed: 0x" << std::hex << status << std::dec << std::endl;
        // Cleanup
        for (int i = 0; i < input_num; i++) {
            free(input_tensors[i].data);
        }
        return status;
    }

    // 9. create output buffers
    for (int i = 0; i < output_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(output_attr[i].data_format);
        size_t output_buffer_size = calculate_buffer_size(data_type, get_element_num(output_attr[i]));
        
        status = ta_runtime_create_buffer(nnrt_context_image.get(), output_buffer_size, &output_buffers[i]);
        if (status != 0) {
            std::cerr << "Create output buffer " << i << " failed: 0x" << std::hex << status << std::dec << std::endl;
            // Cleanup previously created buffers
            for (int j = 0; j < i; j++) {
                ta_runtime_destroy_buffer(nnrt_context_image.get(), &output_buffers[j]);
            }
            // Cleanup input tensors
            for (int i = 0; i < input_num; i++) {
                free(input_tensors[i].data);
            }
            return status;
        }
    }

    // 10. set output
    status = ta_runtime_set_output(nnrt_context_image.get(), output_num, output_buffers.data());
    if (status != 0) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        // Cleanup
        for (int i = 0; i < output_num; i++) {
            ta_runtime_destroy_buffer(nnrt_context_image.get(), &output_buffers[i]);
        }
        for (int i = 0; i < input_num; i++) {
            free(input_tensors[i].data);
        }
        return status;
    }

    return 0;
}



void mat_to_tensor(const cv::Mat& mat, uint8_t* tensor) {
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

void mat_to_tensor(const cv::Mat& mat, float* tensor) {
    /* hwc - (n)chw */
    int channels = mat.channels();
    int total_pixels = mat.rows * mat.cols;

    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);

    // 按 CHW 顺序拷贝每个通道
    for (int c = 0; c < channels; ++c) {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(float));
    }
}



void CLIP::quantize(float* src, void* dst, size_t num_elements, 
                                   const taconn_inout_attr_t& attr) {
    /* f32 to quantized, 无 i8, u8 适配*/
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE) {
        // 无量化处理 - 转换为 float32 或 float16
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
            // 转换为 float32 并标准化
            float* dst_float = static_cast<float*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                dst_float[i] = src[i];
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
            // 转换为 float16 并标准化
            uint16_t* dst_fp16 = static_cast<uint16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i] ;
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
                float normalized = src[i];
                int32_t quantized = static_cast<int32_t>(std::round(normalized / scale)) + zero_point;
                dst_int8[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            uint8_t* dst_uint8 = static_cast<uint8_t*>(dst);
            // 直接复制 uint8 数据
            for (size_t i = 0; i < num_elements; i++){
                int32_t quantized = static_cast<int32_t>(std::round(src[i] / scale)) + zero_point;
                dst_uint8[i] = static_cast<uint8_t>(std::clamp(quantized, 0, 255));

            }
        }
    } 
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_DFP) {
        // DFP量化格式处理
        int fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                float normalized = src[i];
                int32_t scaled = lrintf(normalized * (1 << fixed_point_pos));
                dst_int8[i] = static_cast<int8_t>(std::clamp(scaled, -128, 127));
            }
        }
    }
}

/*  CLIP */

int CLIP::init(const std::string& image_model, const std::string& text_model,
                const std::shared_ptr<ta_runtime_context>& nnrt_context_text,
                const std::shared_ptr<ta_runtime_context>& nnrt_context_images,
                const std::string& text_projection_path,
                bool is_chinese){

    is_chinese_ = is_chinese;

    // 1. init taruntime 
    // model_context = ModelContext();
    int status = ta_runtime_init();
    if (status != 0) {
        std::cerr << "Failed to initialize TA runtime: " << status << std::endl;
        return status;
    }

    status = ta_init_model(nnrt_context_images,
                    image_model,
                    image_model_info.input_tensors,
                    image_model_info.output_buffers,
                    image_model_info.ins_attr,
                    image_model_info.outs_attr,
                    image_model_info.input_num,
                    image_model_info.output_num,
                    image_model_info.model_input_h,
                    image_model_info.model_input_w,
                    CORE_0);
    if (status != 0) {
        std::cerr << "Failed to initialize image model: " << status << std::endl;
        return status;
    }

    status = ta_init_model(nnrt_context_text,
                    text_model,
                    text_model_info.input_tensors,
                    text_model_info.output_buffers,
                    text_model_info.ins_attr,
                    text_model_info.outs_attr,
                    text_model_info.input_num,
                    text_model_info.output_num,
                    text_model_info.batch_size,
                    text_model_info.token_len,
                    CORE_1);
    if (status != 0) {
        std::cerr << "Failed to initialize text model: " << status << std::endl;
        return status;
    }

    if (!is_chinese_ && !text_projection_path.empty()) {
        // 英文模型：从文件加载 text_projection 矩阵
        std::ifstream file(text_projection_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open text projection file: " << text_projection_path << std::endl;
        } else {
            char header[128];
            file.read(header, 128);
            size_t header_length = 0;
            while (header[header_length] != '\n') header_length++;
            file.seekg(header_length + 1, std::ios::beg);
            
            const size_t rows = 512, cols = 512;   // hidden 维度512 
            text_projection = cv::Mat(rows, cols, CV_32FC1);
            std::vector<float> flat_data(rows * cols);
            file.read(reinterpret_cast<char*>(flat_data.data()), flat_data.size() * sizeof(float));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    text_projection.at<float>(i, j) = flat_data[i * cols + j];
                }
            }
            file.close();
            // std::cout << "Loaded text projection matrix for English model." << std::endl;
        }
    } else {
        // 中文模型：不加载投影矩阵，将 cv::Mat 置空
        text_projection = cv::Mat();
        // std::cout << "Running in Chinese mode, text projection is disabled." << std::endl;
    }

    return status;
}
void CLIP::deinit(std::shared_ptr<ta_runtime_context> nnrt_context_images,
                  std::shared_ptr<ta_runtime_context> nnrt_context_text) {
    // 清理图像模型资源
    for (auto& tensor : image_model_info.input_tensors) {
        if (tensor.data) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
    for (auto& buffer : image_model_info.output_buffers) {
        ta_runtime_destroy_buffer(nnrt_context_images.get(), &buffer);
    }
    image_model_info.input_tensors.clear();
    image_model_info.output_buffers.clear();
    image_model_info.ins_attr.clear();
    image_model_info.outs_attr.clear();

    // 清理文本模型资源（类似处理）
    for (auto& tensor : text_model_info.input_tensors) {
        if (tensor.data) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
    for (auto& buffer : text_model_info.output_buffers) {
        ta_runtime_destroy_buffer(nnrt_context_text.get(), &buffer);
    }
    text_model_info.input_tensors.clear();
    text_model_info.output_buffers.clear();
    text_model_info.ins_attr.clear();
    text_model_info.outs_attr.clear();

    ta_runtime_deinit();
}

size_t CLIP::get_token_len() const {

    return text_model_info.token_len;
}




std::tuple<cv::Mat, std::pair<float, float>, std::pair<float, float>> CLIP:: letterbox(const cv::Mat& im, 
                                                                                        const cv::Size& new_shape, 
                                                                                        const cv::Scalar& color, 
                                                                                        bool auto_pad, 
                                                                                        bool scaleFill, 
                                                                                        bool scaleup, 
                                                                                        int stride) {
    cv::Size shape = im.size(); // [width, height]
    float r = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);
    if (!scaleup) {
        r = std::min(r, 1.0f);
    }
 
    cv::Size new_unpad(static_cast<int>(round(shape.width * r)), static_cast<int>(round(shape.height * r)));
    float dw = new_shape.width - new_unpad.width;
    float dh = new_shape.height - new_unpad.height;
    
    if (auto_pad) {
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_unpad = new_shape;
        r = static_cast<float>(new_shape.width) / shape.width;
    }

    dw /= 2;
    dh /= 2;

    cv::Mat resized_img;
    if (shape != new_unpad) {
        cv::resize(im, resized_img, new_unpad, 0, 0, cv::INTER_CUBIC);
    } else {
        resized_img = im.clone();
    }

    int top = static_cast<int>(round(dh - 0.1));
    int bottom = static_cast<int>(round(dh + 0.1));
    int left = static_cast<int>(round(dw - 0.1));
    int right = static_cast<int>(round(dw + 0.1));
    
    cv::Mat letterboxed_img;
    cv::copyMakeBorder(resized_img, letterboxed_img, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return std::make_tuple(letterboxed_img, std::make_pair(r, r), std::make_pair(dw, dh));
}



std::vector<float> CLIP::preprocess_cpu_letterbox(const cv::Mat& image) {
    cv::Size new_shape(224, 224);

    auto [letterbox_img, ratio, padding] = letterbox(image, new_shape);

    // Convert to RGB and normalize
    cv::Mat rgb_image;
    cv::cvtColor(letterbox_img, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0); // Convert to float and scale to [0, 1]

    std::vector<float> mean = {0.48145466, 0.4578275, 0.40821073};
    std::vector<float> std = {0.26862954, 0.26130258, 0.27577711};
    rgb_image.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* position) -> void {
        pixel[0] = (pixel[0] - mean[0]) / std[0];
        pixel[1] = (pixel[1] - mean[1]) / std[1];
        pixel[2] = (pixel[2] - mean[2]) / std[2];
    });

    // return rgb_image;
    cv::Mat blob;
    cv::dnn::blobFromImage(rgb_image, blob);   // hwc - nchw

    std::vector<float> input_tensor;
    int total_elements = blob.total();
    input_tensor.resize(total_elements);
    std::memcpy(input_tensor.data(), blob.data, total_elements * sizeof(float));
    return input_tensor;
}

void CLIP::preprocess(const cv::Mat& image) {
    for(int i = 0; i < image_model_info.input_num; i++){
        // quantize
        if (image_model_info.ins_attr[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE &&
            image_model_info.ins_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            /* with pre_node, image.data */
            memcpy(image_model_info.input_tensors[i].data, image.data, image_model_info.input_tensors[i].size);  
        }
        else{    // prepare input layout to  (N)CHW
            /* without pre_node */
            std::vector<float> input_tensor_f32 = preprocess_cpu_letterbox(image);
            size_t num_elements = get_element_num(image_model_info.ins_attr[i]);
            quantize( &input_tensor_f32[0], 
                    image_model_info.input_tensors[i].data, 
                    num_elements, 
                    image_model_info.ins_attr[i]);
        }
    }
}


void CLIP::normalize(std::vector<float>& features) {
    float norm = std::sqrt(std::inner_product(features.begin(), features.end(), features.begin(), 0.0f));
    for (auto& f : features) {
        f /= norm;
    }
}


void fast_convert_int32_to_uint16(const int32_t* src, uint16_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<uint16_t>(src[i]);
    }
}


std::vector<uint16_t> quantize_clip_input(const std::vector<int32_t>& token_ids, 
                                        size_t batch_size, size_t seq_len) {
    if (token_ids.size() != batch_size * seq_len) {
        throw std::invalid_argument("输入尺寸不匹配");
    }
    
    std::vector<uint16_t> result(batch_size * seq_len);
    fast_convert_int32_to_uint16(token_ids.data(), result.data(), token_ids.size());
    
    return result;
}





std::vector<float> CLIP::encode_text(const std::vector<int>& text, const std::shared_ptr<ta_runtime_context>& nnrt_context_text){
    // 1. 检查模型是否初始化
    if (text_model_info.input_tensors.empty() || text_model_info.ins_attr.empty()) {
        std::cerr << "Text model not properly initialized" << std::endl;
        return std::vector<float>();
    }
    
    // 2. 获取模型输入属性
    taconn_inout_attr_t& input_attr = text_model_info.ins_attr[0];
    taconn_input_t& input_tensor = text_model_info.input_tensors[0];
    
    // 3. 验证输入形状
    if (input_attr.dim_count != 2 || input_attr.dim_size[0] != 52) {
        std::cerr << "Unexpected text model input shape: [";
        for (unsigned int i = 0; i < input_attr.dim_count; ++i) {
            std::cerr << input_attr.dim_size[i];
            if (i != input_attr.dim_count - 1) std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        // return std::vector<float>();
    }
    size_t max_token_len = input_attr.dim_size[0]; // 应该是52
    // std::cout << "max_token_len :" << max_token_len << std::endl;
    // 4. 处理文本输入：填充或截断到固定长度
    if (ts_) ts_->start();
    std::vector<int> processed_text = text;
    if (processed_text.size() > max_token_len) {
        // 截断到最大长度
        processed_text.resize(max_token_len);
        std::cout << "Warning: Text truncated to " << max_token_len << " tokens" << std::endl;
    } else if (processed_text.size() < max_token_len) {
        // 填充到最大长度（用0填充）
        processed_text.resize(max_token_len, 0);
        std::cout << "Text padded to " << max_token_len << " tokens" << std::endl;
    }
    taconn_data_format_t data_type = static_cast<taconn_data_format_t>(input_attr.data_format);
    size_t required_size = calculate_buffer_size(data_type, get_element_num(input_attr));

    std::memcpy(input_tensor.data, processed_text.data(), required_size);
    if (ts_) ts_->time_accumulation("tokenize_time");


    // 7. 运行文本模型推理
    if (ts_) ts_->start();

    int status = ta_runtime_run_network(nnrt_context_text.get());
    if (status != 0) {
        std::cerr << "Text model inference failed: " << status << std::endl;
        return std::vector<float>();
    }
    if (ts_) ts_->time_accumulation("encode_text_time");
    
    // 8. 使输出buffer失效
    if (text_model_info.output_buffers.empty()) {
        std::cerr << "No output buffers available" << std::endl;
        return std::vector<float>();
    }
    
    status = ta_runtime_invalidate_buffer(nnrt_context_text.get(), 
                                         &text_model_info.output_buffers[0]);
    if (status != 0) {
        std::cerr << "Invalidate output buffer failed: " << status << std::endl;
        return std::vector<float>();
    }


    // 9. 获取并反量化输出数据
    std::vector<float> output_data;
    taconn_buffer_t& output_buffer = text_model_info.output_buffers[0];
    taconn_inout_attr_t& output_attr = text_model_info.outs_attr[0];
    
    size_t num_elements = get_element_num(output_attr);
    output_data.resize(num_elements);  // flatten output tensor
    
    
    // 反量化处理
    DataType dtype = convert_to_data_type(output_attr.data_format);
    DequantizeFunc dequant = dequantize_funcs[static_cast<int>(dtype)];
    
    float scale = 1.0f;
    int32_t zp = 0;
    
    if (output_attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        scale = output_attr.quant_data.affine.tf_scale;
        zp = output_attr.quant_data.affine.tf_zero_point;
    }
    
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = dequant(output_buffer.data, i, zp, scale);
    }

    std::vector<float> extracted_row;
    if (is_chinese_) {
        extracted_row.assign(output_data.begin(), output_data.begin() + hidden_dim);
    } else {
        // 找到最后一个非零token（CLIP使用EOS token后的特征）
        int last_valid_index = processed_text.size() - 1;
        for (; last_valid_index >= 0; --last_valid_index) {
            if (processed_text[last_valid_index] != 0) {
                break;
            }
        }
        
        if (last_valid_index < 0) {
            last_valid_index = processed_text.size() - 1; // 如果全部是0，使用最后一个
        }
        
        
        size_t hidden_dim = output_attr.dim_size[0];
        
        int feature_start_index = last_valid_index * hidden_dim;
        if (feature_start_index + hidden_dim > output_data.size()) {
            std::cerr << "Invalid feature extraction index: " << feature_start_index 
                    << " + " << hidden_dim << " > " << output_data.size() << std::endl;
            return std::vector<float>();
        }
        
        extracted_row.assign(output_data.begin() + feature_start_index, 
                                        output_data.begin() + feature_start_index + hidden_dim);
    }

    // 3. 矩阵投影
    std::vector<float> result(embed_dim, 0.0f);
    if (text_projection.empty()) {
        // 如果没有投影矩阵，直接使用原始特征
        result = std::vector<float>(extracted_row.begin(), extracted_row.begin() + embed_dim);

    } else {
        // 将提取的行转换为 OpenCV Mat (1×512 行向量)
        cv::Mat extracted_row_mat(1, hidden_dim, CV_32FC1, extracted_row.data());
        // 执行矩阵乘法: [1×512] × [512×512] = [1×512]
        cv::Mat result_mat = extracted_row_mat * text_projection;
        // 将结果复制回 vector
        std::memcpy(result.data(), result_mat.ptr<float>(0), embed_dim * sizeof(float));

    }

    normalize(result);
    return result;
}

std::vector<float> CLIP::encode_image(const std::string image_path , const std::shared_ptr<ta_runtime_context>& nnrt_context_images){
    // 1. 读取图像
    if (ts_) ts_->start();
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR | cv::IMREAD_RETRY_SOFTDEC);
    if (image.empty()) {
        std::cerr << "Error reading image: " << image_path << std::endl;
        return std::vector<float>();
    }
    if (ts_) ts_->time_accumulation("imread_time");
    std::cout << "Filename: " << image_path << std::endl;
    
    // 2. 预处理图像
    if (ts_) ts_->start();
    preprocess(image);
    if (ts_) ts_->time_accumulation("pre_time");
    
    // 3. 运行图像模型推理
    if (ts_) ts_->start();
    int status = ta_runtime_run_network(nnrt_context_images.get());
    if (status != 0) {
        std::cerr << "Run network failed." << std::endl;
        return std::vector<float>();
    }
    if (ts_) ts_->time_accumulation("encode_image_time");

    
    // 4. 使输出buffer失效
    status = ta_runtime_invalidate_buffer(nnrt_context_images.get(), &image_model_info.output_buffers[0]);

    if (status != 0) {
        std::cerr << "Invalidate output buffer failed." << std::endl;
        return std::vector<float>();
    }

    // 5. 反量化输出数据
    DataType dtype = convert_to_data_type(image_model_info.outs_attr[0].data_format);
    DequantizeFunc dequant = dequantize_funcs[static_cast<int>(dtype)];
    size_t num_elements = get_element_num(image_model_info.outs_attr[0]);

    std::vector<float> result(num_elements);
    float scale = image_model_info.outs_attr[0].quant_data.affine.tf_scale;
    int zp = image_model_info.outs_attr[0].quant_data.affine.tf_zero_point;
    
    for (uint32_t j = 0; j < num_elements; ++j) {
        result[j] = dequant(image_model_info.output_buffers[0].data, j, zp, scale);
    }

    // 6. 返回第一个输出的特征向量
    normalize(result);
    return result;
}


std::pair<std::vector<float>, std::vector<int>> CLIP::topk(const std::vector<float>& x, int k) {
    std::vector<int> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int a, int b) {
        return x[a] > x[b];
    });
    std::vector<float> values(k);
    for (int i = 0; i < k; ++i) {
        values[i] = x[indices[i]];
    }
    return {values, std::vector<int>(indices.begin(), indices.begin() + k)};
}

std::vector<float> CLIP::softmax(const std::vector<float>& x) {
    std::vector<float> e_x(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        e_x[i] = std::exp(x[i] - max_val);
        sum += e_x[i];
    }
    for (size_t i = 0; i < e_x.size(); ++i) {
        e_x[i] /= sum;
    }
    return e_x;
}

std::vector<float> CLIP::calculate_similarity(const std::vector<float>& image_features,
                                        const std::vector<std::vector<float>>& text_features) {
    size_t num_text_features = text_features.size();
    std::vector<float> similarity(num_text_features);
    for (size_t i = 0; i < num_text_features; ++i) {
        similarity[i] = 100.0f * std::inner_product(image_features.begin(), image_features.end(),
                                                      text_features[i].begin(), 0.0f);
        // similarity[i] = 1.0f * std::inner_product(image_features.begin(), image_features.end(),
        //                                               text_features[i].begin(), 0.0f);                                                      
    }
    return softmax(similarity);
}



int CLIP::infer(const std::vector<std::string>& image_paths, const std::vector<std::vector<int>> tokenlized_text,
           const std::vector<std::string>& text_inputs,
           const std::shared_ptr<ta_runtime_context>& nnrt_context_text,
           const std::shared_ptr<ta_runtime_context>& nnrt_context_images){
    /* Bidirectional retrieval */
    // calculate text features
    std::vector<std::vector<float>> text_features;
    for (const auto& text : tokenlized_text){
        std::vector<float> text_feature = encode_text(text, nnrt_context_text);
        text_features.push_back(text_feature);
    }
    // calculate image features
    std::vector<std::vector<float>> image_features;
    for (const auto& filename : image_paths){
        std::vector<float> image_feature = encode_image(filename, nnrt_context_images);
        image_features.push_back(image_feature);
    }

    bool no_result =true;
    // post-process text
    if (text_features.size() > 1){
        no_result = false;
        // process each text
        for (size_t i = 0; i < image_features.size(); ++i) {
            const auto& image_feature = image_features[i];
            std::cout << "Image: " << image_paths[i] << std::endl;
            std::vector<float> similarity(text_inputs.size());
            // calculate similarity per image
            similarity = calculate_similarity(image_feature, text_features);
            int output_size = std::min(text_inputs.size(), static_cast<size_t>(top_k));
            auto [values, indices] = topk(similarity, output_size);
            for (size_t i = 0; i < output_size; ++i) {
                std::cout << "Similarity: " << values[i] << ", Text: " << text_inputs[indices[i]] << std::endl;
            }
        }
    }
    // post-process image
    if (image_features.size() > 1){
        no_result = false;
        std::cout << "\nTotal Similarity per Text:" << std::endl;
        for (size_t i = 0; i < text_features.size(); ++i) {
            const auto& text_feature = text_features[i];
            std::cout << "Text: " << text_inputs[i] << std::endl;
            std::vector<float> similarity(image_features.size());
            // calculate similarity per text
            similarity = calculate_similarity(text_feature, image_features);
            int output_size = std::min(image_features.size(), static_cast<size_t>(top_k));
            auto [values, indices] = topk(similarity, output_size);
            for (size_t i = 0; i < output_size; ++i) {
                std::cout << "Similarity: " << values[i] << ", Image: " << image_paths[indices[i]] << std::endl;
            }
        }
    }
    std::cout << std::defaultfloat << std::setprecision(6);
    if (no_result){
        std::cout << "\nPlease input multiple images or more texts" << std::endl;
    }

    return 0;

}




// 需要为CLIP类添加这个方法来支持内存中的图像处理for cifar100
std::vector<float> CLIP::encode_image_memory(const cv::Mat& image, 
                                        const std::shared_ptr<ta_runtime_context>& nnrt_context_images) {
    if (image.empty()) {
        std::cerr << "Error reading image in encode_image_memory() "  << std::endl;
        return std::vector<float>();
    }
    // 预处理图像
    if (ts_) ts_->start();
    preprocess(image);
    if (ts_) ts_->time_accumulation("pre_time");
    // 运行图像模型推理
    if (ts_) ts_->start();
    int status = ta_runtime_run_network(nnrt_context_images.get());
    if (status != 0) {
        std::cerr << "Run network failed." << std::endl;
        return std::vector<float>();
    }
    if (ts_) ts_->time_accumulation("encode_image_time");

    
    // 使输出buffer失效
    status = ta_runtime_invalidate_buffer(nnrt_context_images.get(), 
                                            &image_model_info.output_buffers[0]);
    
    if (status != 0) {
        std::cerr << "Invalidate output buffer failed." << std::endl;
        return std::vector<float>();
    }
    
    // 反量化输出数据
        DataType dtype = convert_to_data_type(image_model_info.outs_attr[0].data_format);
        DequantizeFunc dequant = dequantize_funcs[static_cast<int>(dtype)];
        size_t num_elements = get_element_num(image_model_info.outs_attr[0]);
        
        std::vector<float> result(num_elements);
        float scale = image_model_info.outs_attr[0].quant_data.affine.tf_scale;
        int zp = image_model_info.outs_attr[0].quant_data.affine.tf_zero_point;
        
        for (uint32_t j = 0; j < num_elements; ++j) {
            result[j] = dequant(image_model_info.output_buffers[0].data, j, zp, scale);
        }
    normalize(result);
    return result;
}

