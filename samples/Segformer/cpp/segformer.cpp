#include "segformer.hpp"
#include <cfloat>  




// ==================== Runtime  ====================

Runtime::Runtime() : nnrt_ctx_(nullptr), rt_flag(false) {
}

Runtime::~Runtime() {
    cleanup();
}

int Runtime::init_ctx(const Config& config) {
    if (rt_flag) {
        std::cerr << "Runtime already initialized" << std::endl;
        return -1;
    }
    config_ = config;
    int status = ta_runtime_init();
    if (status != 0) {
        std::cerr << "Failed to initialize TA runtime: 0x" 
                  << std::hex << status << std::dec << std::endl;
        return status;
    }
    nnrt_ctx_ = std::make_shared<ta_runtime_context>(0);
    rt_flag = true;
    return 0;
}

int Runtime::load_model(const std::string& model_path) {
    if (!rt_flag) {
        std::cerr << "Runtime not initialized" << std::endl;
        return -1;
    }

    int status = ta_runtime_load_model_from_file(nnrt_ctx_.get(),  model_path.c_str(),  config_.core_id);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status << std::dec << std::endl;
    }

    return status;
}

int Runtime::query_io_count(int& input_num, int& output_num) {
    if (!rt_flag) {
        return -1;
    }

    taconn_input_output_num_t num = {};
    int status = ta_runtime_query(
        nnrt_ctx_.get(), 
        TACONN_QUERY_IN_OUT_NUM, 
        &num
    );

    if (status == 0) {
        input_num = num.input_num;
        output_num = num.output_num;
    }

    return status;
}

int Runtime::get_input_attribute(int index, taconn_inout_attr_t& attr) {
    if (!rt_flag) {
        return -1;
    }

    attr.index = index;
    return ta_runtime_query(
        nnrt_ctx_.get(),
        TACONN_QUERY_INPUT_ATTR,
        &attr
    );
}

int Runtime::get_output_attribute(int index, taconn_inout_attr_t& attr) {
    if (!rt_flag) {
        return -1;
    }

    attr.index = index;
    return ta_runtime_query(
        nnrt_ctx_.get(),
        TACONN_QUERY_OUTPUT_ATTR,
        &attr
    );
}

int Runtime::allocate_input_tensor(int index, size_t size, taconn_input_t& tensor) {
    if (posix_memalign((void**)&tensor.data, config_.memory_alignment, size) != 0) {
        std::cerr << "Failed to allocate input tensor memory" << std::endl;
        return -1;
    }

    memset(tensor.data, 0, size);
    tensor.index = index;
    tensor.size = size;

    return 0;
}

int Runtime::allocate_output_buffer(int index, size_t size, taconn_buffer_t& buffer) {
    return ta_runtime_create_buffer(nnrt_ctx_.get(), size, &buffer);
}

int Runtime::set_input_tensors(int count, taconn_input_t* tensors) {
    return ta_runtime_set_input_cva(nnrt_ctx_.get(), count, tensors);
}

int Runtime::set_output_buffers(int count, taconn_buffer_t* buffers) {
    return ta_runtime_set_output(nnrt_ctx_.get(), count, buffers);
}

int Runtime::run_inference() {
    ta_runtime_run_network(nnrt_ctx_.get());
    return 0;
}

void Runtime::cleanup() {
    if (rt_flag) {
        ta_runtime_deinit();
        rt_flag = false;
    }
}



// ================== QuantizationUtils ==================

float QU::dequantize_int32(void* data, size_t idx, int32_t zp, float scale) {
    return static_cast<int32_t*>(data)[idx];
}


float QU::dequantize_float32(void* data, size_t idx, int32_t, float) {
    return static_cast<float*>(data)[idx];
}

float QU::dequantize_float16(void* data, size_t idx, int32_t, float) {
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

float QU::dequantize_uint8(void* data, size_t idx, int32_t zp, float scale) {
    uint8_t val = static_cast<uint8_t*>(data)[idx];
    return ((float)val - (float)zp) * scale;
}

float QU::dequantize_int8(void* data, size_t idx, int32_t zp, float scale) {
    int8_t val = static_cast<int8_t*>(data)[idx];
    return ((float)val - (float)zp) * scale;
}


uint16_t QU::f32_to_f16(float value) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint16_t h = ((x >> 16) & 0x8000) | 
                ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
                ((x >> 13) & 0x03ff);
    return h;
}
static int32_t fp32_to_dfp(float in, int8_t fl, taconn_data_format_t type) {
    int32_t data;
    int32_t min_range, max_range;


    switch(type) {
        case TACONN_DATA_FORMAT_INT8:
            min_range = -128;
            max_range = 127;
            break;
        case TACONN_DATA_FORMAT_INT16:
            min_range = -32768;
            max_range = 32767;
            break;
        case TACONN_DATA_FORMAT_INT32:
            min_range = std::numeric_limits<int32_t>::min();
            max_range = std::numeric_limits<int32_t>::max();
            break;
        case TACONN_DATA_FORMAT_UINT8:
            min_range = 0;
            max_range = 255;
            break;
        case TACONN_DATA_FORMAT_UINT16:
            min_range = 0;
            max_range = 65535;
            break;
        case TACONN_DATA_FORMAT_UINT32:
            min_range = 0;
            max_range = std::numeric_limits<int32_t>::max(); // 注意：返回值是int32_t，所以UINT32的max要限制在int32_t范围内
            break;
        case TACONN_DATA_FORMAT_INT4:
            min_range = -8;
            max_range = 7;
            break;
        case TACONN_DATA_FORMAT_UINT4:
            min_range = 0;
            max_range = 15;
            break;
        case TACONN_DATA_FORMAT_CHAR:
            min_range = -128;
            max_range = 127;
            break;
        case TACONN_DATA_FORMAT_BOOL8:
            min_range = 0;
            max_range = 1;
            break;
        default:

            min_range = std::numeric_limits<int32_t>::min();
            max_range = std::numeric_limits<int32_t>::max();
            break;
    }


    if (fl > 0) {
        data = static_cast<int32_t>(std::rint(in * static_cast<float>(1 << fl)));
    } else if (fl < 0) {

        data = static_cast<int32_t>(std::rint(in * (1.0f / static_cast<float>(1 << -fl))));
    } else {

        data = static_cast<int32_t>(std::rint(in));
    }

    if (data > max_range) {
        data = max_range;
    }
    if (data < min_range) {
        data = min_range;
    }

    return data;
}

static float dfp_to_fp32(
    const int32_t val,
    const int8_t  fl
    )
{
    float result;
    if(fl > 0 ) {
        result = (float)val * (1.0f / ((float) ((int64_t)1 << fl ) ) );
    }
    else
    {
        result = (float)val * ((float) ((int64_t)1 << -fl ) );
    }
    return result;
}
void QU::quantize(float* src, void* dst, size_t num_elements, 
                          const taconn_inout_attr_t& attr) {
    /* f32 to quantized, 无 i8, u8 适配*/
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE) {
        // 无量化处理 - 转换为 float32 或 float16
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
            // 转换为 float32
            float* dst_float = static_cast<float*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                dst_float[i] = src[i];
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
            // 转换为 float16
            uint16_t* dst_fp16 = static_cast<uint16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                dst_fp16[i] = f32_to_f16(src[i]);
            }
        }
        else {
            std::cerr << "Unsupported data format for non-quantized input: " 
                      << attr.data_format << std::endl;
        }
    } 
    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        // 对称量化 (int8, uint8)
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                int32_t quantized = static_cast<int32_t>(std::round(src[i] / scale)) + zero_point;
                dst_int8[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            uint8_t* dst_uint8 = static_cast<uint8_t*>(dst);
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
                // float normalized = src[i] / 255.0f;
                int32_t scaled = lrintf(src[i] * (1 << fixed_point_pos));
                dst_int8[i] = static_cast<int8_t>(std::clamp(scaled, -128, 127));
            }
        }else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT16) {
            int16_t* dst_int8 = static_cast<int16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                dst_int8[i] = (int16_t)fp32_to_dfp(src[i], attr.quant_data.dfp.fixed_point_pos, (taconn_data_format_t)attr.data_format);
            }
           
        }else{
            std::cout << "not support quant_format: " << attr.quant_format << std::endl;
        }
        
    }
}


QU::DequantizeFunc QU::get_dequantize_func(uint32_t data_format) {
    static const std::unordered_map<uint32_t, DequantizeFunc> dequantize_map = {
        {taconn_data_format_e::TACONN_DATA_FORMAT_FP32,   dequantize_float32},
        {taconn_data_format_e::TACONN_DATA_FORMAT_FP16,   dequantize_float16},
        {taconn_data_format_e::TACONN_DATA_FORMAT_UINT8,  dequantize_uint8},
        {taconn_data_format_e::TACONN_DATA_FORMAT_INT8,   dequantize_int8},
        {taconn_data_format_e::TACONN_DATA_FORMAT_INT32,  dequantize_int32},
    };
    auto it = dequantize_map.find(data_format);
    if (it != dequantize_map.end()) {
        return it->second;
    }
    std::cerr << "Warning: No dequantize function for data format: " 
              << data_format << ", using default (float32)" << std::endl;
    return dequantize_float32;
}



// ==================== ModelManager ====================

ModelManager::ModelManager(std::shared_ptr<Runtime> runtime) 
    : runtime_(runtime), model_loaded_(false) {
    if (!runtime_) {
        runtime_ = std::make_shared<Runtime>();
    }
}

ModelManager::~ModelManager() {
    destory_iomem();
}

void ModelManager::set_runtime(std::shared_ptr<Runtime> runtime) {
    if (model_loaded_) {
        std::cerr << "Cannot change runtime while model is loaded" << std::endl;
        return;
    }
    runtime_ = runtime;
}

int ModelManager::init_runtime(const std::string& model_path,
                            const Runtime::Config& runtime_config) {
    if (model_loaded_) {
        destory_iomem();
    }
    // init ctx
    int status = runtime_->init_ctx(runtime_config);
    if (status != 0) {
        std::cerr << "Failed to initialize hardware runtime" << std::endl;
        return status;
    }
    // load model from file
    status = runtime_->load_model(model_path);
    if (status != 0) {
        std::cerr << "Failed to load model" << std::endl;
        return status;
    }

    status = runtime_->query_io_count(model_info_.input_num, model_info_.output_num);  // query io num
    if (status != 0 || model_info_.input_num <= 0 || model_info_.output_num <= 0) {
        std::cerr << "Invalid input/output numbers" << std::endl;
        return -1;
    }

    std::cout << "Model loaded: Inputs=" << model_info_.input_num  << ", Outputs=" << model_info_.output_num << std::endl;
    // allocate input tensors mem
    model_info_.input_attributes.resize(model_info_.input_num);
    model_info_.input_tensors.resize(model_info_.input_num);

    for (int i = 0; i < model_info_.input_num; i++) {
        status = runtime_->get_input_attribute(i, model_info_.input_attributes[i]);
        if (status != 0) {
            std::cerr << "Failed to get input attribute " << i << std::endl;
            destory_iomem();
            return status;
        }

        size_t element_num = get_element_num(model_info_.input_attributes[i]);
        size_t buffer_size = calculate_buffer_size(
            static_cast<taconn_data_format_t>(model_info_.input_attributes[i].data_format),
            element_num
        );

        status = runtime_->allocate_input_tensor(i, buffer_size, model_info_.input_tensors[i]);
        if (status != 0) {
            std::cerr << "Failed to allocate input tensor " << i << std::endl;
            destory_iomem();
            return status;
        }
    }
    // allocate output buffers 
    model_info_.output_attributes.resize(model_info_.output_num);
    model_info_.output_buffers.resize(model_info_.output_num);

    for (int i = 0; i < model_info_.output_num; i++) {
        status = runtime_->get_output_attribute(i, model_info_.output_attributes[i]);
        if (status != 0) {
            std::cerr << "Failed to get output attribute " << i << std::endl;
            destory_iomem();
            return status;
        }

        size_t element_num = get_element_num(model_info_.output_attributes[i]);
        size_t buffer_size = calculate_buffer_size(
            static_cast<taconn_data_format_t>(model_info_.output_attributes[i].data_format),
            element_num
        );

        status = runtime_->allocate_output_buffer(i, buffer_size, model_info_.output_buffers[i]);
        if (status != 0) {
            std::cerr << "Failed to allocate output buffer " << i << std::endl;
            destory_iomem();
            return status;
        }
    }

    status = runtime_->set_input_tensors(model_info_.input_num,  model_info_.input_tensors.data());
    if (status != 0) {
        std::cerr << "Failed to set input tensors" << std::endl;
        destory_iomem();
        return status;
    }

    status = runtime_->set_output_buffers(model_info_.output_num, model_info_.output_buffers.data());
    if (status != 0) {
        std::cerr << "Failed to set output buffers" << std::endl;
        destory_iomem();
        return status;
    }

    if (!model_info_.input_attributes.empty() && model_info_.input_attributes[0].dim_count >= 2) { // dim reverse 
        model_info_.model_input_height = model_info_.input_attributes[0].dim_size[1];
        model_info_.model_input_width = model_info_.input_attributes[0].dim_size[0];

    }

    model_loaded_ = true;
    return 0;
}

void ModelManager::destory_iomem() {
    if (!model_loaded_) {
        return;
    }

    for (auto& tensor : model_info_.input_tensors) {
        if (tensor.data) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }

    if (runtime_) {
        for (auto& buffer : model_info_.output_buffers) {
            if (buffer.data) {  
                ta_runtime_destroy_buffer(runtime_->get_context().get(), &buffer);
            }
        }
    }

    model_info_.input_tensors.clear();
    model_info_.output_buffers.clear();
    model_info_.input_attributes.clear();
    model_info_.output_attributes.clear();
    
    model_loaded_ = false;
}

taconn_input_t* ModelManager::get_input_tensor(int index) {
    if (index < 0 || index >= model_info_.input_tensors.size()) {
        return nullptr;
    }
    return &model_info_.input_tensors[index];
}

taconn_buffer_t* ModelManager::get_output_buffer(int index) {
    if (index < 0 || index >= model_info_.output_buffers.size()) {
        return nullptr;
    }
    return &model_info_.output_buffers[index];
}

int ModelManager::run_network() {
    if (!model_loaded_) {
        std::cerr << "Model not loaded" << std::endl;
        return -1;
    }
    return runtime_->run_inference();
}

int ModelManager::quantize_input_tensor(int index, const cv::Mat& preprocessed_data) {
    if (index < 0 || index >= model_info_.input_tensors.size()) {
        return -1;
    }
    
    auto* input_tensor = get_input_tensor(index);
    if (!input_tensor || !input_tensor->data) {
        return -1;
    }
    
    // 确保数据是连续的并转换为float
    cv::Mat float_data;
    preprocessed_data.convertTo(float_data, CV_32F);
    
    if (!float_data.isContinuous()) {
        float_data = float_data.clone();
    }
    
    // 计算元素数量
    size_t element_count = get_element_num(model_info_.input_attributes[index]);
    // 使用量化函数填充 input tensor
    QU::quantize(float_data.ptr<float>(), input_tensor->data, element_count, 
                model_info_.input_attributes[index]);
    
    return 0;
}

cv::Mat ModelManager::get_output_data(int index) {
    if (index < 0 || index >= model_info_.output_buffers.size()) {
        return cv::Mat();
    }
    
    auto* output_buffer = get_output_buffer(index);
    if (!output_buffer || !output_buffer->data) {
        return cv::Mat();
    }
    
    auto& out_attr = model_info_.output_attributes[index];
    size_t num_elements = get_element_num(out_attr);
    

    QU::DequantizeFunc dequant_func = nullptr;
    

    float scale = 1.0f;
    int32_t zp = 0;
    
    if (out_attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        scale = out_attr.quant_data.affine.tf_scale;
        zp = out_attr.quant_data.affine.tf_zero_point;
    }
    
    // 创建结果矩阵
    cv::Mat result(out_attr.dim_size[1], out_attr.dim_size[0], CV_32SC1); // H, W, int32 ,channel = 1
    
    // 根据数据类型决定返回的矩阵类型
    if (out_attr.data_format == taconn_data_format_t::TACONN_DATA_FORMAT_INT32) {
        // 创建 int32 类型的矩阵
        int32_t* dst_ptr = result.ptr<int32_t>();
        int32_t* src_ptr = static_cast<int32_t*>(output_buffer->data);
        
        // 直接复制数据，不做反量化
        if (out_attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE) {
            memcpy(dst_ptr, src_ptr, num_elements * sizeof(int32_t));
        } else {
            // 如果有量化，应用反量化公式并转换为int32
            for (size_t i = 0; i < num_elements; ++i) {
                float dequantized = (static_cast<float>(src_ptr[i]) - static_cast<float>(zp)) * scale;
                dst_ptr[i] = static_cast<int32_t>(std::round(dequantized));

            }
        }
        return result;
    } 
    else if (out_attr.data_format == taconn_data_format_t::TACONN_DATA_FORMAT_FP32) {
        // 创建 float 类型的矩阵
        cv::Mat result(1, static_cast<int>(num_elements), CV_32FC1);
        float* dst_ptr = result.ptr<float>();
        float* src_ptr = static_cast<float*>(output_buffer->data);
        
        // 直接复制数据
        memcpy(dst_ptr, src_ptr, num_elements * sizeof(float));
        return result;
    }
    else if (out_attr.data_format == taconn_data_format_t::TACONN_DATA_FORMAT_INT8) {
        // 创建 float 类型的矩阵（反量化到float）
        cv::Mat result(1, static_cast<int>(num_elements), CV_32FC1);
        float* dst_ptr = result.ptr<float>();
        int8_t* src_ptr = static_cast<int8_t*>(output_buffer->data);
        
        // 应用反量化
        for (size_t i = 0; i < num_elements; ++i) {
            dst_ptr[i] = (static_cast<float>(src_ptr[i]) - static_cast<float>(zp)) * scale;
        }
        return result;
    }
    else {
        // 默认情况：使用之前的反量化函数，返回float
        cv::Mat result(1, static_cast<int>(num_elements), CV_32FC1);
        float* dst_ptr = result.ptr<float>();
        
        // 选择反量化函数
        QuantizationUtils::DequantizeFunc dequant_func = 
            QuantizationUtils::get_dequantize_func(out_attr.data_format);
        
        // 反量化到float
        for (size_t i = 0; i < num_elements; ++i) {
            dst_ptr[i] = dequant_func(output_buffer->data, i, zp, scale);
        }
        return result;
    }

    return result;
}


size_t ModelManager::calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
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

size_t ModelManager::get_element_num(const taconn_inout_attr_t& attr) {
    size_t element_num = 1;
    for (int i = 0; i < attr.dim_count; i++) {
        element_num *= attr.dim_size[i];
    }
    return element_num;
}

// ==================== Segformer ====================

Segformer::Segformer() : init_flag(false), total_inferences_(0) {
    init_color_palette();
    runtime_ = std::make_shared<Runtime>();  
    model_manager_ = std::make_shared<ModelManager>(runtime_);
}

Segformer::Segformer(const SegformerConfig& config) : config_(config), init_flag(false), total_inferences_(0) {
    init_color_palette();
    runtime_ = std::make_shared<Runtime>();  
    model_manager_ = std::make_shared<ModelManager>(runtime_);
}

Segformer::~Segformer() {
    deinit();
}

int Segformer::init(const std::string& model_path) {  
    if (init_flag) {
        deinit();
    }
    
    config_.model_path = model_path;
    
    Runtime::Config runtime_config;
    runtime_config.core_id = config_.core_id;  // default 0
    runtime_config.memory_alignment = config_.memory_alignment; // default 256
    
    int status = model_manager_->init_runtime(model_path, runtime_config);
    if (status != 0) {
        std::cerr << "Failed to initialize Segformer model" << std::endl;
        return status;
    }
    
    auto& model_info = model_manager_->get_model_info();
    if (model_info.model_input_width > 0 && model_info.model_input_height > 0) {
        config_.target_size = cv::Size(model_info.model_input_width, model_info.model_input_height);
    }
    
    init_flag = true;
    return 0;
}

void Segformer::deinit() { 
    if (!init_flag) {
        return;
    }
    
    model_manager_->destory_iomem();
    init_flag = false;
    total_inferences_ = 0;
    inference_times_.clear();
    
    std::cout << "Segformer deinitialized" << std::endl;
}

cv::Mat Segformer::process(const cv::Mat& image) {
    if (!init_flag || image.empty()) {
        std::cerr << "Segformer not initialized or empty input image" << std::endl;
        return cv::Mat();
    }
    
    TimeStamp timer;
    timer.start();
    
    try {
        cv::Mat preprocessed = preprocess(image);
        
        if (!prepare_model_input(preprocessed)) {
            std::cerr << "Failed to prepare model input" << std::endl;
            return cv::Mat();
        }
        
        timer.stop();
        timer.time_accumulation("preprocess_time");
        
        timer.start();
        if (0 != model_manager_->run_network()) {
            std::cerr << "Inference failed" << std::endl;
            return cv::Mat();
        }
        
        timer.stop();
        timer.time_accumulation("inference_time");

        timer.start();
        cv::Mat result = postprocess_results();        
        timer.stop();
        timer.time_accumulation("postprocess_time");
        
        total_inferences_++;
        inference_times_.push_back(timer.time_map_lab["inference_time"]);
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during processing: " << e.what() << std::endl;
        return cv::Mat();
    }
}




cv::Mat Segformer::preprocess(const cv::Mat& image) {
    cv::Mat processed;
    cv::resize(image, processed, config_.target_size);
    
    if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    processed.convertTo(processed, CV_32F);
    
    if (config_.normalize) {
        cv::subtract(processed, config_.mean, processed);
        cv::divide(processed, config_.stddev, processed);
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(processed, blob);   // hwc - nchw
    return blob;
}

bool Segformer::prepare_model_input(const cv::Mat& preprocessed) {
    return model_manager_->quantize_input_tensor(0, preprocessed) == 0;  
}



cv::Mat Segformer::postprocess_results() {
    cv::Mat output_data = model_manager_->get_output_data(0);
    if (output_data.empty()) {
        return cv::Mat();
    }
    
    // 获取输出属性以确定形状
    auto& model_info = model_manager_->get_model_info();
    if (model_info.output_attributes.empty()) {

        return cv::Mat();
    }
    
    auto& out_attr = model_info.output_attributes[0];
    int height = output_data.rows;
    int width = output_data.cols;
    int channels = output_data.channels();
    
    return output_data;
}




cv::Size Segformer::get_input_size() const {
    if (!init_flag) {
        return config_.target_size;
    }
    
    auto& model_info = model_manager_->get_model_info();
    return cv::Size(model_info.model_input_width, model_info.model_input_height);
}

float Segformer::get_avg_inference_time() const {
    if (inference_times_.empty()) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (float time : inference_times_) {
        sum += time;
    }
    return sum / inference_times_.size();
}

void Segformer::init_color_palette() {
    color_palette_ = {
        cv::Vec3b(128, 64, 128),
        cv::Vec3b(244, 35, 232),
        cv::Vec3b(70, 70, 70),
        cv::Vec3b(102, 102, 156),
        cv::Vec3b(190, 153, 153),
        cv::Vec3b(153, 153, 153),
        cv::Vec3b(250, 170, 30),
        cv::Vec3b(220, 220, 0),
        cv::Vec3b(107, 142, 35),
        cv::Vec3b(152, 251, 152),
        cv::Vec3b(70, 130, 180),
        cv::Vec3b(220, 20, 60),
        cv::Vec3b(255, 0, 0),
        cv::Vec3b(0, 0, 142),
        cv::Vec3b(0, 0, 70),
        cv::Vec3b(0, 60, 100),
        cv::Vec3b(0, 80, 100),
        cv::Vec3b(0, 0, 230),
        cv::Vec3b(119, 11, 32)
    };
    
    while (color_palette_.size() < static_cast<size_t>(config_.num_classes)) {
        cv::RNG rng(static_cast<uint64_t>(color_palette_.size()));
        color_palette_.push_back(cv::Vec3b(
            rng.uniform(0, 256),
            rng.uniform(0, 256),
            rng.uniform(0, 256)
        ));
    }
}