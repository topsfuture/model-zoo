#include "insightface.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>




Runtime::Runtime() : nnrt_ctx_(nullptr), rt_flag(false) {}
Runtime::~Runtime() { cleanup(); }

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
    int status = ta_runtime_load_model_from_file(nnrt_ctx_.get(), model_path.c_str(), config_.core_id);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status << std::dec << std::endl;
    }
    return status;
}

int Runtime::query_io_count(int& input_num, int& output_num) {
    if (!rt_flag) return -1;
    taconn_input_output_num_t num = {};
    int status = ta_runtime_query(nnrt_ctx_.get(), TACONN_QUERY_IN_OUT_NUM, &num);
    if (status == 0) {
        input_num = num.input_num;
        output_num = num.output_num;
    }
    return status;
}

int Runtime::get_input_attribute(int index, taconn_inout_attr_t& attr) {
    if (!rt_flag) return -1;
    attr.index = index;
    return ta_runtime_query(nnrt_ctx_.get(), TACONN_QUERY_INPUT_ATTR, &attr);
}

int Runtime::get_output_attribute(int index, taconn_inout_attr_t& attr) {
    if (!rt_flag) return -1;
    attr.index = index;
    return ta_runtime_query(nnrt_ctx_.get(), TACONN_QUERY_OUTPUT_ATTR, &attr);
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
    return ta_runtime_run_network(nnrt_ctx_.get());
}

void Runtime::cleanup() {
    if (rt_flag) {
        ta_runtime_deinit();
        rt_flag = false;
    }
}

// ==================== QuantizationUtils 实现 ====================
float QuantizationUtils::dequantize_float32(void* data, size_t idx, int32_t, float) {
    return static_cast<float*>(data)[idx];
}

float QuantizationUtils::dequantize_float16(void* data, size_t idx, int32_t, float) {
    uint16_t h = static_cast<uint16_t*>(data)[idx];
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
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

float QuantizationUtils::dequantize_uint8(void* data, size_t idx, int32_t zp, float scale) {
    uint8_t val = static_cast<uint8_t*>(data)[idx];
    return ((float)val - (float)zp) * scale;
}

float QuantizationUtils::dequantize_int8(void* data, size_t idx, int32_t zp, float scale) {
    int8_t val = static_cast<int8_t*>(data)[idx];
    return ((float)val - (float)zp) * scale;
}

float QuantizationUtils::dequantize_int32(void* data, size_t idx, int32_t zp, float scale) {
    int32_t val = static_cast<int32_t*>(data)[idx];
    return (static_cast<float>(val) - static_cast<float>(zp)) * scale;
}

QuantizationUtils::DequantizeFunc QuantizationUtils::get_dequantize_func(uint32_t data_format) {
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

uint16_t QuantizationUtils::f32_to_f16(float value) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint16_t h = ((x >> 16) & 0x8000) | 
                ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
                ((x >> 13) & 0x03ff);
    return h;
}

void QuantizationUtils::quantize(float* src, void* dst, size_t num_elements, 
                                const taconn_inout_attr_t& attr) {
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE) {
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
            float* dst_float = static_cast<float*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                dst_float[i] = src[i];
            }
        } 
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
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
        int fixed_point_pos = attr.quant_data.dfp.fixed_point_pos;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                int32_t scaled = lrintf(src[i] * (1 << fixed_point_pos));
                dst_int8[i] = static_cast<int8_t>(std::clamp(scaled, -128, 127));
            }
        }else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT16) {
            int16_t* dst_int16 = static_cast<int16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                int32_t val = static_cast<int32_t>(std::round(src[i] * (1 << fixed_point_pos)));
                dst_int16[i] = static_cast<int16_t>(std::clamp(val, -32768, 32767));
            }
        }
    }
}

// ==================== ModelManager 实现 ====================
ModelManager::ModelManager(std::shared_ptr<Runtime> runtime) 
    : runtime_(runtime), model_loaded_(false) {
    if (!runtime_) {
        runtime_ = std::make_shared<Runtime>();
    }
}

ModelManager::~ModelManager() {
    destroy_iomem();
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
        destroy_iomem();
    }
    
    int status = runtime_->init_ctx(runtime_config);
    if (status != 0) {
        std::cerr << "Failed to initialize hardware runtime" << std::endl;
        return status;
    }
    
    status = runtime_->load_model(model_path);
    if (status != 0) {
        std::cerr << "Failed to load model" << std::endl;
        return status;
    }

    status = runtime_->query_io_count(model_info_.input_num, model_info_.output_num);
    if (status != 0 || model_info_.input_num <= 0 || model_info_.output_num <= 0) {
        std::cerr << "Invalid input/output numbers" << std::endl;
        return -1;
    }
    
    model_info_.input_attributes.resize(model_info_.input_num);
    model_info_.input_tensors.resize(model_info_.input_num);

    for (int i = 0; i < model_info_.input_num; i++) {
        status = runtime_->get_input_attribute(i, model_info_.input_attributes[i]);
        if (status != 0) {
            std::cerr << "Failed to get input attribute " << i << std::endl;
            destroy_iomem();
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
            destroy_iomem();
            return status;
        }
    }
    
    model_info_.output_attributes.resize(model_info_.output_num);
    model_info_.output_buffers.resize(model_info_.output_num);

    for (int i = 0; i < model_info_.output_num; i++) {
        status = runtime_->get_output_attribute(i, model_info_.output_attributes[i]);
        if (status != 0) {
            std::cerr << "Failed to get output attribute " << i << std::endl;
            destroy_iomem();
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
            destroy_iomem();
            return status;
        }
    }

    status = runtime_->set_input_tensors(model_info_.input_num, model_info_.input_tensors.data());
    if (status != 0) {
        std::cerr << "Failed to set input tensors" << std::endl;
        destroy_iomem();
        return status;
    }

    status = runtime_->set_output_buffers(model_info_.output_num, model_info_.output_buffers.data());
    if (status != 0) {
        std::cerr << "Failed to set output buffers" << std::endl;
        destroy_iomem();
        return status;
    }

    if (!model_info_.input_attributes.empty() && model_info_.input_attributes[0].dim_count >= 2) {
        model_info_.model_input_height = model_info_.input_attributes[0].dim_size[1];
        model_info_.model_input_width = model_info_.input_attributes[0].dim_size[0];
    }

    model_loaded_ = true;
    return 0;
}

void ModelManager::destroy_iomem() {
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
        std::cerr << "  Quantization - Error: Invalid input tensor index: " << index << std::endl;
        return -1;
    }
    
    auto* input_tensor = get_input_tensor(index);
    if (!input_tensor || !input_tensor->data) {
        std::cerr << "  Quantization - Error: Input tensor is null or data is null" << std::endl;
        return -1;
    }
    
    cv::Mat float_data;
    
    if (preprocessed_data.type() != CV_32F) {
        preprocessed_data.convertTo(float_data, CV_32F);
    } else {
        float_data = preprocessed_data;
    }
    
    if (!float_data.isContinuous()) {
        float_data = float_data.clone();
    }
    
    size_t element_count = get_element_num(model_info_.input_attributes[index]);
    
    if (float_data.total() != element_count) {
        std::cerr << "  Quantization - Error: Element count mismatch!" << std::endl;
        std::cerr << "    Blob has " << float_data.total() << " elements" << std::endl;
        std::cerr << "    Model expects " << element_count << " elements" << std::endl;
        return -1;
    }
    
    try {
        QuantizationUtils::quantize(float_data.ptr<float>(), input_tensor->data, element_count, 
                                   model_info_.input_attributes[index]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "  Quantization - Exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "  Quantization - Unknown error" << std::endl;
        return -1;
    }
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
    
    float scale = 1.0f;
    int32_t zp = 0;
    
    if (out_attr.quant_format == taco_qnt_type_t::TACONN_QNT_TYPE_ASYMMETRIC) {
        scale = out_attr.quant_data.affine.tf_scale;
        zp = out_attr.quant_data.affine.tf_zero_point;
    }
    
    cv::Mat result(1, static_cast<int>(num_elements), CV_32FC1);
    float* dst_ptr = result.ptr<float>();
    
    QuantizationUtils::DequantizeFunc dequant_func = 
        QuantizationUtils::get_dequantize_func(out_attr.data_format);
    
    for (size_t i = 0; i < num_elements; ++i) {
        dst_ptr[i] = dequant_func(output_buffer->data, i, zp, scale);
    }
    
    return result;
}

size_t ModelManager::calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
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

size_t ModelManager::get_element_num(const taconn_inout_attr_t& attr) {
    size_t element_num = 1;
    for (int i = 0; i < attr.dim_count; i++) {
        element_num *= attr.dim_size[i];
    }
    return element_num;
}