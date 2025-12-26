#include "lprnet.hpp"

LPRNET::LPRNET(std::shared_ptr<ta_runtime_context> context,std::string model_file) : nnrt_context(context) ,model_name(model_file){
  
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "lprnet create nnrt_context" << std::endl;
  std::cout << "TACO Model: " << model_file << std::endl;
  std::cout << "Input size: " << DEFAULT_IMG_H << " x " << DEFAULT_IMG_W << std::endl;
  std::cout << "--------------------------------------" << std::endl;

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

size_t LPRNET::get_element_num(const taconn_inout_attr_t input_attr) {
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i) {
        size *= input_attr.dim_size[i];
    }
    return size;
}

size_t LPRNET::calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
    switch(format) {
        // 32 bits (4bytes)
        case TACONN_DATA_FORMAT_FP32:
        case TACONN_DATA_FORMAT_INT32:
        case TACONN_DATA_FORMAT_UINT32:
            return sizeof(uint32_t) * element_count;
        
        // 16 bits (2bytes)
        case TACONN_DATA_FORMAT_FP16:
        case TACONN_DATA_FORMAT_BFP16:
        case TACONN_DATA_FORMAT_INT16:
        case TACONN_DATA_FORMAT_UINT16:
            return sizeof(uint16_t) * element_count;
        
        // 8 bits (1bytes)
        case TACONN_DATA_FORMAT_UINT8:
        case TACONN_DATA_FORMAT_INT8:
        case TACONN_DATA_FORMAT_CHAR:
        case TACONN_DATA_FORMAT_BOOL8:
            return sizeof(uint8_t) * element_count;
        
        // 64 bits (8bytes)
        case TACONN_DATA_FORMAT_FP64:
        case TACONN_DATA_FORMAT_INT64:
        case TACONN_DATA_FORMAT_UINT64:
            return sizeof(uint64_t) * element_count;
        
        // 4 bits
        case TACONN_DATA_FORMAT_INT4:
        case TACONN_DATA_FORMAT_UINT4:
            return (element_count + 1) / 2; // round up to an integer
        
        default:
            std::cerr << "Unsupported data format: " << format << std::endl;
            return 0;
    }
}
bool LPRNET::Init() {
    uint32_t model_version = 0;
    taconn_input_output_num_t num;
    taconn_inout_attr_t input_attr;
    taconn_inout_attr_t output_attr;
    size_t input_buffer_size = 0;
    size_t output_buffer_size = 0;

    int status = ta_runtime_init();
    if (status != 0) {
        std::cerr << "Failed to initialize TA runtime: " << status << std::endl;
        return false;
    }
    
    
    status  = ta_runtime_load_model_from_file(nnrt_context.get(), model_name.c_str(), 0);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status<< std::dec<< std::endl;
        goto ERROR_CLEAN;
    }

    // print model's information
    ta_runtime_query(nnrt_context.get(), TACONN_QUERY_DRIVER_VERSION, &model_version);
    std::cout << "TaRuntime driver version: " << model_version << std::endl;
    ta_runtime_query(nnrt_context.get(), TACONN_QUERY_IN_OUT_NUM, &num);
    std::cout << "Input num is : " << num.input_num << ", Output num is : " << num.output_num << std::endl;
    input_num = num.input_num;
    output_num = num.output_num;
    std::cout << "Input:" << std::endl;
    for (int i=0; i<input_num; i++) {
        input_attr.index = i;
        ta_runtime_query(nnrt_context.get(), TACONN_QUERY_INPUT_ATTR, &input_attr);
        print_taconn_inout_attr(input_attr);
        ins_attr.push_back(input_attr);
    }
    std::cout << "Output:" << std::endl;
    for (int i=0; i<output_num; i++) {
        output_attr.index = i;
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_OUTPUT_ATTR, &output_attr);
        print_taconn_inout_attr(output_attr);
        outs_attr.push_back(output_attr);
    }

    input_tensors = new taconn_input_t[input_num];
    if (!input_tensors) {
        std::cerr << "Allocate input_tensors failed" << std::endl;
        goto ERROR_CLEAN;
    }
    for (int i=0; i<input_num; i++){  
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ins_attr[i].data_format);
        input_buffer_size = calculate_buffer_size(data_type, get_element_num(ins_attr[i]));
        input_tensors[i].index = i;
        input_tensors[i].size = input_buffer_size;  // size in bytes

        input_tensors[i].data = nullptr;  // will be set later
        if (posix_memalign((void**)&input_tensors[i].data, 256, input_buffer_size)!=0){
            std::cerr << "Failed to allocate input buffer" << std::endl;
            goto ERROR_CLEAN;
        }
        memset(input_tensors[i].data, 0, input_buffer_size);

    }
    status = ta_runtime_set_input_cva(nnrt_context.get(), input_num, input_tensors);
    if (status != 0) {
        std::cerr << "Set input failed: 0x" << std::hex << status << std::dec << std::endl;
        goto ERROR_CLEAN;
    }


    // 6. set output
    output_buffer = new taconn_buffer_t[output_num];
    if (!output_buffer) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto ERROR_CLEAN;
    }
    memset(output_buffer, 0, sizeof(taconn_buffer_t) * output_num);
    for (int i = 0; i < output_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(outs_attr[i].data_format);
        output_buffer_size = calculate_buffer_size(data_type, get_element_num(outs_attr[i]));

        status = ta_runtime_create_buffer(nnrt_context.get(), output_buffer_size, &output_buffer[i]);
        if (status != 0) {
            
            std::cerr << "Create output buffer "<< i << "failed: 0x%x" << std::hex << status << std::dec << std::endl;
            goto ERROR_CLEAN;
        }
    }
    status = ta_runtime_set_output(nnrt_context.get(), output_num, output_buffer);
    if (status != 0) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto ERROR_CLEAN;
    }
    return true;

ERROR_CLEAN:
    // free
    if (input_tensors) {
        for(int i = 0; i < input_num; i++){
            if(input_tensors[i].data)
                free(input_tensors[i].data);
        }
        delete[] input_tensors;
        input_tensors = nullptr;
    }

    if (output_buffer) {
        for (int i = 0; i < output_num; i++) {
            ta_runtime_destroy_buffer(nnrt_context.get(), &output_buffer[i]);
        }
        delete[] output_buffer;
        output_buffer = nullptr;
    }

    ta_runtime_destroy_context(nnrt_context.get());
    ta_runtime_deinit();
    return false;
}

static uint16_t f32_to_f16(float value) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x7fffff;

    // 处理0
    if (exp == -127 + 15 && mantissa == 0) {
        return sign;
    }

    // 处理NaN和无穷大
    if (exp == 0xff - 127 + 15) {
        if (mantissa) {
            // NaN - 保留部分尾数
            return sign | 0x7c00 | (mantissa >> 13);
        } else {
            // 无穷大
            return sign | 0x7c00;
        }
    }

    // 处理下溢（非规格化数）
    if (exp <= 0) {
        if (exp < -10) {
            // 太小，趋于0
            return sign;
        }
        // 转换为非规格化数
        mantissa = (mantissa | 0x800000) >> (1 - exp);
        // 四舍五入
        mantissa += 0xfff + ((mantissa >> 13) & 1);
        return sign | (mantissa >> 13);
    }

    // 规格化数
    // 四舍五入到最近偶数
    mantissa += 0xfff + ((mantissa >> 13) & 1);
    
    // 处理尾数进位
    if (mantissa & 0x1000000) {
        mantissa >>= 1;
        exp++;
    }
    
    // 检查指数上溢
    if (exp >= 0x1f) {
        return sign | 0x7c00;  // 无穷大
    }

    return sign | (exp << 10) | ((mantissa >> 13) & 0x3ff);
}

int LPRNET::preprocess_image(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat img;
    cv::resize(src, img, cv::Size(DEFAULT_IMG_W, DEFAULT_IMG_H), 0, 0, cv::INTER_LINEAR);

    // no need to convert BGR to RGB for LPRNet
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  
    img.convertTo(img, CV_32FC3);
    img -= 127.5;
    img *= 0.0078125; // 1/128
    
    dst = img.reshape(1, DEFAULT_IMG_H * DEFAULT_IMG_W).t(); 
    dst = dst.reshape(3, DEFAULT_IMG_H);
    return 0;
}

int LPRNET::pre_process(cv::Mat &images) {

  for (int i = 0; i < input_num; i++) {
        size_t num_elements = get_element_num(ins_attr[i]);
        taconn_inout_attr_t attr = ins_attr[i];
        if(attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8){
            cv::Mat temp1;
            cv::resize(images, temp1, cv::Size(DEFAULT_IMG_W, DEFAULT_IMG_H), 0, 0, cv::INTER_LINEAR);
        
            // cv::cvtColor(temp1, temp1, cv::COLOR_BGR2RGB);
            memcpy(input_tensors[i].data, temp1.data, num_elements * sizeof(uint8_t));
            continue;
        }
        cv::Mat input_image;
        cv::resize(images, input_image, cv::Size(DEFAULT_IMG_W, DEFAULT_IMG_H), 0, 0, cv::INTER_LINEAR);
        ts_->start();
        preprocess_image(images, input_image);
        ts_->time_accumulation("pre_time");
        if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
            float* dst_fp32 = static_cast<float*>(input_tensors[i].data);
            memcpy(dst_fp32, input_image.data, num_elements * sizeof(float));
        } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
            uint16_t* dst_fp16 = static_cast<uint16_t*>(input_tensors[i].data);
            float* src_fp32 = (float*)input_image.data;
            for (size_t j = 0; j < num_elements; j++) {
                float val = src_fp32[j];
                val = std::max(-65504.0f, std::min(65504.0f, val));
                dst_fp16[j] = f32_to_f16(val);
            }
        } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
            float scale = attr.quant_data.affine.tf_scale;
            int32_t zero_point = attr.quant_data.affine.tf_zero_point;
            uint8_t* dst_uint8 = static_cast<uint8_t*>(input_tensors[i].data);
            for (size_t j = 0; j < num_elements; j++) {
                float quantized = ((float*)input_image.data)[j] / scale + zero_point;
                dst_uint8[j] = static_cast<uint8_t>(std::round((std::max(0.0f, std::min(255.0f, quantized)))));
            }
        } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            float scale = attr.quant_data.affine.tf_scale;
            int32_t zero_point = attr.quant_data.affine.tf_zero_point;
            int8_t* dst_int8 = static_cast<int8_t*>(input_tensors[i].data);
            for (size_t j = 0; j < num_elements; j++) {
                float quantized = ((float*)input_image.data)[j] / scale + zero_point;
                dst_int8[j] = static_cast<int8_t>(std::round((std::max(-128.0f, std::min(127.0f, quantized)))));
            }

        } else {
            std::cerr << "preprocess Not Support Quantize Type Model!" << std::endl;
            return -1;
        }
    }

  return 0;
}

static float f16_to_f32(uint16_t fp16_data) {
    uint16_t h = fp16_data;
    uint32_t x = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
    return *((float*)&x);
}

int LPRNET::post_process(std::vector<cv::Mat> &images, std::vector<std::string> &results){
    
    taconn_inout_attr_t attr = outs_attr[0];
    int out_number = get_element_num(attr);
    std::vector<float> dequant;

    //1.f16 to f32
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
        for (int i = 0; i < out_number; i++) {
            dequant.push_back(((float*)output_buffer[0].data)[i]);
        }
    } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
        uint16_t* output_data = (uint16_t*)output_buffer[0].data;
        for (int i = 0; i < out_number; i++) {
            float a = f16_to_f32(output_data[i]);
            dequant.push_back(a);
        }
    } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8) {
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        uint8_t* output_data = (uint8_t*)output_buffer[0].data;
        for (int i = 0; i < out_number; i++) {
            float a = (static_cast<float>(output_data[i]) - zero_point) * scale;
            dequant.push_back(a);
        }
    } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        int8_t* output_data = (int8_t*)output_buffer[0].data;
        for (int i = 0; i < out_number; i++) {
            float a = (static_cast<float>(output_data[i]) - zero_point) * scale;
            dequant.push_back(a);
        }
    } else {
        std::cerr << "postprocess Not Support Quantize Type Model!" << std::endl;
        return -1;
    }

    const int blank_index = LPRNET_NUM_CLASSES - 1; // blank character index
    //Get argmax, batch_size = 1
    std::vector<int> argmax_indices(LPRNET_TIME_STEPS, 0);
    for(int t=0; t < LPRNET_TIME_STEPS; t++){
        float max_value = dequant[t];
        int max_index = 0;
        for(int c = 1; c < LPRNET_NUM_CLASSES; c++){
            float value = dequant[c * LPRNET_TIME_STEPS + t];
            if(value > max_value){
                max_value = value;
                max_index = c;
            }
        }
        argmax_indices[t] = max_index;
    }

    std::string decoded_str = "";
    int pre_c = argmax_indices[0];
    if(pre_c != blank_index){
        decoded_str += CHARS[pre_c];
    }
    for(int t = 0; t < LPRNET_TIME_STEPS; t++){
        int c = argmax_indices[t];
        if(c != blank_index && c != pre_c){
            decoded_str += CHARS[c];
        }
        pre_c = c;
    }
    if (decoded_str.empty()) {
        decoded_str = "Empty"; // represent no plate detected
    }
    results.push_back(decoded_str);

    return 0;
}

int LPRNET::deInit(){
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
        delete[] input_tensors;
        input_tensors = nullptr;
    }

    // free output ctx array
    if (output_buffer) {
        delete[] output_buffer;
        output_buffer = nullptr;
    }
    
    // free ctx
    ta_runtime_destroy_context(nnrt_context.get());
    
    // free runtime
    ta_runtime_deinit();
    return 0;
}

int LPRNET::Classify(std::vector<cv::Mat>& input_images, std::vector<std::string> &results){
    
    int ret = 0;
    for(int i = 0; i < input_images.size(); i++){
        ret = pre_process(input_images[i]);
        
        ts_->start();
        ret = ta_runtime_run_network(nnrt_context.get());
        if (ret != 0) {
            std::cerr << "Run network failed." << std::endl;
            return -1;
        }
        
        // 3. invalidate_output
        ret = ta_runtime_invalidate_buffer(nnrt_context.get(),output_buffer);
        if (ret != 0) {
            std::cerr << "Invalidate output buffer failed." << std::endl;
            return -1;
        }
        ts_->time_accumulation("infer_time");

        ts_->start();
        ret = post_process(input_images,results);
        ts_->time_accumulation("post_time");
        
    }
    

    return ret;
}

void LPRNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}