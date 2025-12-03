#include "resnet.hpp"

RESNET::RESNET(std::shared_ptr<ta_runtime_context> context,std::string model_file) : nnrt_context(context) ,model_name(model_file){
  
  input_tensors = nullptr;
  output_buffer = nullptr;
  input_num = 0;
  output_num = 0;
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "ResNet create nnrt_context" << std::endl;
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

size_t RESNET::get_element_num(const taconn_inout_attr_t input_attr) {
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i) {
        size *= input_attr.dim_size[i];
    }
    return size;
}

size_t RESNET::calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
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
int RESNET::Init() {
    
    int status = ta_runtime_init();
    if (status != 0) {
        std::cerr << "Failed to initialize TA runtime: " << status << std::endl;
        return false;
    }
    
    
    status  = ta_runtime_load_model_from_file(nnrt_context.get(), model_name.c_str(), 0);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status<< std::dec<< std::endl;
        return false;
    }

   // 4. print model's information
    // uint32_t model_version = 0;
    // ta_runtime_query(nnrt_context.get(), TACONN_QUERY_DRIVER_VERSION, &model_version);
    // std::cout << "TaRuntime driver version: " << model_version << std::endl;
    taconn_input_output_num_t num;
    ta_runtime_query(nnrt_context.get(), TACONN_QUERY_IN_OUT_NUM, &num);
    std::cout << "Input num is : " << num.input_num << ", Output num is : " << num.output_num << std::endl;
    input_num = num.input_num;
    output_num = num.output_num;
    std::cout << "Input:" << std::endl;
    taconn_inout_attr_t input_attr;
    for (int i=0; i<input_num; i++) {
        input_attr.index = i;
        ta_runtime_query(nnrt_context.get(), TACONN_QUERY_INPUT_ATTR, &input_attr);
        print_taconn_inout_attr(input_attr);
        ins_attr.push_back(input_attr);
    }
    std::cout << "Output:" << std::endl;
    taconn_inout_attr_t output_attr;
    for (int i=0; i<output_num; i++) {
        output_attr.index = i; 
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_OUTPUT_ATTR, &output_attr);
        print_taconn_inout_attr(output_attr);
        outs_attr.push_back(output_attr);
    }

    input_tensors = new taconn_input_t[input_num];
    if (!input_tensors) {
        std::cerr << "Allocate input_tensors failed" << std::endl;
        return -1;
    }
    size_t input_buffer_size = 0;
    for (int i=0; i<input_num; i++){  
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ins_attr[i].data_format);
        input_buffer_size = calculate_buffer_size(data_type, get_element_num(ins_attr[i]));
        input_tensors[i].index = i;
        input_tensors[i].size = input_buffer_size;  // size in bytes

        input_tensors[i].data = nullptr;  // will be set later
        if (posix_memalign((void**)&input_tensors[i].data, 256, input_buffer_size)!=0){
            std::cerr << "Failed to allocate input buffer" << std::endl;
            delete[] input_tensors;
            return false;
        }
        memset(input_tensors[i].data, 0, input_buffer_size);

    }
    status = ta_runtime_set_input_cva(nnrt_context.get(), input_num, input_tensors);
    if (status != 0) {
        std::cerr << "Set input failed: 0x" << std::hex << status << std::dec << std::endl;
        delete[] input_tensors;
        return false;
    }


    // 6. set output
    output_buffer = new taconn_buffer_t[output_num];
    if (!output_buffer) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        delete[] input_tensors;
        return false;
    }
    size_t output_buffer_size = 0;
    for (int i = 0; i < output_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(outs_attr[i].data_format);
        output_buffer_size = calculate_buffer_size(data_type, get_element_num(outs_attr[i]));

        status = ta_runtime_create_buffer(nnrt_context.get(), output_buffer_size, &output_buffer[i]);
        if (status != 0) {
            
            std::cerr << "Create output buffer "<< i << "failed: 0x%x" << std::hex << status << std::dec << std::endl;
            delete[] input_tensors;
            delete[] output_buffer;
            return false;
        }
    }
    status = ta_runtime_set_output(nnrt_context.get(), output_num, output_buffer);
    if (status != 0) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        delete[] input_tensors;
        delete[] output_buffer;
        return false;
    }
  return 0;
}

static uint16_t f32_to_f16(float value) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint16_t h = ((x >> 16) & 0x8000) | 
                ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
                ((x >> 13) & 0x03ff);
    return h;
}

int RESNET::preprocess_image(const cv::Mat& src, cv::Mat& dst) {
    //1.Resize

    cv::Mat temp1;
    cv::resize(src, temp1, cv::Size(DEFAULT_IMG_W, DEFAULT_IMG_H), 0, 0, cv::INTER_LINEAR);


    //2.BGR to RGB
    cv::cvtColor(temp1, temp1, cv::COLOR_BGR2RGB);


    //3.Rescaled and normalized
    cv::Mat temp2;
    temp1.convertTo(temp2, CV_32FC3,1.0/255.0);

    cv::Mat channels[3];
    cv::split(temp2, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;  // R
    channels[1] = (channels[1] - 0.456) / 0.224;  // G
    channels[2] = (channels[2] - 0.406) / 0.225;  // B
    cv::Mat normalizedImage;
    cv::merge(channels, 3, normalizedImage);


    //4.HWC to CHW
    
    cv::Mat chwImage = normalizedImage.reshape(1, 224*224).t();
    
    dst = chwImage.reshape(3, 224);

    return 0;
    
}

int RESNET::pre_process(cv::Mat &images) {


  for (int i = 0; i < input_num; i++) {
        size_t num_elements = get_element_num(ins_attr[i]);
        taconn_inout_attr_t attr = ins_attr[i];
        cv::Mat input_image;
        ts_->start();
        preprocess_image(images, input_image);
        ts_->time_accumulation("pre_time");
        if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32) {
            float* dst_fp32 = static_cast<float*>(input_tensors[i].data);
            memcpy(dst_fp32, input_image.data, num_elements * sizeof(float));
        } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16) {
            uint16_t* dst_fp16 = static_cast<uint16_t*>(input_tensors[i].data);
            for (size_t j = 0; j < num_elements; j++) {
                dst_fp16[j] = f32_to_f16(((float*)input_image.data)[j]);
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
            int8_t* dst_uint8 = static_cast<int8_t*>(input_tensors[i].data);
            for (size_t j = 0; j < num_elements; j++) {
                float quantized = ((float*)input_image.data)[j] / scale + zero_point;
                dst_uint8[j] = static_cast<int8_t>(std::round((std::max(-128.0f, std::min(127.0f, quantized)))));
            }

        }else {
            std::cerr << "Can only infer float16 model for now!" << std::endl;
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
int RESNET::post_process(std::vector<cv::Mat> &images, std::vector<std::pair<int, float>> &results){
    
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
    }else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC && attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        int8_t* output_data = (int8_t*)output_buffer[0].data;
        for (int i = 0; i < out_number; i++) {
            float a = (static_cast<float>(output_data[i]) - zero_point) * scale;
            dequant.push_back(a);
        }
    }

    //2. softmax
    std::vector<float> probs;
    float sum_exp = 0.0f;
    for (int i = 0; i < 1000; i++) {
        dequant[i] = std::exp(dequant[i]);
        sum_exp += dequant[i];
    }
    for (size_t i = 0; i < 1000; ++i) {
        probs.push_back(dequant[i] / sum_exp);
    }

    //3. Top-k
    std::vector<std::pair<int, float>> top_k;
    for (size_t i = 0; i < out_number; ++i) {
        top_k.emplace_back(i, probs[i]);
    }
    std::sort(top_k.begin(), top_k.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );

    
    results.push_back(top_k[0]);
 
    return 0;
}

int RESNET::deInit(){
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
    if(nnrt_context.get())
        ta_runtime_destroy_context(nnrt_context.get());
    
    // free runtime
    ta_runtime_deinit();
    return 0;
}

int RESNET::Classify(std::vector<cv::Mat>& input_images, std::vector<std::pair<int, float>> &results){
    
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

void RESNET::enableProfile(TimeStamp *ts) {
  ts_ = ts;
}