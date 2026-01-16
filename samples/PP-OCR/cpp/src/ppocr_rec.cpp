#include "ppocr_rec.hpp"

static size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
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
static void print_taconn_inout_attr(const taconn_inout_attr_t& attr) {
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

std::vector<std::string> PPOCR_Rec::ReadDict(std::string &path){
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  m_vec.insert(m_vec.begin(),"#"); // blank char for ctc
  m_vec.push_back(" ");
  return m_vec;
}
PPOCR_Rec::PPOCR_Rec(std::shared_ptr<ta_runtime_context> context,std::string m_path,std::string label_path){
    nnrt_context = context;
    model_path = m_path;
    initialized_ = false;
    input_num = 0;
    output_num = 0;
    input_tensors = nullptr;
    output_buffer = nullptr;
    char_charts = ReadDict(label_path);
}
PPOCR_Rec::~PPOCR_Rec(){
    if(initialized_){
        deinit();
    }
}
bool PPOCR_Rec::Init(){
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


    // 3. 加载模型
    status = ta_runtime_load_model_from_file(nnrt_context.get(), model_path.c_str(), 0);
    if (status != 0) {
        std::cerr << "Load model from file failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    // 4. 查询输入输出数量
    status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_IN_OUT_NUM, &num);
    std::cout << "Input num: " << num.input_num << ", Output num: " << num.output_num << std::endl;
    input_num = num.input_num;
    output_num = num.output_num;

    // 5. 查询输入属性
    for (int i = 0; i < input_num; i++) {
        taconn_inout_attr_t input_attr = {0};
        input_attr.index = i;
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_INPUT_ATTR, &input_attr);
        print_taconn_inout_attr(input_attr);
        ins_attr.push_back(input_attr);
    }
    
    //只对yolo11模型有效
    if (input_num > 0 && ins_attr[0].dim_count >= 3) {
        input_height = 48;
        input_width = 320;
        printf("input_height:%d, input_width:%d\n", input_height, input_width);
    }

    
    // 6. 查询输出属性
    for (int i = 0; i < output_num; i++) {
        taconn_inout_attr_t output_attr = {0};
        output_attr.index = i;
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_OUTPUT_ATTR, &output_attr);
        print_taconn_inout_attr(output_attr);
        outs_attr.push_back(output_attr);
    }

    // 7. 设置输入
    input_tensors = (taconn_input_t*)malloc(sizeof(taconn_input_t) * input_num);
    if (!input_tensors) {
        std::cerr << "Allocate input_tensors failed" << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < input_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ins_attr[i].data_format);
        size_t input_buffer_size = calculate_buffer_size(data_type, get_element_num(ins_attr[i]));

        input_tensors[i].index = i;
        input_tensors[i].size = input_buffer_size;
        input_tensors[i].data = nullptr;
        
        if (posix_memalign((void**)&input_tensors[i].data, 256, input_buffer_size) != 0) {
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

    // 8. 设置输出
    output_buffer = (taconn_buffer_t*)malloc(sizeof(taconn_buffer_t) * output_num);
    if (!output_buffer) {
        std::cerr << "Allocate output_buffer failed" << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < output_num; i++) {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(outs_attr[i].data_format);
        size_t output_buffer_size = calculate_buffer_size(data_type, get_element_num(outs_attr[i]));
        
        status = ta_runtime_create_buffer(nnrt_context.get(), output_buffer_size, &output_buffer[i]);
        if (status != 0) {
            std::cerr << "Create output buffer " << i << " failed: 0x" << std::hex << status << std::dec << std::endl;
            goto CLEANUP;
        }
    }

    status = ta_runtime_set_output(nnrt_context.get(), output_num, output_buffer);
    if (status != 0) {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    initialized_ = true;
    std::cout << "Model initialized successfully" << std::endl;
    return true;
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
    if (nnrt_context) {
        ta_runtime_destroy_context(nnrt_context.get());
        nnrt_context = 0;
    }
    ta_runtime_deinit();
    initialized_ = false;
    std::cout << "Model deinitialized" << std::endl;
    return false;
}
void PPOCR_Rec::deinit(){
    if (!initialized_) {
        return;
    }

    // 释放输出buffer
    for (int i = 0; i < output_num; i++) {
        ta_runtime_destroy_buffer(nnrt_context.get(), &output_buffer[i]);
    }

    // 释放输入buffer
    if (input_tensors) {
        for (int i = 0; i < input_num; i++) {
            if (input_tensors[i].data) {
                free(input_tensors[i].data);
            }
        }
        free(input_tensors);
        input_tensors = nullptr;
    }

    // 释放输出数组
    if (output_buffer) {
        free(output_buffer);
        output_buffer = nullptr;
    }

    // 销毁上下文
    if(nnrt_context.get())
        ta_runtime_destroy_context(nnrt_context.get());
    
    // 释放runtime
    ta_runtime_deinit();
    
    initialized_ = false;
    std::cout << "Model deinitialized" << std::endl;
}

static uint16_t float32_to_float16(float value) {
    // 此函数实现将 float32 转换为 float16 格式
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (bits >> 31) & 0x1;
    int exponent = (bits >> 23) & 0xFF;
    uint32_t fraction = bits & 0x7FFFFF;
    
    // 处理特殊情况（0，无穷大，NaN）
    if (exponent == 0 && fraction == 0) {
        return sign << 15;  // 正零或负零
    }
    if (exponent == 0xFF) {
        if (fraction == 0) {
            return (sign << 15) | 0x7C00;  // 无穷大
        } else {
            return (sign << 15) | 0x7E00;  // NaN
        }
    }
    
    // 调整指数（float32 偏移 127 -> float16 偏移 15）
    exponent -= 127;
    
    // 处理下溢情况
    if (exponent < -14) {
        fraction = (0x800000 + fraction) >> (13 - exponent - 14);
        fraction |= (fraction >> 13) & 1;  // 四舍五入
        return (sign << 15) | fraction;
    }
    
    // 处理上溢情况
    if (exponent > 15) {
        return (sign << 15) | 0x7C00;  // 上溢到无穷大
    }
    
    // 正常情况下的转换
    exponent += 15;
    fraction >>= 13;
    
    // 四舍五入
    if (fraction & 0x1000) {
        fraction += 1;
        if (fraction & 0x8000) {
            fraction >>= 1;
            exponent += 1;
        }
    }
    
    return (sign << 15) | (exponent << 10) | (fraction & 0x3FF);
}
static void quantize(float* src, void* dst, size_t num_elements, 
                                   const taconn_inout_attr_t& attr) {
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
            int16_t* dst_fp16 = static_cast<int16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                // float normalized = src[i] / 255.0f;
                dst_fp16[i] = float32_to_float16(src[i]);
            }
        }
        /* pre_proc node */
        else {
            std::cout << "not support quant_format: " << attr.quant_format << std::endl;
        }
    } 

    else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
        // 对称量化 (int8, uint8)
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {
            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                // float normalized = src[i] / 255.0f;
                int32_t quantized = lrintf(src[i] / scale) + zero_point;
                dst_int8[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
            }
        }else {
            std::cout << "not support quant_format: " << attr.quant_format << std::endl;
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
        }
    }
}

static void mat_to_tensor(const cv::Mat& mat, u_int8_t* tensor) {
    /* work as preprocess_node_layer */
   
    const int height = mat.rows;
    const int width = mat.cols;
    const int channels = mat.channels();
  
    const int total_pixels = height * width;

    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);

    // 按 CHW 顺序拷贝每个通道
    for (int c = 0; c < channels; ++c) {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(u_int8_t));
    }
            
}

static void mat_to_tensor(const cv::Mat& mat, float* tensor) {
    /* work as preprocess_node_layer */
 
    const int height = mat.rows;
    const int width = mat.cols;
    const int channels = mat.channels();
  
    const int total_pixels = height * width;

    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);

    // 按 CHW 顺序拷贝每个通道
    for (int c = 0; c < channels; ++c) {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(float));
    }
            
}
void PPOCR_Rec::preprocess(cv::Mat &input,cv::Mat &output_image){
    int src_h = input.rows;
    int src_w = input.cols;
    float ratio = static_cast<float>(src_w) / src_h;
    img_ratio = input_width / (float)input_height;
    int padding_w = 0;
    if(ratio > img_ratio){
        resized_w = input_width;
        resized_h = input_height;
    }else{
        resized_h = input_height;
        resized_w = (int)(input_height * ratio);
        padding_w = (int)(input_height * img_ratio);
    }

    cv::Mat resized;
    if (src_h != resized_h || src_w != resized_w) {
        
        cv::resize(input, resized, cv::Size(resized_w, resized_h));
    } else {
        resized = input.clone();
    }

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    // resized = (resized - 127.5f) / 127.5f;  // 归一化到[-1, 1]
    cv::Mat dst_float;
    
    resized.convertTo(dst_float, CV_32FC3);
    std::vector<cv::Mat> channels;
    cv::split(dst_float, channels);
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean_[c]) * std_[c];
    }
    cv::merge(channels, dst_float);
    if (resized_w < padding_w){
        int right = padding_w - resized_w;
        int bootom = input_height - resized_h;
        cv::copyMakeBorder(dst_float, output_image, 0, bootom, 0, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        if(ins_attr[0].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && ins_attr[0].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8){
            memcpy(input_tensors[0].data, output_image.data, output_image.rows * output_image.cols * output_image.channels());
            return;
        }
        // params.right = right;
    } else {
        output_image = dst_float.clone();
        
    }

    for (int i = 0; i < input_num; i++) {
        
        // 转换为CHW格式
        float* input_tensor_float32 = (float*)malloc(sizeof(float) * input_height * input_width * 3);
        if(input_tensor_float32 == NULL) return;
        mat_to_tensor(output_image, input_tensor_float32);
        
        size_t num_elements = get_element_num(ins_attr[i]);
        quantize(input_tensor_float32, input_tensors[i].data, 
                                num_elements, ins_attr[i]);
        free(input_tensor_float32);
        
    }

}

bool PPOCR_Rec::run(){
    
    int status = ta_runtime_run_network(nnrt_context.get());
    if (status != 0) {
        std::cerr << "Run network failed: 0x" << std::hex << status << std::dec << std::endl;
        return false;
    }
    
    status = ta_runtime_invalidate_buffer(nnrt_context.get(), output_buffer);
    if (status != 0) {
        std::cerr << "Invalidate output buffer failed: 0x" << std::hex << status << std::dec << std::endl;
        return false;
    }
    return true;
}

static void dequantize_float32(float *dst,void* data, size_t element) {
    for(int i = 0; i < element; i++){
        dst[i] = static_cast<float*>(data)[i];
    }
}
static void dequantize_float16(float *dst,void* data, size_t element) {
    for(int i = 0; i < element; i++){
        uint16_t h =  static_cast<uint16_t*>(data)[i];
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                uint32_t f = (sign << 31); // ±0.0
                dst[i] = *reinterpret_cast<float*>(&f); // sign? -0.0f : 0.0f
            }
            // 非正规数直接计算: sign * 2^{-24} * mantissa
            const uint32_t exp_offset = 103;  // 127 - 24
            uint32_t f = (sign << 31) | (exp_offset << 23) | (mantissa << 13);
            dst[i] =*reinterpret_cast<float*>(&f);
        } 
        else if (exponent == 31) {
            uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
            dst[i] =*reinterpret_cast<float*>(&f);
        }
        
        // 正规数
        exponent += (127 - 15);  // 偏置调整
        uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
        dst[i] = *reinterpret_cast<float*>(&f);
    }
    
}

static void dequantize_uint8(float *dst,void* data, size_t element, int32_t zp, float scale) {
    for(int i = 0;i < element; i++){
        uint8_t val = static_cast<uint8_t*>(data)[i];
        dst[i] = ((float)val - (float)zp) * scale;
    }

}

static void dequantize_int8(float *dst,void* data, size_t element, int32_t zp, float scale) {
    for(int i = 0;i < element; i++){
        int8_t val = static_cast<int8_t*>(data)[i];
        dst[i] = ((float)val - (float)zp) * scale;
    }
    
}

static void dequantize_int16(float *dst,void* data, size_t element, int32_t zp, float scale) {
    for(int i = 0;i < element; i++){
        int16_t val = static_cast<int16_t*>(data)[i];
        dst[i] = ((float)val - (float)zp) * scale;
    }
    
}

void PPOCR_Rec::postprocess(std::vector<std::pair<std::string, float>> &result_list, bool beam_search, int beam_width){
    for(int i = 0; i < output_num; i++) {
       
        int element_num = get_element_num(outs_attr[i]);
        void * output_npu = output_buffer[i].data;
        int32_t zp = 0;
        float scale = 1.0f;
        if (outs_attr[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
            zp = outs_attr[i].quant_data.affine.tf_zero_point;
            scale = outs_attr[i].quant_data.affine.tf_scale;
        }

        std::vector<float> float_output(element_num);
    
        if(outs_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8){
            dequantize_int8(float_output.data(),output_npu,element_num,zp,scale);
        }else if(outs_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8){
            dequantize_uint8(float_output.data(),output_npu,element_num,zp,scale);
        }else if(outs_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT16){
            dequantize_int16(float_output.data(),output_npu,element_num,zp,scale);
        }else if(outs_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16){
            dequantize_float16(float_output.data(),output_npu,element_num);
        }
        

        int batch_size = 1;
        int max_seq_len = 40;
        int num_classes = 6625;

        if (beam_search) {
            // Beam Search 解码逻辑
            for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                // 初始化 beam（初始序列为空，得分 1.0）
                std::vector<Beam> beams;
                beams.push_back({{}, 1.0, {}});

                for (int t = 0; t < max_seq_len; ++t) {  // 遍历每个时间步
                    std::vector<Beam> new_beams;

                    // 遍历当前所有 beam
                    for (const auto& beam : beams) {
                        // 获取当前时间步的概率分布 (num_classes)
                        const float* prob_ptr = float_output.data() + t * num_classes;
                        std::vector<float> probs(prob_ptr, prob_ptr + num_classes);

                        // 获取 top beam_width 个候选字符
                        std::vector<int> indices(num_classes);
                        std::iota(indices.begin(), indices.end(), 0);
                        std::sort(indices.begin(), indices.end(), 
                            [&probs](int a, int b) { return probs[a] > probs[b]; });
                        std::vector<int> top_candidates(indices.begin(), indices.begin() + beam_width);

                        // 扩展每个候选字符
                        for (int c : top_candidates) {
                            
                            Beam new_beam = beam;
                            new_beam.prefix.push_back(c);
                            new_beam.score *= probs[c];
                            new_beam.confs.push_back(probs[c]);
                            new_beams.push_back(new_beam);

                        }
                    }

                    // 按得分排序并保留 top beam_width 个
                    std::sort(new_beams.begin(), new_beams.end(),
                        [](const Beam& a, const Beam& b) { return a.score > b.score; });
                    if (new_beams.size() > beam_width) {
                        new_beams.resize(beam_width);
                    }
                    beams = new_beams;
                }

                // 选择得分最高的 beam
                auto best_beam_it = std::max_element(beams.begin(), beams.end(),
                    [](const Beam& a, const Beam& b) { return a.score < b.score; });
                const Beam& best_beam = *best_beam_it;

                // 解码（去重、去空白符）
                std::string text;
                std::vector<float> confs;
                if (!best_beam.prefix.empty()) {
                    int pre_c = best_beam.prefix[0];

                    if (pre_c != 0) {  // 0 为空白符
                        text += char_charts[pre_c];
                        confs.push_back(best_beam.confs[0]);
                    }

                    for (size_t idx = 1; idx < best_beam.prefix.size(); ++idx) {
                        int c = best_beam.prefix[idx];
                        if (c == pre_c || c == 0) {  // 跳过重复或空白符
                            if (c == 0) pre_c = c;
                            continue;
                        }
                        text += char_charts[c];
                        confs.push_back(best_beam.confs[idx]);
                        pre_c = c;
                    }
                }

                // 计算平均置信度
                float avg_conf = 0.0f;
                if (!confs.empty()) {
                    avg_conf = std::accumulate(confs.begin(), confs.end(), 0.0f) / confs.size();
                }
                if (isnan(avg_conf)){
                    avg_conf = 0;
                    text = "###";
                }
                result_list.emplace_back(text, avg_conf);
            }
        } else {
            // 普通解码逻辑（argmax + CTC 去重）
            for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                std::string text;
                std::vector<float> confs;
                int pre_c = -1;

                for (int t = 0; t < max_seq_len; ++t) {
                    // 获取当前时间步的最大概率索引和置信度
                    const float* prob_ptr = float_output.data() + t * num_classes;
                    int c = std::distance(prob_ptr, 
                        std::max_element(prob_ptr, prob_ptr + num_classes));
                    float conf = prob_ptr[c];

                    // 初始化第一个字符
                    if (t == 0) {
                        pre_c = c;
                        if (c != 0) {  // 非空白符
                            text += char_charts[c];
                            confs.push_back(conf);
                        }
                        continue;
                    }

                    // 跳过重复或空白符
                    if (c == pre_c || c == 0) {
                        if (c == 0) pre_c = c;
                        continue;
                    }

                    // 添加有效字符
                    text += char_charts[c];
                    confs.push_back(conf);
                    pre_c = c;
                }

                // 计算平均置信度
                float avg_conf = 0.0f;
                if (!confs.empty()) {
                    avg_conf = std::accumulate(confs.begin(), confs.end(), 0.0f) / confs.size();
                }
                if (isnan(avg_conf)){
                    avg_conf = 0;
                    text = "###";
                }
                result_list.emplace_back(text, avg_conf);
                // result_list.emplace_back(text, avg_conf);
            }
        }
        
    }

}
void PPOCR_Rec::rec_and_save(std::vector<cv::Mat> input, std::vector<std::pair<std::string, float>> &result_list, bool beam_search, int beam_size){
    for(int i = 0; i < input.size(); i++){
        cv::Mat dst;
        
        if (m_ts) m_ts->start();
        preprocess(input[i], dst);
        if (m_ts) m_ts->time_accumulation("pre_time");

        if (m_ts) m_ts->start();
        run();
        if (m_ts) m_ts->time_accumulation("infer_time");

        if (m_ts) m_ts->start();
        postprocess(result_list, beam_search,beam_size);
        if (m_ts) m_ts->time_accumulation("post_time");

        
    }
}


