#include "ppocr_det.hpp"
#include "postprocess.hpp"

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

size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count) {
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

static uint16_t float32_to_float16(float value) {
    // 此函数实现将 float32 转换为 float16 格式
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
            fprintf(stderr, "Unsupported data format for non-quantized input: %d\n", 
                    attr.data_format);
        }
    } else if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_ASYMMETRIC) {
     
        float scale = attr.quant_data.affine.tf_scale;
        int32_t zero_point = attr.quant_data.affine.tf_zero_point;
        
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_INT8) {

            int8_t* dst_int8 = static_cast<int8_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {
                // float normalized = src[i] / 255.0f;
                int32_t quantized = lrintf(src[i] / scale) + zero_point;
                dst_int8[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
            }
        }else if(attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16){

            int16_t* dst_int8 = static_cast<int16_t*>(dst);
            for (size_t i = 0; i < num_elements; i++) {

                dst_int8[i] = float32_to_float16(src[i]);
            }
        } else {
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

static void mat_to_tensor(const cv::Mat& mat, uint8_t* tensor) {
    /* work as preprocess_node_layer */
   
    const int height = mat.rows;
    const int width = mat.cols;
    const int channels = mat.channels();
  
    const int total_pixels = height * width;

    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);

    // 按 CHW 顺序拷贝每个通道
    for (int c = 0; c < channels; ++c) {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(uint8_t));
    }
            
}

static void mat_to_tensor(const cv::Mat& mat, float* tensor) {
    /* work as preprocess_node_layer */
//    printf("mat_to_tensor fp16\n");
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
PPOCR_Detector::PPOCR_Detector(std::shared_ptr<ta_runtime_context> context,std::string m_path){
    nnrt_context = context;
    model_path = m_path;
    std::cout<<"model path "<< model_path << std::endl;
    initialized_ = false;
    input_num = 0;
    output_num = 0;
    input_tensors = nullptr;
    output_buffer = nullptr;
}

PPOCR_Detector::~PPOCR_Detector(){
    if(initialized_)
        deinit();
}
bool PPOCR_Detector::Init(){
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

void PPOCR_Detector::deinit(){
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

void PPOCR_Detector::preprocess(const cv::Mat &input,cv::Mat &dst){

    
    int src_h = input.rows;
    int src_w = input.cols;
    int size_max = std::max(src_h, src_w);
    int limit_side_len = input_height;  // 480
    
    float ratio;
    if (size_max >= limit_side_len) {
        ratio = (float)limit_side_len / size_max;
    } else {
        ratio = 1.0f;
    }
    
    resize_h = (int)(src_h * ratio);
    resize_w = (int)(src_w * ratio);
    
    cv::Mat resized;
    if (src_h != resize_h || src_w != resize_w) {
        cv::resize(input, resized, cv::Size(resize_w, resize_h));
    } else {
        resized = input.clone();
    }

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    if (resize_w < input_width || resize_h < input_height){
        int right = input_width - resize_w;
        int bottom = input_height - resize_h;
        cv::copyMakeBorder(resized, dst, 0, bottom, 0, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        //该模型为单输入
        
        if(ins_attr[0].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE && ins_attr[0].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8){
            memcpy(input_tensors[0].data, dst.data, dst.rows * dst.cols * dst.channels());
            return;
        }
    } else {
        dst = resized.clone();
        
    }
 
    cv::Mat dst_float;
    dst.convertTo(dst_float, CV_32FC3);
    
    std::vector<cv::Mat> channels;
    cv::split(dst_float, channels);
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean_[c]) * std_[c];
    }
    cv::merge(channels, dst_float);

    
    for (int i = 0; i < input_num; i++) {
        size_t num_elements = get_element_num(ins_attr[i]);
        float* input_tensor_float32 = (float*)malloc(sizeof(float) * input_height * input_width * 3);
        if(input_tensor_float32 == NULL) return;
        mat_to_tensor(dst_float, input_tensor_float32);
        
        quantize(input_tensor_float32, input_tensors[i].data, 
                                num_elements, ins_attr[i]);
        free(input_tensor_float32);
        

    }
    
}

bool PPOCR_Detector::run(){
    
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

int PPOCR_Detector::postprocess( cv::Mat &input, std::vector<OCRBoxVec>& batch_boxes)
{ 
    
    int src_h = input.rows;
    int src_w = input.cols;

    
    for(int i = 0; i < output_num; i++){
        void *output_data = output_buffer[i].data;
        
        float scale = outs_attr[i].quant_data.affine.tf_scale;
        int32_t zp = outs_attr[i].quant_data.affine.tf_zero_point;
        int out_h = input_height;  // 480
        int out_w = input_width;   // 480
        int n = out_h * out_w;
        
 
        std::vector<float> pred(n);
        
        if (outs_attr[i].data_format == TACONN_DATA_FORMAT_FP32) {
            float* output_float = (float*)output_data;
            for (int idx = 0; idx < n; idx++) {
                pred[idx] = (float)output_float[idx];
            }
        }
        else if (outs_attr[i].data_format == TACONN_DATA_FORMAT_UINT8) {
            uint8_t* output_uint8 = (uint8_t*)output_data;
            for (int idx = 0; idx < n; idx++) {
                pred[idx] = ((float)output_uint8[idx] - (float)zp) * scale;
            }
        }else if (outs_attr[i].data_format == TACONN_DATA_FORMAT_INT8) {
            int8_t* output_uint8 = (int8_t*)output_data;
            for (int idx = 0; idx < n; idx++) {
                pred[idx] = ((float)output_uint8[idx] - (float)zp) * scale;
            }
        }else if (outs_attr[i].data_format == TACONN_DATA_FORMAT_FP16) {
            
            dequantize_float16(pred.data(),output_data,get_element_num(outs_attr[i]));
        }
        else if (outs_attr[i].data_format == TACONN_DATA_FORMAT_INT16) {
            
            int16_t* output_int16 = (int16_t*)output_data;
            for (int idx = 0; idx < n; idx++) {
                pred[idx] = dfp_to_fp32(output_int16[idx],(outs_attr[i].quant_data.dfp.fixed_point_pos));
            }
        }
        
        // 转换为 uint8 用于可视化
        std::vector<unsigned char> cbuf(n);
        for (int idx = 0; idx < n; idx++) {
            cbuf[idx] = (unsigned char)(std::clamp(pred[idx], 0.0f, 1.0f) * 255.0f);
        }
        
       
        cv::Mat pred_map_full(out_h, out_w, CV_32F, pred.data());
        cv::Mat cbuf_map_full(out_h, out_w, CV_8UC1, cbuf.data());
        

        cv::Rect valid_region(0, 0, resize_w, resize_h);
        cv::Mat pred_map = pred_map_full(valid_region);
        cv::Mat cbuf_map = cbuf_map_full(valid_region);
        
        

        float threshold = 0.3;
        cv::Mat bit_map;
        cv::threshold(cbuf_map, bit_map, threshold * 255, 255, cv::THRESH_BINARY);
        

        int valid_pixels = resize_h * resize_w;
        int white_pixels = cv::countNonZero(bit_map);
        
        // 后处理参数
        double det_db_box_thresh = 0.6;
        double det_db_unclip_ratio = 1.5;
        bool use_polygon_score = false;
        
        PostProcessor post_processor;
        

        std::vector<std::vector<std::vector<int>>> boxes = post_processor.BoxesFromBitmap(
                pred_map, bit_map, det_db_box_thresh, det_db_unclip_ratio, 
                use_polygon_score, src_w, src_h);
        

        OCRBoxVec ocrboxes = post_processor.FilterTagDetRes(boxes, input);

        batch_boxes.push_back(ocrboxes);
    }
  
    return 0;
}

void PPOCR_Detector::detect_and_save( std::vector<cv::Mat> &input,std::vector<OCRBoxVec> &boxes){
    for(int i = 0; i < input.size(); i++){
        cv::Mat dst;
        if (m_ts) m_ts->start();
        preprocess(input[i], dst);
        if (m_ts) m_ts->time_accumulation("pre_time");
        
        if (m_ts) m_ts->start();
        run();
        if (m_ts) m_ts->time_accumulation("infer_time");

        if (m_ts) m_ts->start();
        postprocess(input[i], boxes);
        if (m_ts) m_ts->time_accumulation("post_time");

    }
}


float PPOCR_Detector::dequantize(float value, int32_t, float) {
    return value;
}

float PPOCR_Detector::dequantize(uint16_t value, int32_t, float) {
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

float PPOCR_Detector::dequantize(uint8_t value, int32_t zp, float scale) {
    return ((float)value - (float)zp) * scale;
}
float PPOCR_Detector::dequantize(int8_t value, int32_t zp, float scale) {
    return ((float)value - (float)zp) * scale;
}
