#include "openpose.hpp"
#include <opencv2/imgproc.hpp>
#include <thread>

#define POSE_COLORS_RENDER_CPU \
	255.f, 0.f, 0.f, \
	255.f, 85.f, 0.f, \
	255.f, 170.f, 0.f, \
	255.f, 255.f, 0.f, \
	170.f, 255.f, 0.f, \
	85.f, 255.f, 0.f, \
	0.f, 255.f, 0.f, \
	0.f, 255.f, 85.f, \
	0.f, 255.f, 170.f, \
	0.f, 255.f, 255.f, \
	0.f, 170.f, 255.f, \
	0.f, 85.f, 255.f, \
	0.f, 0.f, 255.f, \
    85.f, 0.f, 255.f, \
    170.f, 0.f, 255.f, \
	255.f, 0.f, 255.f, \
	255.f, 0.f, 170.f, \
    255.f, 0.f, 85.f, \
    255.f, 0.f, 0.f, \
    255.f, 0.f, 255.f, \
    255.f, 85.f, 255.f, \
    255.f, 170.f, 255.f, \
    255.f, 255.f, 255.f, \
    170.f, 255.f, 255.f, \
    85.f, 255.f, 255.f
const std::vector<float> POSE_COLORS_RENDER{ POSE_COLORS_RENDER_CPU };
const unsigned int POSE_MAX_PEOPLE = 96;
OpenPose::OpenPose(std::shared_ptr<ta_runtime_context> context, std::string model_file, float nms_threshold, int core_id) : nnrt_context(context), model_name(model_file),nms_thresh(nms_threshold),core_id_(core_id)
{

    input_tensors = nullptr;
    output_buffer = nullptr;
    input_num = 0;
    output_num = 0;
    initialized_ = false;
}

OpenPose::~OpenPose()
{
    if (initialized_)
    {
        deInit();
    }
}

PoseKeyPoints::EModelType OpenPose::get_model_type(){
  return m_model_type;
};

void print_taconn_inout_attr(const taconn_inout_attr_t &attr)
{
    std::cout << "--------------------------------------------------------------" << "\n";
    std::cout << "    Tensor Attribute index:      |  " << attr.index << "\n";
    std::cout << "    dim_count:                   |  " << attr.dim_count << "\n";

    std::cout << "    dim_size:                    |  [";
    for (unsigned int i = 0; i < attr.dim_count; ++i)
    {
        std::cout << attr.dim_size[i];
        if (i != attr.dim_count - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "    data_format:                 |  " << attr.data_format << "\n";
    std::cout << "    quant_format:                |  " << attr.quant_format << "\n";

    {
        std::cout << "    quant_data (dfp):            \n";
        std::cout << "        fixed_point_pos:         |  " << attr.quant_data.dfp.fixed_point_pos << "\n";
    }
    {
        std::cout << "    quant_data (affine):\n";
        std::cout << "        tf_scale:                |  " << std::fixed << std::setprecision(6)
                  << attr.quant_data.affine.tf_scale << "\n";
        std::cout << "        tf_zero_point:           |  " << attr.quant_data.affine.tf_zero_point << "\n";
    }

    std::cout << "    name:                        |  " << attr.name << "\n";
    std::cout << "--------------------------------------------------------------" << "\n";
}

size_t OpenPose::get_element_num(const taconn_inout_attr_t input_attr)
{
    size_t size = 1;
    for (unsigned int i = 0; i < input_attr.dim_count; ++i)
    {
        size *= input_attr.dim_size[i];
    }
    return size;
}
void OpenPose::mat_to_tensor(const cv::Mat &mat, uint8_t *tensor)
{
    int channels = mat.channels();
    int total_pixels = mat.rows * mat.cols;

    std::vector<cv::Mat> channel_mats;
    cv::split(mat, channel_mats);
    for (int c = 0; c < channels; ++c)
    {
        memcpy(tensor + c * total_pixels, channel_mats[c].data, total_pixels * sizeof(uint8_t));
    }
}

int OpenPose::preprocess_image(const cv::Mat &src, cv::Mat &dst, PreprocessParams &params)
{
    
    params.net_input_w = model_input_w;
    params.net_input_h = model_input_h;
    cv::resize(src, dst, cv::Size(model_input_w, model_input_h), 0, 0, cv::INTER_LINEAR);
    return 0;
}

static uint16_t float32_to_float16(float value) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (bits >> 31) & 0x1;
    int exponent = (bits >> 23) & 0xFF;
    uint32_t fraction = bits & 0x7FFFFF;
    
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

void OpenPose::normalize_and_quantize(uint8_t *src, void *dst, size_t num_elements,
                                    const taconn_inout_attr_t &attr)
{
    
    if (attr.quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE)
    {
        if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP32)
        {
            float *dst_float = static_cast<float *>(dst);
            for (size_t i = 0; i < num_elements; i++)
            {
                dst_float[i] = (src[i] - 128.0f) / 255.0f;
            }
        }
        else if (attr.data_format == taconn_data_format_e::TACONN_DATA_FORMAT_FP16)
        {
            uint16_t *dst_fp16 = static_cast<uint16_t *>(dst);
            for (size_t i = 0; i < num_elements; i++)
            {
                float normalized = (src[i] - 128.0f)  / 255.0f;
                dst_fp16[i] = float32_to_float16(normalized);
            }
        }
        else
        {
            std::cerr << "Unsupported data format for non-quantized input: " << attr.data_format << std::endl;
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
                float normalized = (src[i] - 128.0f) / 255.0f;
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
int OpenPose::pre_process(cv::Mat &images)
{

    for (int i = 0; i < input_num; i++)
    {
        cv::Mat dst;
        PreprocessParams params;
        ts_->start();
        preprocess_image(images, dst, params);
        ts_->time_accumulation("pre_time");
        pre_params.push_back(params);
        if (ins_attr[i].quant_format == taconn_qnt_type_e::TACONN_QNT_TYPE_NONE &&
            ins_attr[i].data_format == taconn_data_format_e::TACONN_DATA_FORMAT_UINT8)
        {
            memcpy(input_tensors[i].data, dst.data, input_tensors[i].size);
        }
        else
        { 
            uint8_t *input_tensor_uint8 = nullptr;
            input_tensor_uint8 = (uint8_t *)malloc(sizeof(uint8_t) * model_input_h * model_input_w * 3);
            if (input_tensor_uint8 == NULL)
                return -1;
            mat_to_tensor(dst, input_tensor_uint8);

            size_t num_elements = get_element_num(ins_attr[i]);
            normalize_and_quantize(
                input_tensor_uint8,
                input_tensors[i].data,
                num_elements,
                ins_attr[i]);
            free(input_tensor_uint8);
        }
    }
    return 0;
}
int OpenPose::post_process(const cv::Mat &image, std::vector<PoseKeyPoints>& vct_keypoints,int img_index)
{
   
    if (outs_attr.empty())
    {
        std::cerr << "Error: outs_attr is empty!" << std::endl;
        return -1;
    }
    for (int i = 0; i < output_num; ++i)
    {
        void *output_data = output_buffer[i].data;
        OpenPosePostProcess::getKeyPointsCPUOpt(outs_attr, image, vct_keypoints,output_data,m_model_type, nms_thresh,pre_params[img_index]);

    }
    return 0;
}

size_t OpenPose::calculate_buffer_size(taconn_data_format_t format, size_t element_count)
{
    switch (format)
    {
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
        std::cerr << "Unsupported data format: " << format << std::endl;
        return 0;
    }
}
int OpenPose::Detect(std::vector<cv::Mat> &input_images, std::vector<PoseKeyPoints> &vct_keypoints)
{
    for (int i = 0; i < input_images.size(); i++)
    {
        
        pre_process(input_images[i]);

        ts_->start();
        int status = ta_runtime_run_network(nnrt_context.get());
        if (status != 0)
        {
            std::cerr << "Run network failed." << std::endl;
            return status;
        }
        // 3. invalidate_output
        status = ta_runtime_invalidate_buffer(nnrt_context.get(), output_buffer);
        if (status != 0)
        {
            std::cerr << "Invalidate output buffer failed." << std::endl;
            return status;
        }
        ts_->time_accumulation("infer_time");
        taconn_profile_t single_haredware_time;
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_PROFILING, &single_haredware_time);
        if (status != 0)
        {
            std::cerr << "Query hardware time failed: " << status << std::endl;
        }
        hardware_time.push_back(single_haredware_time);
        // ts_->time_map_lab["hardware_time"] += single_haredware_time.inference_time /1000.0f;
        ts_->start();
        post_process(input_images[i],vct_keypoints,i);
        ts_->time_accumulation("post_time");

        if (i % 10 == 0 || i == input_images.size())
        {
            std::cout << "Processed " << i << "/" << input_images.size()
                      << " images (" << vct_keypoints.size() << " objects detected)" << std::endl;
        }
    }

    return 0;
}

int OpenPose::Init()
{
    if (initialized_)
    {
        std::cerr << "Detector already initialized" << std::endl;
        return -1;
    }
    m_model_type = PoseKeyPoints::EModelType::COCO_18;
    // 1. init taruntime
    
    taconn_input_output_num_t num = {};

    int status = ta_runtime_init();
    if (status != 0)
    {
        std::cerr << "Failed to initialize TA runtime: " << status << std::endl;
        goto CLEANUP;
    }

    // 2. load model
    status = ta_runtime_load_model_from_file(nnrt_context.get(), model_name.c_str(), core_id_);
    if (status != 0)
    {
        std::cerr << "Load model from file failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_IN_OUT_NUM, &num);
    std::cout << "Input num is : " << num.input_num << ", Output num is : " << num.output_num << std::endl;
    input_num = num.input_num;
    output_num = num.output_num; // cut model

    for (int i = 0; i < input_num; i++)
    {
        taconn_inout_attr_t input_attr = {0};
        input_attr.index = i;
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_INPUT_ATTR, &input_attr);
        print_taconn_inout_attr(input_attr);
        ins_attr.push_back(input_attr);
    }
    // 仅适用于当前单输入情况
    model_input_h = ins_attr[0].dim_size[0];
    model_input_w = ins_attr[0].dim_size[1];

    for (int i = 0; i < output_num; i++)
    {
        taconn_inout_attr_t output_attr = {0};
        output_attr.index = i;
        status = ta_runtime_query(nnrt_context.get(), TACONN_QUERY_OUTPUT_ATTR, &output_attr);
        print_taconn_inout_attr(output_attr);
        outs_attr.push_back(output_attr);
    }

    // 3. set input
    input_tensors = (taconn_input_t *)malloc(sizeof(taconn_input_t) * input_num);
    if (!input_tensors)
    {
        std::cerr << "Allocate input_tensors failed" << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < input_num; i++)
    {

        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(ins_attr[i].data_format);
        size_t input_buffer_size = calculate_buffer_size(data_type, get_element_num(ins_attr[i]));

        input_tensors[i].index = i;
        input_tensors[i].size = input_buffer_size; // size in bytes
        input_tensors[i].data = nullptr;           // will be set later
        if (posix_memalign((void **)&input_tensors[i].data, 256, input_buffer_size) != 0)
        {
            std::cerr << "Failed to allocate input buffer" << std::endl;
            goto CLEANUP;
        }
        memset(input_tensors[i].data, 0, input_buffer_size);
    }
    status = ta_runtime_set_input_cva(nnrt_context.get(), input_num, input_tensors);
    if (status != 0)
    {
        std::cerr << "Set input failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    // 4. set output
    output_buffer = (taconn_buffer_t *)malloc(sizeof(taconn_buffer_t) * output_num);
    if (!output_buffer)
    {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }

    for (int i = 0; i < output_num; i++)
    {
        taconn_data_format_t data_type = static_cast<taconn_data_format_t>(outs_attr[i].data_format);
        size_t output_buffer_size = calculate_buffer_size(data_type, get_element_num(outs_attr[i]));
        status = ta_runtime_create_buffer(nnrt_context.get(), output_buffer_size, &output_buffer[i]);
        if (status != 0)
        {
            std::cerr << "Create output buffer " << i << "failed: 0x%x" << std::hex << status << std::dec << std::endl;
            goto CLEANUP;
        }
        else
        {
            std::cout << "Crated output buffer with size: " << output_buffer_size << std::endl;
        }
    }

    status = ta_runtime_set_output(nnrt_context.get(), output_num, output_buffer);
    if (status != 0)
    {
        std::cerr << "Set output failed: 0x" << std::hex << status << std::dec << std::endl;
        goto CLEANUP;
    }
    initialized_ = true;
    return status;

CLEANUP:
    if (input_tensors)
    {
        for (int i = 0; i < input_num; i++)
        {
            if (input_tensors[i].data)
            {
                free(input_tensors[i].data);
            }
        }
        free(input_tensors);
        input_tensors = nullptr;
    }
    if (output_buffer)
    {
        for (int i = 0; i < output_num; i++)
        {
            ta_runtime_destroy_buffer(nnrt_context.get(), &output_buffer[i]);
        }
        free(output_buffer);
        output_buffer = nullptr;
    }
    if (nnrt_context.get())
    {
        ta_runtime_destroy_context(nnrt_context.get());
    }
    ta_runtime_deinit();
    initialized_ = false;
    std::cout << "Model deinitialized" << std::endl;
    return status;
}
int OpenPose::deInit()
{
    if (!initialized_)
    {
        return -1;
    }
    // free output buffer
    for (int i = 0; i < output_num; i++)
    {
        ta_runtime_destroy_buffer(nnrt_context.get(), &output_buffer[i]);
    }

    // free input buffer
    if (input_tensors)
    {
        for (int i = 0; i < input_num; i++)
        {
            if (input_tensors[i].data)
                free(input_tensors[i].data);
        }
        free(input_tensors);
        input_tensors = nullptr;
    }

    // free output ctx array
    if (output_buffer)
    {
        free(output_buffer);
        output_buffer = nullptr;
    }

    // free ctx
    if (nnrt_context.get())
        ta_runtime_destroy_context(nnrt_context.get());
    // free runtime
    ta_runtime_deinit();
    initialized_ = false;
    return 0;
}
void OpenPose::enableProfile(TimeStamp *ts)
{
    ts_ = ts;
}
// 工具函数
template<typename T>
inline int intRound(const T a)
{
    return static_cast<int>(a + 0.5f);
}

template<typename T>
inline T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

void OpenPosePostProcess::cpuOptNmsFunc(float* ptr, float* top_ptr, int length, int h,
                                  int w, int max_peaks, float threshold,
                                  int plane_offset, int top_plane_offset) {
    for (int c = 0; c < length; c++) {
        int num_peaks = 0;
        for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y) {
            for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x) {
                float value = ptr[y * w + x];
                if (value > threshold) {
                    const float topLeft = ptr[(y - 1) * w + x - 1];
                    const float top = ptr[(y - 1) * w + x];
                    const float topRight = ptr[(y - 1) * w + x + 1];
                    const float left = ptr[y * w + x - 1];
                    const float right = ptr[y * w + x + 1];
                    const float bottomLeft = ptr[(y + 1) * w + x - 1];
                    const float bottom = ptr[(y + 1) * w + x];
                    const float bottomRight = ptr[(y + 1) * w + x + 1];
                    if (value > topLeft && value > top && value > topRight &&
                        value > left && value > right && value > bottomLeft &&
                        value > bottom && value > bottomRight) {
                        float xAcc = 0;
                        float yAcc = 0;
                        float scoreAcc = 0;
                        for (int kx = -3; kx <= 3; ++kx) {
                            int ux = x + kx;
                            if (ux >= 0 && ux < w) {
                                for (int ky = -3; ky <= 3; ++ky) {
                                    int uy = y + ky;
                                    if (uy >= 0 && uy < h) {
                                        float score = ptr[uy * w + ux];
                                        xAcc += ux * score;
                                        yAcc += uy * score;
                                        scoreAcc += score;
                                    }
                                }
                            }
                        }  
                        if (scoreAcc > 0) {
                            xAcc /= scoreAcc;
                            yAcc /= scoreAcc;
                        }
                        scoreAcc = value;
                        top_ptr[(num_peaks + 1) * 3 + 0] = xAcc;
                        top_ptr[(num_peaks + 1) * 3 + 1] = yAcc;
                        top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
                        num_peaks++;
                    }
                }
            }
    }
    top_ptr[0] = num_peaks;
    ptr += plane_offset;
    top_ptr += top_plane_offset;
  }
}

void OpenPosePostProcess::cpuOptNms(PoseBlobPtr bottom_blob, PoseBlobPtr top_blob, float threshold)
{
    
    int w = bottom_blob->width();
    int h = bottom_blob->height();
    int C = bottom_blob->channels();
    int N = bottom_blob->num();
    int plane_offset = w * h;
    float* ptr = bottom_blob->data();
    float* top_ptr = top_blob->data();
    int top_plane_offset = top_blob->width() * top_blob->height();
    int max_peaks = top_blob->height() - 1;

	for (int n = 0; n < N; ++n) {
        const int numThreads = std::min(8, C);
        int length = C / numThreads;
        int last = C % length;
        std::vector<std::thread> threads;
    	for (int c = 0; c < numThreads - 1; ++c) {
        threads.emplace_back(
            &OpenPosePostProcess::cpuOptNmsFunc, ptr + c * length * plane_offset,
            top_ptr + c * length * top_plane_offset, length, h, w, max_peaks,
            threshold, plane_offset, top_plane_offset);
        }
        threads.emplace_back(&OpenPosePostProcess::cpuOptNmsFunc, ptr + (numThreads - 1) * length * plane_offset,
                         top_ptr + (numThreads - 1) * length * top_plane_offset,
                         last ? last : length, h, w, max_peaks, threshold,
                         plane_offset, top_plane_offset);
        for (std::thread& t : threads) {
            t.join();
        }
    }
}
std::vector<unsigned int> OpenPosePostProcess::getPosePairs(PoseKeyPoints::EModelType model_type) {
    switch (model_type) {
        case PoseKeyPoints::EModelType::BODY_25:
            return {
                1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15,
                17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24
            };
        case PoseKeyPoints::EModelType::COCO_18:
        default:
            return {
                1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0,
                15, 15, 17, 2, 16, 5, 17
            };
        }
}

std::vector<unsigned int> OpenPosePostProcess::getPoseMapIdx(PoseKeyPoints::EModelType model_type) {
    switch (model_type) {
        case PoseKeyPoints::EModelType::BODY_25:
            return {
                26,27, 40,41, 48,49, 42,43, 44,45, 50,51, 52,53, 32,33, 28,29, 30,31, 34,35, 36,37, 38,39, 56,57, 58,59, 62,63, 60,61, 64,65, 46,47, 54,55, 66,67, 68,69, 70,71, 72,73, 74,75, 76,77
            };
        case PoseKeyPoints::EModelType::COCO_18:
        default:
            return {
                31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46
            };
    }
}

int OpenPosePostProcess::getNumberBodyParts(PoseKeyPoints::EModelType model_type) {
    switch (model_type) {
        case PoseKeyPoints::EModelType::BODY_25:
            return 25;
        case PoseKeyPoints::EModelType::COCO_18:
        default:
            return 18;
    }
}

void OpenPosePostProcess::connectBodyPartsCpu(std::vector<float>& poseKeypoints, const float* const heatMapPtr,
        const float* const peaksPtr, const cv::Size& heatMapSize, const int maxPeaks,
        const int interMinAboveThreshold, const float interThreshold, const int minSubsetCnt,
        const float minSubsetScore, const float scaleFactor, std::vector<int>& keypointShape, PoseKeyPoints::EModelType modelType)
{
    keypointShape.resize(3);
    const auto bodyPartPairs = getPosePairs(modelType);
    const auto mapIdx = getPoseMapIdx(modelType);
    const auto numberBodyParts = getNumberBodyParts(modelType); //COCO 18 points

    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

    std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
    const auto subsetCounterIndex = numberBodyParts;
    const auto subsetSize = numberBodyParts + 1;

    const auto peaksOffset = 3 * (maxPeaks + 1);
    const auto heatMapOffset = heatMapSize.area();

    for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
    {
        const auto bodyPartA = bodyPartPairs[2 * pairIndex];
        const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
        const auto* candidateA = peaksPtr + bodyPartA*peaksOffset;
        const auto* candidateB = peaksPtr + bodyPartB*peaksOffset;
        const auto nA = intRound(candidateA[0]);
        const auto nB = intRound(candidateB[0]);
        auto addSinglePart = [&](int bodyPart, int count, const auto* candidate){
            for(auto i = 1; i <= count; i++)
            {
                bool found = false;
                    const auto off = (int)bodyPart * peaksOffset + i * 3 + 2;
                    for (auto j = 0u; j < subset.size(); j++)
                    {
                        if (subset[j].first[bodyPart] == off)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        rowVector[bodyPart] = off; //store the index
                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
                        const auto subsetScore = candidate[i * 3 + 2]; //second last number in each row is the total score
                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
            }
        };

        if (nA == 0 || nB == 0)
        {
            
            nA==0 ? addSinglePart(bodyPartB, nB, candidateB): addSinglePart(bodyPartA, nA, candidateA);
        }
        else // if (nA != 0 && nB != 0)
        {
            std::vector<std::tuple<double, int, int>> temp;
            const auto numInter = 10;
            const auto* const mapX = heatMapPtr + mapIdx[2 * pairIndex] * heatMapOffset;
            const auto* const mapY = heatMapPtr + mapIdx[2 * pairIndex + 1] * heatMapOffset;
            for (auto i = 1; i <= nA; i++)
            {
                for (auto j = 1; j <= nB; j++)
                {
                    const auto dX = candidateB[j * 3] - candidateA[i * 3];
                    const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
                    const auto normVec = float(std::sqrt(dX*dX + dY*dY));
                    if (normVec > 1e-6)
                    {
                        const auto sX = candidateA[i * 3];
                        const auto sY = candidateA[i * 3 + 1];
                        const auto vecX = dX / normVec;
                        const auto vecY = dY / normVec;

                        auto sum = 0.;
                        auto count = 0;
                        for (auto lm = 0; lm < numInter; lm++)
                        {
                            const auto mX = fastMin(heatMapSize.width - 1, intRound(sX + lm*dX / numInter));
                            const auto mY = fastMin(heatMapSize.height - 1, intRound(sY + lm*dY / numInter));
                            const auto idx = mY * heatMapSize.width + mX;
                            const auto score = (vecX*mapX[idx] + vecY*mapY[idx]);
                            if (score > interThreshold)
                            {
                                sum += score;
                                count++;
                            }
                        }

                        if (count > interMinAboveThreshold)
                            temp.emplace_back(std::make_tuple(sum / count, i, j));
                    }
                }
            }
            if (!temp.empty())
                std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

            std::vector<std::tuple<int, int, double>> connectionK;

            const auto minAB = fastMin(nA, nB);
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);
            auto counter = 0;
            for (auto row = 0u; row < temp.size(); row++)
            {
                const auto score = std::get<0>(temp[row]);
                const auto x = std::get<1>(temp[row]);
                const auto y = std::get<2>(temp[row]);
                if (!occurA[x - 1] && !occurB[y - 1])
                {
                    connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
                                                             bodyPartB*peaksOffset + y * 3 + 2,
                                                             score));
                    counter++;
                    if (counter == minAB)
                        break;
                    occurA[x - 1] = 1;
                    occurB[y - 1] = 1;
                }
            }
            if (pairIndex == 0)
            {
                for (const auto connectionKI : connectionK)
                {
                    std::vector<int> rowVector(numberBodyParts + 3, 0);
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    const auto score = std::get<2>(connectionKI);
                    rowVector[bodyPartPairs[0]] = indexA;
                    rowVector[bodyPartPairs[1]] = indexB;
                    rowVector[subsetCounterIndex] = 2;
                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                }
            }
            else if (
                    (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                    || ((numberBodyParts == 19 || (numberBodyParts == 25)
                         || numberBodyParts == 59 || numberBodyParts == 65)
                        && (pairIndex==18 || pairIndex==19))
                    )
            {
                for (const auto& connectionKI : connectionK)
                {
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    for (auto& subsetJ : subset)
                    {
                        auto& subsetJFirst = subsetJ.first[bodyPartA];
                        auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
                        if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
                            subsetJFirstPlus1 = indexB;
                        else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
                            subsetJFirst = indexA;
                    }
                }
            }
            else
            {
                if (!connectionK.empty())
                {
                    for (auto i = 0u; i < connectionK.size(); i++)
                    {
                        const auto indexA = std::get<0>(connectionK[i]);
                        const auto indexB = std::get<1>(connectionK[i]);
                        const auto score = std::get<2>(connectionK[i]);
                        auto num = 0;
                        for (auto j = 0u; j < subset.size(); j++)
                        {
                            if (subset[j].first[bodyPartA] == indexA)
                            {
                                subset[j].first[bodyPartB] = indexB;
                                num++;
                                subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
                                subset[j].second = subset[j].second + peaksPtr[indexB] + score;
                            }
                        }
                        if (num == 0)
                        {
                            std::vector<int> rowVector(subsetSize, 0);
                            rowVector[bodyPartA] = indexA;
                            rowVector[bodyPartB] = indexB;
                            rowVector[subsetCounterIndex] = 2;
                            const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                            subset.emplace_back(std::make_pair(rowVector, subsetScore));
                        }
                    }
                }
            }
        }
    }

    auto numberPeople = 0;
    std::vector<int> validSubsetIndexes;
    validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
    for (auto index = 0u; index < subset.size(); index++)
    {
        const auto subsetCounter = subset[index].first[subsetCounterIndex];
        const auto subsetScore = subset[index].second;
        if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
        {
            numberPeople++;
            validSubsetIndexes.emplace_back(index);
            if (numberPeople == POSE_MAX_PEOPLE)
                break;
        }
        else if (subsetCounter < 1)
            printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
    }
    keypointShape = { numberPeople, (int)numberBodyParts, 3 };
    if (numberPeople > 0)
        poseKeypoints.resize(numberPeople * (int)numberBodyParts * 3);
    else
        poseKeypoints.clear();

    for (auto person = 0u; person < validSubsetIndexes.size(); person++)
    {
        const auto& subsetI = subset[validSubsetIndexes[person]].first;
        for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
        {
            const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
            const auto bodyPartIndex = subsetI[bodyPart];
            if (bodyPartIndex > 0)
            {
                poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
                poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
                poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
            }
            else
            {
                poseKeypoints[baseOffset] = 0.f;
                poseKeypoints[baseOffset + 1] = 0.f;
                poseKeypoints[baseOffset + 2] = 0.f;
            }
        }
    }
}


void OpenPosePostProcess::renderKeypointsCpu(cv::Mat& frame, const std::vector<float>& keypoints, std::vector<int> keyshape, const std::vector<unsigned int>& pairs,
                        const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                        const float threshold, float scale)
{
    const auto area = frame.cols * frame.rows;
    const auto lineType = 8;
    const auto shift = 0;
    const auto numberColors = colors.size();
    const auto numberKeypoints = keyshape[1];

    for (auto person = 0; person < keyshape[0]; person++)
    {
        const auto thicknessRatio = fastMax(intRound(std::sqrt(area) * thicknessCircleRatio), 1);
        const auto thicknessCircle = 3;
        const auto thicknessLine = 2;
        const auto radius = thicknessRatio / 2;
        for (auto pair = 0u; pair < pairs.size(); pair += 2)
        {
            const auto index1 = (person * numberKeypoints + pairs[pair]) * keyshape[2];
            const auto index2 = (person * numberKeypoints + pairs[pair + 1]) * keyshape[2];
            if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
            {
                const auto colorIndex = pairs[pair + 1] * 3;
                const cv::Scalar color{ colors[(colorIndex+2) % numberColors],
                                        colors[(colorIndex + 1) % numberColors],
                                        colors[(colorIndex + 0) % numberColors] };
                const cv::Point keypoint1{ intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale) };
                const cv::Point keypoint2{ intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale) };
                cv::line(frame, keypoint1, keypoint2, color, thicknessLine, lineType, shift);
            }
        }

        for (auto part = 0; part < numberKeypoints; part++)
        {
            const auto faceIndex = (person * numberKeypoints + part) * keyshape[2];
            if (keypoints[faceIndex + 2] > threshold)
            {
                const auto colorIndex = part * 3;
                const cv::Scalar color{ colors[(colorIndex+2) % numberColors],
                                        colors[(colorIndex + 1) % numberColors],
                                        colors[(colorIndex + 0) % numberColors] };
                const cv::Point center{ intRound(keypoints[faceIndex] * scale), intRound(keypoints[faceIndex + 1] * scale) };
                cv::circle(frame, center, radius, color, thicknessCircle, lineType, shift);
            }
        }
    }
}

void OpenPosePostProcess::renderPoseKeypointsCpu(cv::Mat& frame, const std::vector<float>& poseKeypoints, std::vector<int> keyshape,
                            const float renderThreshold, float scale, PoseKeyPoints::EModelType modelType, const bool blendOriginalFrame)
{
    if (!blendOriginalFrame)
        frame.setTo(0.f);

    const auto thicknessCircleRatio = 1.f / 75.f;
    const auto thicknessLineRatioWRTCircle = 0.75f;
    const auto& pairs = getPosePairs(modelType);

    renderKeypointsCpu(frame, poseKeypoints, keyshape, pairs, POSE_COLORS_RENDER, thicknessCircleRatio,
                       thicknessLineRatioWRTCircle, renderThreshold, scale);
}
float dequantize_f16(void *data, size_t idx, int32_t, float)
{
    uint16_t h = static_cast<uint16_t *>(data)[idx];
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            uint32_t f = (sign << 31);             // ±0.0
            return *reinterpret_cast<float *>(&f); // sign? -0.0f : 0.0f
        }
        // 非正规数直接计算: sign * 2^{-24} * mantissa
        const uint32_t exp_offset = 103; // 127 - 24
        uint32_t f = (sign << 31) | (exp_offset << 23) | (mantissa << 13);
        return *reinterpret_cast<float *>(&f);
    }
    else if (exponent == 31)
    {
        uint32_t f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float *>(&f);
    }

    // 正规数
    exponent += (127 - 15); // 偏置调整
    uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *reinterpret_cast<float *>(&f);
}

void OpenPosePostProcess::getKeyPointsCPUOpt(
    const std::vector<taconn_inout_attr_t> &outs_attr,
    const cv::Mat& image,
    std::vector<PoseKeyPoints> &body_keypoints,
    void* output_buffer,
    PoseKeyPoints::EModelType model_type,
    float nms_threshold,const PreprocessParams& params)
{
    int chan_num = outs_attr[0].dim_size[2];
    int net_output_height = outs_attr[0].dim_size[0];
    int net_output_width = outs_attr[0].dim_size[1];

    int H = net_output_height;
    int W = net_output_width;
    int C = chan_num;
    int spatial_size = H * W;

    std::vector<float> cnhw_data(C * H * W);

    float scale = outs_attr[0].quant_data.affine.tf_scale;
    int zero_point = outs_attr[0].quant_data.affine.tf_zero_point;
    size_t total = C * H * W;

    if (outs_attr[0].data_format == TACONN_DATA_FORMAT_FP16)
    {
        uint16_t* fp16_data = reinterpret_cast<uint16_t*>(output_buffer);
        for(size_t i = 0; i < total; ++i)
        {
            cnhw_data[i] = dequantize_f16(fp16_data, i, 0, 0.0f);
        }
    }

    if (outs_attr[0].data_format == TACONN_DATA_FORMAT_INT8)
    {
        int8_t* int8_data = reinterpret_cast<int8_t*>(output_buffer);
        for(size_t i = 0; i < total; ++i)
        {
            cnhw_data[i] = static_cast<float>(int8_data[i] - zero_point) * scale;
        }
    }
    cv::Size nmsSize(image.cols, image.rows);
    PoseBlobPtr resizedBlob = std::make_shared<PoseBlob>(1, chan_num, nmsSize.height, nmsSize.width);

    for (int c = 0; c < chan_num; ++c) {
        cv::Mat src(H, W, CV_32F, cnhw_data.data() + c * spatial_size);
        cv::Mat dst(nmsSize.height, nmsSize.width, CV_32F,
                    resizedBlob->data() + c * nmsSize.height * nmsSize.width);
        cv::resize(src, dst, nmsSize, 0, 0, cv::INTER_CUBIC);
    }

    PoseBlobPtr nms_blob;
    nms_blob = std::make_shared<PoseBlob>(1, 57, POSE_MAX_PEOPLE + 1, 3);

    cpuOptNms(resizedBlob, nms_blob, nms_threshold);
    PoseKeyPoints poseKeyPoints;
    poseKeyPoints.width = nmsSize.width;
    poseKeyPoints.height = nmsSize.height;
    poseKeyPoints.modeltype = model_type;

    connectBodyPartsCpu(
        poseKeyPoints.keypoints,
        resizedBlob->data(),
        nms_blob->data(),
        nmsSize,
        POSE_MAX_PEOPLE,
        9,
        0.05f,
        3,
        0.4f,
        1.0f,
        poseKeyPoints.shape,
        model_type
    );

    if (!poseKeyPoints.keypoints.empty())
    {
        
        for (size_t i = 0; i < poseKeyPoints.keypoints.size(); i += 3) {
            poseKeyPoints.keypoints[i] = poseKeyPoints.keypoints[i] * image.cols / nmsSize.width;
            poseKeyPoints.keypoints[i + 1] = poseKeyPoints.keypoints[i + 1] * image.rows / nmsSize.height;
        }
    }
    else
    {
        printf("\nno detected points.\n");
    }

    poseKeyPoints.modeltype = model_type;
    body_keypoints.push_back(poseKeyPoints);
}

