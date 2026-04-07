
#pragma once

#include <chrono>
#include <string>
#include <cstdio>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <dirent.h>
#include <iomanip>
#include <map>

#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ta-runtime-api.h>
#include <unordered_map>

#include <unistd.h>

typedef float (*DequantizeFunc)(void *data, size_t idx, int32_t zp, float scale);

extern const char *CLASS_NAMES[];


class TimeStamp
{
private:
    std::chrono::system_clock::time_point start_time, end_time;

public:
    TimeStamp()
    {
        start();
        time_map_lab["imread_time"] = 0.0f;
        time_map_lab["pre_time"] = 0.0f;
        time_map_lab["infer_time"] = 0.0f;
        time_map_lab["post_time"] = 0.0f;
        // time_map_lab["hardware_time"] = 0.0f;
    }

    void start()
    {
        stop();
        this->start_time = this->end_time;
    }

    void stop()
    {
#ifdef _MSC_VER
        this->end_time = std::chrono::system_clock::now();
#else
        this->end_time = std::chrono::high_resolution_clock::now();
#endif
    }

    std::map<std::string, float> time_map_lab;
    float cost()
    {
        if (this->end_time <= this->start_time)
        {
            this->stop();
        }

        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        return static_cast<float>(ms) / 1000.f;
    }
    void time_accumulation(std::string label)
    {
        if (this->end_time <= this->start_time)
        {
            this->stop();
        }

        auto gap = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        auto it = time_map_lab.find(label);
        if (it == time_map_lab.end())
            return;
        time_map_lab[label] += gap / 1000.f;
    }
};

typedef struct PreprocessParams
{
    float ratio;
    int top;
    int bottom;
    int left;
    int right;
    int net_input_w;
    int net_input_h;
    cv::Size src_size;
} PreprocessParams;


struct KeyPoint
{
    float x, y, score;
};



struct PoseKeyPoints{
  enum EModelType {
      BODY_25 = 0,
      COCO_18 = 1
  };
  std::vector<float> keypoints;
  std::vector<int> shape;
  int width, height;
  EModelType modeltype;
};



class OpenPose
{

    // float conf_thresh;
    float nms_thresh;
    std::shared_ptr<ta_runtime_context> nnrt_context;
    taconn_input_t *input_tensors;
    taconn_buffer_t *output_buffer;
    int input_num;
    int output_num;
    std::vector<taconn_inout_attr_t> ins_attr;
    std::vector<taconn_inout_attr_t> outs_attr;
    PoseKeyPoints::EModelType m_model_type;
    int core_id_;
    // model info

    int model_input_h = 368;
    int model_input_w = 368;

    std::string model_name;
    bool initialized_ = false;
    // for profiling
    TimeStamp *ts_ = NULL;
    std::vector<PreprocessParams> pre_params;

private:
    int preprocess_image(const cv::Mat &src, cv::Mat &dst, PreprocessParams &params);
    int post_process(const cv::Mat &image, std::vector<PoseKeyPoints>& vct_keypoints,int img_index);
    size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count);
    int pre_process(cv::Mat &images);
    size_t get_element_num(const taconn_inout_attr_t input_attr);
    void mat_to_tensor(const cv::Mat &mat, uint8_t *tensor);
    void normalize_and_quantize(uint8_t *src, void *dst, size_t num_elements,
                                const taconn_inout_attr_t &attr);

public:
    std::vector<taconn_profile_t> hardware_time;
    OpenPose(std::shared_ptr<ta_runtime_context> context, std::string model_file, float nms_threshold, int core_id);
    ~OpenPose();
    void enableProfile(TimeStamp *ts);
    int Init();
    int Detect(std::vector<cv::Mat> &input_images, std::vector<PoseKeyPoints> &vct_keypoints);
    int deInit();
    PoseKeyPoints::EModelType get_model_type();
};



class NoCopyable {
  protected:
    NoCopyable() =default;
    ~NoCopyable() = default;
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable& rhs)= delete;
};


class PoseBlob : public NoCopyable, public std::enable_shared_from_this<PoseBlob> {
    int m_count;
    int m_n, m_c, m_h, m_w;
    float *m_data;
public:
    PoseBlob(int n, int c, int h, int w):m_n(n),m_c(c), m_h(h), m_w(w) {
        m_count = n*c*w*h;
        m_data = new float_t[m_count];
    }

    ~PoseBlob() {
        delete[] m_data;
    }

    std::shared_ptr<PoseBlob> getPtr() {
        return shared_from_this();
    }

    int height() { return m_h;}
    int width() { return m_w;}
    int channels() {return m_c;}
    int num() { return m_n;}

    float* data() {
        return m_data;
    }
};
using PoseBlobPtr = std::shared_ptr<PoseBlob>;


class OpenPosePostProcess {
public:
    OpenPosePostProcess();

    ~OpenPosePostProcess();


    static void cpuOptNms(PoseBlobPtr bottom_blob, PoseBlobPtr top_blob, float threshold);

    static void cpuOptNmsFunc(float* ptr, float* top_ptr, int length, int h, int w,
            int max_peaks, float threshold, int plane_offset,
            int top_plane_offset);

    static void connectBodyPartsCpu(std::vector<float> &poseKeypoints, const float *const heatMapPtr,
                                    const float *const peaksPtr, const cv::Size &heatMapSize, const int maxPeaks,
                                    const int interMinAboveThreshold, const float interThreshold,
                                    const int minSubsetCnt,
                                    const float minSubsetScore, const float scaleFactor,
                                    std::vector<int> &keypointShape, PoseKeyPoints::EModelType model_type);


    static void renderKeypointsCpu(cv::Mat &frame, const std::vector<float> &keypoints, std::vector<int> keyshape,
                                   const std::vector<unsigned int> &pairs, const std::vector<float> colors,
                                   const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                                   const float threshold, float scale);

    static void renderPoseKeypointsCpu(cv::Mat &frame, const std::vector<float> &poseKeypoints, std::vector<int> keyshape,
                           const float renderThreshold, float scale, PoseKeyPoints::EModelType modelType, const bool blendOriginalFrame = true);
    
    
     static void getKeyPointsCPUOpt(const std::vector<taconn_inout_attr_t>& outs_attr, const cv::Mat &image,
            std::vector<PoseKeyPoints> &body_keypoints,void* output_buffer,PoseKeyPoints::EModelType model_type, float nms_threshold,const PreprocessParams& params);

    static std::vector<unsigned int> getPosePairs(PoseKeyPoints::EModelType model_type);

    static std::vector<unsigned int> getPoseMapIdx(PoseKeyPoints::EModelType model_type);

    static int getNumberBodyParts(PoseKeyPoints::EModelType model_type);

};
