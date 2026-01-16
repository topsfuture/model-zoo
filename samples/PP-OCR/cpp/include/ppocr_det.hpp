#ifndef PPOCR_DET_HPP
#define PPOCR_DET_HPP

#include <iostream>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include "opencv2/opencv.hpp"
#include <ta-runtime-api.h>
#include "timer.hpp"



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
        time_map_lab["hardware_time"] = 0.0f;
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
    void time_accumulation(std::string label){
        if (this->end_time <= this->start_time)
        {
            this->stop();
        }

        auto gap = std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count();
        auto it = time_map_lab.find(label);
        if (it == time_map_lab.end())
            return; 
        time_map_lab[label] += gap/1000.f;
    }
};

struct OCRBox {
  int x1, y1, x2, y2, x3, y3, x4, y4;
  std::string rec_res;
  float score;
  void printInfo() {
        printf("Box info: (%d, %d); (%d, %d); (%d, %d); (%d, %d) \n", x1, y1, x2, y2, x3, y3, x4, y4);
    }
};

using OCRBoxVec = std::vector<OCRBox>;

class PPOCR_Detector
{   
    std::shared_ptr<ta_runtime_context> nnrt_context;
    TimeStamp* m_ts;
    public:
        PPOCR_Detector(std::shared_ptr<ta_runtime_context> context,std::string m_path);
        virtual ~PPOCR_Detector();
        bool Init();
        void preprocess(const cv::Mat &input,cv::Mat &dst);
        int postprocess(cv::Mat &input, std::vector<OCRBoxVec>& batch_boxes);
        bool run();
        void detect_and_save(std::vector<cv::Mat> &input,std::vector<OCRBoxVec> &boxes);
        void enableProfile(TimeStamp *ts){
            m_ts = ts;
        };
        void deinit();
    private:
        bool initialized_;
        std::string model_path;
        size_t get_element_num(const taconn_inout_attr_t input_attr ) {
            size_t size = 1;
            for (unsigned int i = 0; i < input_attr.dim_count; ++i) {
                size *= input_attr.dim_size[i];
            }
            return size ;
        }
        
        // 反量化相关 - 函数重载
        float dequantize(float value, int32_t zp, float scale);
        float dequantize(uint16_t value, int32_t zp, float scale);
        float dequantize(uint8_t value, int32_t zp, float scale);
        float dequantize(int8_t value, int32_t zp, float scale);
        
        float dequantize_value(void* data, size_t idx, uint32_t data_format, int32_t zp, float scale);

        // buffer of inference results
        float* output_fp32;
        int8_t *output_int8;

        
        float input_scale;
        float output_scale;
        int max_batch;
        int batch_size_;
        int det_limit_len_;
        int net_h_;
        int net_w_;
        int out_net_h_;
        int out_net_w_;

        std::vector<float> mean_ = {123.675f, 116.28f, 103.53f};
        std::vector<float> std_ = { 0.01712471615720524f,  0.017506964285714285f,  0.017429155555555555f};

       
        taconn_input_t* input_tensors;
        taconn_buffer_t * output_buffer;
        int input_num;
        int output_num;
        int output_dtype;  // 输出数据类型
        std::vector<taconn_inout_attr_t> ins_attr;
        std::vector<taconn_inout_attr_t> outs_attr;

        int input_width = 480;
        int input_height = 480;
        
        int resize_h;
        int resize_w;
        

};

#endif //!PPOCR_DET_HPP