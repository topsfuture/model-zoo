#ifndef PPOCR_REC_HPP
#define PPOCR_REC_HPP

#include <iostream>
#include <vector>
#include <string>
#include <numeric> 

#include "opencv2/opencv.hpp"

#include <fstream>
#include "ppocr_det.hpp"



struct Beam {
    std::vector<int> prefix;       // 字符索引序列
    double score;                  // 累计得分（概率乘积）
    std::vector<float> confs;      // 每个字符的置信度
};

class PPOCR_Rec
{   
    std::shared_ptr<ta_runtime_context> nnrt_context;
    std::string model_path;
    TimeStamp* m_ts;
    public:
        PPOCR_Rec(std::shared_ptr<ta_runtime_context> context,std::string m_path,std::string label_path);
        virtual ~PPOCR_Rec();
        bool Init();
        void deinit();
        bool run();
        void rec_and_save(std::vector<cv::Mat> input, std::vector<std::pair<std::string, float>> &result_list, bool beam_search, int beam_size);
        void postprocess(std::vector<std::pair<std::string, float>> &result_list, bool beam_searc, int beam_width);
        void preprocess(cv::Mat &input,cv::Mat &output_image);
   
        std::vector<std::string> ReadDict(std::string &path);
        std::vector<std::string> char_charts;
        void enableProfile(TimeStamp *ts){
            m_ts = ts;
        };
        std::vector<float> mean_ = {127.5f, 127.5f, 127.5f};
        std::vector<float> std_ = { 0.007874016f,  0.007874016f,  0.007874016f};
    private:
        bool initialized_;
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
        
        int input_height; //ppocr network stage shapes.
        int input_width;
        float img_ratio; //ppocr network stage ratios, ratio = w / h
        int resized_w;
        int resized_h;
        

        taconn_input_t* input_tensors;
        taconn_buffer_t * output_buffer;
        int input_num;
        int output_num;
        int output_dtype;  // 输出数据类型
        std::vector<taconn_inout_attr_t> ins_attr;
        std::vector<taconn_inout_attr_t> outs_attr;

};



#endif //!PPOCR_REC_HPP