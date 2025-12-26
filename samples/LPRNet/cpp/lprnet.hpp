
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
#include <array>

#define LPRNET_NUM_CLASSES 68
#define LPRNET_TIME_STEPS 18

const std::array<std::string, LPRNET_NUM_CLASSES> CHARS = {
    "京",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "皖",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "I",
    "O",
    "-",
};

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

class LPRNET {

  std::shared_ptr<ta_runtime_context> nnrt_context;
  taconn_input_t* input_tensors;
  taconn_buffer_t* output_buffer;
  int input_num;
  int output_num;
  std::vector<taconn_inout_attr_t> ins_attr;
  std::vector<taconn_inout_attr_t> outs_attr;

  // model info 
  
  const int DEFAULT_IMG_H = 24;
  const int DEFAULT_IMG_W = 94;
  

  std::string model_name;
  // for profiling
  TimeStamp *ts_ = NULL;

  private:

  int preprocess_image(const cv::Mat& src, cv::Mat& dst);
  int post_process(std::vector<cv::Mat> &images, std::vector<std::string> &results);
  size_t calculate_buffer_size(taconn_data_format_t format, size_t element_count) ;
  int pre_process(cv::Mat &images);
  size_t get_element_num(const taconn_inout_attr_t input_attr);
  public:
  
  LPRNET(std::shared_ptr<ta_runtime_context> context,std::string model_file);
  
  void enableProfile(TimeStamp *ts);
 
  bool Init();
//   int get_output_num();
  int Classify(std::vector<cv::Mat>& input_images, std::vector<std::string>& results);
  int deInit();
};



