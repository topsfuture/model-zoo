#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include "ppocr_det.hpp"
#include "ppocr_rec.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// 计算两点间的欧氏距离
double distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * 对图像进行旋转裁剪，保持文本区域正向
 * @param img 输入图像（cv::Mat）
 * @param points 文本区域的四点坐标（顺时针或逆时针排列，4x2）
 * @return 裁剪并校正后的图像
 */
cv::Mat getRotateCropImage(const cv::Mat& img, OCRBox box) {
    // 检查输入点数量是否为4
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f((float)box.x1, (float)box.y1));
    points.push_back(cv::Point2f((float)box.x2, (float)box.y2));
    points.push_back(cv::Point2f((float)box.x3, (float)box.y3));
    points.push_back(cv::Point2f((float)box.x4, (float)box.y4));
    // CV_Assert(points.size() == 4 && "points must contain 4 points");

    // 计算裁剪区域的宽度和高度（取两组对边的最大值）
    double width1 = distance(points[0], points[1]);
    double width2 = distance(points[2], points[3]);
    int crop_width = static_cast<int>(std::max(width1, width2));

    double height1 = distance(points[0], points[3]);
    double height2 = distance(points[1], points[2]);
    int crop_height = static_cast<int>(std::max(height1, height2));

    // 与C++实现对齐，确保最小尺寸为16
    crop_width = std::max(16, crop_width);
    crop_height = std::max(16, crop_height);

    // 定义标准矩形的四个顶点（目标裁剪区域）
    std::vector<cv::Point2f> pts_std;
    pts_std.emplace_back(0, 0);
    pts_std.emplace_back(crop_width, 0);
    pts_std.emplace_back(crop_width, crop_height);
    pts_std.emplace_back(0, crop_height);

    // 计算透视变换矩阵
    cv::Mat M = cv::getPerspectiveTransform(points, pts_std);

    // 执行透视变换（裁剪并校正）
    cv::Mat dst_img;
    cv::warpPerspective(
        img, 
        dst_img, 
        M, 
        cv::Size(crop_width, crop_height), 
        cv::INTER_CUBIC,  // 插值方式
        cv::BORDER_REPLICATE  // 边界填充方式
    );

    // 如果图像高宽比 >= 1.5，旋转90度（适应竖排文本）
    int dst_height = dst_img.rows;
    int dst_width = dst_img.cols;
    if (static_cast<double>(dst_height) / dst_width >= 1.5) {
        cv::rotate(dst_img, dst_img, cv::ROTATE_90_COUNTERCLOCKWISE);  // 逆时针旋转90度
    }

    return dst_img;
}

cv::Mat drawTextDetRes(const OCRBoxVec& dt_boxes, const std::string& img_path) {
    // 读取图像（对应 Python 的 cv2.imread）
    cv::Mat src_im = cv::imread(img_path,cv::IMREAD_COLOR |cv::IMREAD_RETRY_SOFTDEC);
    
    // 检查图像是否加载成功（避免后续操作崩溃）
    if (src_im.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path);
    }

    // 遍历所有检测框（对应 Python 的 for box in dt_boxes）
    for (int i = 0;i < dt_boxes.size();i++) {
        // 检查单个框的坐标数量是否合法（至少 4 个点，即 8 个坐标值）
        
            OCRBox box = dt_boxes[i];
        // 转换为 OpenCV 绘制所需的点集格式（cv::Point2i 向量）
        std::vector<cv::Point2i> points;
        // for (size_t i = 0; i < 4; i ++) {
            int x = box.x1;
            int y = box.y1;
            points.emplace_back(x, y);  // 添加坐标点
            x = box.x2;
            y = box.y2;
            points.emplace_back(x, y);  // 添加坐标点
            x = box.x3;
            y = box.y3;
            points.emplace_back(x, y);  // 添加坐标点
            x = box.x4;
            y = box.y4;
            points.emplace_back(x, y);  // 添加坐标点
        // }

        // 绘制多边形框（对应 Python 的 cv2.polylines）
        // 参数说明：图像、点集、是否闭合、颜色（BGR 格式）、线宽
        cv::polylines(
            src_im,
            points,
            true,  // 闭合多边形
            cv::Scalar(255, 255, 0),  // BGR 颜色：青色
            2      // 线宽
        );
    }

    return src_im;
}
int main(int argc, char** argv){
    std::cout.setf(std::ios::fixed);
    // get params
    const char* keys =
        "{input | ./datasets/cali_set_det | input path, images directory}"
        "{model_det | ./models/ppocr_det_int16.nb | det nb file path}"
        "{model_rec | ./models/ppocr_rec_float16.nb | rec nb file path}"
        "{rec_thresh | 0.5 | recognize threshold}"
        "{labelnames | ./datasets/ppocr_keys_v1.txt | class names file path}"
        "{help | 0 | print help information.}"
        "{use_beam_search | true | beam search trigger}"
        "{beam_size | 5 | beam size, default 5, available 1-40, only valid when using beam search}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    std::string input = parser.get<std::string>("input");
    std::string model_det = parser.get<std::string>("model_det");
    std::string model_rec = parser.get<std::string>("model_rec");
    std::string label_names = parser.get<std::string>("labelnames");
    float rec_thresh = parser.get<float>("rec_thresh");

    bool beam_search = parser.get<bool>("use_beam_search");
    int beam_size = parser.get<int>("beam_size");
    
    if(beam_size < 1 || beam_size > 40){
        std::cout << "ERROR!!beam_size out of range, should be integer in range(1, 41)" << std::endl;
        exit(1);
    }
    // check params
    struct stat info;
    if (stat(model_det.c_str(), &info) != 0) {
        std::cout << "Cannot find valid det model file." << std::endl;
        exit(1);
    }

    if (stat(model_rec.c_str(), &info) != 0) {
        std::cout << "Cannot find valid rec model file." << std::endl;
        exit(1);
    }
    if (stat(label_names.c_str(), &info) != 0) {
        std::cout << "Cannot find labelnames file." << std::endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0) {
        std::cout << "Cannot find input path." << std::endl;
        exit(1);
    }

    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    std::vector<cv::Mat> src_imgs;
    std::vector<cv::Mat> crop_imgs;
    std::vector<std::string> image_names;
    std::vector<OCRBoxVec> batch_boxes;
    std::vector<std::pair<int, int>> batch_ids;
    std::string result_json_path = "results/ppocr_system_results.json";
    // std::vector<std::string> char_charts = ReadDict("ppocr_keys_v1.txt");
    json results_json;
    
    TimeStamp det_ts,rec_ts;

    std::shared_ptr<ta_runtime_context> nnrt_ctx_det = std::make_shared<ta_runtime_context>(0);
    std::shared_ptr<ta_runtime_context> nnrt_ctx_rec = std::make_shared<ta_runtime_context>(0);
    PPOCR_Detector ppocr_det(nnrt_ctx_det,model_det.c_str());
    ppocr_det.enableProfile(&det_ts);
    ppocr_det.Init();

    PPOCR_Rec ppocr_rec(nnrt_ctx_rec,model_rec.c_str(),label_names.c_str());
    ppocr_rec.Init();
    ppocr_rec.enableProfile(&rec_ts);

    int total_frame_num = 0;
    int crop_frame_num = 0;

    if(info.st_mode & S_IFDIR){
        std::vector<std::string> files_vector;
        DIR* pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        while ((ptr = readdir(pDir)) != 0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);

        std::sort(files_vector.begin(), files_vector.end());

        std::vector<OCRBoxVec> batch_boxes;
        std::vector<std::pair<int, int>> batch_ids;
        std::vector<std::pair<std::string, float>> result_list;
        
        for (std::vector<std::string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            std::string img_file = *iter;
            det_ts.start();
            cv::Mat src_image = cv::imread(img_file,cv::IMREAD_COLOR |cv::IMREAD_RETRY_SOFTDEC);
            det_ts.time_accumulation("imread_time");
            
            size_t index = img_file.rfind("/");
            std::string img_name = img_file.substr(index + 1);
            src_imgs.push_back(src_image);
            image_names.push_back(img_name);
            total_frame_num++; 
            if(total_frame_num % 10 == 0){
                std::cout << "Processed frame " << total_frame_num << std::endl;
            }
            ppocr_det.detect_and_save(src_imgs, batch_boxes);
            for(int i = 0; i < image_names.size(); i++){
                std::string output_image_path = image_names[i] + "_det.jpg";

                // cv::Mat img = drawTextDetRes( batch_boxes[i],files_vector[i]);
                // cv::imwrite(output_image_path, img);
                //crop and warp
                for(int j = 0; j < batch_boxes[i].size(); j++){
                    crop_imgs.push_back(getRotateCropImage(src_imgs[i],batch_boxes[i][j]));
                    // output_image_path = image_names[i] + "_crop_" + std::to_string(j) + ".jpg";
                    // cv::imwrite(output_image_path, crop_imgs[j]);
                    crop_frame_num++;
                }
                    
                
            }

            ppocr_rec.rec_and_save(crop_imgs, result_list, beam_search, beam_size);
            
            int index_rec = 0;
            for (int i = 0; i < batch_boxes.size(); i++) {
                
                size_t index = image_names[i].rfind(".");
                std::string striped_name = image_names[i].substr(0, index);
                std::vector<json> ocrinfo_vec;
                
                for (auto& b : batch_boxes[i]) {
                    
                    if (result_list[index_rec].first != "###" && result_list[index_rec].second > rec_thresh) {
                        
                        json ocr_info;
                        ocr_info["illegibility"] = bool(result_list[index_rec].second < rec_thresh);
                        ocr_info["score"] = result_list[index_rec].second;
                        ocr_info["points"] = {{b.x1, b.y1}, {b.x2, b.y2}, {b.x3, b.y3}, {b.x4, b.y4}};
                        ocr_info["transcription"] = result_list[index_rec].first;
                        ocrinfo_vec.push_back(ocr_info);
                    }
                    index_rec++;
                }
                
                results_json[striped_name] = ocrinfo_vec;
            }
        
            image_names.clear();
            src_imgs.clear();
            crop_imgs.clear();
            batch_boxes.clear();
            result_list.clear();
           
        }
            
    }

    ppocr_det.deinit();    
    ppocr_rec.deinit();   

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Processing completed" << std::endl;
    std::cout << "Total images processed: " << total_frame_num << std::endl;
    
    std::cout << "\n===== Time Statistics (Accumulated) =====" << std::endl;
    std::cout << "Image read time:    " << std::fixed << std::setprecision(2) 
              << det_ts.time_map_lab["imread_time"] << " ms (avg: " 
              << det_ts.time_map_lab["imread_time"] / total_frame_num << " ms/image)" << std::endl;
    std::cout << "PPOCR_Det Preprocess time:    " << det_ts.time_map_lab["pre_time"] << " ms (avg: " 
              << det_ts.time_map_lab["pre_time"] / total_frame_num << " ms/image)" << std::endl;
    std::cout << "PPOCR_Det Inference time:     " << det_ts.time_map_lab["infer_time"] << " ms (avg: " 
              << det_ts.time_map_lab["infer_time"] / total_frame_num << " ms/image)" << std::endl;
    std::cout << "PPOCR_Det Postprocess time:   " << det_ts.time_map_lab["post_time"] << " ms (avg: " 
              << det_ts.time_map_lab["post_time"] / total_frame_num << " ms/image)" << std::endl;
    
    std::cout << "==========================================" << std::endl;
    std::cout <<  std::fixed << std::setprecision(2) ;
    std::cout << "PPOCR_Rec Preprocess time:    " << rec_ts.time_map_lab["pre_time"] << " ms (avg: " 
              << rec_ts.time_map_lab["pre_time"] / crop_frame_num << " ms/image)" << std::endl;
    std::cout << "PPOCR_Rec Inference time:     " << rec_ts.time_map_lab["infer_time"] << " ms (avg: " 
              << rec_ts.time_map_lab["infer_time"] / crop_frame_num << " ms/image)" << std::endl;
    std::cout << "PPOCR_Rec Postprocess time:   " << rec_ts.time_map_lab["post_time"] << " ms (avg: " 
              << rec_ts.time_map_lab["post_time"] / crop_frame_num << " ms/image)" << std::endl;
    
    std::cout << "==========================================" << std::endl;
   
    std::string json_file = "results/ppocr_system_results.json";
    std::cout << "result saved in " << json_file << std::endl;
    std::ofstream(json_file) << std::setw(4) << results_json;

    
    return 0;
}