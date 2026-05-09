// detector.cpp
#include "insightface.hpp"
#include <numeric> 
// ==================== SCRFD 实现 ====================
SCRFD::SCRFD() : initialized_(false) {
    runtime_ = std::make_shared<Runtime>();
    model_manager_ = std::make_shared<ModelManager>(runtime_);
}

SCRFD::SCRFD(const Config& config) : config_(config), initialized_(false) {
    runtime_ = std::make_shared<Runtime>();
    model_manager_ = std::make_shared<ModelManager>(runtime_);
}

SCRFD::~SCRFD() {
    deinit();
}

int SCRFD::init(const std::string& model_path) {
    if (initialized_) {
        deinit();
    }
    
    config_.model_path = model_path;
    
    Runtime::Config runtime_config;
    runtime_config.core_id = config_.core_id;
    runtime_config.memory_alignment = config_.memory_alignment;
    
    std::cout << "Initializing SCRFD with model: " << model_path << std::endl;
    int status = model_manager_->init_runtime(model_path, runtime_config);
    if (status != 0) {
        std::cerr << "Failed to initialize SCRFD model" << std::endl;
        return status;
    }
    
    auto& model_info = model_manager_->get_model_info();
    if (model_info.model_input_width > 0 && model_info.model_input_height > 0) {
        config_.target_size = cv::Size(model_info.model_input_width, model_info.model_input_height);
    }
    
    initialized_ = true;
    return 0;
}

void SCRFD::deinit() {
    if (!initialized_) {
        return;
    }
    
    model_manager_->destroy_iomem();
    initialized_ = false;
    
    std::cout << "SCRFD deinitialized" << std::endl;
}

bool SCRFD::preprocess(const cv::Mat& image, float& det_scale) {
    int input_width = config_.target_size.width;
    int input_height = config_.target_size.height;
    
    float im_ratio = static_cast<float>(image.rows) / image.cols;
    float model_ratio = static_cast<float>(input_height) / input_width;
    
    int new_width, new_height;
    
    if (fabs(im_ratio - model_ratio) < 0.01) {
        new_width = input_width;
        new_height = input_height;
    } else if (im_ratio > model_ratio) {
        new_height = input_height;
        new_width = static_cast<int>(new_height / im_ratio);
    } else {
        new_width = input_width;
        new_height = static_cast<int>(new_width * im_ratio);
    }
    
    det_scale = static_cast<float>(new_height) / image.rows;
    
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(new_width, new_height));
    
    cv::Mat det_img;
    if (new_width == input_width && new_height == input_height) {
        det_img = resized_img;
    } else {
        det_img = cv::Mat::zeros(input_height, input_width, CV_8UC3);
        cv::Mat roi = det_img(cv::Rect(0, 0, new_width, new_height));
        resized_img.copyTo(roi);
    }
    
    cv::Mat blob;
    try {
        cv::dnn::blobFromImage(det_img, blob, config_.scale, 
                               cv::Size(input_width, input_height),
                               config_.mean, config_.swap_rb, false, CV_32F);
    } catch (const cv::Exception& e) {
        std::cerr << "  Preprocessing - OpenCV exception: " << e.what() << std::endl;
        std::cerr << "  Preprocessing - det_img size: " << det_img.cols << "x" << det_img.rows 
                  << ", type: " << det_img.type() << std::endl;
        throw;
    }
    int quantize_result = model_manager_->quantize_input_tensor(0, blob);
    if (quantize_result != 0) {
        std::cerr << "Failed to quantize input tensor, error code: " << quantize_result << std::endl;
        return false;
        // exit(1);
    }

    return true;
}

bool SCRFD::run_model() {
    int run_result = model_manager_->run_network();
    if (run_result != 0) {
        std::cerr << "Inference failed, error code: " << run_result << std::endl;
        return false;
    }
    
    return true;
}

std::vector<cv::Point2f> SCRFD::generate_anchors(int feat_height, int feat_width, int stride) {
    std::vector<cv::Point2f> anchors;
    anchors.reserve(feat_height * feat_width);
    
    for (int i = 0; i < feat_height; ++i) {
        for (int j = 0; j < feat_width; ++j) {
            float cx = (j + 0.5f) * stride;
            float cy = (i + 0.5f) * stride;
            anchors.emplace_back(cx, cy);
        }
    }
    
    return anchors;
}

BBox SCRFD::decode_bbox(const cv::Point2f& anchor, const float* bbox_delta, float stride) {
    if (!bbox_delta) {
        std::cerr << "Error: decode_bbox received null pointer" << std::endl;
        return BBox();
    }
    
    BBox bbox;
    bbox.x1 = anchor.x - bbox_delta[0];
    bbox.y1 = anchor.y - bbox_delta[1];
    bbox.x2 = anchor.x + bbox_delta[2];
    bbox.y2 = anchor.y + bbox_delta[3];
    
    if (bbox.x1 > bbox.x2) std::swap(bbox.x1, bbox.x2);
    if (bbox.y1 > bbox.y2) std::swap(bbox.y1, bbox.y2);
    
    return bbox;
}

Landmarks SCRFD::decode_landmarks(const cv::Point2f& anchor, const float* kps_delta, float stride) {
    Landmarks landmarks;
    for (int i = 0; i < 5; ++i) {
        landmarks.points[i].x = anchor.x + kps_delta[i * 2];
        landmarks.points[i].y = anchor.y + kps_delta[i * 2 + 1];
    }
    return landmarks;
}

float SCRFD::calculate_iou(const BBox& bbox1, const BBox& bbox2) {
    float x1 = std::max(bbox1.x1, bbox2.x1);
    float y1 = std::max(bbox1.y1, bbox2.y1);
    float x2 = std::min(bbox1.x2, bbox2.x2);
    float y2 = std::min(bbox1.y2, bbox2.y2);
    
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }
    
    float inter_area = (x2 - x1) * (y2 - y1);
    float area1 = bbox1.area();
    float area2 = bbox2.area();
    float union_area = area1 + area2 - inter_area;
    
    return inter_area / union_area;
}

std::vector<int> SCRFD::nms(const std::vector<DetectionResult>& detections, float iou_threshold) {
    if (detections.empty()) {
        return {};
    }
    
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&detections](int a, int b) { 
                  return detections[a].score > detections[b].score; 
              });
    
    std::vector<int> keep;
    
    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);
        
        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            float iou = calculate_iou(detections[current].bbox, detections[idx].bbox);
            
            if (iou <= iou_threshold) {
                remaining.push_back(idx);
            }
        }
        
        indices = remaining;
    }
    
    return keep;
}

std::vector<DetectionResult> SCRFD::postprocess(const cv::Mat& image, float det_scale) {
    std::vector<DetectionResult> all_detections;
    
    std::vector<int> strides = {8, 16, 32};
    std::vector<int> feat_sizes = {80, 40, 20};
    
    for (int scale_idx = 0; scale_idx < 3; ++scale_idx) {
        int stride = strides[scale_idx];
        int feat_size = feat_sizes[scale_idx];
        int total_anchors = feat_size * feat_size;
        
        cv::Mat cls_output = model_manager_->get_output_data(scale_idx);
        cv::Mat bbox_output = model_manager_->get_output_data(scale_idx + 3);
        cv::Mat kps_output = model_manager_->get_output_data(scale_idx + 6);
        
        if (cls_output.empty() || bbox_output.empty() || kps_output.empty()) {
            continue;
        }
        
        auto anchors = generate_anchors(feat_size, feat_size, stride);
        
        std::vector<int> high_score_indices;
        for (int i = 0; i < cls_output.cols; i += 2) {
            float face_score = cls_output.at<float>(0, i);
            if (face_score > config_.score_threshold) {
                high_score_indices.push_back(i / 2);
            }
        }
        
        for (int anchor_idx : high_score_indices) {
            float face_score = cls_output.at<float>(0, anchor_idx * 2);
            
            DetectionResult det;
            det.score = face_score;
            
            int bbox_offset = anchor_idx * 8;
            if (bbox_offset + 3 >= bbox_output.cols) {
                continue;
            }
            
            float bbox_delta[4];
            for (int k = 0; k < 4; ++k) {
                bbox_delta[k] = bbox_output.at<float>(0, bbox_offset + k) * stride;
            }
            
            int kps_offset = anchor_idx * 20;
            if (kps_offset + 9 >= kps_output.cols) {
                continue;
            }
            
            float kps_delta[10];
            for (int k = 0; k < 10; ++k) {
                kps_delta[k] = kps_output.at<float>(0, kps_offset + k) * stride;
            }
            
            det.bbox = decode_bbox(anchors[anchor_idx], bbox_delta, stride);
            det.landmarks = decode_landmarks(anchors[anchor_idx], kps_delta, stride);
            
            float inv_scale = 1.0f / det_scale;
            det.bbox.x1 *= inv_scale;
            det.bbox.y1 *= inv_scale;
            det.bbox.x2 *= inv_scale;
            det.bbox.y2 *= inv_scale;
            
            for (int k = 0; k < 5; ++k) {
                det.landmarks.points[k].x *= inv_scale;
                det.landmarks.points[k].y *= inv_scale;
            }
            
            det.bbox.x1 = std::max(0.0f, std::min(det.bbox.x1, static_cast<float>(image.cols - 1)));
            det.bbox.y1 = std::max(0.0f, std::min(det.bbox.y1, static_cast<float>(image.rows - 1)));
            det.bbox.x2 = std::max(0.0f, std::min(det.bbox.x2, static_cast<float>(image.cols - 1)));
            det.bbox.y2 = std::max(0.0f, std::min(det.bbox.y2, static_cast<float>(image.rows - 1)));
            
            all_detections.push_back(det);
        }
    }
    
    std::vector<int> keep_indices = nms(all_detections, config_.nms_threshold);
    std::vector<DetectionResult> final_detections;
    
    for (int idx : keep_indices) {
        final_detections.push_back(all_detections[idx]);
    }
    
    return final_detections;
}


std::vector<DetectionResult> SCRFD::detect(const cv::Mat& image) {
    if (!initialized_ || image.empty()) {
        std::cerr << "SCRFD not initialized or empty input image" << std::endl;
        return {};
    }
    
    try {
        float det_scale = 1.0f;
        
        // 预处理计时
        m_ts->start();
        if (!preprocess(image, det_scale)){
            return {};
        }

        m_ts->stop();
        m_ts->time_accumulation("preprocess_time");

        
        // 推理计时
        m_ts->start();
        if (!run_model()) {
            return {};
        }
        m_ts->stop();
        m_ts->time_accumulation("inference_time");
        
        // 后处理计时
        m_ts->start();
        auto results = postprocess(image, det_scale);
        m_ts->stop();
        m_ts->time_accumulation("postprocess_time");
        
        total_inferences_ ++;
        // inference_times_.push_back(m_ts->time_map_lab["inference_time"]);
    
        return results;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during detection: " << e.what() << std::endl;
        return {};
    }
}




cv::Mat SCRFD::draw(const cv::Mat& image, const std::vector<DetectionResult>& results) {
    cv::Mat display_image = image.clone();
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& det = results[i];
        
        cv::rectangle(display_image, 
                     cv::Point(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1)),
                     cv::Point(static_cast<int>(det.bbox.x2), static_cast<int>(det.bbox.y2)),
                     cv::Scalar(0, 255, 0), 2);
        
        std::string score_text = std::to_string(det.score).substr(0, 4);
        cv::putText(display_image, score_text,
                   cv::Point(static_cast<int>(det.bbox.x1), static_cast<int>(det.bbox.y1) - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        for (int j = 0; j < 5; ++j) {
            cv::circle(display_image, 
                      cv::Point(static_cast<int>(det.landmarks.points[j].x), 
                               static_cast<int>(det.landmarks.points[j].y)),
                      2, cv::Scalar(0, 0, 255), -1);
        }
    }
    
    return display_image;
}