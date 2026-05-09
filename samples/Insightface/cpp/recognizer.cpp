// recognizer.cpp
#include "insightface.hpp"

// ==================== FaceRecognizer 实现 ====================
FaceRecognizer::FaceRecognizer() : initialized_(false) {
    runtime_ = std::make_shared<Runtime>();
    model_manager_ = std::make_shared<ModelManager>(runtime_);
}

FaceRecognizer::FaceRecognizer(const Config& config) : config_(config), initialized_(false) {
    runtime_ = std::make_shared<Runtime>();
    model_manager_ = std::make_shared<ModelManager>(runtime_);
}

FaceRecognizer::~FaceRecognizer() {
    deinit();
}


int FaceRecognizer::init(const std::string& model_path) {
    if (initialized_) {
        deinit();
    }
    
    config_.model_path = model_path;
    
    Runtime::Config runtime_config;
    runtime_config.core_id = config_.core_id;
    runtime_config.memory_alignment = 256;
    
    int status = model_manager_->init_runtime(model_path, runtime_config);
    if (status != 0) {
        std::cerr << "Failed to initialize FaceRecognizer model" << std::endl;
        return status;
    }
    
    initialized_ = true;
    return 0;
}

void FaceRecognizer::deinit() {
    if (!initialized_) {
        return;
    }
    
    model_manager_->destroy_iomem();
    initialized_ = false;
    
    std::cout << "FaceRecognizer deinitialized" << std::endl;
}

bool FaceRecognizer::preprocess(const cv::Mat& aligned_face) {
    if (aligned_face.cols != config_.input_size.width || 
        aligned_face.rows != config_.input_size.height) {
        std::cerr << "Error: FaceRecognizer expects " << config_.input_size 
                  << " input, got " << aligned_face.cols << "x" << aligned_face.rows << std::endl;
        return true;
    }
    
    cv::Mat blob;
    cv::dnn::blobFromImage(aligned_face, blob, config_.scale, 
                          config_.input_size, config_.mean, config_.swap_rb, false, CV_32F);
    
    if (blob.empty()) {
        return false;
    }
    int quantize_result = model_manager_->quantize_input_tensor(0, blob);
    if (quantize_result != 0) {
        std::cerr << "Recognizer: Failed to quantize input tensor" << std::endl;
        return false;
    }

    return true;
}

bool FaceRecognizer::run_model() {
    int run_result = model_manager_->run_network();

    if (run_result != 0) {
        std::cerr << "Recognizer: Inference failed" << std::endl;
        return false;
    }
    
    return true;
}

std::vector<float> FaceRecognizer::postprocess() {
    cv::Mat output = model_manager_->get_output_data(0);
    if (output.empty()) {
        std::cerr << "Recognizer: Failed to get output data" << std::endl;
        return {};
    }
    
    std::vector<float> feature(output.cols);
    for (int i = 0; i < output.cols; ++i) {
        feature[i] = output.at<float>(0, i);
    }
    
    float norm = 0.0f;
    for (float val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-6) {
        for (float& val : feature) {
            val /= norm;
        }
    }
    
    return feature;
}

std::vector<float> FaceRecognizer::extract_feature(const cv::Mat& aligned_face) {
    if (!initialized_) {
        std::cerr << "FaceRecognizer not initialized" << std::endl;
        return {};
    }
    
    if (aligned_face.empty()) {
        std::cerr << "Error: Input face image is empty" << std::endl;
        return {};
    }
    
    try {
        // 预处理计时
        m_ts->start();
        if (!preprocess(aligned_face)) {
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
        std::vector<float> feature = postprocess();
        m_ts->stop();
        m_ts->time_accumulation("postprocess_time");
        
        total_inferences_++;
        // inference_times_.push_back(timer.time_map_lab["inference_time"]);

        return feature;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during feature extraction: " << e.what() << std::endl;
        return {};
    }
}