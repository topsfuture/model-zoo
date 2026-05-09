// alignment.cpp
#include "insightface.hpp"

cv::Mat umeyama_similar_transform(const std::vector<cv::Point2f>& src_points, 
                                 const std::vector<cv::Point2f>& dst_points) {
    if (src_points.size() != dst_points.size() || src_points.empty()) {
        return cv::Mat();
    }
    
    int num_points = src_points.size();
    
    cv::Point2f src_mean(0, 0), dst_mean(0, 0);
    for (int i = 0; i < num_points; ++i) {
        src_mean += src_points[i];
        dst_mean += dst_points[i];
    }
    src_mean /= num_points;
    dst_mean /= num_points;
    
    cv::Mat src_centered(num_points, 2, CV_32F);
    cv::Mat dst_centered(num_points, 2, CV_32F);
    
    for (int i = 0; i < num_points; ++i) {
        src_centered.at<float>(i, 0) = src_points[i].x - src_mean.x;
        src_centered.at<float>(i, 1) = src_points[i].y - src_mean.y;
        
        dst_centered.at<float>(i, 0) = dst_points[i].x - dst_mean.x;
        dst_centered.at<float>(i, 1) = dst_points[i].y - dst_mean.y;
    }
    
    cv::Mat src_cov = src_centered.t() * src_centered;
    cv::Mat dst_cov = dst_centered.t() * dst_centered;
    
    float src_scale = std::sqrt(cv::trace(src_cov)[0] / num_points);
    float dst_scale = std::sqrt(cv::trace(dst_cov)[0] / num_points);
    float scale = dst_scale / src_scale;
    
    cv::Mat A = dst_centered.t() * src_centered;
    
    cv::Mat W, U, Vt;
    cv::SVD::compute(A, W, U, Vt, cv::SVD::FULL_UV);
    
    cv::Mat R = U * Vt;
    
    float det = cv::determinant(R);
    if (det < 0) {
        Vt.at<float>(1, 0) *= -1;
        Vt.at<float>(1, 1) *= -1;
        R = U * Vt;
    }
    
    cv::Mat transform = cv::Mat::eye(2, 3, CV_32F);
    
    R.copyTo(transform(cv::Rect(0, 0, 2, 2)));
    transform.at<float>(0, 0) *= scale;
    transform.at<float>(0, 1) *= scale;
    transform.at<float>(1, 0) *= scale;
    transform.at<float>(1, 1) *= scale;
    
    cv::Mat src_mean_mat = (cv::Mat_<float>(2, 1) << src_mean.x, src_mean.y);
    cv::Mat dst_mean_mat = (cv::Mat_<float>(2, 1) << dst_mean.x, dst_mean.y);
    cv::Mat translation = dst_mean_mat - transform(cv::Rect(0, 0, 2, 2)) * src_mean_mat;
    
    transform.at<float>(0, 2) = translation.at<float>(0, 0);
    transform.at<float>(1, 2) = translation.at<float>(1, 0);
    
    return transform;
}

cv::Mat align_face(const cv::Mat& image, const Landmarks& landmarks) {
    std::vector<cv::Point2f> target_points = {
        cv::Point2f(38.2946f, 51.6963f),
        cv::Point2f(73.5318f, 51.5014f),
        cv::Point2f(56.0252f, 71.7366f),
        cv::Point2f(41.5493f, 92.3655f),
        cv::Point2f(70.7299f, 92.2041f)
    };

    std::vector<cv::Point2f> src_points;
    for (int i = 0; i < 5; ++i) {
        src_points.push_back(landmarks.points[i]);
    }

    cv::Mat transform = umeyama_similar_transform(src_points, target_points);
    if (transform.empty()) {
        std::cerr << "Warning: Failed to compute similarity transform for face alignment" << std::endl;
        return cv::Mat();
    }

    cv::Mat aligned_face;
    cv::warpAffine(image, aligned_face, transform, cv::Size(112, 112), 
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    return aligned_face;
}