cmake_minimum_required(VERSION 3.16)
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
project(pwc_net_ea6530 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TPS_SDK_ROOT "/tps-future" CACHE PATH "TA SDK root")

set(UNIFY   "${TPS_SDK_ROOT}/ta-vsp/ta-unify-9200O")
set(OPENCV  "${TPS_SDK_ROOT}/ta-vsp/ta-opencv")
set(FFMPEG  "${TPS_SDK_ROOT}/ta-vsp/ta-ffmpeg")
set(CV_BASE "${TPS_SDK_ROOT}/ta-base/ta-cv")
set(RUNTIME "${TPS_SDK_ROOT}/ta-base/ta-runtime")
set(VIPLITE "${TPS_SDK_ROOT}/ta-vsp/ta-viplite-9200o")
set(LIBDEC  "${TPS_SDK_ROOT}/ta-vsp/ta-libdec")
set(SYSSDK "${TPS_SDK_ROOT}/ta-vsp/ta-sys")
set(LIBENC  "${TPS_SDK_ROOT}/ta-vsp/ta-libenc")
set(DMABUF  "${TPS_SDK_ROOT}/ta-vsp/ta-libdmabufheap")
set(UMAP    "${TPS_SDK_ROOT}/ta-vsp/ta-libumap")

include_directories(
    "${UNIFY}/include" "${UNIFY}/include/Vivante_ML_Toolkit_OVXLIB_dev"
    "${OPENCV}/include" "${OPENCV}/include/opencv4"
    "${FFMPEG}/include" "${CV_BASE}/include"
    "${RUNTIME}/include" "${VIPLITE}/include"
    "${LIBDEC}/include" "${SYSSDK}/include")

link_directories(
    "${UNIFY}/lib" "${VIPLITE}/lib" "${OPENCV}/lib"
    "${FFMPEG}/lib" "${CV_BASE}/lib" "${RUNTIME}/lib"
    "${LIBDEC}/lib" "${SYSSDK}/lib" "${LIBENC}/lib"
    "${DMABUF}/lib" "${UMAP}/lib")

set(COMMON_FLAGS -O2 -fPIC -Wall -Wextra)

# Common deps shared by all targets:
#   - tacocv: ta_cv_* functions (needed by libopencv_core/imgcodecs)
#   - taco_jpegdec/taco_ppvdec: transitive dep of tacocv
#   - dmabufheap: transitive dep of tacocv
#   - avutil: transitive dep of libopencv_core (ffmpeg)
#   - yuv: needed by libopencv_core
#   - jpeg: needed by libyuv
set(OPENCV_BASE
    enc24 umap
    tacocv taco_jpegdec taco_ppvdec dmabufheap
    avutil yuv jpeg
    pthread dl m)

# 1. pwc_net_video: video -> of_speed (OpenCV VideoCapture + NBG)
add_executable(pwc_net_video pwc_net_video.cpp)
target_compile_options(pwc_net_video PRIVATE ${COMMON_FLAGS})
target_link_libraries(pwc_net_video PRIVATE
    opencv_video opencv_videoio opencv_dnn opencv_calib3d opencv_features2d opencv_flann
    opencv_imgproc opencv_core opencv_imgcodecs
    VIPhal NBGlinker ta_runtime tacosys enc24 umap
    ${OPENCV_BASE})

# 2. pwc_net_imgpair: two images -> .flo (NBG)
add_executable(pwc_net_imgpair pwc_net_imgpair.cpp)
target_compile_options(pwc_net_imgpair PRIVATE ${COMMON_FLAGS})
target_link_libraries(pwc_net_imgpair PRIVATE
    opencv_imgproc opencv_core opencv_imgcodecs
    VIPhal NBGlinker ta_runtime
    ${OPENCV_BASE})

# 3. farneback_imgpair: two images -> .flo (Farneback)
add_executable(farneback_imgpair farneback_imgpair.cpp)
target_compile_options(farneback_imgpair PRIVATE ${COMMON_FLAGS})
target_link_libraries(farneback_imgpair PRIVATE
    opencv_video opencv_videoio opencv_dnn opencv_calib3d opencv_features2d opencv_flann
    opencv_imgproc opencv_core opencv_imgcodecs
    ${OPENCV_BASE})

# 4. pwc_net_eval: predicted.flo + ground_truth.flo -> metrics
add_executable(pwc_net_eval pwc_net_eval.cpp)
target_compile_options(pwc_net_eval PRIVATE ${COMMON_FLAGS})
target_link_libraries(pwc_net_eval PRIVATE
    opencv_imgproc opencv_core opencv_imgcodecs
    ${OPENCV_BASE})
