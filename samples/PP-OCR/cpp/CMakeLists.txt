cmake_minimum_required(VERSION 3.10)
project(tademo)
set(CMAKE_CXX_STANDARD 17)

# set default
set(TARGET_ARCH "soc" CACHE STRING "Target architecture")

# Print the options to confirm.

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
message("PROJECT_ROOT = ${PROJECT_ROOT}")

add_compile_options(-fPIC -O3)
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
set(CMAKE_ASM_COMPILER riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)

set(TPS_SDK_ROOT "/tps-future" CACHE STRING "")
message(STATUS "TPS_SDK_ROOT: ${TPS_SDK_ROOT}")

# 添加头文件搜索路径
include_directories(
    ${TPS_SDK_ROOT}/ta-vsp/ta-opencv/include
    ${TPS_SDK_ROOT}/ta-vsp/ta-unify-9200O/include/ta_op
    ${TPS_SDK_ROOT}/ta-vsp/ta-unify-9200O/include/ovxlib-package-dev
    ${TPS_SDK_ROOT}/ta-vsp/ta-unify-9200O/include/Vivante_ML_Toolkit_OVXLIB_dev
    ${TPS_SDK_ROOT}/ta-vsp/ta-unify-9200O/include/
    ${TPS_SDK_ROOT}/ta-base/ta-cv/include
    ${TPS_SDK_ROOT}/ta-vsp/ta-opencv/include/opencv4/
    ${TPS_SDK_ROOT}/ta-vsp/ta-sys/include
    ${TPS_SDK_ROOT}/ta-vsp/ta-libdec/include
    ${TPS_SDK_ROOT}/ta-vsp/ta-ffmpeg/include
    ${TPS_SDK_ROOT}/ta-vsp/ta-libenc/include
    ${TPS_SDK_ROOT}/ta-base/ta-runtime/include
    ${TPS_SDK_ROOT}/ta-vsp/others/
    "${PROJECT_ROOT}/include"
)

link_directories(
	${TPS_SDK_ROOT}/ta-vsp/ta-unify-9200O/lib
    ${TPS_SDK_ROOT}/ta-vsp/ta-viplite-9200o/lib
	${TPS_SDK_ROOT}/ta-base/ta-runtime/lib
    ${TPS_SDK_ROOT}/ta-vsp/ta-opencv/lib
    ${TPS_SDK_ROOT}/ta-base/ta-cv/lib
    ${TPS_SDK_ROOT}/ta-vsp/ta-libdec/lib
    ${TPS_SDK_ROOT}/ta-vsp/ta-sys/lib
    ${TPS_SDK_ROOT}/ta-vsp/ta-ffmpeg/lib
    ${TPS_SDK_ROOT}/ta-vsp/ta-libenc/lib
)

set(VIPLITE_LIBS VIPhal NBGlinker)
set(OPENCV_LIBS opencv_imgproc opencv_core opencv_imgcodecs opencv_dnn)

set(SRC_ppocr_main	
    src/main.cpp
    src/ppocr_det.cpp
    src/postprocess.cpp	
    src/clipper.cpp
    src/ppocr_rec.cpp)
set(EXE_ppocr_main ppocr_system)
add_executable(${EXE_ppocr_main} ${SRC_ppocr_main})
target_link_libraries(${EXE_ppocr_main} ${OPENCV_LIBS}  ${VIPLITE_LIBS} -ldl -fprofile-arcs -lgcov -lpthread -lta_runtime 
-ltacocv -ltacosys -ltaco_jpegdec -lenc24 -ltaco_ppvdec -lyuv -ljpeg)




