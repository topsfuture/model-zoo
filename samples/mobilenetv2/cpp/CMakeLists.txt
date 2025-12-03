cmake_minimum_required(VERSION 3.10)
project(mobilenetv2)
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

# include_directories(./soc240401_riscv64/include/ta_op)
# include_directories(./soc240401_riscv64/include/Vivante_ML_Toolkit_OVXLIB_dev)
include_directories(/tps-sdk-for-mes20/ta-base/ta-runtime/include)

# 添加头文件搜索路径
include_directories(
    /tps-sdk-for-mes20/ta-vsp/ta-opencv/include
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/ta_op
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/ovxlib-package-dev
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/Vivante_ML_Toolkit_OVXLIB_dev
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/
    /tps-sdk-for-mes20/ta-base/ta-cv/include
    /tps-sdk-for-mes20/ta-vsp/ta-opencv/include/opencv4/
    /tps-sdk-for-mes20/ta-vsp/ta-sys/include
    /tps-sdk-for-mes20/ta-vsp/ta-libdec/include
    /tps-sdk-for-mes20/ta-vsp/ta-ffmpeg/include
    /tps-sdk-for-mes20/ta-vsp/ta-libenc/include
    /tps-sdk-for-mes20/ta-vsp/others/
)


link_directories(
	/tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/lib
    /tps-sdk-for-mes20/ta-vsp/ta-viplite-9200o/lib
	/tps-sdk-for-mes20/ta-base/ta-runtime/lib
    /tps-sdk-for-mes20/ta-vsp/ta-opencv/lib
    /tps-sdk-for-mes20/ta-base/ta-cv/lib
    /tps-sdk-for-mes20/ta-vsp/ta-libdec/lib
    /tps-sdk-for-mes20/ta-vsp/ta-sys/lib
    /tps-sdk-for-mes20/ta-vsp/ta-ffmpeg/lib
    /tps-sdk-for-mes20/ta-vsp/ta-libenc/lib
)

set(VIPLITE_LIBS VIPhal NBGlinker)
set(OPENCV_LIBS opencv_imgproc opencv_core opencv_imgcodecs opencv_dnn)

set(OTHER_LIBS tacocv vx taco_jpegdec taco_ppvdec CLC OpenVX OpenVXU OpenCL SPIRV_viv CLC VSC GAL ovxlib ArchModelSw NNArchPerf  pthread tacosys
        opencv_imgcodecs opencv_videoio opencv_core opencv_imgproc enc24 yuv jpeg)

set(SRC_mobilenetv2_det
    mobilenetv2.cpp
    main.cpp)
set(EXE_mobilenetv2_det mobilenetv2_det_soc)
add_executable(${EXE_mobilenetv2_det} ${SRC_mobilenetv2_det})
target_link_libraries(${EXE_mobilenetv2_det} ${OTHER_LIBS}  ${VIPLITE_LIBS} -ldl -fprofile-arcs -lgcov -lpthread -lta_runtime)


