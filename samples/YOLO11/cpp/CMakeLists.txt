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

# include_directories(./soc240401_riscv64/include/ta_op)
# include_directories(./soc240401_riscv64/include/Vivante_ML_Toolkit_OVXLIB_dev)
include_directories(/tps-sdk-for-mes20/ta-vsp/ta-samples/pipeworks/tademo/soc240401_riscv64/include/opencv4)
include_directories(/tps-sdk-for-mes20/ta-base/ta-runtime/include)

# 添加头文件搜索路径
include_directories(
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/ovxlib-package-dev
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/Vivante_ML_Toolkit_OVXLIB_dev
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/
	/tps-sdk-for-mes20/ta-base/ta-runtime/include
    "${PROJECT_ROOT}/include"
    /tps-sdk-for-mes20/ta-base
    /tps-sdk-for-mes20/ta-vsp/ta-opencv/include/opencv4/
    /tps-sdk-for-mes20/ta-base/ta-cv/include/
    /tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/include/ta_op/
    /tps-sdk-for-mes20/ta-vsp/ta-sys/include/
    /tps-sdk-for-mes20/ta-vsp/ta-ffmpeg/include
    /tps-sdk-for-mes20/ta-vsp/others
)

# 添加库文件搜索路径
# link_directories(./soc240401_riscv64/lib)
link_directories(
	/tps-sdk-for-mes20/ta-vsp/ta-unify-9200O/lib
    /tps-sdk-for-mes20/ta-vsp/ta-viplite-9200o/lib
	/tps-sdk-for-mes20/ta-base/ta-runtime/lib
    /tps-sdk-for-mes20/ta-vsp/ta-opencv/lib/
    /tps-sdk-for-mes20/ta-base/ta-cv/lib/
    /tps-sdk-for-mes20/ta-vsp/ta-sys/lib/
    /tps-sdk-for-mes20/ta-vsp/ta-libenc/lib/
    /tps-sdk-for-mes20/ta-vsp/ta-libdec/lib/
    /tps-sdk-for-mes20/ta-base/ta-cv/lib/
)

set(VIPLITE_LIBS VIPhal NBGlinker)
set(OPENCV_LIBS opencv_imgproc opencv_core opencv_imgcodecs opencv_dnn)

set(SRC_yolo11_main	
    src/main.cpp
    src/yolo11.cpp)	
set(EXE_yolo11_main yolo11s_det_soc)
add_executable(${EXE_yolo11_main} ${SRC_yolo11_main})
target_link_libraries(${EXE_yolo11_main} ${OPENCV_LIBS}  ${VIPLITE_LIBS} -ldl -fprofile-arcs -lgcov -lpthread -lta_runtime 
-ltacocv -ltacosys -ltaco_jpegdec -lenc24 -ltaco_ppvdec -lyuv -ljpeg)




