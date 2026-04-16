cmake_minimum_required(VERSION 3.10)
project(segformer)
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

set(OPENCV_LIBS opencv_imgproc opencv_core opencv_imgcodecs opencv_dnn opencv_videoio)
set(FFMPEG_LIBS avcodec avformat avutil swscale swresample)
set(OTHER_LIBS tacocv taco_jpegdec taco_ppvdec tacosys enc24 yuv jpeg)


set(SRC_segformer
    segformer.cpp
    main.cpp)
set(EXE_segformer segformer_soc)
add_executable(${EXE_segformer} ${SRC_segformer})

find_library(FOUND_AVCODEC avcodec
    PATHS ${TPS_SDK_ROOT}/ta-vsp/ta-ffmpeg/lib
    NO_DEFAULT_PATH
)

if(FOUND_AVCODEC)
    message(STATUS "FFmpeg Lib found, compile with FFmpeg support")
    set(OPENCV_LIBS ${OPENCV_LIBS} opencv_videoio)
    
    set(FFMPEG_LIBS avcodec avformat avutil swscale swresample)
    
    add_definitions(-DHAVE_FFMPEG=1)
    
    find_library(FOUND_AVFORMAT avformat
        PATHS ${TPS_SDK_ROOT}/ta-vsp/ta-ffmpeg/lib
        NO_DEFAULT_PATH
    )
    
    if(NOT FOUND_AVFORMAT)
        message(WARNING "find avcodec, but no avformat , FFmpeg may not support completely")
    endif()
else()
    message(STATUS "FFmpeg not found")
    
    set(FFMPEG_LIBS "")
    
    add_definitions(-DHAVE_FFMPEG=0)
endif()


function(check_and_link_library TARGET_NAME LIB_NAME CHECK_TARGET LIB_PATHS COMPILE_DEFINE)
    message(STATUS "Checking if ${CHECK_TARGET} needs ${LIB_NAME}")
    
    execute_process(
        COMMAND sh -c "readelf -d ${CHECK_TARGET} | grep ${LIB_NAME}"
        RESULT_VARIABLE NEEDS_LIB
        OUTPUT_QUIET
        ERROR_QUIET
    )
    
    if(NEEDS_LIB EQUAL 0)
        message(STATUS "${CHECK_TARGET} needs ${LIB_NAME}")
        string(REPLACE "." "_" LIB_VAR_NAME ${LIB_NAME})
        set(FIND_LIB_VAR "FOUND_${LIB_VAR_NAME}_LIB")
        
        find_library(${FIND_LIB_VAR}
            NAMES ${LIB_NAME}
            PATHS ${LIB_PATHS}
            NO_DEFAULT_PATH
        )

        if(${FIND_LIB_VAR})
            message(STATUS "Found ${LIB_NAME}: ${${FIND_LIB_VAR}}")
            target_link_libraries(${TARGET_NAME} ${${FIND_LIB_VAR}})
            target_compile_definitions(${TARGET_NAME} PRIVATE "${COMPILE_DEFINE}=1")
            set(${COMPILE_DEFINE}_FOUND TRUE PARENT_SCOPE)
        else()
            message(WARNING "${LIB_NAME} not found in specified paths, but ${CHECK_TARGET} needs it")
            target_compile_definitions(${TARGET_NAME} PRIVATE "${COMPILE_DEFINE}=0")
            set(${COMPILE_DEFINE}_FOUND FALSE PARENT_SCOPE)
        endif()
    else()
        message(STATUS "${CHECK_TARGET} does not need ${LIB_NAME}")
        target_compile_definitions(${TARGET_NAME} PRIVATE "${COMPILE_DEFINE}=0")
        set(${COMPILE_DEFINE}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

# 设置要检查的目标库
set(TACO_JPEGDEC_LIB "${TPS_SDK_ROOT}/ta-vsp/ta-libdec/lib/libtaco_jpegdec.so")
set(TACOCV_LIB "${TPS_SDK_ROOT}/ta-base/ta-cv/lib/libtacocv.so")

# 检查 libumap.so
check_and_link_library(
    ${EXE_segformer}  
    "libumap.so"
    ${TACO_JPEGDEC_LIB}
    "${TPS_SDK_ROOT}/ta-base/ta-libumap/lib;${TPS_SDK_ROOT}/ta-vsp/ta-libumap/lib"
    "TPS_HAVE_UMAP"
)

# 检查 libdmabufheap.so
check_and_link_library(
    ${EXE_segformer}  
    "libdmabufheap.so"
    ${TACOCV_LIB}
    "${TPS_SDK_ROOT}/ta-vsp/ta-libdmabufheap/lib;${TPS_SDK_ROOT}/ta-base/ta-libdmabufheap/lib"
    "TPS_HAVE_DMABUFHEAP"
)
target_link_libraries(${EXE_segformer}  
    ${OTHER_LIBS}
    ${OPENCV_LIBS}
    ${FFMPEG_LIBS}
    ${VIPLITE_LIBS}
    -ldl
    -fprofile-arcs
    -lgcov
    -lpthread
    -lta_runtime
)


