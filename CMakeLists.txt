cmake_minimum_required(VERSION 3.16)
project(video_pipeline_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

# =============================================================================
# 平台选择 (编译时指�?
# =============================================================================
# 使用方法:
#   - Rockchip 平台：cmake -DPLATFORM=rockchip ..
#   - NVIDIA 平台�? cmake -DPLATFORM=nvidia ..
#   - 自动检测：    cmake .. (默认)
# =============================================================================

set(PLATFORM "auto" CACHE STRING "Target platform: auto, rockchip, nvidia")
set_property(CACHE PLATFORM PROPERTY STRINGS auto rockchip nvidia)

if(PLATFORM STREQUAL "rockchip")
    set(ROCKCHIP_PLATFORM ON)
    add_definitions(-DROCKCHIP_PLATFORM)
    message(STATUS "Building for Rockchip platform (MPP + RGA + RKNN)")
elseif(PLATFORM STREQUAL "nvidia")
    set(NVIDIA_PLATFORM ON)
    add_definitions(-DNVIDIA_PLATFORM)
    message(STATUS "Building for NVIDIA platform (NVDEC + CUDA + TensorRT)")
else()
    message(STATUS "Building for auto-detect platform")
endif()

# =============================================================================
# 后端选择 (可选，用于裁剪不需要的后端)
# =============================================================================

if(NOT DEFINED CACHE{ENABLE_MPP_DECODER})
    set(ENABLE_MPP_DECODER OFF CACHE BOOL "Enable Rockchip MPP decoder")
endif()
if(NOT DEFINED CACHE{ENABLE_NVDEC_DECODER})
    set(ENABLE_NVDEC_DECODER OFF CACHE BOOL "Enable NVIDIA NVDEC decoder")
endif()
if(NOT DEFINED CACHE{ENABLE_RGA_PREPROC})
    set(ENABLE_RGA_PREPROC OFF CACHE BOOL "Enable Rockchip RGA preprocessor")
endif()
if(NOT DEFINED CACHE{ENABLE_CUDA_PREPROC})
    set(ENABLE_CUDA_PREPROC OFF CACHE BOOL "Enable NVIDIA CUDA preprocessor")
endif()
if(NOT DEFINED CACHE{ENABLE_RKNN_INFER})
    set(ENABLE_RKNN_INFER OFF CACHE BOOL "Enable Rockchip RKNN inference")
endif()
if(NOT DEFINED CACHE{ENABLE_TRT_INFER})
    set(ENABLE_TRT_INFER OFF CACHE BOOL "Enable NVIDIA TensorRT inference")
endif()

# 根据平台设置默认后端
if(ROCKCHIP_PLATFORM)
    if(NOT DEFINED CACHE{ENABLE_MPP_DECODER})
        set(ENABLE_MPP_DECODER ON CACHE BOOL "Enable Rockchip MPP decoder" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_RGA_PREPROC})
        set(ENABLE_RGA_PREPROC ON CACHE BOOL "Enable Rockchip RGA preprocessor" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_RKNN_INFER})
        set(ENABLE_RKNN_INFER ON CACHE BOOL "Enable Rockchip RKNN inference" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_NVDEC_DECODER})
        set(ENABLE_NVDEC_DECODER OFF CACHE BOOL "Enable NVIDIA NVDEC decoder" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_CUDA_PREPROC})
        set(ENABLE_CUDA_PREPROC OFF CACHE BOOL "Enable NVIDIA CUDA preprocessor" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_TRT_INFER})
        set(ENABLE_TRT_INFER OFF CACHE BOOL "Enable NVIDIA TensorRT inference" FORCE)
    endif()
elseif(NVIDIA_PLATFORM)
    if(NOT DEFINED CACHE{ENABLE_NVDEC_DECODER})
        set(ENABLE_NVDEC_DECODER ON CACHE BOOL "Enable NVIDIA NVDEC decoder" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_CUDA_PREPROC})
        set(ENABLE_CUDA_PREPROC OFF CACHE BOOL "Enable NVIDIA CUDA preprocessor" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_TRT_INFER})
        set(ENABLE_TRT_INFER ON CACHE BOOL "Enable NVIDIA TensorRT inference" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_MPP_DECODER})
        set(ENABLE_MPP_DECODER OFF CACHE BOOL "Enable Rockchip MPP decoder" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_RGA_PREPROC})
        set(ENABLE_RGA_PREPROC OFF CACHE BOOL "Enable Rockchip RGA preprocessor" FORCE)
    endif()
    if(NOT DEFINED CACHE{ENABLE_RKNN_INFER})
        set(ENABLE_RKNN_INFER OFF CACHE BOOL "Enable Rockchip RKNN inference" FORCE)
    endif()
endif()

# =============================================================================
# 依赖查找
# =============================================================================

# FFmpeg (必需，用于解复用)
# 优先使用 pkg-config，如果失败则使用手动配置
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
        libavformat
        libavcodec
        libavutil
    )
    message(STATUS "Using pkg-config for FFmpeg")
else()
    # 手动配置 FFmpeg (Windows)
    set(FFMPEG_ROOT "D:/Software/ffmpeg/ffmpeg-8.1-full_build-shared" CACHE PATH "FFmpeg root directory")
    
    find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
        PATHS ${FFMPEG_ROOT}/include
        NO_DEFAULT_PATH
    )
    find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h
        PATHS ${FFMPEG_ROOT}/include
        NO_DEFAULT_PATH
    )
    find_path(FFMPEG_INCLUDE_DIR libavutil/avutil.h
        PATHS ${FFMPEG_ROOT}/include
        NO_DEFAULT_PATH
    )
    
    find_library(AVCODEC_LIBRARY NAMES avcodec libavcodec
        PATHS ${FFMPEG_ROOT}/lib
        NO_DEFAULT_PATH
    )
    find_library(AVFORMAT_LIBRARY NAMES avformat libavformat
        PATHS ${FFMPEG_ROOT}/lib
        NO_DEFAULT_PATH
    )
    find_library(AVUTIL_LIBRARY NAMES avutil libavutil
        PATHS ${FFMPEG_ROOT}/lib
        NO_DEFAULT_PATH
    )
    
    message(STATUS "FFMPEG_ROOT: ${FFMPEG_ROOT}")
    message(STATUS "AVCODEC_LIBRARY: ${AVCODEC_LIBRARY}")
    message(STATUS "AVFORMAT_LIBRARY: ${AVFORMAT_LIBRARY}")
    message(STATUS "AVUTIL_LIBRARY: ${AVUTIL_LIBRARY}")
    
    if(AVCODEC_LIBRARY AND AVFORMAT_LIBRARY AND AVUTIL_LIBRARY)
        message(STATUS "Found FFmpeg libraries manually")
        set(FFMPEG_LIBRARIES ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY})
        set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Could not find FFmpeg libraries")
    endif()
endif()

# OpenCV (可选，用于可视�?
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "OpenCV not found, visualization disabled")
endif()

# Rockchip MPP
if(ENABLE_MPP_DECODER OR ROCKCHIP_PLATFORM)
    find_path(MPP_INCLUDE_DIR rk_mpi.h
        PATHS /usr/include/rockchip /usr/local/include/rockchip /usr/include /usr/local/include
    )
    find_library(MPP_LIBRARY NAMES rockchip_mpp mpp
        PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/lib
    )
    if(MPP_INCLUDE_DIR AND MPP_LIBRARY)
        message(STATUS "MPP found: ${MPP_INCLUDE_DIR}")
    elseif(ROCKCHIP_PLATFORM)
        message(FATAL_ERROR "Rockchip MPP not found but PLATFORM=rockchip")
    endif()
endif()

# Rockchip RGA
if(ENABLE_RGA_PREPROC OR ROCKCHIP_PLATFORM)
    find_path(RGA_INCLUDE_DIR im2d.h
        PATHS /usr/include/rga /usr/local/include/rga /usr/include /usr/local/include
    )
    find_library(RGA_LIBRARY NAMES rga
        PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/lib
    )
    if(RGA_INCLUDE_DIR AND RGA_LIBRARY)
        message(STATUS "RGA found: ${RGA_INCLUDE_DIR}")
    elseif(ROCKCHIP_PLATFORM)
        message(FATAL_ERROR "Rockchip RGA not found but PLATFORM=rockchip")
    endif()
endif()

# RKNN
if(ENABLE_RKNN_INFER OR ROCKCHIP_PLATFORM)
    set(PROJECT_RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/include)
    set(PROJECT_RKNN_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/lib)

    find_path(RKNN_INCLUDE_DIR rknn_api.h
        PATHS ${PROJECT_RKNN_INCLUDE_DIR} /usr/include /usr/local/include /usr/include/rknn
    )
    find_library(RKNN_LIBRARY NAMES rknnrt
        PATHS ${PROJECT_RKNN_LIBRARY_DIR} /usr/lib /usr/lib/aarch64-linux-gnu
    )
    if(RKNN_INCLUDE_DIR AND RKNN_LIBRARY)
        message(STATUS "RKNN found: ${RKNN_INCLUDE_DIR}")
    elseif(ROCKCHIP_PLATFORM)
        message(FATAL_ERROR "RKNN not found but PLATFORM=rockchip")
    endif()
endif()

# NVIDIA CUDA
if(ENABLE_CUDA_PREPROC OR ENABLE_NVDEC_DECODER OR NVIDIA_PLATFORM)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        find_path(CUDA_HEADER_DIR cuda.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
        if(CUDA_HEADER_DIR)
            message(STATUS "CUDA found: ${CUDAToolkit_TARGET_DIR}")
            if(ENABLE_CUDA_PREPROC)
                if(WIN32 AND MINGW)
                    message(FATAL_ERROR "ENABLE_CUDA_PREPROC=ON requires an MSVC toolchain on Windows because nvcc needs cl.exe")
                endif()
                enable_language(CUDA)
            endif()
        elseif(NVIDIA_PLATFORM)
            message(FATAL_ERROR "CUDA toolkit found but cuda.h is missing")
        else()
            message(WARNING "CUDA toolkit found but cuda.h is missing; CUDA/NVDEC backends will be disabled")
        endif()
    elseif(NVIDIA_PLATFORM)
        message(FATAL_ERROR "CUDA not found but PLATFORM=nvidia")
    endif()
endif()

if(NOT CUDAToolkit_FOUND OR NOT CUDA_HEADER_DIR)
    set(ENABLE_NVDEC_DECODER OFF CACHE BOOL "Enable NVIDIA NVDEC decoder" FORCE)
    set(ENABLE_CUDA_PREPROC OFF CACHE BOOL "Enable NVIDIA CUDA preprocessor" FORCE)
endif()

# TensorRT
if(ENABLE_TRT_INFER)
    if(WIN32 AND MINGW)
        message(FATAL_ERROR "ENABLE_TRT_INFER=ON is not supported with MinGW on Windows; use an MSVC generator so TensorRT libraries and nvcc can link correctly")
    endif()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS 
            ${TENSORRT_ROOT}/include
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include"
            /usr/include/x86_64-linux-gnu
            /usr/local/cuda/include
            /usr/include
        NO_DEFAULT_PATH
    )
    find_library(TENSORRT_LIBRARY NAMES nvinfer libnvinfer
        PATHS 
            ${TENSORRT_ROOT}/lib
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64"
            /usr/lib/x86_64-linux-gnu
            /usr/local/cuda/lib64
            /usr/lib
        NO_DEFAULT_PATH
    )
    message(STATUS "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TENSORRT_LIBRARY: ${TENSORRT_LIBRARY}")
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
    endif()
endif()

# =============================================================================
# 源文件列�?
# =============================================================================

set(COMMON_SOURCES
    src/app_config.cpp
    src/backend_registry.cpp
    src/main.cpp
    src/ffmpeg_packet_source.cpp
    src/pipeline_runner.cpp
)

set(DECODER_SOURCES "")
set(PREPROC_SOURCES "")
set(INFER_SOURCES "")
set(POSTPROC_SOURCES "")
set(VISUAL_SOURCES "")
set(ENCODER_SOURCES "")

# MPP 解码�?
if(ENABLE_MPP_DECODER)
    if(NOT MPP_INCLUDE_DIR OR NOT MPP_LIBRARY)
        message(WARNING "MPP decoder requested but not found, disabling")
    else()
        list(APPEND DECODER_SOURCES
            src/backends/mpp_decoder.cpp
        )
        add_compile_definitions(ENABLE_MPP_DECODER)
    endif()
endif()

# NVDEC decoder
if(ENABLE_NVDEC_DECODER)
    if(CUDAToolkit_FOUND AND CUDA_HEADER_DIR)
        list(APPEND DECODER_SOURCES
            src/backends/nvdec_decoder.cpp
        )
        add_compile_definitions(ENABLE_NVDEC_DECODER)
    else()
        message(WARNING "NVDEC requested but CUDA headers were not found, disabling")
    endif()
endif()

# Decoder factory
list(APPEND DECODER_SOURCES
    src/backends/decoder_factory.cpp
)
# RGA 预处�?- Windows 禁用 (Rockchip only)
if(ENABLE_RGA_PREPROC AND NOT WIN32)
    if(NOT RGA_INCLUDE_DIR OR NOT RGA_LIBRARY)
        message(WARNING "RGA preprocessor requested but not found, disabling")
    else()
        list(APPEND PREPROC_SOURCES
            src/backends/rga_preprocessor.cpp
        )
        add_compile_definitions(ENABLE_RGA_PREPROC)
    endif()
endif()

# CUDA 预处�?- 启用
if(ENABLE_CUDA_PREPROC)
    if(CUDAToolkit_FOUND)
        list(APPEND PREPROC_SOURCES
            src/backends/cuda_preprocessor.cu
        )
        add_compile_definitions(ENABLE_CUDA_PREPROC)
    else()
        message(WARNING "CUDA preprocessor requested but CUDA not found, disabling")
    endif()
endif()

# 使用 CPU fallback 预处�?
list(APPEND PREPROC_SOURCES
    src/backends/preproc_factory.cpp
)

# 推理后端 - 禁用 (Windows)
if(ENABLE_RKNN_INFER AND NOT WIN32)
    if(RKNN_INCLUDE_DIR AND RKNN_LIBRARY)
        list(APPEND INFER_SOURCES
            src/backends/rknn_infer.cpp
        )
        add_compile_definitions(ENABLE_RKNN_INFER)
    else()
        message(WARNING "RKNN inference requested but not found, disabling")
    endif()
endif()

if(ENABLE_TRT_INFER)
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        list(APPEND INFER_SOURCES
            src/backends/trt_infer.cpp
        )
        add_compile_definitions(ENABLE_TRT_INFER)
    else()
        message(WARNING "TensorRT inference requested but not found, disabling")
    endif()
endif()

# 添加空的推理工厂以避免链接错�?
list(APPEND INFER_SOURCES
    src/backends/infer_factory.cpp
)

# 后处�?(YOLO 后处�?
list(APPEND POSTPROC_SOURCES
    src/backends/yolo_postproc.cpp
    src/backends/postproc_factory.cpp
)

# 编码�?(NVENC + MPP)
if(ENABLE_NVDEC_DECODER AND FALSE)
    list(APPEND ENCODER_SOURCES
        src/backends/nvenc_encoder.cpp
    )
    add_compile_definitions(ENABLE_NVENC_ENCODER)
endif()

# MPP 编码�?
if(ENABLE_MPP_DECODER AND FALSE)
    list(APPEND ENCODER_SOURCES
        src/backends/mpp_encoder.cpp
    )
    add_compile_definitions(ENABLE_MPP_ENCODER)
endif()

list(APPEND ENCODER_SOURCES
    src/backends/encoder_factory.cpp
)

# 可视�?(OpenCV) - 禁用
if(OpenCV_FOUND)
    list(APPEND VISUAL_SOURCES
        src/backends/opencv_visualizer.cpp
    )
    add_compile_definitions(ENABLE_VISUAL)
else()
    # 添加空的 stub 可视化器
    list(APPEND VISUAL_SOURCES
        src/backends/null_visualizer.cpp
    )
endif()

# =============================================================================
# 可执行文�?
# =============================================================================

add_executable(video_pipeline
    ${COMMON_SOURCES}
    ${DECODER_SOURCES}
    ${PREPROC_SOURCES}
    ${INFER_SOURCES}
    ${POSTPROC_SOURCES}
    ${ENCODER_SOURCES}
    ${VISUAL_SOURCES}
)

if(ENABLE_CUDA_PREPROC AND CUDAToolkit_FOUND)
    set_source_files_properties(src/backends/cuda_preprocessor.cu PROPERTIES LANGUAGE CUDA)
endif()

target_include_directories(video_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FFMPEG_INCLUDE_DIRS}
)

if(MPP_INCLUDE_DIR)
    target_include_directories(video_pipeline PRIVATE ${MPP_INCLUDE_DIR})
endif()

if(RGA_INCLUDE_DIR)
    target_include_directories(video_pipeline PRIVATE ${RGA_INCLUDE_DIR})
endif()

if(RKNN_INCLUDE_DIR)
    target_include_directories(video_pipeline PRIVATE ${RKNN_INCLUDE_DIR})
endif()

if(CUDAToolkit_FOUND)
    target_include_directories(video_pipeline PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(TENSORRT_INCLUDE_DIR)
    target_include_directories(video_pipeline PRIVATE ${TENSORRT_INCLUDE_DIR})
endif()

if(PkgConfig_FOUND)
    target_link_libraries(video_pipeline PRIVATE PkgConfig::FFMPEG)
else()
    target_link_libraries(video_pipeline PRIVATE ${FFMPEG_LIBRARIES})
    target_include_directories(video_pipeline PRIVATE ${FFMPEG_INCLUDE_DIRS})
endif()

if(OpenCV_FOUND)
    target_link_libraries(video_pipeline PRIVATE ${OpenCV_LIBS})
    target_include_directories(video_pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(MPP_LIBRARY)
    target_link_libraries(video_pipeline PRIVATE ${MPP_LIBRARY})
endif()

if(RGA_LIBRARY)
    target_link_libraries(video_pipeline PRIVATE ${RGA_LIBRARY})
endif()

if(RKNN_LIBRARY)
    target_link_libraries(video_pipeline PRIVATE ${RKNN_LIBRARY})
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(video_pipeline PRIVATE CUDA::cudart)
endif()

if(TENSORRT_LIBRARY)
    target_link_libraries(video_pipeline PRIVATE ${TENSORRT_LIBRARY})
endif()

# RPATH 设置
set_target_properties(video_pipeline PROPERTIES
    BUILD_RPATH "${PROJECT_RKNN_LIBRARY_DIR}"
    INSTALL_RPATH "${PROJECT_RKNN_LIBRARY_DIR}"
)

# =============================================================================
# 编译配置摘要
# =============================================================================

message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Platform: ${PLATFORM}")
message(STATUS "MPP Decoder: ${ENABLE_MPP_DECODER}")
message(STATUS "NVDEC Decoder: ${ENABLE_NVDEC_DECODER}")
message(STATUS "RGA Preproc: ${ENABLE_RGA_PREPROC}")
message(STATUS "CUDA Preproc: ${ENABLE_CUDA_PREPROC}")
message(STATUS "RKNN Infer: ${ENABLE_RKNN_INFER}")
message(STATUS "TensorRT Infer: ${ENABLE_TRT_INFER}")
message(STATUS "Postproc: ON")
message(STATUS "NVENC Encoder: ${ENABLE_NVENC_ENCODER}")
message(STATUS "===========================")
message(STATUS "")

if(BUILD_TESTING)
    add_executable(app_config_test
        src/app_config.cpp
        tests/app_config_test.cpp
    )
    if(MSVC)
        set_property(TARGET app_config_test PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    target_include_directories(app_config_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    add_test(NAME app_config_test COMMAND $<TARGET_FILE:app_config_test>)
endif()





