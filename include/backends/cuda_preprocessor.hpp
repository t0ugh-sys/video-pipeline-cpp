#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

#include <vector>
#include <cstdint>

// CUDA forward declarations
struct CUstream_st;
typedef CUstream_st* CUstream;

/**
 * NVIDIA CUDA 预处理器
 * 使用 CUDA 进行 NV12 到 RGB 的转换和缩放
 */
class CudaPreprocessor : public IPreprocessorBackend {
 public:
  CudaPreprocessor();
  ~CudaPreprocessor() override;

  RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight) const override;

  std::string name() const override { return "NVIDIA CUDA"; }

  /** 设置 GPU 设备 ID */
  void setGpuId(int gpu_id);

 private:
  int gpu_id_ = 0;
  mutable std::vector<std::uint8_t> device_buffer_;
  mutable size_t device_buffer_size_ = 0;
};
