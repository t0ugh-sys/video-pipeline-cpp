#pragma once

#include "preproc_interface.hpp"
#include "pipeline_types.hpp"

#include <cstdint>

class CudaPreprocessor : public IPreprocessorBackend {
 public:
  CudaPreprocessor();
  ~CudaPreprocessor() override;

  RgbImage convertAndResize(
      const DecodedFrame& frame,
      int outputWidth,
      int outputHeight,
      const PreprocessOptions& options = {}) override;

  std::string name() const override { return "NVIDIA CUDA"; }

  void setGpuId(int gpu_id);

 private:
  int gpu_id_ = 0;
};
