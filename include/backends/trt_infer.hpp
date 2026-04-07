#pragma once

#include "infer_interface.hpp"
#include "pipeline_types.hpp"

#include <vector>
#include <memory>
#include <string>

// TensorRT forward declarations
namespace nvinfer1 {
  class ICudaEngine;
  class IExecutionContext;
}

/**
 * NVIDIA TensorRT 推理后端
 * 适用于 NVIDIA GPU 平台的高性能推理
 */
class TrtInfer : public IInferenceBackend {
 public:
  TrtInfer() = default;
  ~TrtInfer() override;

  TrtInfer(const TrtInfer&) = delete;
  TrtInfer& operator=(const TrtInfer&) = delete;

  void open(const ModelConfig& config) override;
  std::vector<float> infer(const RgbImage& image) override;
  int inputWidth() const override { return input_width_; }
  int inputHeight() const override { return input_height_; }
  std::string name() const override { return "NVIDIA TensorRT"; }

  /** 设置 GPU 设备 ID */
  void setGpuId(int gpu_id) { gpu_id_ = gpu_id; }

 private:
  void loadEngine(const std::string& path);
  void close();

  int gpu_id_ = 0;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  int input_width_ = 0;
  int input_height_ = 0;
  int input_channels_ = 3;
  size_t input_binding_ = 0;
  size_t input_size_ = 0;
};
