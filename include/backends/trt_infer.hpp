#pragma once

#include "infer_interface.hpp"
#include "pipeline_types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
}

class TrtInfer : public IInferenceBackend {
 public:
  TrtInfer() = default;
  ~TrtInfer() override;

  TrtInfer(const TrtInfer&) = delete;
  TrtInfer& operator=(const TrtInfer&) = delete;

  void open(const ModelConfig& config) override;
  InferenceOutput infer(const RgbImage& image) override;
  int inputWidth() const override { return input_width_; }
  int inputHeight() const override { return input_height_; }
  std::string name() const override { return "NVIDIA TensorRT"; }

  void setGpuId(int gpu_id) { gpu_id_ = gpu_id; }

 private:
  void loadEngine(const std::string& path);
  void releaseBuffers();
  void close();

  int gpu_id_ = 0;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  int input_width_ = 0;
  int input_height_ = 0;
  int input_channels_ = 3;
  bool input_is_nchw_ = true;
  size_t input_binding_ = 0;
  size_t output_binding_ = 1;
  std::size_t input_bytes_ = 0;
  std::size_t output_elements_ = 0;
  std::vector<std::int64_t> output_shape_;
  std::string output_name_;
  void* input_buffer_ = nullptr;
  void* output_buffer_ = nullptr;
  std::vector<std::uint8_t> host_input_buffer_;
};
