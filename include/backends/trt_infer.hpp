#pragma once

#include "infer_interface.hpp"
#include "pipeline_types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
enum class DataType : int;
}

class TrtInfer : public IInferenceBackend {
 public:
  TrtInfer() = default;
  ~TrtInfer() override;

  TrtInfer(const TrtInfer&) = delete;
  TrtInfer& operator=(const TrtInfer&) = delete;

  void open(const ModelConfig& config, const InferRuntimeConfig& runtime = {}) override;
  InferenceOutput infer(const RgbImage& image) override;
  int inputWidth() const override { return input_width_; }
  int inputHeight() const override { return input_height_; }
  std::string name() const override { return "NVIDIA TensorRT"; }

  void setGpuId(int gpu_id) { gpu_id_ = gpu_id; }

 private:
  void loadEngine(const std::string& path);
  void configureBindings();
  const char* copyInputToDevice(const RgbImage& image);
  void releaseBuffers();
  void close();

  struct BindingInfo {
    std::size_t index = 0;
    std::string name;
    bool isInput = false;
    bool isNchw = true;
    int channels = 0;
    int width = 0;
    int height = 0;
    std::size_t bytes = 0;
    std::size_t elementCount = 0;
    TensorDataType dataType = TensorDataType::kUnknown;
    std::vector<std::int64_t> shape;
    void* deviceBuffer = nullptr;
  };

  int gpu_id_ = 0;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  int input_width_ = 0;
  int input_height_ = 0;
  int input_channels_ = 3;
  bool input_is_nchw_ = true;
  TensorDataType input_data_type_ = TensorDataType::kUnknown;
  std::size_t input_binding_ = 0;
  std::size_t input_bytes_ = 0;
  void* owned_input_buffer_ = nullptr;
  bool verbose_ = false;
  bool logged_input_mode_ = false;
  std::vector<BindingInfo> output_bindings_;
  std::vector<std::uint8_t> host_input_buffer_;
  std::vector<float> host_input_f32_buffer_;
};
