#include "backends/trt_infer.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace {

void checkTrtStatus(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void checkCudaStatus(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      // 可以输出日志，但默认沉默
    }
  }
};

static Logger gLogger;

}  // namespace

TrtInfer::~TrtInfer() {
  close();
}

void TrtInfer::open(const ModelConfig& config) {
  close();
  loadEngine(config.modelPath);
}

std::vector<float> TrtInfer::infer(const RgbImage& image) {
  if (image.width != input_width_ || image.height != input_height_) {
    throw std::runtime_error("RGB image size does not match TensorRT input");
  }

  // 分配输入输出内存
  void* input_buffer = nullptr;
  void* output_buffer = nullptr;

  checkCudaStatus(cudaMalloc(&input_buffer, input_size_), "Failed to allocate input buffer");

  // 获取输出维度并分配
  nvinfer1::Dims output_dims = context_->getBindingDimensions(1);
  size_t output_size = sizeof(float);
  for (int i = 0; i < output_dims.nbDims; ++i) {
    output_size *= output_dims.d[i];
  }
  checkCudaStatus(cudaMalloc(&output_buffer, output_size), "Failed to allocate output buffer");

  // 拷贝输入数据到 GPU
  checkCudaStatus(
      cudaMemcpy(input_buffer, image.data.data(), input_size_, cudaMemcpyHostToDevice),
      "Failed to copy input to device");

  // 执行推理
  void* bindings[] = {input_buffer, output_buffer};
  checkTrtStatus(context_->execute(1, bindings), "TensorRT execute failed");

  // 拷贝结果回 CPU
  std::vector<float> result(output_size / sizeof(float));
  checkCudaStatus(
      cudaMemcpy(result.data(), output_buffer, output_size, cudaMemcpyDeviceToHost),
      "Failed to copy output to host");

  // 清理
  checkCudaStatus(cudaFree(input_buffer), "Failed to free input buffer");
  checkCudaStatus(cudaFree(output_buffer), "Failed to free output buffer");

  return result;
}

void TrtInfer::loadEngine(const std::string& path) {
  // 设置 CUDA 设备
  checkCudaStatus(cudaSetDevice(gpu_id_), "Failed to set CUDA device");

  // 读取引擎文件
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open TensorRT engine: " + path);
  }

  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  file.read(buffer.data(), size);
  file.close();

  // 创建 Runtime
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(gLogger));
  checkTrtStatus(runtime != nullptr, "Failed to create TensorRT runtime");

  // 加载引擎
  engine_.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  checkTrtStatus(engine_ != nullptr, "Failed to deserialize TensorRT engine");

  // 创建执行上下文
  context_.reset(engine_->createExecutionContext());
  checkTrtStatus(context_ != nullptr, "Failed to create TensorRT execution context");

  // 获取输入绑定信息
  input_binding_ = engine_->getBindingIndex("input");
  if (input_binding_ == static_cast<size_t>(-1)) {
    input_binding_ = 0;  // 假设第一个绑定是输入
  }

  nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_binding_);
  if (input_dims.nbDims >= 3) {
    // NCHW 或 NHWC
    input_height_ = input_dims.d[2];
    input_width_ = input_dims.d[3];
    input_channels_ = input_dims.d[1];
  }

  input_size_ = input_width_ * input_height_ * input_channels_;
}

void TrtInfer::close() {
  context_.reset();
  engine_.reset();
  input_width_ = 0;
  input_height_ = 0;
  input_channels_ = 0;
  input_size_ = 0;
}
