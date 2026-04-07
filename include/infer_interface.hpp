#pragma once

#include "pipeline_types.hpp"

#include <memory>
#include <string>
#include <vector>

/**
 * 推理后端抽象接口
 * 支持：Rockchip RKNN | NVIDIA TensorRT | ONNX Runtime
 */
class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  /** 加载模型 */
  virtual void open(const ModelConfig& config) = 0;

  /** 推理 */
  virtual std::vector<float> infer(const RgbImage& image) = 0;

  /** 获取输入宽度 */
  virtual int inputWidth() const = 0;

  /** 获取输入高度 */
  virtual int inputHeight() const = 0;

  /** 获取后端名称 */
  virtual std::string name() const = 0;
};

/**
 * 后端类型枚举
 */
enum class InferBackendType {
  kAuto,         ///< 自动选择
  kRockchipRknn, ///< Rockchip RKNN NPU
  kNvidiaTrt,    ///< NVIDIA TensorRT
  kOnnxRuntime,  ///< ONNX Runtime (CPU/GPU)
};

/**
 * 创建推理后端实例
 */
std::unique_ptr<IInferenceBackend> createInferBackend(InferBackendType type = InferBackendType::kAuto);

/**
 * 检测当前平台可用的推理后端
 */
InferBackendType detectAvailableInferBackend();
