#include "backends/trt_infer.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

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

template <typename T>
struct TrtDestroy {
  void operator()(T* value) const {
    if (value != nullptr) {
      delete value;
    }
  }
};

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    (void)severity;
    (void)message;
  }
};

Logger gLogger;

std::size_t dimsElementCount(const nvinfer1::Dims& dims) {
  std::size_t count = 1;
  for (int index = 0; index < dims.nbDims; ++index) {
    if (dims.d[index] <= 0) {
      throw std::runtime_error("TensorRT binding has dynamic or invalid dimensions");
    }
    count *= static_cast<std::size_t>(dims.d[index]);
  }
  return count;
}

std::vector<std::int64_t> dimsToShape(const nvinfer1::Dims& dims) {
  std::vector<std::int64_t> shape;
  shape.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    shape.push_back(dims.d[i]);
  }
  return shape;
}

TensorDataType toTensorDataType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return TensorDataType::kFloat32;
    case nvinfer1::DataType::kINT8:
      return TensorDataType::kInt8;
    case nvinfer1::DataType::kINT32:
      return TensorDataType::kInt32;
    default:
      return TensorDataType::kUnknown;
  }
}

const char* tensorDataTypeName(TensorDataType type) {
  switch (type) {
    case TensorDataType::kFloat32:
      return "float32";
    case TensorDataType::kUint8:
      return "uint8";
    case TensorDataType::kInt8:
      return "int8";
    case TensorDataType::kInt32:
      return "int32";
    default:
      return "unknown";
  }
}

std::size_t bytesPerElement(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      throw std::runtime_error("Unsupported TensorRT binding datatype");
  }
}

void packRgbToNchw(
    const RgbImage& image,
    int channels,
    std::vector<std::uint8_t>& destination) {
  const std::size_t planeSize = static_cast<std::size_t>(image.width * image.height);
  destination.resize(planeSize * static_cast<std::size_t>(channels));

  for (std::size_t index = 0; index < planeSize; ++index) {
    const std::size_t src = index * 3;
    destination[index] = image.data[src];
    if (channels > 1) {
      destination[planeSize + index] = image.data[src + 1];
    }
    if (channels > 2) {
      destination[planeSize * 2 + index] = image.data[src + 2];
    }
  }
}

void packRgbToNchwFloat(
    const RgbImage& image,
    int channels,
    std::vector<float>& destination) {
  const std::size_t planeSize = static_cast<std::size_t>(image.width * image.height);
  destination.resize(planeSize * static_cast<std::size_t>(channels));

  for (std::size_t index = 0; index < planeSize; ++index) {
    const std::size_t src = index * 3;
    destination[index] = static_cast<float>(image.data[src]);
    if (channels > 1) {
      destination[planeSize + index] = static_cast<float>(image.data[src + 1]);
    }
    if (channels > 2) {
      destination[planeSize * 2 + index] = static_cast<float>(image.data[src + 2]);
    }
  }
}

__global__ void rgbNhwcUint8ToNchwUint8Kernel(
    const std::uint8_t* src,
    std::uint8_t* dst,
    int width,
    int height,
    int channels) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const std::size_t pixelIndex = static_cast<std::size_t>(y) * width + x;
  const std::size_t srcBase = pixelIndex * channels;
  const std::size_t planeSize = static_cast<std::size_t>(width) * height;
  for (int c = 0; c < channels; ++c) {
    dst[static_cast<std::size_t>(c) * planeSize + pixelIndex] = src[srcBase + c];
  }
}

__global__ void rgbNhwcUint8ToNhwcFloatKernel(
    const std::uint8_t* src,
    float* dst,
    int width,
    int height,
    int channels) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const std::size_t pixelIndex = static_cast<std::size_t>(y) * width + x;
  const std::size_t base = pixelIndex * channels;
  for (int c = 0; c < channels; ++c) {
    dst[base + c] = static_cast<float>(src[base + c]);
  }
}

__global__ void rgbNhwcUint8ToNchwFloatKernel(
    const std::uint8_t* src,
    float* dst,
    int width,
    int height,
    int channels) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const std::size_t pixelIndex = static_cast<std::size_t>(y) * width + x;
  const std::size_t srcBase = pixelIndex * channels;
  const std::size_t planeSize = static_cast<std::size_t>(width) * height;
  for (int c = 0; c < channels; ++c) {
    dst[static_cast<std::size_t>(c) * planeSize + pixelIndex] =
        static_cast<float>(src[srcBase + c]);
  }
}

}  // namespace

TrtInfer::~TrtInfer() {
  close();
}

void TrtInfer::open(const ModelConfig& config, const InferRuntimeConfig& runtime) {
  close();
  verbose_ = runtime.verbose;
  loadEngine(config.modelPath);
}

InferenceOutput TrtInfer::infer(const RgbImage& image) {
  if (!context_ || !engine_) {
    throw std::runtime_error("TensorRT backend is not initialized");
  }
  if (image.width != input_width_ || image.height != input_height_) {
    throw std::runtime_error("RGB image size does not match TensorRT input");
  }

  std::vector<void*> bindings(static_cast<std::size_t>(engine_->getNbBindings()), nullptr);
  for (const auto& binding : output_bindings_) {
    bindings[binding.index] = binding.deviceBuffer;
  }

  const char* inputMode = nullptr;
  if (input_data_type_ == TensorDataType::kUint8 &&
      !input_is_nchw_ &&
      image.isOnDevice &&
      image.devicePtr != 0) {
    bindings[input_binding_] = reinterpret_cast<void*>(image.devicePtr);
    inputMode = "device-direct-nhwc-u8";
  } else {
    inputMode = copyInputToDevice(image);
    bindings[input_binding_] = owned_input_buffer_;
  }

  if (verbose_ && !logged_input_mode_) {
    logged_input_mode_ = true;
    std::cerr << "[TRT] input_mode=" << inputMode
              << " input_layout=" << (input_is_nchw_ ? "NCHW" : "NHWC")
              << " input_dtype=" << tensorDataTypeName(input_data_type_)
              << " outputs=" << output_bindings_.size() << "\n";
  }

  checkTrtStatus(context_->executeV2(bindings.data()), "TensorRT execute failed");

  InferenceOutput output;
  output.reserve(output_bindings_.size());
  for (const auto& binding : output_bindings_) {
    InferenceTensor tensor;
    tensor.name = binding.name.empty() ? "output_" + std::to_string(binding.index) : binding.name;
    tensor.layout = binding.isNchw ? "NCHW" : "NHWC";
    tensor.shape = binding.shape;
    tensor.dataType = binding.dataType;
    if (binding.dataType == TensorDataType::kFloat32) {
      tensor.data.resize(binding.elementCount);
      checkCudaStatus(
          cudaMemcpy(
              tensor.data.data(),
              binding.deviceBuffer,
              binding.bytes,
              cudaMemcpyDeviceToHost),
          "Failed to copy TensorRT float output to host");
    } else {
      tensor.rawData.resize(binding.bytes);
      checkCudaStatus(
          cudaMemcpy(
              tensor.rawData.data(),
              binding.deviceBuffer,
              binding.bytes,
              cudaMemcpyDeviceToHost),
          "Failed to copy TensorRT raw output to host");
    }
    output.push_back(std::move(tensor));
  }

  return output;
}

void TrtInfer::loadEngine(const std::string& path) {
  checkCudaStatus(cudaSetDevice(gpu_id_), "Failed to set CUDA device");

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open TensorRT engine: " + path);
  }

  const auto size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("TensorRT engine file is empty: " + path);
  }
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(static_cast<std::size_t>(size));
  file.read(buffer.data(), size);
  file.close();

  std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime(
      nvinfer1::createInferRuntime(gLogger));
  checkTrtStatus(runtime != nullptr, "Failed to create TensorRT runtime");

  engine_.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  checkTrtStatus(engine_ != nullptr, "Failed to deserialize TensorRT engine");

  context_.reset(engine_->createExecutionContext());
  checkTrtStatus(context_ != nullptr, "Failed to create TensorRT execution context");
  checkTrtStatus(engine_->getNbBindings() >= 2, "TensorRT engine must have at least one input and one output");

  configureBindings();
}

void TrtInfer::configureBindings() {
  input_binding_ = 0;
  output_bindings_.clear();

  for (int index = 0; index < engine_->getNbBindings(); ++index) {
    const nvinfer1::Dims dims = context_->getBindingDimensions(index);
    const std::size_t elementCount = dimsElementCount(dims);
    const auto dataType = engine_->getBindingDataType(index);
    const std::size_t bytes = elementCount * bytesPerElement(dataType);
    const bool isInput = engine_->bindingIsInput(index);
    if (isInput) {
      input_binding_ = static_cast<std::size_t>(index);
      checkTrtStatus(dims.nbDims == 4, "TensorRT input must be a 4D tensor");
      if (dims.d[1] > 0 && dims.d[1] <= 4) {
        input_is_nchw_ = true;
        input_channels_ = dims.d[1];
        input_height_ = dims.d[2];
        input_width_ = dims.d[3];
      } else if (dims.d[3] > 0 && dims.d[3] <= 4) {
        input_is_nchw_ = false;
        input_height_ = dims.d[1];
        input_width_ = dims.d[2];
        input_channels_ = dims.d[3];
      } else {
        throw std::runtime_error("Unsupported TensorRT input layout");
      }
      checkTrtStatus(
          input_channels_ == 3,
          "TensorRT input channel count must be 3 for the current RGB pipeline");
      input_data_type_ = toTensorDataType(dataType);
      input_bytes_ = elementCount * bytesPerElement(dataType);
      checkCudaStatus(cudaMalloc(&owned_input_buffer_, input_bytes_), "Failed to allocate TensorRT input buffer");
      if (verbose_) {
        std::cerr << "[TRT] input_binding name=" << engine_->getBindingName(index)
                  << " shape=[";
        for (int dim = 0; dim < dims.nbDims; ++dim) {
          if (dim > 0) {
            std::cerr << ",";
          }
          std::cerr << dims.d[dim];
        }
        std::cerr << "] dtype=" << tensorDataTypeName(input_data_type_)
                  << " layout=" << (input_is_nchw_ ? "NCHW" : "NHWC") << "\n";
      }
      continue;
    }

    BindingInfo binding;
    binding.index = static_cast<std::size_t>(index);
    binding.name = engine_->getBindingName(index);
    binding.isInput = false;
    binding.isNchw = true;
    binding.bytes = bytes;
    binding.elementCount = elementCount;
    binding.dataType = toTensorDataType(dataType);
    binding.shape = dimsToShape(dims);
    if (dims.nbDims == 4) {
      if (dims.d[1] > 0 && dims.d[1] <= 4096) {
        binding.channels = dims.d[1];
        binding.height = dims.d[2];
        binding.width = dims.d[3];
      } else {
        binding.isNchw = false;
        binding.height = dims.d[1];
        binding.width = dims.d[2];
        binding.channels = dims.d[3];
      }
    }
    checkCudaStatus(
        cudaMalloc(&binding.deviceBuffer, binding.bytes),
        "Failed to allocate TensorRT output buffer");
    if (verbose_) {
      std::cerr << "[TRT] output_binding name=" << binding.name
                << " shape=[";
      for (std::size_t dim = 0; dim < binding.shape.size(); ++dim) {
        if (dim > 0) {
          std::cerr << ",";
        }
        std::cerr << binding.shape[dim];
      }
      std::cerr << "] dtype=" << tensorDataTypeName(binding.dataType)
                << " layout=" << (binding.isNchw ? "NCHW" : "NHWC") << "\n";
    }
    output_bindings_.push_back(binding);
  }
}

const char* TrtInfer::copyInputToDevice(const RgbImage& image) {
  if (image.isOnDevice && image.devicePtr != 0) {
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((image.width + block.x - 1) / block.x),
        static_cast<unsigned int>((image.height + block.y - 1) / block.y));

    const auto* deviceInput = reinterpret_cast<const std::uint8_t*>(image.devicePtr);
    if (input_data_type_ == TensorDataType::kUint8) {
      if (input_is_nchw_) {
        rgbNhwcUint8ToNchwUint8Kernel<<<grid, block>>>(
            deviceInput,
            reinterpret_cast<std::uint8_t*>(owned_input_buffer_),
            image.width,
            image.height,
            input_channels_);
        checkCudaStatus(cudaGetLastError(), "Failed to launch TensorRT uint8 NCHW input kernel");
      } else {
        checkCudaStatus(
            cudaMemcpy(owned_input_buffer_, deviceInput, input_bytes_, cudaMemcpyDeviceToDevice),
            "Failed to copy TensorRT uint8 NHWC input on device");
      }
      checkCudaStatus(cudaDeviceSynchronize(), "TensorRT uint8 input staging failed");
      return input_is_nchw_ ? "device-kernel-nchw-u8" : "device-copy-nhwc-u8";
    }

    if (input_data_type_ == TensorDataType::kFloat32) {
      if (input_is_nchw_) {
        rgbNhwcUint8ToNchwFloatKernel<<<grid, block>>>(
            deviceInput,
            reinterpret_cast<float*>(owned_input_buffer_),
            image.width,
            image.height,
            input_channels_);
        checkCudaStatus(cudaGetLastError(), "Failed to launch TensorRT float NCHW input kernel");
      } else {
        rgbNhwcUint8ToNhwcFloatKernel<<<grid, block>>>(
            deviceInput,
            reinterpret_cast<float*>(owned_input_buffer_),
            image.width,
            image.height,
            input_channels_);
        checkCudaStatus(cudaGetLastError(), "Failed to launch TensorRT float NHWC input kernel");
      }
      checkCudaStatus(cudaDeviceSynchronize(), "TensorRT float input staging failed");
      return input_is_nchw_ ? "device-kernel-nchw-f32" : "device-kernel-nhwc-f32";
    }
  }

  if (image.data.empty()) {
    throw std::runtime_error(
        "TensorRT input requires CPU RGB data when direct CUDA input is not available");
  }
  if (image.data.size() != static_cast<std::size_t>(image.width * image.height * 3)) {
    throw std::runtime_error("RGB image buffer size does not match TensorRT input buffer");
  }

  if (input_data_type_ == TensorDataType::kUint8) {
    const void* hostInput = image.data.data();
    if (input_is_nchw_) {
      packRgbToNchw(image, input_channels_, host_input_buffer_);
      hostInput = host_input_buffer_.data();
    }

    checkCudaStatus(
        cudaMemcpy(owned_input_buffer_, hostInput, input_bytes_, cudaMemcpyHostToDevice),
        "Failed to copy TensorRT uint8 input to device");
    return input_is_nchw_ ? "host-pack-nchw-u8" : "host-copy-nhwc-u8";
  }

  if (input_data_type_ == TensorDataType::kFloat32) {
    if (input_is_nchw_) {
      packRgbToNchwFloat(image, input_channels_, host_input_f32_buffer_);
    } else {
      host_input_f32_buffer_.resize(static_cast<std::size_t>(image.width * image.height * input_channels_));
      for (std::size_t index = 0; index < host_input_f32_buffer_.size(); ++index) {
        host_input_f32_buffer_[index] = static_cast<float>(image.data[index]);
      }
    }

    checkCudaStatus(
        cudaMemcpy(
            owned_input_buffer_,
            host_input_f32_buffer_.data(),
            input_bytes_,
            cudaMemcpyHostToDevice),
        "Failed to copy TensorRT float input to device");
    return input_is_nchw_ ? "host-pack-nchw-f32" : "host-pack-nhwc-f32";
  }

  throw std::runtime_error("Unsupported TensorRT input datatype for RGB pipeline");
}

void TrtInfer::releaseBuffers() {
  if (owned_input_buffer_ != nullptr) {
    cudaFree(owned_input_buffer_);
    owned_input_buffer_ = nullptr;
  }
  for (auto& binding : output_bindings_) {
    if (binding.deviceBuffer != nullptr) {
      cudaFree(binding.deviceBuffer);
      binding.deviceBuffer = nullptr;
    }
  }
  output_bindings_.clear();
}

void TrtInfer::close() {
  releaseBuffers();
  context_.reset();
  engine_.reset();
  input_width_ = 0;
  input_height_ = 0;
  input_channels_ = 0;
  input_is_nchw_ = true;
  input_data_type_ = TensorDataType::kUnknown;
  input_binding_ = 0;
  input_bytes_ = 0;
  verbose_ = false;
  logged_input_mode_ = false;
  host_input_buffer_.clear();
  host_input_f32_buffer_.clear();
}
