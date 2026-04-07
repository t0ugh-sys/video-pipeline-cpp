#include "backends/cuda_preprocessor.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace {

void checkCudaStatus(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

/**
 * CUDA kernel: NV12 转 RGB + Resize
 * 使用双线性插值进行缩放
 */
__global__ void nv12ToRgbKernel(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    int srcStride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight) {
    return;
  }

  // 计算源图像中的对应位置 (双线性插值)
  const float srcX = static_cast<float>(x) * srcWidth / dstWidth;
  const float srcY = static_cast<float>(y) * srcHeight / dstHeight;

  const int x0 = static_cast<int>(srcX);
  const int y0 = static_cast<int>(srcY);
  const int x1 = min(x0 + 1, srcWidth - 1);
  const int y1 = min(y0 + 1, srcHeight - 1);

  const float dx = srcX - x0;
  const float dy = srcY - y0;

  // Y 平面采样
  const float y00 = y_plane[y0 * srcStride + x0];
  const float y01 = y_plane[y0 * srcStride + x1];
  const float y10 = y_plane[y1 * srcStride + x0];
  const float y11 = y_plane[y1 * srcStride + x1];
  const float Y = (1 - dx) * (1 - dy) * y00 + dx * (1 - dy) * y01 +
                  (1 - dx) * dy * y10 + dx * dy * y11;

  // UV 平面采样 (UV 分辨率是 Y 的一半)
  const int uvX = x0 / 2;
  const int uvY = y0 / 2;
  const U8 U = uv_plane[(srcHeight * srcStride) + uvY * srcStride + uvX * 2];
  const U8 V = uv_plane[(srcHeight * srcStride) + uvY * srcStride + uvX * 2 + 1];

  // YUV 转 RGB (BT.601)
  const float C = Y - 16.0f;
  const float Uf = U - 128.0f;
  const float Vf = V - 128.0f;

  int R = static_cast<int>(clamp(298.0f * C + 409.0f * Vf + 128.0f, 0.0f, 255.0f));
  int G = static_cast<int>(clamp(298.0f * C - 100.0f * Uf - 208.0f * Vf + 128.0f, 0.0f, 255.0f));
  int B = static_cast<int>(clamp(298.0f * C + 516.0f * Uf + 128.0f, 0.0f, 255.0f));

  // 输出 RGB
  const int idx = (y * dstWidth + x) * 3;
  rgb_output[idx] = R;
  rgb_output[idx + 1] = G;
  rgb_output[idx + 2] = B;
}

}  // namespace

CudaPreprocessor::CudaPreprocessor() {
  // 初始化 CUDA
  checkCudaStatus(cudaSetDevice(gpu_id_), "Failed to set CUDA device");
}

CudaPreprocessor::~CudaPreprocessor() {
  if (!device_buffer_.empty()) {
    cudaFree(device_buffer_.data());
  }
}

void CudaPreprocessor::setGpuId(int gpu_id) {
  gpu_id_ = gpu_id;
  checkCudaStatus(cudaSetDevice(gpu_id_), "Failed to set CUDA device");
}

RgbImage CudaPreprocessor::convertAndResize(
    const DecodedFrame& frame,
    int outputWidth,
    int outputHeight) const {
  // 注意：NVDEC 解码后的帧在 CPU 内存中 (通过 av_hwframe_transfer_data 下载)
  // 如果需要零拷贝，需要修改解码器直接输出 CUDA 设备内存

  RgbImage output;
  output.width = outputWidth;
  output.height = outputHeight;
  output.data.resize(static_cast<std::size_t>(outputWidth * outputHeight * 3));

  // 分配设备内存
  const size_t srcSize = frame.verticalStride * frame.height * 3 / 2;  // NV12
  if (device_buffer_size_ < srcSize) {
    if (device_buffer_size_ > 0) {
      cudaFree(device_buffer_.data());
    }
    device_buffer_.resize(srcSize);
    device_buffer_size_ = srcSize;
  }

  // 这里简化处理：实际使用 CUDA 图像处理库 (如 NPP) 会更高效
  // 由于 NV12 数据需要从 CPU 拷贝到 GPU，这里暂时用 CPU 实现
  // 完整实现需要解码器直接输出 CUDA 设备内存

  // TODO: 使用 CUDA NPP 或自定义 kernel 进行 NV12->RGB + Resize
  // 目前返回空数据，实际使用需要实现完整的 CUDA 路径

  throw std::runtime_error(
      "CUDA preprocessor: Full implementation requires zero-copy from NVDEC. "
      "Use RGA preprocessor on Rockchip or CPU fallback for now.");
}
