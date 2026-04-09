#include "backends/mpp_encoder.hpp"

extern "C" {
#include <rk_mpi.h>
#include <mpp_buffer.h>
#include <mpp_err.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
}

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

void checkMppStatus(MPP_RET status, const char* message) {
  if (status != MPP_OK) {
    throw std::runtime_error(message);
  }
}

}  // namespace

MppEncoder::MppEncoder() = default;

MppEncoder::~MppEncoder() {
  close();
}

void MppEncoder::init(const EncoderConfig& config) {
  close();

  // 1. 创建编码器 context
  checkMppStatus(mpp_create(&context_, &api_), "mpp_create failed");

  // 2. 初始化编码器
  MppCtxType codecType = MPP_CTX_ENC;
  MppCodingType codingType = (config.codec == "hevc") ? MPP_VIDEO_CodingHEVC : MPP_VIDEO_CodingAVC;
  
  checkMppStatus(mpp_init(context_, codecType, codingType), "mpp_init failed");

  // 3. 获取编码器句柄
  MppEncApiCtx* encCtx = nullptr;
  api_->control(context_, MPP_GET_ENC_CTX, &encCtx);

  if (!encCtx) {
    throw std::runtime_error("Failed to get encoder context");
  }
  encoder_ = encCtx;

  // 4. 设置编码参数
  MppVideoFormatCam param = {};
  param.width = config.width;
  param.height = config.height;
  param.hdr_stride = config.width;
  param.vdr_stride = config.width;
  param.frame_rate = config.fps;
  param.bit_rate = config.bitrate / 1000;  // kbps
  param.rc_mode = MPP_RC_CBR;
  param.stream_type = MPP_STREAM_TYPE_NV12;
  param.enc_level = MPP_ENC_LEVEL_DEFAULT;
  
  // H.264/H.265 参数
  api_->control(context_, MPP_ENC_SET_FORMAT, &param);

  // 打开输出文件
  outputFile_ = std::ofstream(config.outputPath, std::ios::binary);
  if (!outputFile_.is_open()) {
    throw std::runtime_error("Failed to open output file: " + config.outputPath);
  }

  initialized_ = true;
}

void MppEncoder::encode(const RgbImage& frame, int64_t pts) {
  if (!initialized_) {
    return;
  }

  // TODO: 将 RGB 转换为 NV12 并编码
  // 需要使用 MPP 的编码 API
}

void MppEncoder::flush() {
  if (!initialized_ || !outputFile_.is_open()) {
    return;
  }

  // 刷新编码器
  MppPacket packet = nullptr;
  api_->flush(context_, 0, &packet);
  
  while (packet) {
    // 写入文件
    void* data = nullptr;
    std::size_t size = 0;
    mpp_packet_get_data(packet, &data, &size);
    
    if (data && size > 0) {
      outputFile_.write(static_cast<const char*>(data), size);
    }
    
    mpp_packet_deinit(&packet);
    
    // 获取下一个 packet
    api_->flush(context_, 0, &packet);
  }

  outputFile_.flush();
}

void MppEncoder::close() {
  if (outputFile_.is_open()) {
    outputFile_.close();
  }

  if (context_) {
    mpp_destroy(context_);
    context_ = nullptr;
    api_ = nullptr;
    encoder_ = nullptr;
  }

  initialized_ = false;
}