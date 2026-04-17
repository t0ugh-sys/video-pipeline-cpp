#include "backends/mpp_encoder.hpp"

extern "C" {
#include <mpp_buffer.h>
#include <mpp_err.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <rk_mpi.h>
#include <rk_mpi_cmd.h>
}

#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
#include <im2d.h>
#include <im2d_buffer.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <stdexcept>

namespace {

constexpr size_t kPacketBufferMinSize = 512 * 1024;
constexpr auto kDrainRetrySleep = std::chrono::milliseconds(2);
constexpr int kDrainRetryCount = 50;

void checkMppStatus(MPP_RET status, const char* message) {
  if (status != MPP_OK) {
    throw std::runtime_error(message);
  }
}

MppCodingType toCodingType(const std::string& codec) {
  if (codec == "hevc" || codec == "h265") {
    return MPP_VIDEO_CodingHEVC;
  }
  return MPP_VIDEO_CodingAVC;
}

size_t packetBufferSize(const EncoderConfig& config) {
  const size_t frameBytes = static_cast<size_t>(std::max(config.horStride, config.width)) *
                            static_cast<size_t>(std::max(config.verStride, config.height)) * 3 / 2;
  return std::max(frameBytes, kPacketBufferMinSize);
}

int alignUp(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

bool verboseMppRgbEncodeLogsEnabled() {
  const char* value = std::getenv("MPP_RGB_ENCODER_VERBOSE_LOG");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void logMppRgbEncodeStep(const char* step) {
  if (!verboseMppRgbEncodeLogsEnabled()) {
    return;
  }
  std::cerr << "[MPP-RGB] " << step << "\n";
  std::cerr.flush();
}

}  // namespace

MppEncoder::MppEncoder() = default;

MppEncoder::~MppEncoder() {
  close();
}

void MppEncoder::init(const EncoderConfig& config) {
  close();
  logMppRgbEncodeStep("init_begin");

  if (config.width <= 0 || config.height <= 0) {
    throw std::runtime_error("MPP encoder requires a positive frame size");
  }
  if (config.inputFormat != PixelFormat::kNv12 &&
      config.inputFormat != PixelFormat::kRgb888) {
    throw std::runtime_error("MPP encoder only supports NV12 decoded-frame input or RGB888 frames");
  }

  width_ = config.width;
  height_ = config.height;
  horStride_ = config.horStride > 0 ? config.horStride : alignUp(config.width, 16);
  verStride_ = config.verStride > 0 ? config.verStride : alignUp(config.height, 16);
  inputFormat_ = config.inputFormat;
  flushSubmitted_ = false;
  frameIndex_ = 0;

  outputFile_ = std::ofstream(config.outputPath, std::ios::binary);
  if (!outputFile_.is_open()) {
    throw std::runtime_error("Failed to open output file: " + config.outputPath);
  }
  logMppRgbEncodeStep("init_output_opened");

  if (inputFormat_ == PixelFormat::kRgb888) {
    checkMppStatus(mpp_buffer_group_get_internal(
                       &bufferGroup_,
                       static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)),
                   "mpp_buffer_group_get_internal failed");
    logMppRgbEncodeStep("init_buffer_group_done");
    const size_t frameBytes =
        static_cast<size_t>(horStride_) * static_cast<size_t>(verStride_) * 3 / 2;
    checkMppStatus(mpp_buffer_group_limit_config(bufferGroup_, frameBytes, 2),
                   "mpp_buffer_group_limit_config failed");
    logMppRgbEncodeStep("init_buffer_limit_done");
    checkMppStatus(mpp_buffer_get(bufferGroup_, &inputBuffer_, frameBytes),
                   "mpp_buffer_get for encoder input failed");
    logMppRgbEncodeStep("init_input_buffer_done");
  } else {
    checkMppStatus(mpp_buffer_group_get_internal(
                       &bufferGroup_,
                       static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)),
                   "mpp_buffer_group_get_internal failed");
    logMppRgbEncodeStep("init_buffer_group_done");
  }

  checkMppStatus(mpp_buffer_get(bufferGroup_, &packetBuffer_, packetBufferSize(config)),
                 "mpp_buffer_get for output packet failed");
  logMppRgbEncodeStep("init_packet_buffer_done");

  checkMppStatus(mpp_create(&context_, &api_), "mpp_create failed");
  logMppRgbEncodeStep("init_mpp_create_done");
  checkMppStatus(mpp_init(context_, MPP_CTX_ENC, toCodingType(config.codec)), "mpp_init failed");
  logMppRgbEncodeStep("init_mpp_init_done");

  MppPollType timeout = MPP_POLL_BLOCK;
  checkMppStatus(api_->control(context_, MPP_SET_OUTPUT_TIMEOUT, &timeout),
                 "MPP_SET_OUTPUT_TIMEOUT failed");
  logMppRgbEncodeStep("init_set_timeout_done");

  MppEncPrepCfg prepCfg = {};
  prepCfg.change = MPP_ENC_PREP_CFG_CHANGE_INPUT | MPP_ENC_PREP_CFG_CHANGE_FORMAT;
  prepCfg.width = width_;
  prepCfg.height = height_;
  prepCfg.hor_stride = horStride_;
  prepCfg.ver_stride = verStride_;
  prepCfg.format = MPP_FMT_YUV420SP;
  checkMppStatus(api_->control(context_, MPP_ENC_SET_PREP_CFG, &prepCfg), "MPP_ENC_SET_PREP_CFG failed");
  logMppRgbEncodeStep("init_set_prep_cfg_done");

  MppEncRcCfg rcCfg = {};
  rcCfg.change = MPP_ENC_RC_CFG_CHANGE_RC_MODE |
                 MPP_ENC_RC_CFG_CHANGE_BPS |
                 MPP_ENC_RC_CFG_CHANGE_FPS_IN |
                 MPP_ENC_RC_CFG_CHANGE_FPS_OUT |
                 MPP_ENC_RC_CFG_CHANGE_GOP;
  rcCfg.rc_mode = MPP_ENC_RC_MODE_CBR;
  rcCfg.bps_target = config.bitrate;
  rcCfg.bps_max = config.bitrate * 17 / 16;
  rcCfg.bps_min = std::max(1, config.bitrate * 9 / 10);
  rcCfg.fps_in_flex = 0;
  rcCfg.fps_in_num = config.fps;
  rcCfg.fps_in_denorm = 1;
  rcCfg.fps_out_flex = 0;
  rcCfg.fps_out_num = config.fps;
  rcCfg.fps_out_denorm = 1;
  rcCfg.gop = std::max(1, config.fps * 2);
  checkMppStatus(api_->control(context_, MPP_ENC_SET_RC_CFG, &rcCfg), "MPP_ENC_SET_RC_CFG failed");
  logMppRgbEncodeStep("init_set_rc_cfg_done");

  if (config.codec == "hevc" || config.codec == "h265") {
    throw std::runtime_error("Struct-based Rockchip encoder init currently supports h264 output only");
  }

  MppEncCodecCfg codecCfg = {};
  codecCfg.coding = MPP_VIDEO_CodingAVC;
  codecCfg.h264.change =
      MPP_ENC_H264_CFG_STREAM_TYPE |
      MPP_ENC_H264_CFG_CHANGE_PROFILE |
      MPP_ENC_H264_CFG_CHANGE_ENTROPY;
  codecCfg.h264.stream_type = 0;
  codecCfg.h264.profile = 100;
  codecCfg.h264.level = (width_ >= 1920 || height_ >= 1080) ? 40 : 31;
  codecCfg.h264.entropy_coding_mode = 1;
  codecCfg.h264.cabac_init_idc = 0;
  checkMppStatus(api_->control(context_, MPP_ENC_SET_CODEC_CFG, &codecCfg), "MPP_ENC_SET_CODEC_CFG failed");
  logMppRgbEncodeStep("init_set_codec_cfg_done");

  MppEncHeaderMode headerMode = MPP_ENC_HEADER_MODE_EACH_IDR;
  checkMppStatus(api_->control(context_, MPP_ENC_SET_HEADER_MODE, &headerMode),
                 "MPP_ENC_SET_HEADER_MODE failed");
  logMppRgbEncodeStep("init_set_header_mode_done");

  MppPacket headerPacket = nullptr;
  checkMppStatus(mpp_packet_init_with_buffer(&headerPacket, packetBuffer_), "mpp_packet_init_with_buffer failed");
  logMppRgbEncodeStep("init_header_packet_done");
  mpp_packet_set_length(headerPacket, 0);
  checkMppStatus(api_->control(context_, MPP_ENC_GET_HDR_SYNC, headerPacket), "MPP_ENC_GET_HDR_SYNC failed");
  logMppRgbEncodeStep("init_get_hdr_done");
  writePacket(headerPacket);
  mpp_packet_deinit(&headerPacket);
  logMppRgbEncodeStep("init_done");

  initialized_ = true;
}

void MppEncoder::encode(const RgbImage& frame, int64_t pts) {
#if defined(ENABLE_RGA_PREPROC) && !defined(WIN32)
  logMppRgbEncodeStep("encode_begin");
  if (!initialized_) {
    throw std::runtime_error("MPP encoder is not initialized");
  }
  if (inputFormat_ != PixelFormat::kRgb888) {
    throw std::runtime_error("MPP encoder is not configured for RGB input");
  }
  if (frame.width != width_ || frame.height != height_) {
    throw std::runtime_error("MPP encoder RGB input size mismatch");
  }
  if (frame.data.size() != static_cast<std::size_t>(frame.width * frame.height * 3)) {
    throw std::runtime_error("MPP encoder received an invalid RGB frame buffer");
  }
  if (inputBuffer_ == nullptr) {
    throw std::runtime_error("MPP encoder RGB input buffer is not allocated");
  }

  void* framePtr = mpp_buffer_get_ptr(inputBuffer_);
  logMppRgbEncodeStep("got_input_ptr");
  if (framePtr == nullptr) {
    throw std::runtime_error("mpp_buffer_get_ptr for encoder input failed");
  }

  {
    logMppRgbEncodeStep("rga_begin");
    std::lock_guard<std::mutex> lock(rgaMutex_);
    const size_t rgbBytes = frame.data.size();
    const size_t nv12Bytes = static_cast<size_t>(horStride_) * static_cast<size_t>(verStride_) * 3 / 2;
    rga_buffer_handle_t srcHandle =
        importbuffer_virtualaddr(const_cast<std::uint8_t*>(frame.data.data()), rgbBytes);
    if (srcHandle == 0) {
      throw std::runtime_error("RGA importbuffer_virtualaddr failed for RGB source");
    }
    rga_buffer_handle_t dstHandle = importbuffer_virtualaddr(framePtr, nv12Bytes);
    if (dstHandle == 0) {
      releasebuffer_handle(srcHandle);
      throw std::runtime_error("RGA importbuffer_virtualaddr failed for NV12 destination");
    }

    rga_buffer_t src = wrapbuffer_handle(srcHandle, width_, height_, RK_FORMAT_RGB_888);
    rga_buffer_t dst =
        wrapbuffer_handle(dstHandle, width_, height_, RK_FORMAT_YCbCr_420_SP, horStride_, verStride_);
    const IM_STATUS status = imcvtcolor(src, dst, RK_FORMAT_RGB_888, RK_FORMAT_YCbCr_420_SP);
    releasebuffer_handle(dstHandle);
    releasebuffer_handle(srcHandle);
    if (status != IM_STATUS_SUCCESS) {
      throw std::runtime_error("RGA RGB to NV12 conversion failed");
    }
  }
  logMppRgbEncodeStep("rga_done");

  MppFrame inputFrame = nullptr;
  MppPacket packet = nullptr;
  try {
    checkMppStatus(mpp_frame_init(&inputFrame), "mpp_frame_init failed");
    logMppRgbEncodeStep("frame_init_done");
    mpp_frame_set_width(inputFrame, width_);
    mpp_frame_set_height(inputFrame, height_);
    mpp_frame_set_hor_stride(inputFrame, horStride_);
    mpp_frame_set_ver_stride(inputFrame, verStride_);
    mpp_frame_set_fmt(inputFrame, MPP_FMT_YUV420SP);
    mpp_frame_set_pts(inputFrame, pts >= 0 ? pts : frameIndex_++);
    mpp_frame_set_buffer(inputFrame, inputBuffer_);

    checkMppStatus(mpp_packet_init_with_buffer(&packet, packetBuffer_),
                   "mpp_packet_init_with_buffer failed");
    logMppRgbEncodeStep("packet_init_done");
    mpp_packet_set_length(packet, 0);

    MppMeta meta = mpp_frame_get_meta(inputFrame);
    checkMppStatus(mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet), "mpp_meta_set_packet failed");
    logMppRgbEncodeStep("meta_set_packet_done");

    checkMppStatus(api_->encode_put_frame(context_, inputFrame), "encode_put_frame failed");
    logMppRgbEncodeStep("encode_put_frame_done");
    mpp_frame_deinit(&inputFrame);
    MppPacket encodedPacket = nullptr;
    checkMppStatus(api_->encode_get_packet(context_, &encodedPacket), "encode_get_packet failed");
    logMppRgbEncodeStep("encode_get_packet_done");
    if (encodedPacket != nullptr) {
      void* data = mpp_packet_get_data(encodedPacket);
      const size_t length = mpp_packet_get_length(encodedPacket);
      if (data != nullptr && length > 0) {
        outputFile_.write(static_cast<const char*>(data), static_cast<std::streamsize>(length));
      }
      if (encodedPacket == packet) {
        packet = nullptr;
      }
      mpp_packet_deinit(&encodedPacket);
    }
    if (packet != nullptr) {
      mpp_packet_deinit(&packet);
    }
    logMppRgbEncodeStep("encode_done");
  } catch (...) {
    if (packet != nullptr) {
      mpp_packet_deinit(&packet);
    }
    if (inputFrame != nullptr) {
      mpp_frame_deinit(&inputFrame);
    }
    throw;
  }
#else
  (void)frame;
  (void)pts;
  throw std::runtime_error("MPP encoder RGB input requires Rockchip RGA support");
#endif
}

void MppEncoder::encodeDecodedFrame(const DecodedFrame& frame, int64_t pts) {
  if (!initialized_) {
    throw std::runtime_error("MPP encoder is not initialized");
  }
  if (frame.dmaFd < 0) {
    throw std::runtime_error("MPP encoder requires a valid dma-buf fd");
  }
  if (frame.format != PixelFormat::kNv12 && frame.format != PixelFormat::kUnknown) {
    throw std::runtime_error("MPP encoder currently only supports NV12 decoded frames");
  }
  if (frame.width != width_ || frame.height != height_) {
    throw std::runtime_error("MPP encoder does not support resolution changes yet");
  }

  MppBuffer inputBuffer = nullptr;
  MppFrame inputFrame = nullptr;
  MppPacket outputPacket = nullptr;

  try {
    MppBufferInfo bufferInfo;
    std::memset(&bufferInfo, 0, sizeof(bufferInfo));
    bufferInfo.type = MPP_BUFFER_TYPE_EXT_DMA;
    bufferInfo.fd = frame.dmaFd;
    bufferInfo.size = static_cast<size_t>(frame.horizontalStride) * static_cast<size_t>(frame.verticalStride) * 3 / 2;
    checkMppStatus(mpp_buffer_import(&inputBuffer, &bufferInfo), "mpp_buffer_import failed");

    checkMppStatus(mpp_frame_init(&inputFrame), "mpp_frame_init failed");
    mpp_frame_set_width(inputFrame, frame.width);
    mpp_frame_set_height(inputFrame, frame.height);
    mpp_frame_set_hor_stride(inputFrame, frame.horizontalStride > 0 ? frame.horizontalStride : horStride_);
    mpp_frame_set_ver_stride(inputFrame, frame.verticalStride > 0 ? frame.verticalStride : verStride_);
    mpp_frame_set_fmt(inputFrame, MPP_FMT_YUV420SP);
    mpp_frame_set_pts(inputFrame, pts);
    mpp_frame_set_buffer(inputFrame, inputBuffer);

    checkMppStatus(mpp_packet_init_with_buffer(&outputPacket, packetBuffer_), "mpp_packet_init_with_buffer failed");
    mpp_packet_set_length(outputPacket, 0);

    MppMeta meta = mpp_frame_get_meta(inputFrame);
    mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, outputPacket);

    checkMppStatus(api_->encode_put_frame(context_, inputFrame), "encode_put_frame failed");
    mpp_frame_deinit(&inputFrame);
    mpp_buffer_put(inputBuffer);
    inputBuffer = nullptr;

    drainPackets(false);
  } catch (...) {
    if (outputPacket != nullptr) {
      mpp_packet_deinit(&outputPacket);
    }
    if (inputFrame != nullptr) {
      mpp_frame_deinit(&inputFrame);
    }
    if (inputBuffer != nullptr) {
      mpp_buffer_put(inputBuffer);
    }
    throw;
  }
}

void MppEncoder::flush() {
  if (!initialized_ || flushSubmitted_) {
    return;
  }

  MppFrame eosFrame = nullptr;
  try {
    checkMppStatus(mpp_frame_init(&eosFrame), "mpp_frame_init failed");
    mpp_frame_set_eos(eosFrame, 1);
    checkMppStatus(api_->encode_put_frame(context_, eosFrame), "encode_put_frame(eos) failed");
    flushSubmitted_ = true;
    mpp_frame_deinit(&eosFrame);
    drainPackets(true);
    outputFile_.flush();
  } catch (...) {
    if (eosFrame != nullptr) {
      mpp_frame_deinit(&eosFrame);
    }
    throw;
  }
}

void MppEncoder::close() {
  if (outputFile_.is_open()) {
    outputFile_.close();
  }
  if (inputBuffer_ != nullptr) {
    mpp_buffer_put(inputBuffer_);
    inputBuffer_ = nullptr;
  }
  if (packetBuffer_ != nullptr) {
    mpp_buffer_put(packetBuffer_);
    packetBuffer_ = nullptr;
  }
  if (bufferGroup_ != nullptr) {
    mpp_buffer_group_put(bufferGroup_);
    bufferGroup_ = nullptr;
  }
  if (config_ != nullptr) {
    mpp_enc_cfg_deinit(config_);
    config_ = nullptr;
  }
  if (context_ != nullptr) {
    mpp_destroy(context_);
    context_ = nullptr;
    api_ = nullptr;
  }

  initialized_ = false;
  flushSubmitted_ = false;
  width_ = 0;
  height_ = 0;
  horStride_ = 0;
  verStride_ = 0;
  frameIndex_ = 0;
  inputFormat_ = PixelFormat::kUnknown;
}

void MppEncoder::writePacket(void* opaquePacket) {
  MppPacket packet = static_cast<MppPacket>(opaquePacket);
  if (!packet) {
    return;
  }

  void* data = mpp_packet_get_pos(packet);
  const size_t length = mpp_packet_get_length(packet);
  if (data != nullptr && length > 0) {
    outputFile_.write(static_cast<const char*>(data), static_cast<std::streamsize>(length));
  }
}

void MppEncoder::drainPackets(bool untilEos) {
  int retries = 0;
  while (true) {
    MppPacket packet = nullptr;
    checkMppStatus(api_->encode_get_packet(context_, &packet), "encode_get_packet failed");
    if (packet == nullptr) {
      if (untilEos && retries++ < kDrainRetryCount) {
        std::this_thread::sleep_for(kDrainRetrySleep);
        continue;
      }
      return;
    }
    retries = 0;

    const bool packetEos = mpp_packet_get_eos(packet) != 0;
    writePacket(packet);
    mpp_packet_deinit(&packet);

    if (!untilEos || packetEos) {
      break;
    }
  }
}
