#include "backends/nvenc_encoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}

#include <cstring>
#include <stdexcept>

namespace {

std::string toLowerAscii(std::string value) {
  for (char& ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return value;
}

bool startsWithIgnoreCase(const std::string& value, const std::string& prefix) {
  const std::string lowerValue = toLowerAscii(value);
  const std::string lowerPrefix = toLowerAscii(prefix);
  return lowerValue.size() >= lowerPrefix.size() &&
         lowerValue.compare(0, lowerPrefix.size(), lowerPrefix) == 0;
}

bool isRtspUrl(const std::string& value) {
  return startsWithIgnoreCase(value, "rtsp://");
}

void checkAvStatus(int status, const char* message) {
  if (status < 0) {
    char err[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(status, err, sizeof(err));
    throw std::runtime_error(std::string(message) + ": " + err);
  }
}

}  // namespace

NvencEncoder::NvencEncoder() = default;

NvencEncoder::~NvencEncoder() {
  close();
}

void NvencEncoder::init(const EncoderConfig& config) {
  close();
  try {
    if (config.width <= 0 || config.height <= 0) {
      throw std::runtime_error("NVENC encoder requires a positive frame size");
    }
    if (config.fps <= 0) {
      throw std::runtime_error("NVENC encoder requires a positive fps");
    }

    int ret = av_hwdevice_ctx_create(&hwDeviceCtx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    checkAvStatus(ret, "Failed to create CUDA hardware device context");

    const AVCodec* encoder = avcodec_find_encoder_by_name(
        config.codec == "h265" ? "hevc_nvenc" : "h264_nvenc");
    if (!encoder) {
      throw std::runtime_error("Requested NVENC encoder is not available in FFmpeg");
    }

    codecCtx_ = avcodec_alloc_context3(encoder);
    if (!codecCtx_) {
      throw std::runtime_error("Failed to allocate NVENC codec context");
    }

    width_ = config.width;
    height_ = config.height;

    codecCtx_->width = config.width;
    codecCtx_->height = config.height;
    codecCtx_->framerate = {config.fps, 1};
    codecCtx_->time_base = {1, config.fps};
    if (config.fpsNum > 0 && config.fpsDen > 0) {
      codecCtx_->framerate = {config.fpsNum, config.fpsDen};
      codecCtx_->time_base = {config.fpsDen, config.fpsNum};
    }
    codecCtx_->bit_rate = config.bitrate;
    codecCtx_->gop_size = 30;
    codecCtx_->max_b_frames = 0;
    codecCtx_->pix_fmt = AV_PIX_FMT_CUDA;
    codecCtx_->hw_device_ctx = av_buffer_ref(hwDeviceCtx_);
    if (!codecCtx_->hw_device_ctx) {
      throw std::runtime_error("Failed to reference CUDA device context for NVENC");
    }

    hwFramesCtx_ = av_hwframe_ctx_alloc(hwDeviceCtx_);
    if (!hwFramesCtx_) {
      throw std::runtime_error("Failed to allocate NVENC hardware frame context");
    }

    auto* framesCtx = reinterpret_cast<AVHWFramesContext*>(hwFramesCtx_->data);
    framesCtx->format = AV_PIX_FMT_CUDA;
    framesCtx->sw_format = AV_PIX_FMT_RGB24;
    framesCtx->width = config.width;
    framesCtx->height = config.height;
    framesCtx->initial_pool_size = 2;
    ret = av_hwframe_ctx_init(hwFramesCtx_);
    checkAvStatus(ret, "Failed to initialize NVENC hardware frame context");

    codecCtx_->hw_frames_ctx = av_buffer_ref(hwFramesCtx_);
    if (!codecCtx_->hw_frames_ctx) {
      throw std::runtime_error("Failed to reference NVENC hardware frame context");
    }

    av_opt_set(codecCtx_->priv_data, "preset", "p4", 0);
    av_opt_set(codecCtx_->priv_data, "tune", "ull", 0);

    ret = avcodec_open2(codecCtx_, encoder, nullptr);
    checkAvStatus(ret, "Failed to open NVENC encoder");

    avformat_network_init();
    const char* formatName = isRtspUrl(config.outputPath) ? "rtsp" : nullptr;
    ret = avformat_alloc_output_context2(&formatCtx_, nullptr, formatName, config.outputPath.c_str());
    checkAvStatus(ret, "Failed to create output format context");
    if (!formatCtx_) {
      throw std::runtime_error("Failed to create output format context");
    }

    stream_ = avformat_new_stream(formatCtx_, nullptr);
    if (!stream_) {
      throw std::runtime_error("Failed to create output stream");
    }
    stream_->time_base = codecCtx_->time_base;
    ret = avcodec_parameters_from_context(stream_->codecpar, codecCtx_);
    checkAvStatus(ret, "Failed to copy NVENC codec parameters");

    if (!(formatCtx_->oformat->flags & AVFMT_NOFILE)) {
      ret = avio_open(&formatCtx_->pb, config.outputPath.c_str(), AVIO_FLAG_WRITE);
      checkAvStatus(ret, "Failed to open output muxer target");
    }

    AVDictionary* muxOptions = nullptr;
    if (isRtspUrl(config.outputPath)) {
      av_dict_set(&muxOptions, "rtsp_transport", "tcp", 0);
      av_dict_set(&muxOptions, "muxdelay", "0.1", 0);
    }
    ret = avformat_write_header(formatCtx_, &muxOptions);
    av_dict_free(&muxOptions);
    checkAvStatus(ret, "Failed to write output container header");

    swFrame_ = av_frame_alloc();
    if (!swFrame_) {
      throw std::runtime_error("Failed to allocate NVENC software frame");
    }
    swFrame_->format = AV_PIX_FMT_RGB24;
    swFrame_->width = config.width;
    swFrame_->height = config.height;
    ret = av_frame_get_buffer(swFrame_, 32);
    checkAvStatus(ret, "Failed to allocate NVENC software frame buffer");

    hwFrame_ = av_frame_alloc();
    if (!hwFrame_) {
      throw std::runtime_error("Failed to allocate NVENC hardware frame");
    }
    ret = av_hwframe_get_buffer(hwFramesCtx_, hwFrame_, 0);
    checkAvStatus(ret, "Failed to allocate NVENC hardware frame buffer");

    initialized_ = true;
  } catch (...) {
    close();
    throw;
  }
}

void NvencEncoder::encode(const RgbImage& frame, int64_t pts) {
  if (!initialized_) {
    return;
  }
  if (frame.width != width_ || frame.height != height_) {
    throw std::runtime_error("NVENC encoder received an RGB frame with a mismatched size");
  }
  if (frame.data.size() != static_cast<std::size_t>(frame.width * frame.height * 3)) {
    throw std::runtime_error("NVENC encoder received an invalid RGB frame buffer");
  }

  int ret = av_frame_make_writable(swFrame_);
  checkAvStatus(ret, "Failed to make NVENC software frame writable");
  for (int y = 0; y < height_; ++y) {
    std::memcpy(
        swFrame_->data[0] + static_cast<std::size_t>(y) * static_cast<std::size_t>(swFrame_->linesize[0]),
        frame.data.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) * 3,
        static_cast<std::size_t>(width_) * 3);
  }
  swFrame_->pts = pts;

  ret = av_hwframe_transfer_data(hwFrame_, swFrame_, 0);
  checkAvStatus(ret, "Failed to upload RGB frame to NVENC hardware memory");
  hwFrame_->pts = pts;

  ret = avcodec_send_frame(codecCtx_, hwFrame_);
  checkAvStatus(ret, "Failed to submit frame to NVENC encoder");

  AVPacket* packet = av_packet_alloc();
  if (!packet) {
    throw std::runtime_error("Failed to allocate NVENC packet");
  }
  try {
    while (true) {
      ret = avcodec_receive_packet(codecCtx_, packet);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      }
      checkAvStatus(ret, "Failed to receive packet from NVENC encoder");
      av_packet_rescale_ts(packet, codecCtx_->time_base, stream_->time_base);
      packet->stream_index = stream_->index;
      const int writeRet = av_interleaved_write_frame(formatCtx_, packet);
      checkAvStatus(writeRet, "Failed to write NVENC packet to muxer");
      av_packet_unref(packet);
    }
  } catch (...) {
    av_packet_free(&packet);
    throw;
  }
  av_packet_free(&packet);
}

void NvencEncoder::flush() {
  if (!initialized_) {
    return;
  }

  int ret = avcodec_send_frame(codecCtx_, nullptr);
  checkAvStatus(ret, "Failed to flush NVENC encoder");

  AVPacket* packet = av_packet_alloc();
  if (!packet) {
    throw std::runtime_error("Failed to allocate NVENC flush packet");
  }
  try {
    while (true) {
      ret = avcodec_receive_packet(codecCtx_, packet);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      }
      checkAvStatus(ret, "Failed to receive flush packet from NVENC encoder");
      av_packet_rescale_ts(packet, codecCtx_->time_base, stream_->time_base);
      packet->stream_index = stream_->index;
      const int writeRet = av_interleaved_write_frame(formatCtx_, packet);
      checkAvStatus(writeRet, "Failed to write flush packet to muxer");
      av_packet_unref(packet);
    }
  } catch (...) {
    av_packet_free(&packet);
    throw;
  }
  av_packet_free(&packet);
  if (formatCtx_) {
    const int trailerRet = av_write_trailer(formatCtx_);
    checkAvStatus(trailerRet, "Failed to finalize output container");
  }
}

void NvencEncoder::close() {
  if (swFrame_) {
    av_frame_free(&swFrame_);
  }
  if (hwFrame_) {
    av_frame_free(&hwFrame_);
  }
  if (codecCtx_) {
    avcodec_free_context(&codecCtx_);
  }
  if (formatCtx_) {
    if (!(formatCtx_->oformat->flags & AVFMT_NOFILE) && formatCtx_->pb) {
      avio_closep(&formatCtx_->pb);
    }
    avformat_free_context(formatCtx_);
    formatCtx_ = nullptr;
  }
  stream_ = nullptr;
  if (hwFramesCtx_) {
    av_buffer_unref(&hwFramesCtx_);
  }
  if (hwDeviceCtx_) {
    av_buffer_unref(&hwDeviceCtx_);
  }
  initialized_ = false;
  width_ = 0;
  height_ = 0;
}
