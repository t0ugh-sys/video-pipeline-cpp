#pragma once

#include "pipeline_types.hpp"

#include <memory>
#include <optional>
#include <string>

class IDecoderBackend {
 public:
  virtual ~IDecoderBackend() = default;

  virtual void open(VideoCodec codec) = 0;

  // Submits one encoded packet into the decoder. The decoder retains any
  // internally queued output frames until receiveFrame() drains them.
  virtual void submitPacket(const EncodedPacket& packet) = 0;

  // Retrieves the next decoded frame if available. The returned frame owns the
  // backend-native decode buffer for as long as the DecodedFrame object lives,
  // via DecodedFrame::nativeHandle.
  virtual std::optional<DecodedFrame> receiveFrame() = 0;

  virtual std::string name() const = 0;
};

enum class DecoderBackendType {
  kAuto,
  kRockchipMpp,
  kNvidiaNvdec,
  kCpu,
};

std::unique_ptr<IDecoderBackend> createDecoderBackend(DecoderBackendType type = DecoderBackendType::kAuto);

DecoderBackendType detectAvailableDecoderBackend();
