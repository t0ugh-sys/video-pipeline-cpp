#pragma once

#include "app_config.hpp"
#include "pipeline_types.hpp"

#include <cstddef>

int resolveEncoderFps(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo);
int resolveEncoderFpsNum(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo);
int resolveEncoderFpsDen(const AppConfig& config, const SourceVideoInfo& sourceVideoInfo);
bool shouldKeepEncodedFrame(
    std::size_t zeroBasedFrameIndex,
    const SourceVideoInfo& sourceVideoInfo,
    int targetFps);
