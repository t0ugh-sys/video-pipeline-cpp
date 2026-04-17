#include "visualizer.hpp"

#include <opencv2/opencv.hpp>

#include "../../../rknn_model_zoo/utils/font.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int COLOR_BLUE = 0xFF0000FFU;
constexpr unsigned int COLOR_RED = 0xFFFF0000U;
constexpr int kModelZooBoxThickness = 3;
constexpr int kModelZooFontPixelSize = 10;

int clampValue(float value, int minValue, int maxValue) {
  if (value < static_cast<float>(minValue)) {
    return minValue;
  }
  if (value > static_cast<float>(maxValue)) {
    return maxValue;
  }
  return static_cast<int>(value);
}

unsigned int convertColorRgb888(unsigned int srcColor) {
  unsigned int dstColor = 0;
  unsigned char* src = reinterpret_cast<unsigned char*>(&srcColor);
  unsigned char* dst = reinterpret_cast<unsigned char*>(&dstColor);
  const unsigned char r = src[2];
  const unsigned char g = src[1];
  const unsigned char b = src[0];
  dst[0] = r;
  dst[1] = g;
  dst[2] = b;
  return dstColor;
}

void drawRectangleC3(
    unsigned char* pixels,
    int width,
    int height,
    int rx,
    int ry,
    int rw,
    int rh,
    unsigned int color,
    int thickness) {
  const unsigned char* penColor = reinterpret_cast<unsigned char*>(&color);
  const int stride = width * 3;

  if (thickness == -1) {
    for (int y = ry; y < ry + rh; ++y) {
      if (y < 0) {
        continue;
      }
      if (y >= height) {
        break;
      }
      unsigned char* p = pixels + stride * y;
      for (int x = rx; x < rx + rw; ++x) {
        if (x < 0) {
          continue;
        }
        if (x >= width) {
          break;
        }
        p[x * 3 + 0] = penColor[0];
        p[x * 3 + 1] = penColor[1];
        p[x * 3 + 2] = penColor[2];
      }
    }
    return;
  }

  const int t0 = thickness / 2;
  const int t1 = thickness - t0;

  for (int y = ry - t0; y < ry + t1; ++y) {
    if (y < 0) {
      continue;
    }
    if (y >= height) {
      break;
    }
    unsigned char* p = pixels + stride * y;
    for (int x = rx - t0; x < rx + rw + t1; ++x) {
      if (x < 0) {
        continue;
      }
      if (x >= width) {
        break;
      }
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }

  for (int y = ry + rh - t0; y < ry + rh + t1; ++y) {
    if (y < 0) {
      continue;
    }
    if (y >= height) {
      break;
    }
    unsigned char* p = pixels + stride * y;
    for (int x = rx - t0; x < rx + rw + t1; ++x) {
      if (x < 0) {
        continue;
      }
      if (x >= width) {
        break;
      }
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }

  for (int x = rx - t0; x < rx + t1; ++x) {
    if (x < 0) {
      continue;
    }
    if (x >= width) {
      break;
    }
    for (int y = ry + t1; y < ry + rh - t0; ++y) {
      if (y < 0) {
        continue;
      }
      if (y >= height) {
        break;
      }
      unsigned char* p = pixels + stride * y;
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }

  for (int x = rx + rw - t0; x < rx + rw + t1; ++x) {
    if (x < 0) {
      continue;
    }
    if (x >= width) {
      break;
    }
    for (int y = ry + t1; y < ry + rh - t0; ++y) {
      if (y < 0) {
        continue;
      }
      if (y >= height) {
        break;
      }
      unsigned char* p = pixels + stride * y;
      p[x * 3 + 0] = penColor[0];
      p[x * 3 + 1] = penColor[1];
      p[x * 3 + 2] = penColor[2];
    }
  }
}

int resizeBilinearC1(
    const unsigned char* srcPixels,
    int srcWidth,
    int srcHeight,
    unsigned char* dstPixels,
    int dstWidth,
    int dstHeight) {
  const int xRatio = static_cast<int>((static_cast<float>(srcWidth - 1) / dstWidth) * (1 << 16));
  const int yRatio = static_cast<int>((static_cast<float>(srcHeight - 1) / dstHeight) * (1 << 16));

  for (int i = 0; i < dstHeight; ++i) {
    for (int j = 0; j < dstWidth; ++j) {
      const int x = (xRatio * j) >> 16;
      const int y = (yRatio * i) >> 16;
      const int xDiff = (xRatio * j) & 0xffff;
      const int yDiff = (yRatio * i) & 0xffff;

      const int index = y * srcWidth + x;
      const int a = srcPixels[index];
      const int b = srcPixels[index + 1];
      const int c = srcPixels[index + srcWidth];
      const int d = srcPixels[index + srcWidth + 1];

      const std::uint64_t accum =
          static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(65536 - xDiff) * static_cast<std::uint64_t>(65536 - yDiff) +
          static_cast<std::uint64_t>(b) * static_cast<std::uint64_t>(xDiff) * static_cast<std::uint64_t>(65536 - yDiff) +
          static_cast<std::uint64_t>(c) * static_cast<std::uint64_t>(yDiff) * static_cast<std::uint64_t>(65536 - xDiff) +
          static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(xDiff) * static_cast<std::uint64_t>(yDiff);
      dstPixels[i * dstWidth + j] = static_cast<unsigned char>(accum >> 32);
    }
  }

  return 0;
}

void drawTextC3(
    unsigned char* pixels,
    int width,
    int height,
    const char* text,
    int x,
    int y,
    int fontPixelSize,
    unsigned int color) {
  const unsigned char* penColor = reinterpret_cast<unsigned char*>(&color);
  const int stride = width * 3;
  std::vector<unsigned char> resizedFontBitmap(
      static_cast<std::size_t>(fontPixelSize * fontPixelSize * 2));

  const int n = static_cast<int>(std::strlen(text));
  int cursorX = x;
  int cursorY = y;
  for (int i = 0; i < n; ++i) {
    const char ch = text[i];
    if (ch == '\n') {
      cursorX = x;
      cursorY += fontPixelSize * 2;
      continue;
    }
    if (std::isprint(static_cast<unsigned char>(ch)) == 0) {
      continue;
    }

    const int fontBitmapIndex = ch - ' ';
    if (fontBitmapIndex < 0 || fontBitmapIndex >= 95) {
      continue;
    }
    const unsigned char* fontBitmap = mono_font_data[fontBitmapIndex];
    resizeBilinearC1(fontBitmap, 20, 40, resizedFontBitmap.data(), fontPixelSize, fontPixelSize * 2);

    for (int j = cursorY; j < cursorY + fontPixelSize * 2; ++j) {
      if (j < 0) {
        continue;
      }
      if (j >= height) {
        break;
      }

      const unsigned char* alpha = resizedFontBitmap.data() +
          static_cast<std::size_t>(j - cursorY) * fontPixelSize;
      unsigned char* p = pixels + stride * j;

      for (int k = cursorX; k < cursorX + fontPixelSize; ++k) {
        if (k < 0) {
          continue;
        }
        if (k >= width) {
          break;
        }

        const unsigned char a = alpha[k - cursorX];
        p[k * 3 + 0] = static_cast<unsigned char>((p[k * 3 + 0] * (255 - a) + penColor[0] * a) / 255);
        p[k * 3 + 1] = static_cast<unsigned char>((p[k * 3 + 1] * (255 - a) + penColor[1] * a) / 255);
        p[k * 3 + 2] = static_cast<unsigned char>((p[k * 3 + 2] * (255 - a) + penColor[2] * a) / 255);
      }
    }

    cursorX += fontPixelSize;
  }
}

void drawRectangle(
    RgbImage& image,
    int x,
    int y,
    int width,
    int height,
    unsigned int color,
    int thickness) {
  if (image.data.empty() || image.width <= 0 || image.height <= 0) {
    return;
  }
  const unsigned int drawColor = convertColorRgb888(color);
  drawRectangleC3(image.data.data(), image.width, image.height, x, y, width, height, drawColor, thickness);
}

void drawText(
    RgbImage& image,
    const char* text,
    int x,
    int y,
    unsigned int color,
    int fontPixelSize) {
  if (image.data.empty() || image.width <= 0 || image.height <= 0) {
    return;
  }
  const unsigned int drawColor = convertColorRgb888(color);
  drawTextC3(image.data.data(), image.width, image.height, text, x, y, fontPixelSize, drawColor);
}

cv::Mat rgbImageToMat(const RgbImage& frame) {
  if (frame.width <= 0 || frame.height <= 0) {
    throw std::runtime_error("Visualizer received an invalid frame size");
  }
  if (frame.data.size() != static_cast<std::size_t>(frame.width * frame.height * 3)) {
    throw std::runtime_error("Visualizer received an invalid RGB frame buffer");
  }

  cv::Mat rgb(frame.height, frame.width, CV_8UC3, const_cast<std::uint8_t*>(frame.data.data()));
  return rgb.clone();
}

class OpenCVVisualizer : public IVisualizer {
 public:
  OpenCVVisualizer() = default;
  ~OpenCVVisualizer() override { close(); }

  void init(int width, int height, const VisualConfig& config) override {
    close();
    width_ = width;
    height_ = height;
    config_ = config;

    if (!config_.outputVideo.empty()) {
      writer_.open(config_.outputVideo, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, cv::Size(width_, height_));
      if (!writer_.isOpened()) {
        throw std::runtime_error("Failed to open video writer: " + config_.outputVideo);
      }
    }

    if (config_.display) {
      window_name_ = "video_pipeline";
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    }
  }

  RgbImage draw(const RgbImage& frame, const DetectionResult& result) override {
    RgbImage output = frame;
    for (const auto& box : result.boxes) {
      const int x1 = clampValue(box.x1, 0, std::max(0, output.width - 1));
      const int y1 = clampValue(box.y1, 0, std::max(0, output.height - 1));
      const int x2 = clampValue(box.x2, 0, std::max(0, output.width - 1));
      const int y2 = clampValue(box.y2, 0, std::max(0, output.height - 1));
      const int w = std::max(1, x2 - x1);
      const int h = std::max(1, y2 - y1);

      drawRectangle(output, x1, y1, w, h, COLOR_BLUE, kModelZooBoxThickness);

      char text[256] = {};
      if (config_.showLabel && !box.label.empty() && config_.showConf) {
        std::snprintf(text, sizeof(text), "%s %.1f%%", box.label.c_str(), box.score * 100.0f);
      } else if (config_.showLabel && !box.label.empty()) {
        std::snprintf(text, sizeof(text), "%s", box.label.c_str());
      } else if (config_.showConf) {
        std::snprintf(text, sizeof(text), "%.1f%%", box.score * 100.0f);
      }
      if (text[0] != '\0') {
        drawText(output, text, x1, y1 - 20, COLOR_RED, kModelZooFontPixelSize);
      }
    }

    display_image_ = rgbImageToMat(output);
    if (writer_.isOpened()) {
      cv::Mat bgr;
      cv::cvtColor(display_image_, bgr, cv::COLOR_RGB2BGR);
      writer_.write(bgr);
    }
    return output;
  }

  void show() override {
    if (config_.display && !display_image_.empty()) {
      cv::Mat bgr;
      cv::cvtColor(display_image_, bgr, cv::COLOR_RGB2BGR);
      cv::imshow(window_name_, bgr);
      cv::waitKey(1);
    }
  }

  void close() override {
    if (writer_.isOpened()) {
      writer_.release();
    }
    if (!window_name_.empty()) {
      cv::destroyWindow(window_name_);
      window_name_.clear();
    }
    display_image_.release();
  }

  std::string name() const override { return "OpenCV"; }
  bool isAvailable() const override { return true; }

 private:
  int width_ = 0;
  int height_ = 0;
  VisualConfig config_;
  cv::VideoWriter writer_;
  cv::Mat display_image_;
  std::string window_name_;
};

}  // namespace

std::unique_ptr<IVisualizer> createVisualizer() {
  return std::make_unique<OpenCVVisualizer>();
}
