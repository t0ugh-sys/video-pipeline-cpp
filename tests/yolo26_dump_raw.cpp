#include "backends/rknn_infer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

RgbImage letterboxToRgb(const cv::Mat& bgrImage, int targetWidth, int targetHeight) {
  const float scale = std::min(
      static_cast<float>(targetWidth) / static_cast<float>(bgrImage.cols),
      static_cast<float>(targetHeight) / static_cast<float>(bgrImage.rows));
  const int resizedWidth = std::max(1, static_cast<int>(std::round(static_cast<float>(bgrImage.cols) * scale)));
  const int resizedHeight = std::max(1, static_cast<int>(std::round(static_cast<float>(bgrImage.rows) * scale)));
  const int padLeft = (targetWidth - resizedWidth) / 2;
  const int padTop = (targetHeight - resizedHeight) / 2;

  cv::Mat resized;
  cv::resize(bgrImage, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat canvas(targetHeight, targetWidth, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(padLeft, padTop, resizedWidth, resizedHeight)));

  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

  RgbImage image;
  image.width = targetWidth;
  image.height = targetHeight;
  image.wstride = targetWidth;
  image.hstride = targetHeight;
  image.format = PixelFormat::kRgb888;
  image.data.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());
  image.letterbox.enabled = true;
  image.letterbox.scale = scale;
  image.letterbox.resizedWidth = resizedWidth;
  image.letterbox.resizedHeight = resizedHeight;
  image.letterbox.padLeft = padLeft;
  image.letterbox.padTop = padTop;
  image.letterbox.padRight = targetWidth - resizedWidth - padLeft;
  image.letterbox.padBottom = targetHeight - resizedHeight - padTop;
  return image;
}

void dumpTensor(const InferenceTensor& tensor, int maxRows) {
  if (tensor.shape.size() < 3) {
    throw std::runtime_error("expected [1, proposals, 6] or [1, 6, proposals]");
  }
  const bool proposalFirst = tensor.shape[1] >= tensor.shape[2];
  const int proposals = static_cast<int>(proposalFirst ? tensor.shape[1] : tensor.shape[2]);
  const int attrs = static_cast<int>(proposalFirst ? tensor.shape[2] : tensor.shape[1]);
  if (attrs != 6) {
    throw std::runtime_error("expected 6 attributes, got " + std::to_string(attrs));
  }
  if (tensor.data.empty()) {
    throw std::runtime_error("tensor.data is empty; expected float-decoded RKNN output");
  }

  std::cout << "tensor=" << tensor.name
            << " layout=" << tensor.layout
            << " shape=[";
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << tensor.shape[i];
  }
  std::cout << "] proposal_first=" << (proposalFirst ? "true" : "false") << "\n";

  const int rowCount = std::min(proposals, maxRows);
  for (int row = 0; row < rowCount; ++row) {
    std::cout << row << ":";
    for (int attr = 0; attr < attrs; ++attr) {
      const std::size_t index = proposalFirst
          ? static_cast<std::size_t>(row * attrs + attr)
          : static_cast<std::size_t>(attr * proposals + row);
      std::cout << (attr == 0 ? " " : ", ") << tensor.data[index];
    }
    std::cout << "\n";
  }
}

cv::Mat makeLetterboxedBgrPreview(const RgbImage& image) {
  cv::Mat rgb(image.height, image.width, CV_8UC3, const_cast<std::uint8_t*>(image.data.data()));
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  return bgr.clone();
}

void drawRawBoxes(
    cv::Mat& image,
    const InferenceTensor& tensor,
    int maxRows,
    bool swapClassScore) {
  const bool proposalFirst = tensor.shape[1] >= tensor.shape[2];
  const int proposals = static_cast<int>(proposalFirst ? tensor.shape[1] : tensor.shape[2]);
  const int attrs = static_cast<int>(proposalFirst ? tensor.shape[2] : tensor.shape[1]);
  if (attrs != 6 || tensor.data.empty()) {
    return;
  }

  const int classIndex = swapClassScore ? 5 : 4;
  const int scoreIndex = swapClassScore ? 4 : 5;
  const int rowCount = std::min(proposals, maxRows);
  for (int row = 0; row < rowCount; ++row) {
    auto valueAt = [&](int attr) -> float {
      const std::size_t index = proposalFirst
          ? static_cast<std::size_t>(row * attrs + attr)
          : static_cast<std::size_t>(attr * proposals + row);
      return tensor.data[index];
    };
    const float x1 = valueAt(0);
    const float y1 = valueAt(1);
    const float x2 = valueAt(2);
    const float y2 = valueAt(3);
    const float cls = valueAt(classIndex);
    const float score = valueAt(scoreIndex);
    const cv::Scalar color = swapClassScore ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
    cv::rectangle(
        image,
        cv::Point(static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1))),
        cv::Point(static_cast<int>(std::round(x2)), static_cast<int>(std::round(y2))),
        color,
        2);
    if (row < 10) {
      const std::string label =
          (swapClassScore ? "swap" : "raw") + std::string(" r=") + std::to_string(row) +
          " c=" + std::to_string(static_cast<int>(std::round(cls))) +
          " s=" + std::to_string(score);
      cv::putText(
          image,
          label,
          cv::Point(static_cast<int>(std::round(x1)), std::max(15, static_cast<int>(std::round(y1)) - 4)),
          cv::FONT_HERSHEY_SIMPLEX,
          0.45,
          color,
          1,
          cv::LINE_AA);
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const std::string modelPath =
        argc > 1 ? argv[1] : "/edge/workspace/yolo26n-rk3588-int8-coco200.rknn";
    const std::string imagePath =
        argc > 2 ? argv[2] : "/edge/workspace/yolo26_source_frame1.png";
    const int maxRows = argc > 3 ? std::max(1, std::stoi(argv[3])) : 30;

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("failed to load image: " + imagePath);
    }

    ModelConfig config;
    config.modelPath = modelPath;

    RknnInfer infer;
    InferRuntimeConfig runtime;
    runtime.verbose = true;
    infer.open(config, runtime);

    const RgbImage modelInput = letterboxToRgb(image, infer.inputWidth(), infer.inputHeight());
    const InferenceOutput output = infer.infer(modelInput);
    if (output.empty()) {
      throw std::runtime_error("inference returned no tensors");
    }

    std::cout << "model=" << modelPath << "\n";
    std::cout << "image=" << imagePath << "\n";
    std::cout << "letterbox scale=" << modelInput.letterbox.scale
              << " pad_left=" << modelInput.letterbox.padLeft
              << " pad_top=" << modelInput.letterbox.padTop
              << " resized=" << modelInput.letterbox.resizedWidth
              << "x" << modelInput.letterbox.resizedHeight << "\n";
    dumpTensor(output.front(), maxRows);

    cv::Mat preview = makeLetterboxedBgrPreview(modelInput);
    drawRawBoxes(preview, output.front(), maxRows, false);
    drawRawBoxes(preview, output.front(), maxRows, true);
    const std::string previewPath = "/edge/workspace/yolo26_raw_boxes_preview.png";
    cv::imwrite(previewPath, preview);
    std::cout << "preview=" << previewPath << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "yolo26_dump_raw failed: " << error.what() << "\n";
    return 1;
  }
}
