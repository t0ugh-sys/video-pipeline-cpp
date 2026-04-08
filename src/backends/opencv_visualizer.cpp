#include "visualizer.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>

/**
 * OpenCV 可视化后端
 */
class OpenCVVisualizer : public IVisualizer {
 public:
  OpenCVVisualizer() = default;
  ~OpenCVVisualizer() override { close(); }

  void init(int width, int height, const VisualConfig& config) override {
    close();
    width_ = width;
    height_ = height;
    config_ = config;

    // 打开视频写入器
    if (!config_.outputVideo.empty()) {
      cvWriter_.open(
          config_.outputVideo,
          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
          30,
          cv::Size(width_, height_));
      if (!cvWriter_.isOpened()) {
        throw std::runtime_error("Failed to open video writer: " + config_.outputVideo);
      }
    }

    // 打开 RTSP 流 (可选)
    // 注意: RTSP 需要 FFmpeg 支持
    // if (!config_.outputRtsp.empty()) {
    //   rtspWriter_.open(config_.outputRtsp, ...);
    // }
  }

  RgbImage draw(const RgbImage& frame, const DetectionResult& result) override {
    // 转换 RgbImage 到 cv::Mat
    cv::Mat img(frame.height, frame.width, CV_8UC3, frame.data.data());

    // 绘制检测框
    for (const auto& box : result.boxes) {
      // 随机颜色
      cv::Scalar color = getColor(box.classId);

      // 绘制框
      cv::rectangle(
          img,
          cv::Point(box.x1, box.y1),
          cv::Point(box.x2, box.y2),
          color,
          static_cast<int>(config_.bboxThickness));

      // 构建标签文本
      std::string label;
      if (config_.showLabel && !box.label.empty()) {
        label = box.label;
      }
      if (config_.showConf) {
        if (!label.empty()) {
          label += " ";
        }
        label += std::to_string(static_cast<int>(box.score * 100)) + "%";
      }

      // 绘制标签背景
      if (!label.empty()) {
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, config_.fontScale, 1, &baseline);
        cv::rectangle(
            img,
            cv::Point(box.x1, box.y1 - textSize.height - 8),
            cv::Point(box.x1 + textSize.width + 8, box.y1),
            color,
            -1);

        // 绘制文本
        cv::putText(
            img,
            label,
            cv::Point(box.x1 + 4, box.y1 - 4),
            cv::FONT_HERSHEY_SIMPLEX,
            config_.fontScale,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
      }
    }

    // 转回 RgbImage
    RgbImage output;
    output.width = img.cols;
    output.height = img.rows;
    output.data.resize(img.total() * img.elemSize());
    std::memcpy(output.data.data(), img.data, output.data.size());

    return output;
  }

  void show() override {
    // 如果没有配置任何输出，不做处理
    if (!config_.display && config_.outputVideo.empty() && config_.outputRtsp.empty()) {
      return;
    }
    // 窗口显示在主循环中处理
  }

  void close() override {
    if (cvWriter_.isOpened()) {
      cvWriter_.release();
    }
    width_ = 0;
    height_ = 0;
  }

  std::string name() const override { return "OpenCV"; }

  bool isAvailable() const override { return true; }

  // 获取窗口显示用的图像
  cv::Mat getDisplayImage() const { return displayImage_; }

  // 设置显示图像
  void setDisplayImage(const cv::Mat& img) { displayImage_ = img.clone(); }

  // 获取写入器
  cv::VideoWriter& getWriter() { return cvWriter_; }

 private:
  cv::Scalar getColor(int classId) {
    static const cv::Scalar colors[] = {
        cv::Scalar(255, 0, 0),     // blue
        cv::Scalar(0, 255, 0),     // green
        cv::Scalar(0, 0, 255),     // red
        cv::Scalar(255, 255, 0),   // cyan
        cv::Scalar(255, 0, 255),   // magenta
        cv::Scalar(0, 255, 255),   // yellow
        cv::Scalar(128, 0, 128),   // purple
        cv::Scalar(255, 165, 0),   // orange
        cv::Scalar(128, 128, 0),   // olive
        cv::Scalar(0, 128, 128),   // teal
    };
    return colors[classId % 10];
  }

  int width_ = 0;
  int height_ = 0;
  VisualConfig config_;
  cv::VideoWriter cvWriter_;
  cv::Mat displayImage_;
};

// 简单的占位实现 (无 OpenCV 时)
class DummyVisualizer : public IVisualizer {
 public:
  void init(int width, int height, const VisualConfig& config) override {
    (void)width;
    (void)height;
    (void)config;
  }

  RgbImage draw(const RgbImage& frame, const DetectionResult& result) override {
    (void)result;
    return frame;
  }

  void show() override {}
  void close() override {}

  std::string name() const override { return "Dummy"; }
  bool isAvailable() const override { return false; }
};

std::unique_ptr<IVisualizer> createVisualizer() {
#ifdef OpenCV_FOUND
  return std::make_unique<OpenCVVisualizer>();
#else
  return std::make_unique<DummyVisualizer>();
#endif
}
