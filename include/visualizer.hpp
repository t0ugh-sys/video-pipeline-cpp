#pragma once

#include "detection.hpp"
#include "pipeline_types.hpp"

#include <memory>
#include <string>

/**
 * 可视化输出配置
 */
struct VisualConfig {
  bool display = false;          // 显示窗口
  std::string outputVideo;       // 输出视频文件路径
  std::string outputRtsp;        // 输出 RTSP 流地址
  int displayMaxWidth = 0;       // 显示路径最大宽度，0 表示不限制
  int displayMaxHeight = 0;      // 显示路径最大高度，0 表示不限制
  bool showLabel = true;         // 显示类别标签
  bool showConf = true;          // 显示置信度
  float bboxThickness = 2.0f;    // 框线粗细
  float fontScale = 0.5f;        // 字体大小
};

/**
 * 可视化后端接口
 */
class IVisualizer {
 public:
  virtual ~IVisualizer() = default;

  /**
   * 初始化
   * @param width 视频宽度
   * @param height 视频高度
   * @param config 可视化配置
   */
  virtual void init(int width, int height, const VisualConfig& config) = 0;

  /**
   * 绘制检测结果
   * @param frame 原始帧 (RGB)
   * @param result 检测结果
   * @return 绘制后的帧
   */
  virtual RgbImage draw(const RgbImage& frame, const DetectionResult& result) = 0;

  /**
   * 显示/保存当前帧
   */
  virtual void show() = 0;

  /**
   * 释放资源
   */
  virtual void close() = 0;

  /** 获取后端名称 */
  virtual std::string name() const = 0;

  /** 是否可用 */
  virtual bool isAvailable() const = 0;
};

/**
 * 创建可视化后端
 */
std::unique_ptr<IVisualizer> createVisualizer();
