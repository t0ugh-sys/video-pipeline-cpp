#include "backends/yolo_postproc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

namespace {
struct DenseLayout { int proposals = 0; int attributes = 0; bool transposed = false; };
struct TensorView { const InferenceTensor* tensor = nullptr; int channels = 0; int height = 0; int width = 0; bool nchw = true; };
struct BranchSummary { int boxCount = 0; int clsCount = 0; int scoreCount = 0; };
struct DenseOrientationStats {
  float bestScore = -1.0f;
  float classMin = std::numeric_limits<float>::infinity();
  float classMax = -std::numeric_limits<float>::infinity();
};
constexpr std::size_t kMaxCandidatesBeforeNms = 200;
constexpr std::size_t kMaxDetectionsAfterNms = 50;

std::vector<std::string> loadLabels(const std::string& path) {
  std::vector<std::string> labels;
  if (path.empty()) return labels;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) if (!line.empty()) labels.push_back(line);
  return labels;
}

bool buildDenseLayout(const InferenceTensor& tensor, DenseLayout& layout) {
  if (tensor.shape.size() == 3 && tensor.shape[0] == 1) {
    const int a = static_cast<int>(tensor.shape[1]);
    const int b = static_cast<int>(tensor.shape[2]);
    layout.attributes = std::min(a, b);
    layout.proposals = std::max(a, b);
    layout.transposed = a <= b;
    return true;
  }
  if (tensor.shape.size() == 2) {
    const int a = static_cast<int>(tensor.shape[0]);
    const int b = static_cast<int>(tensor.shape[1]);
    layout.attributes = std::min(a, b);
    layout.proposals = std::max(a, b);
    layout.transposed = a <= b;
    return true;
  }
  return false;
}

bool buildTensorView(const InferenceTensor& tensor, TensorView& view) {
  if (tensor.shape.size() != 4) return false;
  view.tensor = &tensor;
  view.nchw = tensor.layout != "NHWC";
  if (view.nchw) {
    view.channels = static_cast<int>(tensor.shape[1]);
    view.height = static_cast<int>(tensor.shape[2]);
    view.width = static_cast<int>(tensor.shape[3]);
  } else {
    view.height = static_cast<int>(tensor.shape[1]);
    view.width = static_cast<int>(tensor.shape[2]);
    view.channels = static_cast<int>(tensor.shape[3]);
  }
  return view.channels > 0 && view.height > 0 && view.width > 0;
}

bool looksLikeYolo26E2E(const InferenceTensor& tensor) {
  DenseLayout layout;
  return buildDenseLayout(tensor, layout) && layout.attributes == 6;
}

bool looksLikeYolov8Flat(const InferenceTensor& tensor) {
  DenseLayout layout;
  if (!buildDenseLayout(tensor, layout)) {
    return false;
  }
  // Typical flat YOLOv8 export:
  //   [1, 84, 8400] or [1, 8400, 84]
  // Some variants keep objectness as 85 attrs.
  return layout.proposals >= 100 && (layout.attributes == 84 || layout.attributes == 85);
}

std::map<std::pair<int, int>, BranchSummary> summarizeYoloBranches(
    const InferenceOutput& output,
    int expectedClassChannels) {
  std::map<std::pair<int, int>, BranchSummary> summaries;
  for (const auto& tensor : output) {
    TensorView view;
    if (!buildTensorView(tensor, view)) {
      continue;
    }

    auto& summary = summaries[{view.height, view.width}];
    if (view.channels == 1) {
      summary.scoreCount++;
    } else if (view.channels == expectedClassChannels) {
      summary.clsCount++;
    } else if (view.channels == 4 || (view.channels % 4 == 0 && view.channels <= 64)) {
      summary.boxCount++;
    }
  }
  return summaries;
}

ModelOutputLayout inferBranchLayout(const InferenceOutput& output, int expectedClassChannels) {
  const auto summaries = summarizeYoloBranches(output, expectedClassChannels);
  if (summaries.empty()) {
    return output.size() <= 6 ? ModelOutputLayout::kYolov8RknnBranch6
                              : ModelOutputLayout::kYolov8RknnBranch9;
  }

  bool allHaveBoxAndCls = true;
  bool anyHaveScore = false;
  bool anyMissingScore = false;
  for (const auto& [_, summary] : summaries) {
    if (summary.boxCount <= 0 || summary.clsCount <= 0) {
      allHaveBoxAndCls = false;
      break;
    }
    if (summary.scoreCount > 0) {
      anyHaveScore = true;
    } else {
      anyMissingScore = true;
    }
  }

  if (!allHaveBoxAndCls) {
    return output.size() <= 6 ? ModelOutputLayout::kYolov8RknnBranch6
                              : ModelOutputLayout::kYolov8RknnBranch9;
  }

  if (anyHaveScore && !anyMissingScore) {
    return ModelOutputLayout::kYolov8RknnBranch9;
  }
  return ModelOutputLayout::kYolov8RknnBranch6;
}

float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float denseValueAt(
    const std::vector<float>& values,
    int proposals,
    int attributes,
    bool transposed,
    int proposalIndex,
    int attributeIndex) {
  return transposed
      ? values[static_cast<std::size_t>(attributeIndex * proposals + proposalIndex)]
      : values[static_cast<std::size_t>(proposalIndex * attributes + attributeIndex)];
}

DenseOrientationStats evaluateDenseOrientation(
    const std::vector<float>& values,
    int proposals,
    int attributes,
    bool transposed,
    bool hasObjectness,
    int classOffset,
    int classCount) {
  DenseOrientationStats stats;
  for (int i = 0; i < proposals; ++i) {
    float objectness = 1.0f;
    if (hasObjectness) {
      objectness = denseValueAt(values, proposals, attributes, transposed, i, 4);
      if (objectness > 1.0f || objectness < 0.0f) {
        objectness = sigmoid(objectness);
      }
    }
    float bestScore = 0.0f;
    for (int c = 0; c < classCount; ++c) {
      const float rawCls = denseValueAt(values, proposals, attributes, transposed, i, classOffset + c);
      stats.classMin = std::min(stats.classMin, rawCls);
      stats.classMax = std::max(stats.classMax, rawCls);
      float cls = rawCls;
      if (cls > 1.0f || cls < 0.0f) {
        cls = sigmoid(cls);
      }
      bestScore = std::max(bestScore, cls * objectness);
    }
    stats.bestScore = std::max(stats.bestScore, bestScore);
  }
  return stats;
}

float tensorValueAt(const InferenceTensor& tensor, const TensorView& view, int c, int y, int x) {
  const std::size_t index = view.nchw
      ? (static_cast<std::size_t>(c) * view.height + y) * view.width + x
      : (static_cast<std::size_t>(y) * view.width + x) * view.channels + c;

  if (!tensor.data.empty()) {
    return tensor.data[index];
  }

  switch (tensor.dataType) {
    case TensorDataType::kInt8: {
      const auto* raw = reinterpret_cast<const std::int8_t*>(tensor.rawData.data());
      const float rawValue = static_cast<float>(raw[index]);
      if (tensor.quantization == TensorQuantizationType::kAffineAsymmetric) {
        return (rawValue - static_cast<float>(tensor.zeroPoint)) * tensor.scale;
      }
      if (tensor.quantization == TensorQuantizationType::kDfp) {
        return std::ldexp(rawValue, -tensor.zeroPoint);
      }
      return rawValue;
    }
    case TensorDataType::kUint8: {
      const float rawValue = static_cast<float>(tensor.rawData[index]);
      if (tensor.quantization == TensorQuantizationType::kAffineAsymmetric) {
        return (rawValue - static_cast<float>(tensor.zeroPoint)) * tensor.scale;
      }
      if (tensor.quantization == TensorQuantizationType::kDfp) {
        return std::ldexp(rawValue, -tensor.zeroPoint);
      }
      return rawValue;
    }
    case TensorDataType::kInt32: {
      const auto* raw = reinterpret_cast<const std::int32_t*>(tensor.rawData.data());
      return static_cast<float>(raw[index]);
    }
    case TensorDataType::kFloat32:
    default:
      return 0.0f;
  }
}

void clampBoxes(std::vector<BoundingBox>& boxes, int originalWidth, int originalHeight) {
  for (auto& box : boxes) {
    box.x1 = std::clamp(box.x1, 0.0f, static_cast<float>(originalWidth - 1));
    box.y1 = std::clamp(box.y1, 0.0f, static_cast<float>(originalHeight - 1));
    box.x2 = std::clamp(box.x2, 0.0f, static_cast<float>(originalWidth - 1));
    box.y2 = std::clamp(box.y2, 0.0f, static_cast<float>(originalHeight - 1));
  }
}

}  // namespace

YoloPostprocessor::YoloPostprocessor(YoloVersion version, PostprocessOptions options)
    : version_(version), options_(std::move(options)) {
  if (options_.labels.empty() && !options_.labelsPath.empty()) options_.labels = loadLabels(options_.labelsPath);
}

std::string YoloPostprocessor::name() const {
  return version_ == YoloVersion::kYolo26 ? "YOLO26" : "YOLOv8";
}

const std::vector<std::string>& YoloPostprocessor::labelsForClassCount(int classCount) const {
  if (!options_.labels.empty()) return options_.labels;
  if (static_cast<int>(cachedGeneratedLabels_.size()) < classCount) {
    cachedGeneratedLabels_.clear();
    for (int i = 0; i < classCount; ++i) cachedGeneratedLabels_.push_back("class_" + std::to_string(i));
  }
  return cachedGeneratedLabels_;
}

std::vector<float> YoloPostprocessor::valuesAsFloat(const InferenceTensor& tensor) const {
  if (!tensor.data.empty()) {
    return tensor.data;
  }

  std::vector<float> values;
  switch (tensor.dataType) {
    case TensorDataType::kInt8: {
      const auto count = tensor.rawData.size();
      values.resize(count);
      const auto* raw = reinterpret_cast<const std::int8_t*>(tensor.rawData.data());
      for (std::size_t i = 0; i < count; ++i) {
        const int v = raw[i];
        values[i] = tensor.quantization == TensorQuantizationType::kAffineAsymmetric ? (static_cast<float>(v - tensor.zeroPoint) * tensor.scale) : static_cast<float>(v);
      }
      break;
    }
    case TensorDataType::kUint8: {
      const auto count = tensor.rawData.size();
      values.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        const int v = tensor.rawData[i];
        values[i] = tensor.quantization == TensorQuantizationType::kAffineAsymmetric ? (static_cast<float>(v - tensor.zeroPoint) * tensor.scale) : static_cast<float>(v);
      }
      break;
    }
    case TensorDataType::kInt32: {
      const auto count = tensor.rawData.size() / sizeof(std::int32_t);
      values.resize(count);
      const auto* raw = reinterpret_cast<const std::int32_t*>(tensor.rawData.data());
      for (std::size_t i = 0; i < count; ++i) values[i] = static_cast<float>(raw[i]);
      break;
    }
    case TensorDataType::kFloat32:
    default:
      break;
  }
  return values;
}

int YoloPostprocessor::classChannelCount(const InferenceOutput& output) const {
  int maxChannels = 0;
  for (const auto& tensor : output) {
    TensorView view;
    if (!buildTensorView(tensor, view)) continue;
    if (view.channels == 1) continue;
    if (view.channels == 4 || (view.channels % 4 == 0 && view.channels <= 64)) continue;
    maxChannels = std::max(maxChannels, view.channels);
  }
  return maxChannels > 0 ? maxChannels : 1;
}

ModelOutputLayout YoloPostprocessor::inferLayout(const InferenceOutput& output) const {
  ModelOutputLayout layout = options_.outputLayout;
  if (layout != ModelOutputLayout::kAuto) {
    return layout;
  }

  ModelOutputLayout resolved = ModelOutputLayout::kYolov8Flat;
  std::string reason;

  if (output.size() == 1) {
    if (looksLikeYolo26E2E(output.front())) {
      const ModelOutputLayout resolved = ModelOutputLayout::kYolo26E2E;
      reason = "single-output shape matches [N,6] end-to-end layout";
      return resolved;
    }
    resolved = looksLikeYolov8Flat(output.front())
                   ? ModelOutputLayout::kYolov8Flat
                   : ModelOutputLayout::kYolov8Flat;
    reason = "single-output dense proposal tensor";
  } else {
    resolved = inferBranchLayout(output, classChannelCount(output));
    reason = "multi-output branch-head tensor set";
  }

  if (options_.verbose && !autoLayoutLogged_) {
    std::cerr << "[POST] auto layout resolved to " << layoutToString(resolved)
              << " from " << reason;
    if (!output.empty() && output.size() == 1 && !output.front().shape.empty()) {
      std::cerr << " [";
      for (std::size_t i = 0; i < output.front().shape.size(); ++i) {
        if (i != 0) std::cerr << ", ";
        std::cerr << output.front().shape[i];
      }
      std::cerr << "]";
    } else {
      std::cerr << " output_count=" << output.size();
    }
    std::cerr << "\n";
    autoLayoutLogged_ = true;
  }
  return resolved;
}

DetectionResult YoloPostprocessor::postprocess(const InferenceOutput& output, const RgbImage& modelInput, int originalWidth, int originalHeight, int64_t pts) {
  switch (inferLayout(output)) {
    case ModelOutputLayout::kYolo26E2E:
      return postprocessYolo26E2E(output.front(), modelInput, originalWidth, originalHeight, pts);
    case ModelOutputLayout::kYolov8Flat:
      return postprocessDenseTensor(output.front(), modelInput, originalWidth, originalHeight, pts);
    default:
      return postprocessBranchOutputs(output, modelInput, originalWidth, originalHeight, pts);
  }
}

DetectionResult YoloPostprocessor::postprocessDenseTensor(const InferenceTensor& tensor, const RgbImage& modelInput, int originalWidth, int originalHeight, int64_t pts) const {
  DenseLayout layout;
  if (!buildDenseLayout(tensor, layout) || layout.attributes < 5) throw std::runtime_error("Unsupported dense YOLO tensor shape");
  const std::vector<float> values = valuesAsFloat(tensor);
  const bool hasObjectness = layout.attributes == 85;
  const int classOffset = hasObjectness ? 5 : 4;
  const int classCount = std::max(1, layout.attributes - classOffset);
  const auto& labels = labelsForClassCount(classCount);
  DetectionResult result{pts, {}, originalWidth, originalHeight};
  std::vector<BoundingBox> boxes;
  const DenseOrientationStats primaryOrientation = evaluateDenseOrientation(
      values, layout.proposals, layout.attributes, layout.transposed, hasObjectness, classOffset, classCount);
  const DenseOrientationStats alternateOrientation = evaluateDenseOrientation(
      values, layout.proposals, layout.attributes, !layout.transposed, hasObjectness, classOffset, classCount);
  bool useTransposed = layout.transposed;
  // Some single-head exports report [1,84,8400] but must be read proposal-first.
  // Prefer the orientation that exposes non-degenerate class channels.
  const bool primaryLooksDead =
      primaryOrientation.classMin == 0.0f && primaryOrientation.classMax == 0.0f;
  const bool alternateHasSignal =
      alternateOrientation.classMin != 0.0f || alternateOrientation.classMax != 0.0f;
  if (primaryLooksDead && alternateHasSignal) {
    useTransposed = !layout.transposed;
  } else if (alternateOrientation.bestScore > primaryOrientation.bestScore + 1e-5f &&
             (alternateOrientation.classMax - alternateOrientation.classMin) >
                 (primaryOrientation.classMax - primaryOrientation.classMin) + 1e-5f) {
    useTransposed = !layout.transposed;
  }
  auto proposalValue = [&](int proposalIndex, int attributeIndex) -> float {
    return denseValueAt(values, layout.proposals, layout.attributes, useTransposed, proposalIndex, attributeIndex);
  };
  float debugBestScore = -1.0f;
  int debugBestProposal = -1;
  int debugBestClass = -1;
  float debugBestObjectness = 1.0f;
  std::array<float, 4> boxMin = {
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity()};
  std::array<float, 4> boxMax = {
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()};
  float classMin = std::numeric_limits<float>::infinity();
  float classMax = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < layout.proposals; ++i) {
    for (int c = 0; c < 4; ++c) {
      const float value = proposalValue(i, c);
      boxMin[c] = std::min(boxMin[c], value);
      boxMax[c] = std::max(boxMax[c], value);
    }
    float objectness = 1.0f;
    if (hasObjectness) {
      objectness = proposalValue(i, 4);
      if (objectness > 1.0f || objectness < 0.0f) objectness = sigmoid(objectness);
    }
    float bestScore = 0.0f;
    int bestClass = 0;
    for (int c = 0; c < classCount; ++c) {
      const float rawCls = proposalValue(i, classOffset + c);
      classMin = std::min(classMin, rawCls);
      classMax = std::max(classMax, rawCls);
      float cls = rawCls;
      if (cls > 1.0f || cls < 0.0f) cls = sigmoid(cls);
      if (cls > bestScore) { bestScore = cls; bestClass = c; }
    }
    const float score = bestScore * objectness;
    if (score > debugBestScore) {
      debugBestScore = score;
      debugBestProposal = i;
      debugBestClass = bestClass;
      debugBestObjectness = objectness;
    }
    if (score < options_.confThreshold) continue;
    float cx = proposalValue(i, 0);
    float cy = proposalValue(i, 1);
    float w = proposalValue(i, 2);
    float h = proposalValue(i, 3);
    if (std::max({std::fabs(cx), std::fabs(cy), std::fabs(w), std::fabs(h)}) <= 2.0f) {
      cx *= modelInput.width; cy *= modelInput.height; w *= modelInput.width; h *= modelInput.height;
    }
    BoundingBox box;
    box.x1 = cx - w * 0.5f; box.y1 = cy - h * 0.5f; box.x2 = cx + w * 0.5f; box.y2 = cy + h * 0.5f;
    box.score = score; box.classId = bestClass;
    if (bestClass >= 0 && bestClass < static_cast<int>(labels.size())) box.label = labels[bestClass];
    boxes.push_back(box);
  }
  if (options_.verbose && !denseTensorStatsLogged_) {
    std::cerr << "[POST] dense tensor stats: proposals=" << layout.proposals
              << " attrs=" << layout.attributes
              << " has_objectness=" << (hasObjectness ? "true" : "false")
              << " orientation=" << (useTransposed ? "attr_first" : "proposal_first")
              << " best_score=" << debugBestScore
              << " best_class=" << debugBestClass
              << " best_proposal=" << debugBestProposal
              << " best_objectness=" << debugBestObjectness
              << " box_minmax=["
              << boxMin[0] << ".." << boxMax[0] << ", "
              << boxMin[1] << ".." << boxMax[1] << ", "
              << boxMin[2] << ".." << boxMax[2] << ", "
              << boxMin[3] << ".." << boxMax[3] << "]"
              << " class_minmax=[" << classMin << ".." << classMax << "]"
              << " primary_best=" << primaryOrientation.bestScore
              << " primary_class_minmax=[" << primaryOrientation.classMin << ".." << primaryOrientation.classMax << "]"
              << " alternate_best=" << alternateOrientation.bestScore
              << " alternate_class_minmax=[" << alternateOrientation.classMin << ".." << alternateOrientation.classMax << "]";
    if (debugBestProposal >= 0) {
      std::cerr << " raw_box=["
                << proposalValue(debugBestProposal, 0) << ", "
                << proposalValue(debugBestProposal, 1) << ", "
                << proposalValue(debugBestProposal, 2) << ", "
                << proposalValue(debugBestProposal, 3) << "]";
    }
    std::cerr << "\n";
    denseTensorStatsLogged_ = true;
  }
  boxes = nms(boxes, options_.nmsThreshold);
  mapBoxesToOriginal(boxes, modelInput, originalWidth, originalHeight);
  result.boxes = std::move(boxes);
  return result;
}

DetectionResult YoloPostprocessor::postprocessBranchOutputs(const InferenceOutput& output, const RgbImage& modelInput, int originalWidth, int originalHeight, int64_t pts) const {
  struct Branch {
    InferenceTensor box;
    InferenceTensor cls;
    InferenceTensor score;
    std::vector<InferenceTensor> aux;
    bool hasBox = false;
    bool hasCls = false;
    bool hasScore = false;
  };

  const int expectedClassChannels = classChannelCount(output);
  std::map<std::pair<int, int>, Branch> branches;
  for (const auto& tensor : output) {
    TensorView view;
    if (!buildTensorView(tensor, view)) continue;
    auto& branch = branches[{view.height, view.width}];
    if (!branch.hasBox && (view.channels == 4 || (view.channels % 4 == 0 && view.channels <= 64))) {
      branch.box = tensor;
      branch.hasBox = true;
      continue;
    }
    if (!branch.hasCls && view.channels == expectedClassChannels) {
      branch.cls = tensor;
      branch.hasCls = true;
      continue;
    }
    if (!branch.hasScore && view.channels == 1) {
      branch.score = tensor;
      branch.hasScore = true;
      continue;
    }
    branch.aux.push_back(tensor);
  }

  std::vector<BoundingBox> boxes;
  for (auto& [_, branch] : branches) {
    if (!branch.hasBox) continue;
    if (!branch.hasCls) {
      if (branch.hasScore && expectedClassChannels == 1) {
        branch.cls = branch.score;
        branch.hasCls = true;
        branch.hasScore = false;
      } else {
        continue;
      }
    }

    TensorView boxView{};
    TensorView clsView{};
    TensorView scoreView{};
    if (!buildTensorView(branch.box, boxView) || !buildTensorView(branch.cls, clsView)) continue;
    if (branch.hasScore && !buildTensorView(branch.score, scoreView)) continue;

    const auto& labels = labelsForClassCount(clsView.channels);
    const float strideX = static_cast<float>(modelInput.width) / static_cast<float>(boxView.width);
    const float strideY = static_cast<float>(modelInput.height) / static_cast<float>(boxView.height);

    for (int y = 0; y < boxView.height; ++y) {
      for (int x = 0; x < boxView.width; ++x) {
        float objectness = 1.0f;
        if (branch.hasScore) {
          objectness = tensorValueAt(branch.score, scoreView, 0, y, x);
          if (objectness > 1.0f || objectness < 0.0f) objectness = sigmoid(objectness);
        }
        float bestScore = 0.0f;
        int bestClass = 0;
        for (int c = 0; c < clsView.channels; ++c) {
          float cls = tensorValueAt(branch.cls, clsView, c, y, x);
          if (cls > 1.0f || cls < 0.0f) cls = sigmoid(cls);
          if (cls > bestScore) { bestScore = cls; bestClass = c; }
        }
        const float score = bestScore * objectness;
        if (score < options_.confThreshold) continue;
        float left = 0.0f, top = 0.0f, right = 0.0f, bottom = 0.0f;
        if (boxView.channels == 4) {
          left = tensorValueAt(branch.box, boxView, 0, y, x);
          top = tensorValueAt(branch.box, boxView, 1, y, x);
          right = tensorValueAt(branch.box, boxView, 2, y, x);
          bottom = tensorValueAt(branch.box, boxView, 3, y, x);
        } else {
          const int bins = boxView.channels / 4;
          auto decode = [&](int base) {
            float maxLogit = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < bins; ++i) {
              maxLogit = std::max(maxLogit, tensorValueAt(branch.box, boxView, base + i, y, x));
            }
            float denominator = 0.0f;
            float numerator = 0.0f;
            for (int i = 0; i < bins; ++i) {
              const float prob = std::exp(tensorValueAt(branch.box, boxView, base + i, y, x) - maxLogit);
              denominator += prob;
              numerator += prob * static_cast<float>(i);
            }
            return denominator > 0.0f ? numerator / denominator : 0.0f;
          };
          left = decode(0); top = decode(bins); right = decode(bins * 2); bottom = decode(bins * 3);
        }
        BoundingBox box;
        const float centerX = (static_cast<float>(x) + 0.5f) * strideX;
        const float centerY = (static_cast<float>(y) + 0.5f) * strideY;
        box.x1 = centerX - left * strideX; box.y1 = centerY - top * strideY; box.x2 = centerX + right * strideX; box.y2 = centerY + bottom * strideY;
        box.score = score; box.classId = bestClass;
        if (bestClass >= 0 && bestClass < static_cast<int>(labels.size())) box.label = labels[bestClass];
        boxes.push_back(box);
      }
    }
  }

  if (boxes.size() > kMaxCandidatesBeforeNms) {
    std::partial_sort(
        boxes.begin(),
        boxes.begin() + static_cast<std::ptrdiff_t>(kMaxCandidatesBeforeNms),
        boxes.end(),
        [](const BoundingBox& a, const BoundingBox& b) { return a.score > b.score; });
    boxes.resize(kMaxCandidatesBeforeNms);
  }

  boxes = nms(boxes, options_.nmsThreshold);
  if (boxes.size() > kMaxDetectionsAfterNms) {
    boxes.resize(kMaxDetectionsAfterNms);
  }
  mapBoxesToOriginal(boxes, modelInput, originalWidth, originalHeight);
  return DetectionResult{pts, std::move(boxes), originalWidth, originalHeight};
}

DetectionResult YoloPostprocessor::postprocessYolo26E2E(const InferenceTensor& tensor, const RgbImage& modelInput, int originalWidth, int originalHeight, int64_t pts) const {
  DenseLayout layout;
  if (!buildDenseLayout(tensor, layout) || layout.attributes != 6) throw std::runtime_error("Unsupported YOLO26 end-to-end tensor shape");
  const auto values = valuesAsFloat(tensor);
  const auto& labels = labelsForClassCount(1);
  std::vector<BoundingBox> boxes;
  auto proposalValue = [&](int proposalIndex, int attributeIndex) -> float {
    return layout.transposed ? values[static_cast<std::size_t>(attributeIndex * layout.proposals + proposalIndex)] : values[static_cast<std::size_t>(proposalIndex * layout.attributes + attributeIndex)];
  };
  for (int i = 0; i < layout.proposals; ++i) {
    const float conf = proposalValue(i, 4);
    if (conf < options_.confThreshold) continue;
    BoundingBox box;
    box.x1 = proposalValue(i, 0); box.y1 = proposalValue(i, 1); box.x2 = proposalValue(i, 2); box.y2 = proposalValue(i, 3);
    if (std::max({std::fabs(box.x1), std::fabs(box.y1), std::fabs(box.x2), std::fabs(box.y2)}) <= 2.0f) {
      box.x1 *= modelInput.width; box.y1 *= modelInput.height; box.x2 *= modelInput.width; box.y2 *= modelInput.height;
    }
    box.score = conf; box.classId = static_cast<int>(proposalValue(i, 5));
    if (box.classId >= 0 && box.classId < static_cast<int>(labels.size())) box.label = labels[box.classId];
    boxes.push_back(box);
  }
  mapBoxesToOriginal(boxes, modelInput, originalWidth, originalHeight);
  return DetectionResult{pts, std::move(boxes), originalWidth, originalHeight};
}

float YoloPostprocessor::computeIoU(const BoundingBox& a, const BoundingBox& b) {
  const float x1 = std::max(a.x1, b.x1); const float y1 = std::max(a.y1, b.y1); const float x2 = std::min(a.x2, b.x2); const float y2 = std::min(a.y2, b.y2);
  const float interW = std::max(0.0f, x2 - x1); const float interH = std::max(0.0f, y2 - y1); const float interArea = interW * interH;
  const float unionArea = a.area() + b.area() - interArea; return unionArea <= 0.0f ? 0.0f : interArea / unionArea;
}

std::vector<BoundingBox> YoloPostprocessor::nms(std::vector<BoundingBox>& boxes, float iouThreshold) {
  if (boxes.empty()) return {};
  std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) { return a.score > b.score; });
  std::vector<BoundingBox> result; std::vector<bool> suppressed(boxes.size(), false);
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (suppressed[i]) continue;
    result.push_back(boxes[i]);
    for (std::size_t j = i + 1; j < boxes.size(); ++j) {
      if (suppressed[j] || boxes[i].classId != boxes[j].classId) continue;
      if (computeIoU(boxes[i], boxes[j]) > iouThreshold) suppressed[j] = true;
    }
  }
  return result;
}

void YoloPostprocessor::mapBoxesToOriginal(std::vector<BoundingBox>& boxes, const RgbImage& modelInput, int originalWidth, int originalHeight) {
  if (modelInput.letterbox.enabled) {
    for (auto& box : boxes) {
      box.x1 = (box.x1 - static_cast<float>(modelInput.letterbox.padLeft)) / modelInput.letterbox.scale;
      box.y1 = (box.y1 - static_cast<float>(modelInput.letterbox.padTop)) / modelInput.letterbox.scale;
      box.x2 = (box.x2 - static_cast<float>(modelInput.letterbox.padLeft)) / modelInput.letterbox.scale;
      box.y2 = (box.y2 - static_cast<float>(modelInput.letterbox.padTop)) / modelInput.letterbox.scale;
    }
  } else {
    const float scaleX = static_cast<float>(originalWidth) / static_cast<float>(modelInput.width);
    const float scaleY = static_cast<float>(originalHeight) / static_cast<float>(modelInput.height);
    for (auto& box : boxes) {
      box.x1 *= scaleX; box.y1 *= scaleY; box.x2 *= scaleX; box.y2 *= scaleY;
    }
  }
  clampBoxes(boxes, originalWidth, originalHeight);
}

const char* YoloPostprocessor::layoutToString(ModelOutputLayout layout) {
  switch (layout) {
    case ModelOutputLayout::kAuto:
      return "auto";
    case ModelOutputLayout::kYolov8Flat:
      return "yolov8_flat_8400x84";
    case ModelOutputLayout::kYolov8RknnBranch6:
      return "yolov8_rknn_branch_6";
    case ModelOutputLayout::kYolov8RknnBranch9:
      return "yolov8_rknn_branch_9";
    case ModelOutputLayout::kYolo26E2E:
      return "yolo26_e2e";
  }
  return "unknown";
}
