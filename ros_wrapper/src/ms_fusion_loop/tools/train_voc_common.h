#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace msf_loop_tools {

struct TrainVocOptions {
  std::string image_dir;
  std::string model_path;
  std::string output_path;
  // Optional: dump extracted per-image descriptors here instead of (or in
  // addition to) clustering into a .dbow3 on this machine. Format is a
  // simple custom binary (see saveFeatures()), NOT DBoW3's own format --
  // meant to be consumed offline by scripts/dbow3_train_from_features.py,
  // so the slow hierarchical k-means step can run on a faster local
  // machine instead of the NPU board.
  std::string features_out;
  int k = 10;
  int L = 5;
  int max_keypoints = 400;
  float score_threshold = 0.015f;
  bool score_threshold_set = false;
  int nms_radius = 4;
  int max_images = 0;  // 0 = all
  bool recursive = false;
};

bool parseTrainArgs(int argc, char** argv, TrainVocOptions& opt, const char* tool_name);

std::vector<std::string> listImageFiles(const std::string& image_dir, bool recursive);

cv::Mat loadGrayImage(const std::string& path);

// features: each Mat is NxD CV_32F (one image). Saves DBoW3 binary (.dbow3).
bool trainAndSaveVocabulary(const std::vector<cv::Mat>& features, const TrainVocOptions& opt);

// Custom binary dump (little-endian, not a DBoW3 format):
//   uint32 magic ('MSFV' = 0x4D534656)
//   uint32 desc_dim
//   uint32 num_images
//   per image: uint32 num_rows, then num_rows*desc_dim float32 (row-major)
// Consumed by scripts/dbow3_train_from_features.py.
bool saveFeatures(const std::string& path, const std::vector<cv::Mat>& features);

void printUsage(const char* tool_name);

}  // namespace msf_loop_tools
