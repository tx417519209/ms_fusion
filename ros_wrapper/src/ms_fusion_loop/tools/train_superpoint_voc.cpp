#include "train_voc_common.h"

#include "config_params.h"
#include "ms_fusion/vision/superpoint_detector.h"

#include <iostream>
#include "ms_fusion/common/runtime_log.h"

#ifndef MSF_USE_AXERA_SUPERPOINT
#error "train_superpoint_voc requires MSF_USE_AXERA_SUPERPOINT=ON (Axera NPU SuperPoint)"
#endif

int main(int argc, char** argv) {
  using namespace msf_loop_tools;
  TrainVocOptions opt;
  try {
    if (!parseTrainArgs(argc, argv, opt, "train_superpoint_voc")) {
      return 1;
    }
  } catch (const std::exception& e) {
    MSF_WLOG << e.what() << "\n";
    printUsage("train_superpoint_voc");
    return 1;
  }

  MSF_ILOG << "[train_superpoint_voc] SoC NPU required (.axmodel). "
              "Run this binary on the Axera board.\n";

  ms_fusion::SuperPointConfigParams cfg;
  cfg.model_path = opt.model_path;
  cfg.score_threshold = opt.score_threshold;
  cfg.nms_radius = opt.nms_radius;
  cfg.enable_hardware_preprocess = false;  // file images: CPU resize path

  ms_fusion::SuperPointDetector detector(cfg);
  if (!detector.initialized()) {
    MSF_ELOG << "[train_superpoint_voc] SuperPoint init failed: "
             << ms_fusion::superPointInitFailToString(detector.initFailReason()) << "\n";
    return 2;
  }

  std::vector<std::string> images;
  try {
    images = listImageFiles(opt.image_dir, opt.recursive);
  } catch (const std::exception& e) {
    MSF_WLOG << e.what() << "\n";
    return 1;
  }
  if (opt.max_images > 0 && static_cast<int>(images.size()) > opt.max_images) {
    images.resize(static_cast<size_t>(opt.max_images));
  }
  if (images.empty()) {
    MSF_WLOG << "[train_superpoint_voc] No images found in " << opt.image_dir << "\n";
    return 1;
  }

  MSF_ILOG << "[train_superpoint_voc] images=" << images.size()
           << " model=" << opt.model_path << std::endl;

  std::vector<cv::Mat> features;
  features.reserve(images.size());
  int ok = 0;
  int fail = 0;

  for (size_t i = 0; i < images.size(); ++i) {
    const auto& path = images[i];
    cv::Mat gray = loadGrayImage(path);
    if (gray.empty()) {
      ++fail;
      continue;
    }

    std::vector<ms_fusion::ScoredKeypoint> kps;
    cv::Mat descs;
    if (!detector.detectWithDescriptors(gray, opt.max_keypoints, kps, descs) || descs.empty()) {
      ++fail;
      if ((i + 1) % 50 == 0) {
        MSF_ILOG << "[train_superpoint_voc] " << (i + 1) << "/" << images.size()
                 << " ok=" << ok << " fail=" << fail << std::endl;
      }
      continue;
    }

    features.push_back(descs);
    ++ok;
    if ((i + 1) % 50 == 0 || i + 1 == images.size()) {
      MSF_ILOG << "[train_superpoint_voc] " << (i + 1) << "/" << images.size()
               << " ok=" << ok << " fail=" << fail
               << " last_kps=" << descs.rows << " dim=" << descs.cols << std::endl;
    }
  }

  if (!trainAndSaveVocabulary(features, opt)) {
    return 3;
  }
  return 0;
}
