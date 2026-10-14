#include "train_voc_common.h"

#include "config_params.h"
#include "ms_fusion/vision/xfeat_detector.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "ms_fusion/common/runtime_log.h"

#ifndef MSF_USE_AXERA_SUPERPOINT
#error "train_xfeat_voc requires MSF_USE_AXERA_SUPERPOINT=ON (Axera NPU XFeat)"
#endif

namespace {

std::string basenameOf(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

void scoreRange(const std::vector<ms_fusion::ScoredKeypoint>& kps, float& min_s, float& max_s,
                float& mean_s) {
  min_s = 0.f;
  max_s = 0.f;
  mean_s = 0.f;
  if (kps.empty()) {
    return;
  }
  min_s = kps.front().score;
  max_s = kps.front().score;
  double sum = 0.0;
  for (const auto& kp : kps) {
    min_s = std::min(min_s, kp.score);
    max_s = std::max(max_s, kp.score);
    sum += kp.score;
  }
  mean_s = static_cast<float>(sum / static_cast<double>(kps.size()));
}

void appendTiming(std::ostream& os, const ms_fusion::NpuDetectTiming& t) {
  if (!t.valid) {
    os << " timing=n/a";
    return;
  }
  os << std::fixed << std::setprecision(2) << " preprocess_ms=" << t.preprocess_ms
     << " infer_ms=" << t.infer_ms << " post_ms=" << t.post_ms << " detect_ms=" << t.detect_ms;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace msf_loop_tools;
  TrainVocOptions opt;
  try {
    if (!parseTrainArgs(argc, argv, opt, "train_xfeat_voc")) {
      return 1;
    }
  } catch (const std::exception& e) {
    MSF_WLOG << e.what() << "\n";
    printUsage("train_xfeat_voc");
    return 1;
  }

  MSF_ILOG << "[train_xfeat_voc] SoC NPU required (.axmodel). "
              "Run this binary on the Axera board.\n";

  // Match runtime msf yaml (xfeat.score_threshold: 0.005)
  if (!opt.score_threshold_set) {
    opt.score_threshold = 0.005f;
  }

  ms_fusion::XFeatConfigParams cfg;
  cfg.model_path = opt.model_path;
  cfg.score_threshold = opt.score_threshold;
  cfg.enable_hardware_preprocess = false;  // file images: CPU resize

  MSF_ILOG << "[train_xfeat_voc] config:\n"
           << "  image_dir=" << opt.image_dir << " recursive=" << (opt.recursive ? 1 : 0)
           << " max_images=" << opt.max_images << "\n"
           << "  model=" << opt.model_path << "\n"
           << "  output=" << opt.output_path << "\n"
           << "  score_threshold=" << opt.score_threshold
           << " max_keypoints=" << opt.max_keypoints << "\n"
           << "  hw_preprocess=" << (cfg.enable_hardware_preprocess ? 1 : 0) << "\n"
           << "  dbow k=" << opt.k << " L=" << opt.L << std::endl;

  ms_fusion::XFeatDetector detector(cfg);
  if (!detector.initialized()) {
    MSF_ELOG << "[train_xfeat_voc] XFeat init failed: "
             << ms_fusion::xfeatInitFailToString(detector.initFailReason()) << "\n";
    return 2;
  }
  MSF_ILOG << "[train_xfeat_voc] XFeat detector ready\n";

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
    MSF_WLOG << "[train_xfeat_voc] No images found in " << opt.image_dir << "\n";
    return 1;
  }

  MSF_ILOG << "[train_xfeat_voc] images=" << images.size() << " first=" << basenameOf(images.front())
           << " last=" << basenameOf(images.back()) << std::endl;

  std::vector<cv::Mat> features;
  features.reserve(images.size());
  int ok = 0;
  int fail = 0;
  int fail_load = 0;
  int fail_detect = 0;
  int fail_desc = 0;
  int first_ok_idx = -1;
  const int diag_limit = 8;
  int diag_printed = 0;

  std::map<std::string, int> detect_reason_hist;
  long long sum_kps = 0;
  long long sum_desc_rows = 0;
  double sum_detect_ms = 0.0;
  double sum_infer_ms = 0.0;
  int timing_count = 0;
  int desc_dim = -1;

  for (size_t i = 0; i < images.size(); ++i) {
    const auto& path = images[i];
    cv::Mat gray = loadGrayImage(path);
    if (gray.empty()) {
      ++fail;
      ++fail_load;
      if (diag_printed < diag_limit) {
        MSF_ELOG << "[train_xfeat_voc] FAIL[" << (diag_printed + 1) << "/" << diag_limit
                 << "] load empty image idx=" << i << " path=" << path << "\n";
        ++diag_printed;
      }
      continue;
    }

    std::vector<ms_fusion::ScoredKeypoint> kps;
    cv::Mat descs;
    const bool detect_ok =
        detector.detectWithDescriptors(gray, opt.max_keypoints, kps, descs) && !descs.empty();
    const auto timing = detector.lastDetectTiming();
    const auto reason = detector.lastDetectFailReason();
    const char* reason_str = ms_fusion::xfeatDetectFailToString(reason);
    const auto& dmap = detector.lastDescriptorMap();

    if (!detect_ok) {
      ++fail;
      if (!kps.empty()) {
        ++fail_desc;
        detect_reason_hist["DESC_SAMPLE"]++;
      } else {
        ++fail_detect;
        detect_reason_hist[reason_str]++;
      }

      if (diag_printed < diag_limit) {
        float smin = 0.f, smax = 0.f, smean = 0.f;
        scoreRange(kps, smin, smax, smean);
        MSF_ELOG << "[train_xfeat_voc] FAIL[" << (diag_printed + 1) << "/" << diag_limit
                 << "] idx=" << i << " file=" << basenameOf(path) << "\n"
                 << "  path=" << path << "\n"
                 << "  gray=" << gray.cols << "x" << gray.rows << " type=" << gray.type()
                 << " channels=" << gray.channels() << " continuous=" << (gray.isContinuous() ? 1 : 0)
                 << "\n"
                 << "  kps=" << kps.size() << " descs_empty=" << (descs.empty() ? 1 : 0)
                 << " detect_reason=" << reason_str;
        if (!kps.empty()) {
          MSF_WLOG << std::fixed << std::setprecision(4) << " score[min/mean/max]=" << smin << "/"
                   << smean << "/" << smax;
        }
        MSF_WLOG << "\n"
                 << "  desc_map valid=" << (dmap.valid() ? 1 : 0) << " spatial=" << dmap.spatial_h
                 << "x" << dmap.spatial_w << " C=" << dmap.channels
                 << " chw=" << (dmap.chw ? 1 : 0) << " img=" << dmap.image_w << "x" << dmap.image_h
                 << " data_n=" << dmap.data.size() << "\n"
                 << " ";
        std::ostringstream timing_text;
        appendTiming(timing_text, timing);
        MSF_WLOG << timing_text.str();
        MSF_WLOG << "\n";
        ++diag_printed;
      }
    } else {
      features.push_back(descs);
      ++ok;
      sum_kps += static_cast<long long>(kps.size());
      sum_desc_rows += static_cast<long long>(descs.rows);
      if (desc_dim < 0) {
        desc_dim = descs.cols;
      }
      if (timing.valid) {
        sum_detect_ms += timing.detect_ms;
        sum_infer_ms += timing.infer_ms;
        ++timing_count;
      }

      if (first_ok_idx < 0) {
        first_ok_idx = static_cast<int>(i);
        float smin = 0.f, smax = 0.f, smean = 0.f;
        scoreRange(kps, smin, smax, smean);
        MSF_ILOG << "[train_xfeat_voc] FIRST_OK idx=" << i << " file=" << basenameOf(path)
                 << " gray=" << gray.cols << "x" << gray.rows << " kps=" << kps.size()
                 << " desc=" << descs.rows << "x" << descs.cols << std::fixed << std::setprecision(4)
                 << " score[min/mean/max]=" << smin << "/" << smean << "/" << smax
                 << " desc_map=" << dmap.spatial_h << "x" << dmap.spatial_w << "x" << dmap.channels;
        std::ostringstream timing_text;
        appendTiming(timing_text, timing);
        MSF_ILOG << timing_text.str();
        MSF_ILOG << std::endl;
      }
    }

    if ((i + 1) % 50 == 0 || i + 1 == images.size()) {
      const double ok_rate =
          images.empty() ? 0.0 : 100.0 * static_cast<double>(ok) / static_cast<double>(i + 1);
      MSF_ILOG << "[train_xfeat_voc] progress " << (i + 1) << "/" << images.size()
               << std::fixed << std::setprecision(1) << " ok_rate=" << ok_rate << "%"
               << " ok=" << ok << " fail=" << fail << " (load=" << fail_load
               << " detect=" << fail_detect << " desc=" << fail_desc << ")";
      if (ok > 0) {
        MSF_ILOG << " avg_kps=" << std::setprecision(1)
                 << (static_cast<double>(sum_kps) / static_cast<double>(ok))
                 << " dim=" << desc_dim;
      }
      if (timing_count > 0) {
        MSF_ILOG << std::setprecision(2)
                 << " avg_detect_ms=" << (sum_detect_ms / timing_count)
                 << " avg_infer_ms=" << (sum_infer_ms / timing_count);
      }
      if (!detect_reason_hist.empty()) {
        MSF_ILOG << " reasons={";
        bool first = true;
        for (const auto& kv : detect_reason_hist) {
          if (!first) {
            MSF_ILOG << ", ";
          }
          first = false;
          MSF_ILOG << kv.first << ":" << kv.second;
        }
        MSF_ILOG << "}";
      }
      MSF_ILOG << std::endl;
    }
  }

  MSF_ILOG << "[train_xfeat_voc] extract done:\n"
           << "  total=" << images.size() << " ok=" << ok << " fail=" << fail
           << " (load=" << fail_load << " detect=" << fail_detect << " desc=" << fail_desc << ")\n"
           << "  first_ok_idx=" << first_ok_idx;
  if (ok > 0) {
    MSF_ILOG << " avg_kps=" << std::fixed << std::setprecision(1)
             << (static_cast<double>(sum_kps) / static_cast<double>(ok))
             << " total_descs=" << sum_desc_rows << " dim=" << desc_dim;
  }
  MSF_ILOG << "\n";
  if (!detect_reason_hist.empty()) {
    MSF_ILOG << "  detect_fail_reasons:\n";
    for (const auto& kv : detect_reason_hist) {
      MSF_ILOG << "    " << kv.first << ": " << kv.second << "\n";
    }
  }
  if (timing_count > 0) {
    MSF_ILOG << std::fixed << std::setprecision(2)
             << "  timing_samples=" << timing_count
             << " avg_detect_ms=" << (sum_detect_ms / timing_count)
             << " avg_infer_ms=" << (sum_infer_ms / timing_count) << std::endl;
  } else {
    MSF_ILOG << "  timing: no successful detect samples (timing only filled on detect success)\n";
  }

  bool ok_overall = true;

  if (!opt.features_out.empty()) {
    if (!saveFeatures(opt.features_out, features)) {
      ok_overall = false;
    }
  }

  if (!opt.output_path.empty()) {
    if (!trainAndSaveVocabulary(features, opt)) {
      if (ok == 0 && fail_desc > 0) {
        MSF_ELOG << "[train_xfeat_voc] all frames failed descriptor sampling "
                    "(detect OK). Rebuild msf_core after descriptor_map/xfeat fix; "
                    "check [XFeat] model outputs / copy_descriptors_to_map logs.\n";
      } else if (ok == 0) {
        MSF_ELOG << "[train_xfeat_voc] vocabulary train/save failed "
                    "(see fail reasons above; try lower --score_threshold e.g. 0.005)\n";
      } else {
        MSF_ELOG << "[train_xfeat_voc] vocabulary train/save failed\n";
      }
      ok_overall = false;
    }
  }

  return ok_overall ? 0 : 3;
}
