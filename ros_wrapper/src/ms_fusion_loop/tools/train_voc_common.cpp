#include "train_voc_common.h"

#include "DBoW3.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "ms_fusion/common/runtime_log.h"

namespace fs = std::filesystem;

namespace msf_loop_tools {

void printUsage(const char* tool_name) {
  MSF_ILOG
      << "Usage:\n  " << tool_name
      << " --image_dir DIR --model MODEL.axmodel (--output VOC.dbow3 | --features_out DUMP) "
         "[options]\n\n"
      << "Note: must run on Axera SoC (NPU loads --model .axmodel).\n\n"
      << "Required:\n"
      << "  --image_dir DIR       Directory of training images (jpg/png/bmp/tif)\n"
      << "  --model PATH          Axera .axmodel for this detector (SoC only)\n\n"
      << "At least one of:\n"
      << "  --output PATH         Cluster into a vocabulary and save it here (.dbow3 binary;\n"
      << "                        .yml.gz auto→.dbow3). Clustering is CPU-only but runs on\n"
      << "                        this machine, i.e. the NPU board -- can be slow.\n"
      << "  --features_out PATH   Dump extracted per-image descriptors here instead, to\n"
      << "                        cluster later on a faster machine with\n"
      << "                        scripts/dbow3_train_from_features.py. Both flags may be\n"
      << "                        given together.\n\n"
      << "Optional:\n"
      << "  --k N                 DBoW3 branching factor (default 10)\n"
      << "  --L N                 DBoW3 tree depth (default 5)\n"
      << "  --max_keypoints N     Max keypoints per image (default 400)\n"
      << "  --score_threshold F   Detector score threshold (default 0.015)\n"
      << "  --nms_radius N        NMS radius (default 4)\n"
      << "  --max_images N        Cap number of images (0=all)\n"
      << "  --recursive           Recurse into subdirectories\n"
      << "  -h, --help            Show this help\n";
}

bool parseTrainArgs(int argc, char** argv, TrainVocOptions& opt, const char* tool_name) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      printUsage(tool_name);
      std::exit(0);
    } else if (a == "--image_dir") {
      opt.image_dir = need("--image_dir");
    } else if (a == "--model") {
      opt.model_path = need("--model");
    } else if (a == "--output") {
      opt.output_path = need("--output");
    } else if (a == "--features_out") {
      opt.features_out = need("--features_out");
    } else if (a == "--k") {
      opt.k = std::stoi(need("--k"));
    } else if (a == "--L") {
      opt.L = std::stoi(need("--L"));
    } else if (a == "--max_keypoints") {
      opt.max_keypoints = std::stoi(need("--max_keypoints"));
    } else if (a == "--score_threshold") {
      opt.score_threshold = std::stof(need("--score_threshold"));
      opt.score_threshold_set = true;
    } else if (a == "--nms_radius") {
      opt.nms_radius = std::stoi(need("--nms_radius"));
    } else if (a == "--max_images") {
      opt.max_images = std::stoi(need("--max_images"));
    } else if (a == "--recursive") {
      opt.recursive = true;
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (opt.image_dir.empty() || opt.model_path.empty() ||
      (opt.output_path.empty() && opt.features_out.empty())) {
    printUsage(tool_name);
    return false;
  }
  if (opt.k < 2 || opt.L < 1) {
    MSF_ELOG << "Invalid --k/--L\n";
    return false;
  }
  return true;
}

namespace {

bool isImageExt(std::string ext) {
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" ||
         ext == ".tiff" || ext == ".webp";
}

}  // namespace

std::vector<std::string> listImageFiles(const std::string& image_dir, bool recursive) {
  std::vector<std::string> files;
  const fs::path root(image_dir);
  if (!fs::exists(root) || !fs::is_directory(root)) {
    throw std::runtime_error("image_dir is not a directory: " + image_dir);
  }

  auto consider = [&](const fs::directory_entry& entry) {
    if (!entry.is_regular_file()) {
      return;
    }
    if (isImageExt(entry.path().extension().string())) {
      files.push_back(entry.path().string());
    }
  };

  if (recursive) {
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
      consider(entry);
    }
  } else {
    for (const auto& entry : fs::directory_iterator(root)) {
      consider(entry);
    }
  }

  std::sort(files.begin(), files.end());
  return files;
}

cv::Mat loadGrayImage(const std::string& path) {
  cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
  if (img.empty()) {
    return {};
  }
  if (img.channels() == 1) {
    return img;
  }
  if (img.channels() == 3) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
  }
  if (img.channels() == 4) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    return gray;
  }
  return {};
}

bool trainAndSaveVocabulary(const std::vector<cv::Mat>& features, const TrainVocOptions& opt) {
  if (features.empty()) {
    MSF_WLOG << "[train_voc] No features collected\n";
    return false;
  }

  size_t total_desc = 0;
  int desc_dim = -1;
  for (const auto& f : features) {
    if (f.empty() || f.type() != CV_32FC1) {
      continue;
    }
    total_desc += static_cast<size_t>(f.rows);
    if (desc_dim < 0) {
      desc_dim = f.cols;
    } else if (f.cols != desc_dim) {
      MSF_WLOG << "[train_voc] Inconsistent descriptor dim: " << f.cols << " vs " << desc_dim
               << "\n";
      return false;
    }
  }

  MSF_ILOG << "[train_voc] Images with features: " << features.size()
           << " total_descriptors=" << total_desc << " dim=" << desc_dim << " k=" << opt.k
           << " L=" << opt.L << std::endl;
  MSF_ILOG << "[train_voc] Clustering (can take a long time)..." << std::endl;

  DBoW3::Vocabulary voc(opt.k, opt.L);
  voc.create(features);

  // OpenCV YAML (.yml/.yml.gz) cannot serialize large float trees ("string too long").
  // DBoW3 binary path is used whenever the filename does not contain ".yml".
  std::string out_path = opt.output_path;
  {
    const auto yml_pos = out_path.find(".yml");
    if (yml_pos != std::string::npos) {
      out_path = out_path.substr(0, yml_pos) + ".dbow3";
      MSF_ILOG << "[train_voc] YAML too large for OpenCV FileStorage; saving binary instead:\n"
               << "  " << out_path << std::endl;
    }
  }

  const fs::path out(out_path);
  if (out.has_parent_path()) {
    fs::create_directories(out.parent_path());
  }

  MSF_ILOG << "[train_voc] Saving " << out_path << " ..." << std::endl;
  try {
    voc.save(out_path, /*binary_compressed=*/true);
  } catch (const cv::Exception& e) {
    MSF_ELOG << "[train_voc] save failed: " << e.what() << "\n";
    return false;
  } catch (const std::exception& e) {
    MSF_ELOG << "[train_voc] save failed: " << e.what() << "\n";
    return false;
  }
  MSF_ILOG << "[train_voc] Done. words=" << voc.size() << " path=" << out_path << std::endl;
  return true;
}

bool saveFeatures(const std::string& path, const std::vector<cv::Mat>& features) {
  if (features.empty()) {
    MSF_WLOG << "[train_voc] saveFeatures: no features to save\n";
    return false;
  }
  int desc_dim = -1;
  for (const auto& f : features) {
    if (f.empty()) {
      continue;
    }
    if (f.type() != CV_32FC1) {
      MSF_WLOG << "[train_voc] saveFeatures: expected CV_32FC1, got type=" << f.type() << "\n";
      return false;
    }
    if (desc_dim < 0) {
      desc_dim = f.cols;
    } else if (f.cols != desc_dim) {
      MSF_WLOG << "[train_voc] saveFeatures: inconsistent descriptor dim " << f.cols << " vs "
               << desc_dim << "\n";
      return false;
    }
  }
  if (desc_dim < 0) {
    MSF_WLOG << "[train_voc] saveFeatures: all feature mats empty\n";
    return false;
  }

  const fs::path out(path);
  if (out.has_parent_path()) {
    fs::create_directories(out.parent_path());
  }

  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    MSF_ELOG << "[train_voc] saveFeatures: cannot open " << path << " for writing\n";
    return false;
  }

  const uint32_t magic = 0x4D534656u;  // 'MSFV'
  const uint32_t dim = static_cast<uint32_t>(desc_dim);
  const uint32_t num_images = static_cast<uint32_t>(features.size());
  ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  ofs.write(reinterpret_cast<const char*>(&num_images), sizeof(num_images));

  long long total_rows = 0;
  for (const auto& f : features) {
    const uint32_t rows = static_cast<uint32_t>(f.rows);
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    if (rows > 0) {
      // f is guaranteed CV_32FC1 and continuous (single Mat from detector output).
      ofs.write(reinterpret_cast<const char*>(f.ptr<float>(0)),
                static_cast<std::streamsize>(sizeof(float)) * f.rows * f.cols);
    }
    total_rows += f.rows;
  }
  ofs.close();

  MSF_ILOG << "[train_voc] saveFeatures: wrote " << path << " images=" << num_images
           << " dim=" << dim << " total_descriptors=" << total_rows << std::endl;
  return true;
}

}  // namespace msf_loop_tools
