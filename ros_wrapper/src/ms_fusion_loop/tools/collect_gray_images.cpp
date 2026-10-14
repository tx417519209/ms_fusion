// Collect tracking-resolution grayscale frames from Axera IVPS into a local folder
// for DBoW vocabulary training. Must run on SoC with camera/IVPS pyramid up.
// Do not run concurrently with ms_fusion (same IVPS channels).

#include <ax_global_type.h>
#include <ax_ivps_api.h>
#include <ax_sys_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include "ms_fusion/common/runtime_log.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string output_dir = "/data/voc_images";
  int left_grp = 7;
  int right_grp = 8;
  int chn = 0;
  int interval_ms = 200;
  int max_frames = 0;  // 0 = until Ctrl-C
  int timeout_ms = 200;
  bool save_right = false;
};

void printUsage() {
  MSF_ILOG
      << "Usage:\n  collect_gray_images --output_dir DIR [options]\n\n"
      << "Saves IVPS pyramid grayscale (tracking mono) as PNG under DIR.\n"
      << "Run on Axera SoC with MIPI/IVPS running; stop ms_fusion first.\n\n"
      << "Required:\n"
      << "  --output_dir DIR     Local folder for gray PNGs\n\n"
      << "Optional:\n"
      << "  --left_grp N         Left IVPS GRP (default 7)\n"
      << "  --right_grp N        Right IVPS GRP (default 8)\n"
      << "  --chn N              IVPS channel (default 0 = 640 pyramid)\n"
      << "  --interval_ms N      Save period (default 200)\n"
      << "  --max_frames N       Stop after N saves (0=forever)\n"
      << "  --timeout_ms N       IVPS GetChnFrame timeout (default 200)\n"
      << "  --save_right         Also write right/ subfolder\n"
      << "  -h, --help\n";
}

bool parseArgs(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      printUsage();
      std::exit(0);
    } else if (a == "--output_dir") {
      opt.output_dir = need("--output_dir");
    } else if (a == "--left_grp") {
      opt.left_grp = std::stoi(need("--left_grp"));
    } else if (a == "--right_grp") {
      opt.right_grp = std::stoi(need("--right_grp"));
    } else if (a == "--chn") {
      opt.chn = std::stoi(need("--chn"));
    } else if (a == "--interval_ms") {
      opt.interval_ms = std::stoi(need("--interval_ms"));
    } else if (a == "--max_frames") {
      opt.max_frames = std::stoi(need("--max_frames"));
    } else if (a == "--timeout_ms") {
      opt.timeout_ms = std::stoi(need("--timeout_ms"));
    } else if (a == "--save_right") {
      opt.save_right = true;
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }
  if (opt.output_dir.empty()) {
    printUsage();
    return false;
  }
  if (opt.interval_ms < 1) {
    MSF_ELOG << "Invalid --interval_ms\n";
    return false;
  }
  return true;
}

AX_U32 frameMmapSize(const AX_VIDEO_FRAME_T& frame) {
  if (frame.u32FrameSize > 0) {
    return frame.u32FrameSize;
  }
  if (frame.u32Height > 0 && frame.u32PicStride[0] > 0) {
    return frame.u32PicStride[0] * frame.u32Height;
  }
  return 0;
}

bool frameToGray(const AX_VIDEO_FRAME_T& frame, void* vir, cv::Mat& gray) {
  if (vir == nullptr || frame.u32Width == 0 || frame.u32Height == 0) {
    return false;
  }
  cv::Mat view(static_cast<int>(frame.u32Height), static_cast<int>(frame.u32Width), CV_8UC1, vir,
               static_cast<size_t>(frame.u32PicStride[0]));
  gray = view.clone();
  return !gray.empty();
}

void releaseFrame(IVPS_GRP grp, IVPS_CHN chn, AX_VIDEO_FRAME_T& frame) {
  if (frame.u64PhyAddr[0] == 0) {
    return;
  }
  if (frame.u64VirAddr[0] != 0) {
    AX_SYS_Munmap(reinterpret_cast<AX_VOID*>(frame.u64VirAddr[0]), frameMmapSize(frame));
  }
  AX_IVPS_ReleaseChnFrame(grp, chn, &frame);
  std::memset(&frame, 0, sizeof(frame));
}

bool grabOne(IVPS_GRP grp, IVPS_CHN chn, int timeout_ms, AX_VIDEO_FRAME_T& frame, AX_S32* err) {
  std::memset(&frame, 0, sizeof(frame));
  const AX_S32 ret = AX_IVPS_GetChnFrame(grp, chn, &frame, timeout_ms);
  if (err) {
    *err = ret;
  }
  return ret == 0;
}

volatile std::sig_atomic_t g_stop = 0;

void onSignal(int) { g_stop = 1; }

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  try {
    if (!parseArgs(argc, argv, opt)) {
      return 1;
    }
  } catch (const std::exception& e) {
    MSF_WLOG << e.what() << "\n";
    printUsage();
    return 1;
  }

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  const fs::path out_root(opt.output_dir);
  const fs::path left_dir = opt.save_right ? (out_root / "left") : out_root;
  const fs::path right_dir = out_root / "right";
  fs::create_directories(left_dir);
  if (opt.save_right) {
    fs::create_directories(right_dir);
  }

  const AX_S32 sys_ret = AX_SYS_Init();
  if (sys_ret != 0) {
    MSF_ELOG << "[collect_gray_images] AX_SYS_Init failed: 0x" << std::hex << sys_ret << std::dec
             << "\n";
    return 2;
  }

  MSF_ILOG << "[collect_gray_images] SoC IVPS → local PNG\n"
           << "  output_dir=" << opt.output_dir << " left_grp=" << opt.left_grp
           << " right_grp=" << opt.right_grp << " chn=" << opt.chn
           << " interval_ms=" << opt.interval_ms << " max_frames=" << opt.max_frames
           << " save_right=" << (opt.save_right ? "1" : "0") << std::endl;
  MSF_ILOG << "[collect_gray_images] Ensure camera/IVPS is up and ms_fusion is stopped.\n";

  int saved = 0;
  int fail = 0;
  int consecutive_fail = 0;
  auto next_save = std::chrono::steady_clock::now();

  while (!g_stop) {
    if (opt.max_frames > 0 && saved >= opt.max_frames) {
      break;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now < next_save) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    next_save = now + std::chrono::milliseconds(opt.interval_ms);

    AX_VIDEO_FRAME_T fl{}, fr{};
    AX_S32 err = 0;
    const IVPS_GRP lgrp = static_cast<IVPS_GRP>(opt.left_grp);
    const IVPS_GRP rgrp = static_cast<IVPS_GRP>(opt.right_grp);
    const IVPS_CHN chn = static_cast<IVPS_CHN>(opt.chn);

    if (!grabOne(lgrp, chn, opt.timeout_ms, fl, &err)) {
      ++fail;
      ++consecutive_fail;
      if (consecutive_fail == 1 || consecutive_fail % 50 == 0) {
        MSF_ELOG << "[collect_gray_images] left GetChnFrame failed ret=0x" << std::hex << err
                 << std::dec << " (fail=" << fail << ")\n";
      }
      continue;
    }
    consecutive_fail = 0;

    void* vir_l = nullptr;
    if (fl.u64VirAddr[0] != 0) {
      vir_l = reinterpret_cast<void*>(fl.u64VirAddr[0]);
    } else {
      const AX_U32 sz = frameMmapSize(fl);
      if (sz > 0 && fl.u64PhyAddr[0] != 0) {
        vir_l = AX_SYS_Mmap(fl.u64PhyAddr[0], sz);
        if (vir_l) {
          fl.u64VirAddr[0] = reinterpret_cast<AX_U64>(vir_l);
        }
      }
    }

    cv::Mat gray_l;
    if (!frameToGray(fl, vir_l, gray_l)) {
      ++fail;
      releaseFrame(lgrp, chn, fl);
      continue;
    }

    char name[64];
    std::snprintf(name, sizeof(name), "gray_%06d.png", saved);
    const fs::path left_path = left_dir / name;
    if (!cv::imwrite(left_path.string(), gray_l)) {
      MSF_ELOG << "[collect_gray_images] imwrite failed: " << left_path << "\n";
      ++fail;
      releaseFrame(lgrp, chn, fl);
      continue;
    }

    if (opt.save_right) {
      AX_S32 err_r = 0;
      if (grabOne(rgrp, chn, opt.timeout_ms, fr, &err_r)) {
        void* vir_r = nullptr;
        if (fr.u64VirAddr[0] != 0) {
          vir_r = reinterpret_cast<void*>(fr.u64VirAddr[0]);
        } else {
          const AX_U32 sz = frameMmapSize(fr);
          if (sz > 0 && fr.u64PhyAddr[0] != 0) {
            vir_r = AX_SYS_Mmap(fr.u64PhyAddr[0], sz);
            if (vir_r) {
              fr.u64VirAddr[0] = reinterpret_cast<AX_U64>(vir_r);
            }
          }
        }
        cv::Mat gray_r;
        if (frameToGray(fr, vir_r, gray_r)) {
          cv::imwrite((right_dir / name).string(), gray_r);
        }
        releaseFrame(rgrp, chn, fr);
      }
    }

    releaseFrame(lgrp, chn, fl);
    ++saved;
    if (saved == 1 || saved % 20 == 0) {
      MSF_ILOG << "[collect_gray_images] saved=" << saved << " last=" << left_path
               << " size=" << gray_l.cols << "x" << gray_l.rows << " fail=" << fail << std::endl;
    }
  }

  MSF_ILOG << "[collect_gray_images] done. saved=" << saved << " fail=" << fail
           << " dir=" << opt.output_dir << std::endl;
  MSF_ILOG << "Next: ros2 run ms_fusion_loop train_xfeat_voc -- --image_dir " << opt.output_dir
           << " ...\n";
  return saved > 0 ? 0 : 3;
}
