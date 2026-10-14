#!/usr/bin/env python3
"""Prepare a TUM-VIE raw sequence (skate-hard/running-hard style) for offline replay.

TUM-VIE ships fisheye (Kannala-Brandt / kb4) stereo images, but msf_core's
create_camera.h only implements pinhole-radtan4/5. This script:

  1. Undistorts the stereo image sequence from kb4 fisheye into an equivalent
     pinhole projection (OpenCV cv2.fisheye), so it can be fed to the existing
     pinhole-radtan4 pipeline unmodified.
  2. Derives T_virtual_imu_from_real_imu from the recording's initial static
     segment: msf_core's tiltFromSpecificForce() (initializer.cpp) requires the virtual
     IMU z-axis to carry gravity; we measure the real gravity direction while
     the rig is at rest and solve for the rotation that maps it to -z.
  3. Converts Basalt calibration noise densities into msf_core's arw/vrw/gb_std/
     ab_std convention (see ms_fusion_manager.cpp:73-79 for the forward formula
     this inverts).
  4. Prints a ready-to-paste camera:/imu: YAML block; does NOT touch any repo
     config file itself so the numbers get reviewed before landing.

Usage:
  python3 tumvie_prepare_dataset.py dataset/skate-hard-vi_gt_data.tar.gz \\
      --calib /path/to/camera-calibrationB.json \\
      --out /tmp/tumvie_skate
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path

import cv2
import numpy as np
import yaml

D2R = math.pi / 180.0
STATIC_WINDOW_ROWS = 400  # ~2s at 200Hz; matches the manually-checked static segment

# msf_core's runtime tracking canvas (NPU models are all 640x384 native, see
# tracking.extractor/matcher model_path in msf_robot.yaml). Undistorting
# straight onto this canvas means the replayed images already equal
# measurement_manager.cpp's `already_tracking_size`, so msf_core skips its own
# resize_tracking_image step entirely -- smaller files (less disk I/O on the
# board) and no redundant runtime resize. Fallback only -- read_tracking_canvas()
# below prefers camera.cam0.resolution from an existing profile yaml so this
# script can't silently drift out of sync with the profile it's meant to feed.
TRACKING_CANVAS_W = 640
TRACKING_CANVAS_H = 384


def read_tracking_canvas(profile_path: Path) -> tuple:
    """camera.cam0.resolution from an existing msf_robot.yaml -- the yaml is
    the source of truth for the tracking canvas size (it's what msf_core actually
    reads at runtime); falls back to TRACKING_CANVAS_W/H if the profile doesn't
    exist yet (bootstrapping a brand new profile)."""
    if profile_path is None or not profile_path.exists():
        print(f"[canvas] no profile yaml at {profile_path}, using default "
              f"{TRACKING_CANVAS_W}x{TRACKING_CANVAS_H}")
        return TRACKING_CANVAS_W, TRACKING_CANVAS_H
    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    w, h = profile["camera"]["cam0"]["resolution"]
    print(f"[canvas] read {w}x{h} from {profile_path} (camera.cam0.resolution)")
    return int(w), int(h)


def load_calibration(calib_path: Path) -> dict:
    with calib_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["value0"]


def quat_to_matrix(px: float, py: float, pz: float, qx: float, qy: float, qz: float,
                    qw: float) -> np.ndarray:
    """Basalt T_imu_cam entry -> 4x4 matrix, X_imu = T * X_cam (see plan section 2)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]
    return T


def solve_virtual_imu_rotation(gravity_body: np.ndarray) -> np.ndarray:
    """Return R such that R @ gravity_body_hat == [0, 0, -1]: virtual-IMU-frame
    specific force at rest must have its bulk on NEGATIVE z, not positive.

    Verified from the actual code, not assumed:
    preintegration_base.cpp: gravity_=[0,0,+g], dvel = R*fb*dt + gravity_*dt; at
    rest dvel=0 => R*fb = [0,0,-g]. tiltFromSpecificForce()'s
    roll=-asin(fb_y/g)/pitch=asin(fb_x/g) combined with euler2matrix's
    Rz(yaw)*Ry(pitch)*Rx(roll) only reproduces that [0,0,-g] result when fb's
    dominant component is on -z (checked numerically against a real device's actual
    production T_virtual_imu_from_real_imu=diag(1,-1,-1): a tilted synthetic FLU
    reading transformed through it and run through the exact same formula lands
    on R@fb=[0,0,-9.80999...], not +g). An earlier version of this function
    targeted +z, which is backwards -- it produces a self-inconsistent initial
    attitude that leaves gravity uncompensated and the integrated position/
    rotation diverge (this is what caused the ATE blowup: rmse~1145m/150deg).

    x/y axes of the orthogonal complement are picked arbitrarily (yaw origin is
    free, see plan section 3) via Gram-Schmidt against a reference axis.
    """
    g_hat = gravity_body / np.linalg.norm(gravity_body)
    target_z = np.array([0.0, 0.0, -1.0])
    v = np.cross(g_hat, target_z)
    s = np.linalg.norm(v)
    c = float(np.dot(g_hat, target_z))
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def measure_static_gravity(imu_txt: Path) -> np.ndarray:
    data = np.loadtxt(imu_txt, comments="#")
    ax, ay, az = data[:STATIC_WINDOW_ROWS, 4], data[:STATIC_WINDOW_ROWS, 5], data[:STATIC_WINDOW_ROWS, 6]
    gx, gy, gz = data[:STATIC_WINDOW_ROWS, 1], data[:STATIC_WINDOW_ROWS, 2], data[:STATIC_WINDOW_ROWS, 3]
    gyro_std = float(np.linalg.norm([gx.std(), gy.std(), gz.std()]))
    g_vec = np.array([ax.mean(), ay.mean(), az.mean()])
    g_norm = float(np.linalg.norm(g_vec))
    print(f"[gravity] static-window gyro |std|={gyro_std:.5f} rad/s, "
          f"gravity vec={g_vec}, |g|={g_norm:.4f}")
    if gyro_std > 0.05:
        print("[gravity][WARN] gyro std looks too high for a static segment; "
              "check STATIC_WINDOW_ROWS / recording start.")
    if not (9.5 < g_norm < 10.1):
        print("[gravity][WARN] |g| far from 9.81; result may be unreliable.")
    return g_vec


def convert_noise_params(calib: dict, corr_time_s: float = 3600.0) -> dict:
    gyro_noise_std = np.array(calib["gyro_noise_std"])  # rad/s / sqrt(Hz) == rad/sqrt(s)
    accel_noise_std = np.array(calib["accel_noise_std"])  # m/s^2 / sqrt(Hz) == m/s^1.5
    gyro_bias_std = np.array(calib["gyro_bias_std"])
    accel_bias_std = np.array(calib["accel_bias_std"])

    arw = float(np.mean(gyro_noise_std)) * 60.0 / D2R
    vrw = float(np.mean(accel_noise_std)) * 60.0
    k = math.sqrt(corr_time_s / 2.0)
    gb_std = float(np.mean(gyro_bias_std)) * k * 3600.0 / D2R
    ab_std = float(np.mean(accel_bias_std)) * k / 1.0e-5
    return {
        "arw": arw, "vrw": vrw, "gb_std": gb_std, "ab_std": ab_std,
        "corr_time_hr": corr_time_s / 3600.0,
    }


def compute_canvas_geometry(source_w: int, source_h: int, canvas_w: int, canvas_h: int) -> dict:
    """Mirrors msf_core's makeTopLeftPaddedImageGeometry
    (msf_core/include/ms_fusion/common/padded_image_geometry.h): aspect-preserving
    scale, content anchored top-left, padding only added on the right/bottom."""
    scale = min(canvas_w / source_w, canvas_h / source_h)
    content_w = min(max(1, round(source_w * scale)), canvas_w)
    content_h = min(max(1, round(source_h * scale)), canvas_h)
    return {
        "scale_x": content_w / source_w, "scale_y": content_h / source_h,
        "content_w": content_w, "content_h": content_h,
        "pad_right": canvas_w - content_w, "pad_bottom": canvas_h - content_h,
    }


def undistort_fisheye_dir(src_dir: Path, dst_dir: Path, fx: float, fy: float, cx: float,
                           cy: float, k: list, source_resolution: tuple, canvas_w: int,
                           canvas_h: int) -> tuple:
    """Undistort a kb4 fisheye sequence directly onto msf_core's tracking canvas
    (see compute_canvas_geometry): virtual pinhole is centered and uses the mean
    of the calibrated fx/fy at source resolution (moderate crop, avoids extreme
    fisheye edge stretching), then scaled onto the canvas content area with the
    same linear scale msf_core applies at runtime (no principal-point offset,
    content begins at (0, 0) -- scalePinholeIntrinsicsTopLeft)."""
    w_src, h_src = source_resolution
    geo = compute_canvas_geometry(w_src, h_src, canvas_w, canvas_h)
    f_virtual_src = 0.5 * (fx + fy)
    cx_src, cy_src = w_src / 2.0, h_src / 2.0
    f_content = f_virtual_src * geo["scale_x"]
    cx_content = cx_src * geo["scale_x"]
    cy_content = cy_src * geo["scale_y"]

    K_src = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D_src = np.array(k, dtype=np.float64).reshape(4, 1)
    K_content = np.array([[f_content, 0, cx_content], [0, f_content, cy_content], [0, 0, 1]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_src, D_src, np.eye(3), K_content, (geo["content_w"], geo["content_h"]), cv2.CV_16SC2)

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for jpg in sorted(src_dir.glob("*.jpg")):
        img = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[undistort][WARN] failed to read {jpg}")
            continue
        content = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        canvas = cv2.copyMakeBorder(content, 0, geo["pad_bottom"], 0, geo["pad_right"],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(str(dst_dir / jpg.name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    return count, {"fx": f_content, "fy": f_content, "cx": cx_content, "cy": cy_content}


def format_matrix_yaml(name: str, T: np.ndarray, comment: str = "", indent: int = 2) -> str:
    pad = " " * indent
    lines = [f"{pad}{name}:" + (f"  {comment}" if comment else "")]
    for row in T:
        vals = ", ".join(f"{v:.9g}" for v in row)
        lines.append(f"{pad}- [{vals}]")
    return "\n".join(lines)


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract only regular files/directories that stay below destination.

    Links are refused outright: a link member could point outside destination
    and a later member written through it would escape the check below."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for item in members:
            if not (item.isfile() or item.isdir()):
                raise ValueError(f"refusing non-regular tar member: {item.name}")
            if Path(item.name).is_absolute() or not (root / item.name).resolve().is_relative_to(root):
                raise ValueError(f"refusing unsafe tar member: {item.name}")
        tf.extractall(destination, members=members)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_tgz", type=Path)
    ap.add_argument("--calib", type=Path, required=True,
                     help="Basalt camera-calibrationB.json (TUM-VIE style)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corr-time-hr", type=float, default=1.0)
    ap.add_argument("--profile", type=Path,
                     default=Path("calib_files/tumvie_stereoimu/msf_robot.yaml"),
                     help="existing msf_robot.yaml to read the tracking canvas size "
                          f"from (camera.cam0.resolution); falls back to "
                          f"{TRACKING_CANVAS_W}x{TRACKING_CANVAS_H} if it doesn't exist yet")
    args = ap.parse_args()

    canvas_w, canvas_h = read_tracking_canvas(args.profile)

    args.out.mkdir(parents=True, exist_ok=True)
    extract_dir = args.out / "raw"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {args.dataset_tgz} -> {extract_dir}")
    safe_extract(args.dataset_tgz, extract_dir)

    calib = load_calibration(args.calib)
    cam0, cam1 = calib["intrinsics"][0]["intrinsics"], calib["intrinsics"][1]["intrinsics"]
    res0 = tuple(calib["resolution"][0])
    res1 = tuple(calib["resolution"][1])
    T_imu_cam0 = quat_to_matrix(**calib["T_imu_cam"][0])
    T_imu_cam1 = quat_to_matrix(**calib["T_imu_cam"][1])
    # Basalt's single global cam-to-imu offset (t_imu = t_cam + offset), matches
    # td_cam_to_imu's usage in measurement_manager.cpp:625
    # (timestamp_corrected = image_message.timestamp + td_cam_to_imu_).
    td_cam_to_imu = calib["cam_time_offset_ns"] * 1e-9

    print(f"[undistort] cam0 {res0} fx={cam0['fx']:.2f} fy={cam0['fy']:.2f} "
          f"-> canvas {canvas_w}x{canvas_h}")
    n0, virt0 = undistort_fisheye_dir(
        extract_dir / "left_images", args.out / "left",
        cam0["fx"], cam0["fy"], cam0["cx"], cam0["cy"],
        [cam0["k1"], cam0["k2"], cam0["k3"], cam0["k4"]], res0,
        canvas_w, canvas_h)
    print(f"[undistort] cam0: {n0} images -> {args.out/'left'}, canvas K={virt0}")

    print(f"[undistort] cam1 {res1} fx={cam1['fx']:.2f} fy={cam1['fy']:.2f} "
          f"-> canvas {canvas_w}x{canvas_h}")
    n1, virt1 = undistort_fisheye_dir(
        extract_dir / "right_images", args.out / "right",
        cam1["fx"], cam1["fy"], cam1["cx"], cam1["cy"],
        [cam1["k1"], cam1["k2"], cam1["k3"], cam1["k4"]], res1,
        canvas_w, canvas_h)
    print(f"[undistort] cam1: {n1} images -> {args.out/'right'}, canvas K={virt1}")

    g_vec = measure_static_gravity(extract_dir / "imu_data.txt")
    R_virtual_from_real = solve_virtual_imu_rotation(g_vec)
    T_virtual_from_real = np.eye(4)
    T_virtual_from_real[:3, :3] = R_virtual_from_real
    check = R_virtual_from_real @ (g_vec / np.linalg.norm(g_vec))
    print(f"[gravity] R@g_hat = {check} (expect ~[0,0,-1])")

    noise = convert_noise_params(calib, corr_time_s=args.corr_time_hr * 3600.0)
    print(f"[noise] arw={noise['arw']:.6f} deg/sqrt(hr)  vrw={noise['vrw']:.6f} m/s/sqrt(hr)  "
          f"gb_std={noise['gb_std']:.6f} deg/hr  ab_std={noise['ab_std']:.6f} mGal  "
          f"(corr_time={noise['corr_time_hr']} hr)")

    for name, path in [("left_images/image_timestamps_left.txt",
                         extract_dir / "left_images" / "image_timestamps_left.txt"),
                        ("right_images/image_timestamps_right.txt",
                         extract_dir / "right_images" / "image_timestamps_right.txt"),
                        ("imu_data.txt", extract_dir / "imu_data.txt"),
                        ("mocap_data.txt", extract_dir / "mocap_data.txt")]:
        dst = args.out / Path(name).name
        dst.write_bytes(path.read_bytes())
    print(f"[copy] timestamps/imu/mocap copied flat into {args.out}")

    yaml_block = f"""
# ---- paste into calib_files/tumvie_stereoimu/msf_robot.yaml ----
imu:
{format_matrix_yaml('T_virtual_imu_from_real_imu', T_virtual_from_real, '# derived from static gravity, see tumvie_prepare_dataset.py')}
  arw: {noise['arw']:.6f}                            # IMU的角度随机游走（deg/sqrt(hr)），从 Basalt gyro_noise_std 换算
  vrw: {noise['vrw']:.6f}                            # IMU的速度随机游走（m/s/sqrt(hr)），从 Basalt accel_noise_std 换算
  gb_std: {noise['gb_std']:.6f}                       # 陀螺仪偏置标准差（deg/hr），corr_time={noise['corr_time_hr']}hr 反解
  ab_std: {noise['ab_std']:.6f}                       # 加表偏置标准差（mGal），corr_time={noise['corr_time_hr']}hr 反解
  corr_time: {noise['corr_time_hr']:.4f}

camera:
  cam0:
    model: pinhole-radtan4
    resolution: [{canvas_w}, {canvas_h}]
    intrinsic: [{virt0['fx']:.6f}, {virt0['fy']:.6f}, {virt0['cx']:.6f}, {virt0['cy']:.6f}]
    distortion: [0.0, 0.0, 0.0, 0.0]  # 已在 tumvie_prepare_dataset.py 里去畸变
    td_cam_to_imu: {td_cam_to_imu:.9f}   # 从 Basalt cam_time_offset_ns 换算，t_imu = t_cam + td
{format_matrix_yaml('T_imu_cam', T_imu_cam0, indent=4)}

  cam1:
    model: pinhole-radtan4
    resolution: [{canvas_w}, {canvas_h}]
    intrinsic: [{virt1['fx']:.6f}, {virt1['fy']:.6f}, {virt1['cx']:.6f}, {virt1['cy']:.6f}]
    distortion: [0.0, 0.0, 0.0, 0.0]
    td_cam_to_imu: {td_cam_to_imu:.9f}
{format_matrix_yaml('T_imu_cam', T_imu_cam1, indent=4)}

tracking:
  tracking:
    downsize_camera: false   # 图片已在预处理阶段 resize 到 tracking canvas 尺寸，运行时跳过 resize
"""
    print(yaml_block)
    (args.out / "generated_profile_snippet.yaml").write_text(yaml_block, encoding="utf-8")
    print(f"[done] snippet also written to {args.out/'generated_profile_snippet.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
