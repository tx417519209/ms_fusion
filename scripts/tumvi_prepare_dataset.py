#!/usr/bin/env python3
"""Prepare a TUM VI EuRoC export for ms_fusion's offline board replay.

The official 512_16 export contains 16-bit fisheye PNGs and EuRoC CSV files.
ms_fusion accepts only pinhole-radtan images, so this tool undistorts both eyes
into the configured tracking canvas, emits JPEGs plus the flat text layout used
by tumvie_replay_node.py, and generates a dedicated pure-VIO profile.
"""
from __future__ import annotations

import argparse
import math
import tarfile
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", comments="#", dtype=str, ndmin=2)


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


def find_mav0(root: Path) -> Path:
    matches = list(root.rglob("mav0/cam0/data.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one mav0/cam0/data.csv below {root}, found {len(matches)}")
    return matches[0].parent.parent


def virtual_imu_from_cam0(calib: dict) -> np.ndarray:
    """Build real-IMU -> virtual-FRD from the official cam0-IMU calibration.

    TUM VI's calibrated export already applies IMU scale/axis corrections and
    supplies T_cam_imu.  OpenCV camera coordinates are right/down/forward;
    ms_fusion's virtual IMU convention is front/right/down.  Therefore this is
    a fixed rig transform, not an estimate from a possibly moving room sequence.
    """
    R_cam_from_imu = np.asarray(calib["cam0"]["T_cam_imu"], dtype=float)[:3, :3]
    R_frd_from_cam = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    return R_frd_from_cam @ R_cam_from_imu


def matrix_rows(T: np.ndarray) -> list:
    return [[float(x) for x in row] for row in T]


def prepare_images(cam: dict, src: Path, dst: Path, rows: np.ndarray, canvas: tuple[int, int]) -> tuple:
    width, height = canvas
    fx, fy, cx, cy = map(float, cam["intrinsics"])
    distortion = np.asarray(cam["distortion_coeffs"], dtype=np.float64).reshape(4, 1)
    src_w, src_h = map(int, cam["resolution"])
    scale = min(width / src_w, height / src_h)
    content_w, content_h = round(src_w * scale), round(src_h * scale)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    f = 0.5 * (fx + fy) * scale
    Knew = np.array([[f, 0, 0.5 * src_w * scale], [0, f, 0.5 * src_h * scale], [0, 0, 1]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), Knew,
                                                       (content_w, content_h), cv2.CV_16SC2)
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for timestamp, filename in rows[:, :2]:
        image = cv2.imread(str(src / filename), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cannot read {src / filename}")
        # 16-bit calibrated images are mapped deterministically to 8-bit before JPEG.
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=1.0 / 256.0)
        image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        image = cv2.copyMakeBorder(image, 0, height - content_h, 0, width - content_w,
                                   cv2.BORDER_CONSTANT, value=0)
        name = f"{timestamp}.jpg"
        if not cv2.imwrite(str(dst / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"cannot write {dst / name}")
        names.append(name)
    return (f, f, Knew[0, 2], Knew[1, 2]), names


def write_profile(template: Path, output: Path, calib: dict, intr0: tuple, intr1: tuple,
                  R_virtual: np.ndarray) -> None:
    with template.open() as f:
        profile = yaml.safe_load(f)
    profile["base"]["log_imu_propagation_debug"] = False
    profile["imu"].update({"topic": "/tumvi/imu",
        "arw": float(calib["imu0"]["gyroscope_noise_density"]) * 60 / (math.pi / 180),
        "vrw": float(calib["imu0"]["accelerometer_noise_density"]) * 60,
        # TUM VI provides random-walk density.  Convert it to this project's
        # Gauss-Markov stationary bias std using the configured one-hour tau.
        "gb_std": float(calib["imu0"]["gyroscope_random_walk"]) * math.sqrt(1800) * 3600 / (math.pi / 180),
        "ab_std": float(calib["imu0"]["accelerometer_random_walk"]) * math.sqrt(1800) / 1e-5,
        "corr_time": 1.0, "T_virtual_imu_from_real_imu": matrix_rows(np.block([[R_virtual, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))})
    for key, intr in (("cam0", intr0), ("cam1", intr1)):
        cam = calib[key]
        T_imu_cam = np.linalg.inv(np.asarray(cam["T_cam_imu"], dtype=float))
        profile["camera"][key].update({"resolution": [640, 384], "intrinsic": list(map(float, intr)),
            "distortion": [0., 0., 0., 0.], "td_cam_to_imu": 0.0,
            "T_imu_cam": matrix_rows(T_imu_cam)})
        profile["camera"][key]["offline"].update({"topic": f"/tumvi/{key}", "image_type": "compressed"})
    profile["tracking"]["tracking"]["downsize_camera"] = False
    for feature in ("odom", "reloc", "matched_cloud", "loop"):
        if feature in profile:
            profile[feature]["enable"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        yaml.safe_dump(profile, f, sort_keys=False)


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("archive", type=Path, help="dataset-room1_512_16.tar")
    p.add_argument("--calib", required=True, type=Path, help="official camchain-imucam-imucalib.yaml")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--extract", type=Path, default=None, help="raw extraction directory")
    p.add_argument("--template", type=Path, default=Path("calib_files/tumvie_stereoimu/msf_robot.yaml"))
    p.add_argument("--profile-out", type=Path, default=Path("calib_files/tumvi_room1/msf_robot.yaml"))
    args = p.parse_args()
    extraction = args.extract or args.out.parent / "tumvi_room1_raw"
    if not (extraction / ".extracted").exists():
        safe_extract(args.archive, extraction)
        (extraction / ".extracted").touch()
    mav0 = find_mav0(extraction)
    with args.calib.open() as f:
        calib = yaml.safe_load(f)
    # The official Kalibr zip stores camera-chain and IMU noise metadata in
    # separate files.  Keep --calib pointing at the camera-chain source and
    # load its sibling explicitly, rather than silently substituting values.
    imu_calib = args.calib.with_name("imu-imucalib.yaml")
    if not imu_calib.exists():
        raise FileNotFoundError(f"TUM VI IMU calibration missing beside camera chain: {imu_calib}")
    with imu_calib.open() as f:
        calib.update(yaml.safe_load(f))
    cam0_rows, cam1_rows = load_csv(mav0 / "cam0/data.csv"), load_csv(mav0 / "cam1/data.csv")
    if len(cam0_rows) != len(cam1_rows) or not np.array_equal(cam0_rows[:, 0], cam1_rows[:, 0]):
        raise RuntimeError("TUM VI stereo CSVs are not one-to-one timestamp aligned")
    imu = load_csv(mav0 / "imu0/data.csv").astype(float)
    R_virtual = virtual_imu_from_cam0(calib)
    first_accel = imu[:400, 4:7].mean(axis=0)
    print("[frame] official cam0-IMU calibration -> virtual FRD; first-2s mean accel "
          f"real={first_accel}, virtual={R_virtual @ first_accel}")
    args.out.mkdir(parents=True, exist_ok=True)
    intr0, _ = prepare_images(calib["cam0"], mav0 / "cam0/data", args.out / "left", cam0_rows, (640, 384))
    intr1, _ = prepare_images(calib["cam1"], mav0 / "cam1/data", args.out / "right", cam1_rows, (640, 384))
    np.savetxt(args.out / "image_timestamps_left.txt", cam0_rows[:, 0].astype(np.int64) // 1000, fmt="%d")
    imu_out = np.column_stack((imu[:, 0].astype(np.int64) // 1000, imu[:, 1:7]))
    np.savetxt(args.out / "imu_data.txt", imu_out, fmt=["%d"] + ["%.12g"] * 6)
    # TUM VI calls this stream mocap0; retain the common EuRoC name as a
    # fallback so a compatible export remains usable.
    gt_path = mav0 / "mocap0/data.csv"
    if not gt_path.exists():
        gt_path = mav0 / "state_groundtruth_estimate0/data.csv"
    if gt_path.exists():
        gt = load_csv(gt_path).astype(float)
        # EuRoC: t, p xyz, q_wxyz, ... -> our evaluator: t_us, p xyz, q_xyzw.
        out_gt = np.column_stack((gt[:, 0].astype(np.int64) // 1000, gt[:, 1:4], gt[:, 5:8], gt[:, 4]))
        np.savetxt(args.out / "mocap_data.txt", out_gt, fmt=["%d"] + ["%.12g"] * 7)
    write_profile(args.template, args.profile_out, calib, intr0, intr1, R_virtual)
    print(f"[done] prepared {len(cam0_rows)} stereo pairs, {len(imu)} IMU rows in {args.out}")
    print(f"[done] profile: {args.profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
