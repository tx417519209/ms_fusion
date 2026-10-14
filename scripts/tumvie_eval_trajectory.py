#!/usr/bin/env python3
"""Record ms_fusion's odometry output while a TUM-VIE replay runs, then compute
Absolute Trajectory Error (ATE) against mocap_data.txt ground truth.

Two subcommands:
  record   Subscribe to a nav_msgs/Odometry topic, dump to a TUM-format file
           extended with world-frame velocity (timestamp tx ty tz qx qy qz qw
           vx vy vz) until Ctrl-C or --duration-s elapses. vx/vy/vz come from
           msg.twist.twist.linear -- only populated by ms_fusion_node builds
           that fill OdomInfo::velocity (see ms_fusion_manager.cpp); older
           builds publish an all-zero twist, which still parses fine here but
           makes velocity columns meaningless -- rebuild+redeploy first.
  eval     Compare a recorded TUM trajectory against mocap_data.txt: time-align
           (linear interpolation), Umeyama SE3 (rotation+translation, no scale --
           VIO is metric) rigid alignment, report ATE RMSE/max for position and
           rotation. No external deps beyond numpy (evo isn't installed here).

Usage:
  # terminal running ms_fusion + tumvie_replay_node.py already going:
  python3 tumvie_eval_trajectory.py record --topic /odometry_vio --out skate_est.tum --duration-s 90

  python3 tumvie_eval_trajectory.py eval --gt /tmp/tumvie_skate/mocap_data.txt \\
      --est skate_est.tum --out skate_ate.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def cmd_record(args: argparse.Namespace) -> int:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    rows = []

    class Recorder(Node):
        def __init__(self):
            super().__init__("tumvie_eval_recorder")
            qos = QoSProfile(depth=2000, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              durability=QoSDurabilityPolicy.VOLATILE)
            self.sub = self.create_subscription(Odometry, args.topic, self.cb, qos)
            self.get_logger().info(f"recording {args.topic} -> {args.out}")

        def cb(self, msg: Odometry):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            rows.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z))

    import time as _time

    rclpy.init()
    node = Recorder()
    start = _time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.duration_s > 0 and (_time.monotonic() - start) >= args.duration_s:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT 先被 rclpy 的信号处理器接走并关掉 context，spin_once 抛的是
        # ExternalShutdownException 而不是 KeyboardInterrupt，两者都要接住再写文件
        pass
    finally:
        node.destroy_node()
        # rclpy.init() installs its own SIGINT handler that shuts the context down
        # at the rcl level as soon as Ctrl-C arrives -- it usually wins the race
        # against this thread's `except KeyboardInterrupt`, so by the time we get
        # here rclpy.shutdown() has often already run once. Calling it again raises
        # RCLError("rcl_shutdown already called"), which -- since this is a bare
        # finally, not try/except -- propagates out of cmd_record() and skips the
        # write below entirely, silently losing every row recorded so far.
        if rclpy.ok():
            rclpy.shutdown()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(f"{v:.9f}" for v in row) + "\n")
    print(f"[record] wrote {len(rows)} poses to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def quat_to_R(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Batch quaternion (N,) each -> (N,3,3) rotation matrices."""
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    N = len(qx)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


def load_tum(path: Path) -> np.ndarray:
    """timestamp tx ty tz qx qy qz qw [vx vy vz], whitespace separated, '#'
    comments. The trailing velocity columns are optional (older recordings
    from before OdomInfo::velocity existed only have 8 columns) -- callers
    that need them must check est.shape[1] > 8 rather than assume they're
    there."""
    return np.loadtxt(path, comments="#")


def load_mocap(path: Path) -> np.ndarray:
    """time(us) px py pz qx qy qz qw -> same layout as TUM but time in seconds."""
    data = np.loadtxt(path, comments="#")
    data[:, 0] *= 1e-6
    return data


def load_imu_marker_extrinsic(path: Path) -> tuple:
    """TUM-VIE's official mocap-imu-calibration{B,...}.json: T_imu_marker is the
    marker rig's pose expressed in the IMU frame (same T_A_B = "B's pose in frame
    A" convention as T_imu_cam elsewhere in this repo -- X_imu = T_imu_marker @
    X_marker). Returns (R_im (3,3), p_im (3,)). mocap_data.txt's own quaternion is
    the marker rig's orientation, not the IMU's -- without this correction,
    rotation ATE is dominated by this fixed, uncalibrated-looking offset instead
    of actual attitude tracking error (confirmed: this file's rotation angle is
    ~75.7 deg, matching the ~65-90 deg constant offset empirically observed before
    this fix)."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)["value0"]["T_imu_marker"]
    q = np.array([raw["qx"], raw["qy"], raw["qz"], raw["qw"]])
    q = q / np.linalg.norm(q)
    R_im = quat_to_R(q[0:1], q[1:2], q[2:3], q[3:4])[0]
    p_im = np.array([raw["px"], raw["py"], raw["pz"]])
    return R_im, p_im


def convert_marker_gt_to_imu_gt(gt: np.ndarray, R_im: np.ndarray, p_im: np.ndarray) -> np.ndarray:
    """gt columns are (t, px,py,pz, qx,qy,qz,qw) of the mocap MARKER pose in the
    mocap world frame (T_world_marker). Returns the same layout but for the IMU
    pose instead (T_world_imu = T_world_marker @ T_marker_imu, T_marker_imu =
    inverse(T_imu_marker)): R_wi = R_wm @ R_im.T, p_wi = p_wm - R_wm @ (R_im.T @ p_im)."""
    R_wm = quat_to_R(gt[:, 4], gt[:, 5], gt[:, 6], gt[:, 7])
    R_wi = R_wm @ R_im.T
    p_wi = gt[:, 1:4] - np.einsum("nij,j->ni", R_wm, R_im.T @ p_im)

    out = gt.copy()
    out[:, 1:4] = p_wi
    out[:, 4:8] = R_to_quat(R_wi)
    return out


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices -> (N,4) unit quaternions (qx,qy,qz,qw).
    Shepperd's method: divide by the largest of 4*qw, 4*qx, 4*qy, 4*qz, so a
    rotation near 180 deg (qw -> 0) stays well-conditioned instead of collapsing
    to a zero / identity quaternion."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.empty((R.shape[0], 4))
    case = np.argmax(np.stack([tr, m00, m11, m22], axis=1), axis=1)

    i = case == 0
    s = 2.0 * np.sqrt(np.maximum(1.0 + tr[i], 1e-12))
    q[i, 3] = 0.25 * s
    q[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / s
    q[i, 1] = (R[i, 0, 2] - R[i, 2, 0]) / s
    q[i, 2] = (R[i, 1, 0] - R[i, 0, 1]) / s

    i = case == 1
    s = 2.0 * np.sqrt(np.maximum(1.0 + m00[i] - m11[i] - m22[i], 1e-12))
    q[i, 3] = (R[i, 2, 1] - R[i, 1, 2]) / s
    q[i, 0] = 0.25 * s
    q[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / s
    q[i, 2] = (R[i, 0, 2] + R[i, 2, 0]) / s

    i = case == 2
    s = 2.0 * np.sqrt(np.maximum(1.0 - m00[i] + m11[i] - m22[i], 1e-12))
    q[i, 3] = (R[i, 0, 2] - R[i, 2, 0]) / s
    q[i, 0] = (R[i, 0, 1] + R[i, 1, 0]) / s
    q[i, 1] = 0.25 * s
    q[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / s

    i = case == 3
    s = 2.0 * np.sqrt(np.maximum(1.0 - m00[i] - m11[i] + m22[i], 1e-12))
    q[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / s
    q[i, 0] = (R[i, 0, 2] + R[i, 2, 0]) / s
    q[i, 1] = (R[i, 1, 2] + R[i, 2, 1]) / s
    q[i, 2] = 0.25 * s

    return q / np.linalg.norm(q, axis=1, keepdims=True)


def interpolate_gt(gt: np.ndarray, query_t: np.ndarray, max_gap_s: float = 0.5) -> tuple:
    """Linearly interpolate gt position at query_t; SLERP-free nearest-quaternion
    (fine for ATE where ground truth sampling is dense relative to motion).

    Returns (interpolated (N,8), valid_mask (N,) bool). Mocap tracking can drop out
    for real (e.g. TUM-VIE skate-hard has a genuine ~32s gap where the subject left
    the capture volume) -- straight-line interpolating across a gap that wide
    fabricates ground truth that has no relation to the real trajectory. valid_mask
    is False for any query_t whose straddling gt samples are more than max_gap_s
    apart, so cmd_eval can exclude those points instead of scoring against noise."""
    t_gt = gt[:, 0]
    out = np.zeros((len(query_t), 8))
    out[:, 0] = query_t
    for col in range(1, 4):
        out[:, col] = np.interp(query_t, t_gt, gt[:, col])
    right_idx = np.clip(np.searchsorted(t_gt, query_t), 1, len(t_gt) - 1)
    left_idx = right_idx - 1
    out[:, 4:8] = gt[right_idx, 4:8]
    gap = t_gt[right_idx] - t_gt[left_idx]
    valid_mask = gap <= max_gap_s
    return out, valid_mask


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Rigid (no scale) alignment: find R,t minimizing ||R@src+t - dst||^2.
    src, dst: (N,3). Returns (R (3,3), t (3,))."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,) rotation-angle magnitude in degrees."""
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def detect_gt_segments(gt: np.ndarray, gap_factor: float = 50.0, min_gap_s: float = 0.5) -> tuple:
    """Split gt into gap-free (start_idx, end_idx) segments (end_idx exclusive).
    A gap is declared where consecutive gt samples are more than
    max(gap_factor / f_gt, min_gap_s) apart, f_gt = 1/median(diff(gt[:,0])) the
    nominal sampling rate -- catches real mocap tracking dropouts (e.g.
    skate-hard has one ~32s gap) so callers can evaluate each gap-free run
    separately instead of letting one straight-line-interpolated gap corrupt a
    single whole-trajectory ATE.

    The min_gap_s floor matters because gap_factor/f_gt alone isn't a portable
    threshold across TUM-VIE sequences: real mocap data has frequent brief
    ~100-135ms occlusion hiccups during fast motion (confirmed on skate-hard)
    that are fine to interpolate across, not real segment boundaries. A purely
    relative threshold tuned to avoid flagging those on one sequence could still
    misfire on another with a different nominal rate or hiccup pattern; the
    absolute floor guarantees anything under min_gap_s is never treated as a
    structural gap regardless of f_gt."""
    t = gt[:, 0]
    dt = np.diff(t)
    f_gt = 1.0 / np.median(dt)
    gap_threshold_s = max(gap_factor / f_gt, min_gap_s)
    gap_idx = np.where(dt > gap_threshold_s)[0]  # dt[i] is gap between row i and i+1
    bounds = [0] + [i + 1 for i in gap_idx] + [len(t)]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    return segments, f_gt, gap_threshold_s


def se3_from_row(row: np.ndarray) -> tuple:
    """row: (t,px,py,pz,qx,qy,qz,qw) -> (R (3,3), t (3,))."""
    R = quat_to_R(row[4:5], row[5:6], row[6:7], row[7:8])[0]
    return R, row[1:4]


def se3_relative_error(R_before_est, t_before_est, R_after_est, t_after_est,
                        R_before_gt, t_before_gt, R_after_gt, t_after_gt) -> tuple:
    """How much the estimated relative motion between two timestamps deviates from
    the true (gt) relative motion -- alignment-invariant (no Umeyama needed): for
    any fixed global transform applied equally to both endpoints of a trajectory,
    inv(before)@after is unchanged, so this works directly on raw (unaligned)
    poses. This is exactly "how much extra drift accumulated" across a stretch
    (e.g. a mocap tracking gap) where gt itself can't directly verify the middle,
    but its two endpoints are still trustworthy on their own.
    Returns (translation_error_m, rotation_error_deg)."""
    R_rel_est = R_before_est.T @ R_after_est
    t_rel_est = R_before_est.T @ (t_after_est - t_before_est)
    R_rel_gt = R_before_gt.T @ R_after_gt
    t_rel_gt = R_before_gt.T @ (t_after_gt - t_before_gt)

    R_drift = R_rel_est.T @ R_rel_gt
    t_drift = R_rel_est.T @ (t_rel_gt - t_rel_est)
    return float(np.linalg.norm(t_drift)), float(rotation_angle_deg(R_drift[np.newaxis, :, :])[0])


def evaluate_segment(est_all: np.ndarray, gt_segment: np.ndarray, max_gt_gap_s: float,
                      use_official_extrinsic: bool, gauge: dict | None = None) -> dict | None:
    """Core ATE computation (overlap window, interpolation, Umeyama, rotation
    error) restricted to one gap-free gt segment. gt_segment must already be in
    the IMU frame if a mocap-imu-calib was applied (use_official_extrinsic=True --
    then R_gt needs no further correction). Otherwise a constant body-frame
    rotation offset is fit from this segment's own data (see
    solve_body_rotation_offset).

    gauge=None (default): fit position (Umeyama) and rotation (Wahba) alignment
    from THIS segment's own data -- "own-gauge" mode. Characterizes this
    segment's internal shape/consistency independent of any other segment, but
    says nothing about absolute long-term drift, and (for position) can be
    numerically unstable if this segment's own position track is badly
    degenerate/diverged.

    gauge={"R_align","t_align","R_world","R_body"} (from a previous
    evaluate_segment's returned gauge): reuse a FIXED alignment fit from
    elsewhere (typically the first trustworthy segment) instead of refitting
    here -- "reference-gauge" mode. This is what actually reflects cumulative
    real-world drift across gaps/segments; own-gauge mode would silently refit
    away exactly the drift you're trying to measure.

    Returns None if there aren't enough overlapping poses to evaluate, else a
    dict that also carries "gauge" (this segment's own fit, own-gauge mode
    only) for a caller to reuse on later segments."""
    t_lo = max(est_all[0, 0], gt_segment[0, 0])
    t_hi = min(est_all[-1, 0], gt_segment[-1, 0])
    # A too-short/too-few-points segment makes the per-segment Umeyama fit
    # numerically unstable (near-collinear samples over tens of ms can yield an
    # almost arbitrary best-fit rotation) -- refuse rather than report noise.
    if t_hi - t_lo < 1.0:
        return None
    mask = (est_all[:, 0] >= t_lo) & (est_all[:, 0] <= t_hi)
    est = est_all[mask]
    if len(est) < 10:
        return None

    gt_aligned, gt_valid = interpolate_gt(gt_segment, est[:, 0], max_gap_s=max_gt_gap_s)
    est = est[gt_valid]
    gt_aligned = gt_aligned[gt_valid]
    if len(est) < 10:
        return None

    R_est = quat_to_R(est[:, 4], est[:, 5], est[:, 6], est[:, 7])
    R_gt = quat_to_R(gt_aligned[:, 4], gt_aligned[:, 5], gt_aligned[:, 6], gt_aligned[:, 7])

    if gauge is not None:
        R_align, t_align = gauge["R_align"], gauge["t_align"]
        R_world, R_body = gauge["R_world"], gauge["R_body"]
    else:
        R_align, t_align = umeyama_se3(est[:, 1:4], gt_aligned[:, 1:4])
        # Rotation alignment is solved independently from the position (Umeyama)
        # fit above -- see solve_wahba_rotation docstring for why reusing R_align
        # here would make rotation ATE unreliable whenever position tracking has
        # diverged badly within this segment.
        R_world = solve_wahba_rotation(R_est, R_gt)
        R_body = None
        if not use_official_extrinsic:
            R_est_aligned_tmp = np.einsum("ij,njk->nik", R_world, R_est)
            R_body = solve_body_rotation_offset(R_est_aligned_tmp, R_gt)

    est_pos_aligned = (R_align @ est[:, 1:4].T).T + t_align
    pos_err = np.linalg.norm(est_pos_aligned - gt_aligned[:, 1:4], axis=1)

    R_est_aligned = np.einsum("ij,njk->nik", R_world, R_est)
    body_offset_deg = None
    if use_official_extrinsic:
        R_est_final = R_est_aligned
    else:
        body_offset_deg = float(rotation_angle_deg(R_body[np.newaxis, :, :])[0])
        R_est_final = np.einsum("nij,jk->nik", R_est_aligned, R_body)

    R_err = np.einsum("nij,nkj->nik", R_est_final, R_gt)
    rot_err_deg = rotation_angle_deg(R_err)

    return {
        "t_lo": t_lo, "t_hi": t_hi, "n_poses": len(est),
        "pos_rmse": float(np.sqrt(np.mean(pos_err**2))), "pos_mean": float(pos_err.mean()),
        "pos_max": float(pos_err.max()),
        "rot_rmse": float(np.sqrt(np.mean(rot_err_deg**2))), "rot_mean": float(rot_err_deg.mean()),
        "rot_max": float(rot_err_deg.max()), "body_offset_deg": body_offset_deg,
        "gauge": {"R_align": R_align, "t_align": t_align, "R_world": R_world, "R_body": R_body},
        # Endpoints of the poses actually evaluated, with gt interpolated at the
        # same timestamps (the est track may cover only part of the gt segment).
        "first_est_row": est[0], "last_est_row": est[-1],
        "first_gt_row": gt_aligned[0], "last_gt_row": gt_aligned[-1],
    }


def solve_wahba_rotation(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Best-fit constant LEFT (world-frame) rotation R_world (Wahba's problem --
    same chordal-L2 idea as Umeyama/solve_body_rotation_offset, but solved
    directly from rotation data only) such that R_world @ R_a[i] ~= R_b[i] for
    every i. R_a, R_b: (N,3,3).

    Deliberately NOT reusing the position-based Umeyama rotation for this: if a
    segment's estimated position has diverged far outside the ground-truth's own
    range (confirmed on skate-hard's post-gap segment, positions wandering ~50m+
    against a room only a few meters across), Umeyama's rotation is fit jointly
    with a poorly-conditioned translation problem and can land far from the true
    world alignment -- producing a rotation ATE that's actually measuring a bad
    gauge, not real attitude error (verified: reusing an earlier, trustworthy
    segment's own gauge instead showed rotation error flat around 10 deg
    throughout, not the 127 deg the per-segment Umeyama rotation implied).
    Solving rotation alignment independently from translation makes rotation ATE
    robust to that failure mode regardless of how badly position tracking has
    drifted."""
    S = np.einsum("nij,nkj->ik", R_b, R_a)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    return U @ D @ Vt


def solve_body_rotation_offset(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Best-fit constant rotation R_body (chordal L2 / rotation-averaging, same idea
    as Umeyama but for SO(3) instead of R^3) such that R_a[i] @ R_body ~= R_b[i] for
    every i. R_a, R_b: (N,3,3).

    TUM-VIE's mocap quaternion is the orientation of the *marker rig*, not our
    IMU/virtual-IMU frame -- there's a fixed, uncalibrated body-frame rotation
    between the two that this dataset doesn't publish. Umeyama only solves a
    world-frame (left-multiply) alignment, which can't remove a body-frame
    (right-multiply) constant offset; without correcting for it, rotation ATE is
    dominated by this unknown marker extrinsic rather than actual attitude
    tracking error (verified: aligning only the first 15s of a real run showed
    rotation error flat at ~65-90 deg from t=0, not growing with time like
    position error does -- a constant offset, not drift)."""
    S = np.einsum("nij,nik->jk", R_a, R_b)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    return U @ D @ Vt


def cmd_eval(args: argparse.Namespace) -> int:
    est = load_tum(args.est)
    gt = load_mocap(args.gt)
    if len(est) < 10:
        print(f"[eval][ERROR] only {len(est)} estimated poses, refusing to evaluate", file=sys.stderr)
        return 1

    use_official_extrinsic = args.mocap_imu_calib is not None
    if use_official_extrinsic:
        R_im, p_im = load_imu_marker_extrinsic(args.mocap_imu_calib)
        offset_deg = float(rotation_angle_deg(R_im[np.newaxis, :, :])[0])
        print(f"[eval] converting mocap marker-rig ground truth to IMU ground truth "
              f"via {args.mocap_imu_calib} (T_imu_marker rotation={offset_deg:.2f} deg, "
              f"translation={np.linalg.norm(p_im):.3f} m)")
        gt = convert_marker_gt_to_imu_gt(gt, R_im, p_im)

    if args.start_s is not None:
        gt = gt[gt[:, 0] >= est[0, 0] + args.start_s]
    if args.end_s is not None:
        gt = gt[gt[:, 0] <= est[0, 0] + args.end_s]
    if len(gt) < 10:
        print(f"[eval][ERROR] only {len(gt)} gt pose(s) left after --start-s/--end-s, "
              f"refusing to evaluate", file=sys.stderr)
        return 1

    segments, f_gt, gap_threshold_s = detect_gt_segments(gt, gap_factor=args.gap_factor,
                                                          min_gap_s=args.min_gap_s)
    print(f"[eval] gt sampling rate ~{f_gt:.2f}Hz, gap threshold {gap_threshold_s:.3f}s "
          f"(max({args.gap_factor}x nominal period, {args.min_gap_s}s floor)) "
          f"-> {len(segments)} gap-free gt segment(s)")

    lines = ["=== own-gauge evaluation (each segment aligned independently) ===\n"
             "Characterizes each segment's internal shape/consistency in isolation -- does\n"
             "NOT reflect cumulative drift across segments (own-gauge silently refits away\n"
             "exactly that). See 'reference-gauge evaluation' below for actual long-term\n"
             "drift, and prefer that section for judging overall tracking quality.\n\n"]
    seg_results = []
    for k, (i0, i1) in enumerate(segments):
        gt_seg = gt[i0:i1]
        res = evaluate_segment(est, gt_seg, args.max_gt_gap_s, use_official_extrinsic)
        seg_results.append(res)
        if res is None:
            lines.append(f"GT segment {k}: [{gt_seg[0,0]:.3f}, {gt_seg[-1,0]:.3f}]s "
                          f"({gt_seg[-1,0]-gt_seg[0,0]:.2f}s, {len(gt_seg)} gt samples) "
                          f"-- no overlapping estimated poses, skipped\n")
            continue
        offset_note = (f"  (fitted body-frame offset: {res['body_offset_deg']:.2f} deg -- no "
                        f"--mocap-imu-calib given)" if res["body_offset_deg"] is not None else "")
        lines.append(
            f"GT segment {k}: [{res['t_lo']:.3f}, {res['t_hi']:.3f}]s  n_poses={res['n_poses']}\n"
            f"  position ATE (m):  rmse={res['pos_rmse']:.4f}  mean={res['pos_mean']:.4f}  "
            f"max={res['pos_max']:.4f}\n"
            f"  rotation ATE (deg): rmse={res['rot_rmse']:.4f}  mean={res['rot_mean']:.4f}  "
            f"max={res['rot_max']:.4f}{offset_note}\n"
        )

    lines.append("\n=== reference-gauge evaluation "
                  "(all segments aligned to the first evaluable segment) ===\n"
                  "This is the one that reflects actual cumulative drift.\n\n")
    reference_gauge = next((r["gauge"] for r in seg_results if r is not None), None)
    if reference_gauge is None:
        lines.append("(no segment had enough poses to establish a reference gauge)\n")
    else:
        for k, (i0, i1) in enumerate(segments):
            gt_seg = gt[i0:i1]
            res = evaluate_segment(est, gt_seg, args.max_gt_gap_s, use_official_extrinsic,
                                    gauge=reference_gauge)
            if res is None:
                lines.append(f"GT segment {k}: [{gt_seg[0,0]:.3f}, {gt_seg[-1,0]:.3f}]s "
                              f"-- no overlapping estimated poses, skipped\n")
                continue
            lines.append(
                f"GT segment {k}: [{res['t_lo']:.3f}, {res['t_hi']:.3f}]s  n_poses={res['n_poses']}\n"
                f"  position ATE (m):  rmse={res['pos_rmse']:.4f}  mean={res['pos_mean']:.4f}  "
                f"max={res['pos_max']:.4f}\n"
                f"  rotation ATE (deg): rmse={res['rot_rmse']:.4f}  mean={res['rot_mean']:.4f}  "
                f"max={res['rot_max']:.4f}\n"
            )
    lines.append("\n=== alignment-independent inter-segment drift ===\n\n")

    for k in range(len(segments) - 1):
        prev, nxt = seg_results[k], seg_results[k + 1]
        if prev is None or nxt is None:
            continue
        gt_before, gt_after = prev["last_gt_row"], nxt["first_gt_row"]
        est_before, est_after = prev["last_est_row"], nxt["first_est_row"]
        R_before_est, t_before_est = se3_from_row(est_before)
        R_after_est, t_after_est = se3_from_row(est_after)
        R_before_gt, t_before_gt = se3_from_row(gt_before)
        R_after_gt, t_after_gt = se3_from_row(gt_after)
        drift_t, drift_r = se3_relative_error(R_before_est, t_before_est, R_after_est, t_after_est,
                                               R_before_gt, t_before_gt, R_after_gt, t_after_gt)
        gap_s = gt_after[0] - gt_before[0]
        lines.append(
            f"segment {k} -> segment {k + 1} accumulated drift (across {gap_s:.2f}s gap, "
            f"alignment-independent):\n"
            f"  translation={drift_t:.4f} m  rotation={drift_r:.4f} deg\n"
        )

    report = "".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"[eval] report written to {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_record = sub.add_parser("record")
    p_record.add_argument("--topic", default="/odometry_vio")
    p_record.add_argument("--out", type=Path, required=True)
    p_record.add_argument("--duration-s", type=float, default=0.0,
                           help="0 = record until Ctrl-C")
    p_record.set_defaults(func=cmd_record)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--gt", type=Path, required=True, help="mocap_data.txt")
    p_eval.add_argument("--est", type=Path, required=True, help="TUM file from `record`")
    p_eval.add_argument("--out", type=Path, default=None)
    p_eval.add_argument("--gap-factor", type=float, default=50.0,
                         help="split --gt into gap-free segments wherever consecutive gt "
                              "samples are more than max(gap_factor/f_gt, --min-gap-s) apart "
                              "(f_gt = nominal gt sampling rate, from the median inter-sample "
                              "period). Each segment gets its own ATE (Umeyama alignment refit "
                              "per segment) instead of one straight-line-interpolated whole-"
                              "trajectory ATE across a real mocap tracking dropout (e.g. "
                              "skate-hard has a ~32s one).")
    p_eval.add_argument("--min-gap-s", type=float, default=0.5,
                         help="absolute floor for the segment-split threshold above (see "
                              "--gap-factor) -- makes the default portable across TUM-VIE "
                              "sequences with different f_gt, instead of relying on a purely "
                              "relative factor tuned to one sequence's nominal rate. Real "
                              "mocap data has frequent brief ~100-135ms occlusion hiccups "
                              "during fast motion that are fine to interpolate across (confirmed "
                              "on skate-hard); a relative-only threshold small enough to avoid "
                              "flagging a low-f_gt sequence's routine hiccups could flag dozens "
                              "of those as false segment boundaries on a higher-f_gt one, "
                              "fragmenting the trajectory into unusable few-sample slivers with "
                              "numerically unstable per-segment alignment.")
    p_eval.add_argument("--max-gt-gap-s", type=float, default=0.5,
                         help="secondary, finer safety net within a segment: drop estimated "
                              "poses whose straddling ground-truth samples are still more "
                              "than this far apart (should rarely trigger once --gap-factor "
                              "has already split out the big dropouts)")
    p_eval.add_argument("--mocap-imu-calib", type=Path, default=None,
                         help="TUM-VIE official mocap-imu-calibrationB.json (T_imu_marker); "
                              "converts --gt from the mocap marker rig's pose to the IMU's "
                              "pose before scoring. Without it, rotation ATE is dominated by "
                              "the marker/IMU extrinsic instead of real attitude error.")
    p_eval.add_argument("--start-s", type=float, default=None,
                         help="only score poses from this many seconds after the first "
                              "estimated pose onward (relative, not absolute time). Useful "
                              "to isolate a well-tracked window from a badly-diverged tail "
                              "-- the Umeyama position alignment is refit on just this "
                              "window too, so a diverged tail elsewhere can't contaminate "
                              "the rotation/position alignment of the window you care about.")
    p_eval.add_argument("--end-s", type=float, default=None,
                         help="only score poses up to this many seconds after the first "
                              "estimated pose (relative). See --start-s.")
    p_eval.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
