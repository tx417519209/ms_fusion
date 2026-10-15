#!/usr/bin/env python3
"""Offline report over one or more run_tumvi_benchmark.py campaign directories.

Never contacts the board. For every run it recomputes, from the stored trajectory + the prepared GT:
  ate_global  one SE3 (Umeyama, no scale) fitted on ALL valid GT poses of the sequence, RMSE over them
              (the single-alignment protocol of ORB-SLAM3's evaluate_ate_scale.py; keeps cross-segment
              drift but lets the fit absorb part of it)
  ate_ref     SE3 fitted on the first evaluable GT segment only (status.json ate_m; pure cumulative drift)
  ate_own     each GT segment aligned independently (local shape quality only; hides drift)
  rpe_1s      translation RPE over a 1 s interval, error of the displacement vector after the global
              alignment rotation, RMSE over all pairs inside one GT-gap-free span
and from the board log: front-end fps, front-end e2e / back-end total P95 (over the 1 s timing windows),
back-end optimisation P95, initialisation success, queue drops.

Usage: tumvi_report.py --cache dataset/tumvi_benchmark --out report_dir CAMPAIGN_DIR [CAMPAIGN_DIR...]
Later campaigns override earlier ones per sequence only if listed with --label; by default each
campaign is reported separately (one table per campaign).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tumvie_eval_trajectory as ev  # noqa: E402

ORDER = ["room1", "room2", "room3", "room4", "room5", "room6",
         "corridor1", "corridor2", "corridor3", "corridor4", "corridor5",
         "magistrale1", "magistrale2", "magistrale3", "magistrale4", "magistrale5", "magistrale6",
         "outdoors1", "outdoors2", "outdoors3", "outdoors4", "outdoors5", "outdoors6", "outdoors7", "outdoors8",
         "slides1", "slides2", "slides3"]


def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def percentile(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def gt_info(data: Path):
    gt = ev.load_mocap(data / "mocap_data.txt")
    segs, _, _ = ev.detect_gt_segments(gt)
    return gt, segs


def trajectory_metrics(est, gt, segs):
    """est: (N,8+) TUM; gt: (M,8) seconds. Returns dict with global ATE, own ATE, RPE and GT coverage."""
    out = {}
    pts_e, pts_g, span_id = [], [], []
    own = []
    for k, (i0, i1) in enumerate(segs):
        seg = gt[i0:i1]
        t_lo, t_hi = max(est[0, 0], seg[0, 0]), min(est[-1, 0], seg[-1, 0])
        if t_hi - t_lo < 0.1:
            continue
        sel = (est[:, 0] >= t_lo) & (est[:, 0] <= t_hi)
        e = est[sel]
        g, ok = ev.interpolate_gt(seg, e[:, 0])
        e, g = e[ok], g[ok]
        if len(e) < 10:
            continue
        pts_e.append(e)
        pts_g.append(g)
        span_id.append(np.full(len(e), k))
        R, t = ev.umeyama_se3(e[:, 1:4], g[:, 1:4])
        own.append((len(e), rmse(np.linalg.norm((e[:, 1:4] @ R.T + t) - g[:, 1:4], axis=1))))
    if not pts_e:
        return None
    E, G, S = np.vstack(pts_e), np.vstack(pts_g), np.concatenate(span_id)
    R, t = ev.umeyama_se3(E[:, 1:4], G[:, 1:4])
    err = np.linalg.norm((E[:, 1:4] @ R.T + t) - G[:, 1:4], axis=1)
    out["ate_global"] = rmse(err)
    out["ate_own"] = float(np.sqrt(sum(n * v * v for n, v in own) / sum(n for n, _ in own)))
    # translation RPE, 1 s displacement, inside one GT segment
    errs = []
    for k in np.unique(S):
        m = S == k
        te, pe, pg = E[m, 0], E[m, 1:4] @ R.T, G[m, 1:4]
        j = np.searchsorted(te, te + 1.0)
        valid = j < len(te)
        valid[valid] &= np.abs(te[j[valid]] - te[valid] - 1.0) < 0.02
        i, jj = np.nonzero(valid)[0], j[valid]
        errs.append(np.linalg.norm((pe[jj] - pe[i]) - (pg[jj] - pg[i]), axis=1))
    errs = np.concatenate(errs) if errs else np.array([])
    out["rpe_1s"] = rmse(errs)
    seq_dur = gt[-1, 0] - gt[0, 0]
    out["gt_covered_s"] = float(sum(gt[i1 - 1, 0] - gt[i0, 0] for i0, i1 in segs))
    out["gt_span_s"] = float(seq_dur)
    out["gt_segments"] = len(segs)
    return out


def log_metrics(run: Path):
    logs = [p for p in run.glob("data_*.log") if not p.name.endswith(("_launch.log", "_replay.log", "_rec_vio.log"))]
    if len(logs) != 1:
        return {}
    text = logs[0].read_text(errors="replace")
    fps = [float(x) for x in re.findall(r"frontend\s+SCORE.*?fps=([\d.]+)", text)][1:]
    fe_e2e = [float(x) for x in re.findall(r"frontend\s+SCORE\s+e2e=([\d.]+)ms", text)][1:]
    be_tot = [float(x) for x in re.findall(r"backend\s+SCORE\s+total=([\d.]+)ms", text)][1:]
    be_opt = [float(x) for x in re.findall(r"backend\s+SCORE\s+total=[\d.]+ms.*?opt=([\d.]+)ms", text)][1:]
    drops = sum(float(x) for x in re.findall(r"frontend\s+queue\s+.*?drop=([\d.]+)", text))
    drops += sum(float(x) for x in re.findall(r"backend\s+queue\s+.*?dropped=([\d.]+)", text))
    return {"init_ok": int("Initialization Optimization successfual" in text),
            "fe_fps": statistics.median(fps) if fps else float("nan"),
            "fe_e2e_p95_ms": percentile(fe_e2e, 95), "be_total_p95_ms": percentile(be_tot, 95),
            "be_opt_p95_ms": percentile(be_opt, 95), "queue_drops": drops}


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return statistics.median(v) if v else None


def fmt(v, nd=2):
    return "N/A" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", nargs="+", type=Path)
    ap.add_argument("--cache", type=Path, required=True, help="benchmark cache holding prepared/<seq>/mocap_data.txt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gts = {}
    for campaign in args.campaign:
        rows = []
        for status_path in sorted(campaign.glob("*/run_*/status.json")):
            run, seq = status_path.parent, status_path.parent.parent.name
            st = json.loads(status_path.read_text())
            row = {"campaign": campaign.name, "sequence": seq, "repeat": int(run.name.split("_")[1]),
                   "status": st["status"], "ate_ref": st.get("ate_m"), "drift_m": st.get("drift_m"),
                   "coverage": st.get("coverage"), "path_ratio": st.get("path_ratio")}
            row.update(log_metrics(run))
            data = args.cache / "prepared" / seq
            if seq not in gts and (data / "mocap_data.txt").is_file():
                gts[seq] = gt_info(data)
            tum = list(run.glob("*_vio_est.tum"))
            if seq in gts and tum and st["status"] in ("evaluated", "diverged"):
                try:
                    m = trajectory_metrics(np.loadtxt(tum[0], comments="#", ndmin=2), *gts[seq])
                except Exception as exc:  # keep the row, record why
                    m, row["error"] = None, str(exc)
                if m:
                    row.update(m)
            rows.append(row)
        fields = ["campaign", "sequence", "repeat", "status", "ate_global", "ate_ref", "ate_own", "rpe_1s",
                  "drift_m", "coverage", "path_ratio", "init_ok", "fe_fps", "fe_e2e_p95_ms", "be_total_p95_ms",
                  "be_opt_p95_ms", "queue_drops", "gt_segments", "gt_covered_s", "gt_span_s", "error"]
        with (args.out / f"{campaign.name}_runs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        lines = ["| 序列 | 完成/发散 | ATE Global-SE3 (m) | ATE Reference-SE3 (m) | ATE Own-SE3 (m) | RPE 1s (m) | 初始化失败 | 回环 | 前端 Hz | 前端 e2e P95 (ms) | 后端 P95 (ms) | 后端优化 P95 (ms) | GT 覆盖 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        seqs = [s for s in ORDER if any(r["sequence"] == s for r in rows)]
        for s in seqs:
            g = [r for r in rows if r["sequence"] == s]
            ev_rows = [r for r in g if r["status"] == "evaluated"]
            div = sum(r["status"] == "diverged" for r in g)
            r0 = [r for r in g if r.get("gt_span_s")]
            cover = (f"{r0[0]['gt_segments']} 段 / {r0[0]['gt_covered_s']:.0f} s（序列 GT 跨度 {r0[0]['gt_span_s']:.0f} s）"
                     if r0 else "N/A")
            lines.append(
                f"| {s} | {len(ev_rows)}/{len(g)}, 发散 {div} | {fmt(med(ev_rows, 'ate_global'))} | {fmt(med(ev_rows, 'ate_ref'))} | "
                f"{fmt(med(ev_rows, 'ate_own'))} | {fmt(med(ev_rows, 'rpe_1s'), 3)} | "
                f"{sum(1 - r.get('init_ok', 1) for r in g)} | 关闭 | {fmt(med(g, 'fe_fps'), 1)} | {fmt(med(g, 'fe_e2e_p95_ms'), 0)} | "
                f"{fmt(med(g, 'be_total_p95_ms'), 0)} | {fmt(med(g, 'be_opt_p95_ms'), 0)} | {cover} |")
        (args.out / f"{campaign.name}_table.md").write_text("\n".join(lines) + "\n")
        print(f"{campaign.name}: {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
