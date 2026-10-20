#!/usr/bin/env python3
"""Trajectory-vs-GT plots for selected sequences of a run_tumvi_benchmark.py campaign.

Per sequence the run with the median global-SE3 ATE is drawn (first run if none was evaluated), est
restricted to the GT-covered spans and aligned with one SE3 over all of them (XY top view, equal axes).
Usage: tumvi_plot.py --cache dataset/tumvi_benchmark --out fig.png CAMPAIGN_DIR seq [seq...]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tumvi_report as rep  # noqa: E402
import tumvie_eval_trajectory as ev  # noqa: E402


def aligned(est, gt, segs):
    pe, pg, ids = [], [], []
    for k, (i0, i1) in enumerate(segs):
        seg = gt[i0:i1]
        sel = (est[:, 0] >= seg[0, 0]) & (est[:, 0] <= seg[-1, 0])
        e = est[sel]
        if len(e) < 10:
            continue
        g, ok = ev.interpolate_gt(seg, e[:, 0])
        pe.append(e[ok, 1:4]); pg.append(g[ok, 1:4]); ids.append(np.full(ok.sum(), k))
    E, G, I = np.vstack(pe), np.vstack(pg), np.concatenate(ids)
    R, t = ev.umeyama_se3(E, G)
    return E @ R.T + t, G, I, rep.rmse(np.linalg.norm(E @ R.T + t - G, axis=1))


def full_reference(est, gt, segs):
    """Whole estimate, aligned only on the first evaluable GT segment (shows cumulative drift)."""
    for i0, i1 in segs:
        seg = gt[i0:i1]
        e = est[(est[:, 0] >= seg[0, 0]) & (est[:, 0] <= seg[-1, 0])]
        if len(e) >= 10:
            g, ok = ev.interpolate_gt(seg, e[:, 0])
            R, t = ev.umeyama_se3(e[ok, 1:4], g[ok, 1:4])
            return est[:, 1:4] @ R.T + t
    raise RuntimeError("no evaluable GT segment")


def error_curves(est, gt, segs):
    """Per-pose position (m) and rotation (deg) error vs time, one reference gauge fitted on the first
    evaluable GT segment and reused for every segment (same gauge as ATE(ref)/drift in the evaluator)."""
    gauge, out = None, []
    for k, (i0, i1) in enumerate(segs):
        seg = gt[i0:i1]
        t_lo, t_hi = max(est[0, 0], seg[0, 0]), min(est[-1, 0], seg[-1, 0])
        if t_hi - t_lo < 1.0:
            continue
        e = est[(est[:, 0] >= t_lo) & (est[:, 0] <= t_hi)]
        g, ok = ev.interpolate_gt(seg, e[:, 0])
        e, g = e[ok], g[ok]
        if len(e) < 10:
            continue
        R_e = ev.quat_to_R(e[:, 4], e[:, 5], e[:, 6], e[:, 7])
        R_g = ev.quat_to_R(g[:, 4], g[:, 5], g[:, 6], g[:, 7])
        if gauge is None:
            R_a, t_a = ev.umeyama_se3(e[:, 1:4], g[:, 1:4])
            R_w = ev.solve_wahba_rotation(R_e, R_g)
            R_b = ev.solve_body_rotation_offset(np.einsum("ij,njk->nik", R_w, R_e), R_g)
            gauge = (R_a, t_a, R_w, R_b)
        R_a, t_a, R_w, R_b = gauge
        pos = np.linalg.norm(e[:, 1:4] @ R_a.T + t_a - g[:, 1:4], axis=1)
        R_fin = np.einsum("nij,jk->nik", np.einsum("ij,njk->nik", R_w, R_e), R_b)
        rot = ev.rotation_angle_deg(np.einsum("nij,nkj->nik", R_fin, R_g))
        out.append((e[:, 0], pos, rot))
    if not out:
        raise RuntimeError("no evaluable GT segment")
    return out


def plot_errors(a, picks):
    """picks: [(seq, est, gt, segs, status)] -> two rows (position / rotation error) per sequence column."""
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6), squeeze=False, sharex="col")
    for col, (seq, est, gt, segs, st) in enumerate(picks):
        t0 = est[0, 0]
        for k, (t, pos, rot) in enumerate(error_curves(est, gt, segs)):
            axes[0, col].plot(t - t0, pos, "r-", lw=1.0)
            axes[1, col].plot(t - t0, rot, "b-", lw=1.0)
        d = (f"  drift={st['drift_m']:.2f} m / {st['drift_deg']:.1f}°"
             if st.get("drift_m") is not None and st.get("drift_deg") is not None else "")
        axes[0, col].set_title(f"{seq}{d}", fontsize=10)
        axes[0, col].set_ylabel("position error (m)")
        axes[1, col].set_ylabel("rotation error (deg)")
        axes[1, col].set_xlabel("time since start (s)")
        for ax in axes[:, col]:
            ax.grid(alpha=.3)
    fig.tight_layout()
    out = a.out.with_name(a.out.stem + "_errors" + a.out.suffix)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("sequences", nargs="+")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full", action="store_true",
                    help="draw the whole estimate aligned on the first GT segment (drift view) instead of GT-only spans")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--errors", action="store_true",
                    help="also write <out>_errors.png: position and rotation error vs time (reference gauge)")
    a = ap.parse_args()
    n = len(a.sequences)
    rows = (n + a.cols - 1) // a.cols
    fig, axes = plt.subplots(rows, a.cols, figsize=(5 * a.cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    picks = []
    for ax, seq in zip(axes.flat, a.sequences):
        gt, segs = rep.gt_info(a.cache / "prepared" / seq)
        cand = []
        for run in sorted((a.campaign / seq).glob("run_*")):
            tum = list(run.glob("*_vio_est.tum"))
            if tum:
                cand.append((run, np.loadtxt(tum[0], comments="#", ndmin=2)))
        res = []
        for run, est in cand:
            try:
                res.append((aligned(est, gt, segs), run))
            except Exception:
                pass
        res.sort(key=lambda x: x[0][3])
        (E, G, I, ate), run = res[len(res) // 2]
        st = json.loads((run / "status.json").read_text())
        status = st["status"]
        picks.append((seq, dict((str(r), e) for r, e in cand)[str(run)], gt, segs, st))
        drift = ""
        if st.get("drift_m") is not None and st.get("drift_deg") is not None:
            drift = f"\ndrift={st['drift_m']:.2f} m / {st['drift_deg']:.1f}°"
        if a.full:
            est = dict((str(r), e) for r, e in cand)[str(run)]
            P = full_reference(est, gt, segs)
            ax.plot(P[:, 0], P[:, 1], "r-", lw=0.8, label="ms_fusion (full)")
            for k, (i0, i1) in enumerate(segs):
                ax.plot(gt[i0:i1, 1], gt[i0:i1, 2], "k-", lw=2.2, label="GT segments" if k == 0 else None)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{seq}  ATE(ref)={json.loads((run / 'status.json').read_text())['ate_m']:.1f} m" + drift, fontsize=10)
            ax.grid(alpha=.3); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            continue
        for k in np.unique(I):
            m = I == k
            ax.plot(G[m, 0], G[m, 1], "k-", lw=1.6, label="GT" if k == I[0] else None)
            ax.plot(E[m, 0], E[m, 1], "r-", lw=0.9, label="ms_fusion" if k == I[0] else None)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{seq}  ATE(global)={ate:.2f} m" + ("  [diverged]" if status == "diverged" else "") + drift, fontsize=10)
        ax.grid(alpha=.3); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    if a.errors:
        plot_errors(a, picks)


if __name__ == "__main__":
    main()
