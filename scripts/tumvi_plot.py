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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("sequences", nargs="+")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full", action="store_true",
                    help="draw the whole estimate aligned on the first GT segment (drift view) instead of GT-only spans")
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()
    n = len(a.sequences)
    rows = (n + a.cols - 1) // a.cols
    fig, axes = plt.subplots(rows, a.cols, figsize=(5 * a.cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
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
        status = json.loads((run / "status.json").read_text())["status"]
        if a.full:
            est = dict((str(r), e) for r, e in cand)[str(run)]
            P = full_reference(est, gt, segs)
            ax.plot(P[:, 0], P[:, 1], "r-", lw=0.8, label="ms_fusion (full)")
            for k, (i0, i1) in enumerate(segs):
                ax.plot(gt[i0:i1, 1], gt[i0:i1, 2], "k-", lw=2.2, label="GT segments" if k == 0 else None)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{seq}  ATE(ref)={json.loads((run / 'status.json').read_text())['ate_m']:.1f} m", fontsize=10)
            ax.grid(alpha=.3); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            continue
        for k in np.unique(I):
            m = I == k
            ax.plot(G[m, 0], G[m, 1], "k-", lw=1.6, label="GT" if k == I[0] else None)
            ax.plot(E[m, 0], E[m, 1], "r-", lw=0.9, label="ms_fusion" if k == I[0] else None)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{seq}  ATE(global)={ate:.2f} m" + ("  [diverged]" if status == "diverged" else ""), fontsize=10)
        ax.grid(alpha=.3); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)


if __name__ == "__main__":
    main()
