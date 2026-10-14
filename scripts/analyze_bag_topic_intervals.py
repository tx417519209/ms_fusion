#!/usr/bin/env python3
"""
Analyze per-topic timestamp intervals in a ROS 2 rosbag2 recording.

For each topic, computes mean and standard deviation of consecutive
message intervals (seconds), prints a summary table, and plots the results.

Requires Python 3.10 with ROS 2 rosbag2 (e.g. source /opt/ros/humble/setup.bash)
for non-sqlite3 bags. sqlite3 bags can be read with stdlib sqlite3 only.

Example:
  python3.10 analyze_bag_topic_intervals.py \\
      /path/to/rosbag2_dir
  python3.10 analyze_bag_topic_intervals.py bag_path --save plot.png
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required: pip install matplotlib (or apt install python3-matplotlib)"
    ) from exc


@dataclass
class TopicStats:
    name: str
    count: int
    mean_s: float
    std_s: float
    mean_hz: float

    @property
    def mean_ms(self) -> float:
        return self.mean_s * 1e3

    @property
    def std_ms(self) -> float:
        return self.std_s * 1e3


def _interval_stats(timestamps_ns: np.ndarray) -> Tuple[int, float, float]:
    if timestamps_ns.size < 2:
        return int(timestamps_ns.size), float("nan"), float("nan")
    ts = np.sort(timestamps_ns.astype(np.float64))
    dt = np.diff(ts) * 1e-9
    return int(ts.size), float(np.mean(dt)), float(np.std(dt))


def _load_metadata(bag_dir: Path) -> Tuple[str, List[Path]]:
    """Return (storage_id, list of db3/mcap files) from metadata.yaml if present."""
    meta_path = bag_dir / "metadata.yaml"
    if not meta_path.is_file():
        return "sqlite3", sorted(bag_dir.glob("*.db3"))

    try:
        import yaml
    except ImportError:
        return "sqlite3", sorted(bag_dir.glob("*.db3"))

    with meta_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    info = meta.get("rosbag2_bagfile_information", meta)
    storage_id = info.get("storage_identifier", "sqlite3")
    rel_paths = info.get("relative_file_paths") or []
    files = info.get("files") or []

    paths: List[Path] = []
    if rel_paths:
        paths = [bag_dir / p for p in rel_paths]
    elif files:
        for entry in files:
            p = entry.get("path") if isinstance(entry, dict) else entry
            if p:
                paths.append(bag_dir / p)
    if not paths:
        paths = sorted(bag_dir.glob("*.db3")) or sorted(bag_dir.glob("*.mcap"))
    return storage_id, paths


def read_timestamps_sqlite(db_paths: List[Path]) -> Dict[str, List[int]]:
    per_topic: Dict[str, List[int]] = {}
    for db_path in db_paths:
        if not db_path.is_file():
            raise FileNotFoundError(f"Bag database not found: {db_path}")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                SELECT topics.name, messages.timestamp
                FROM messages
                JOIN topics ON messages.topic_id = topics.id
                ORDER BY topics.name, messages.timestamp
                """
            )
            for name, ts in cur:
                per_topic.setdefault(name, []).append(int(ts))
        finally:
            conn.close()
    return per_topic


def read_timestamps_rosbag2(bag_uri: str, storage_id: str) -> Dict[str, List[int]]:
    try:
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    except ImportError as exc:
        raise SystemExit(
            "rosbag2_py not available. Use Python 3.10 and source your ROS 2 setup, e.g.\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  python3.10 analyze_bag_topic_intervals.py <bag_path>"
        ) from exc

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_uri, storage_id=storage_id),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    per_topic: Dict[str, List[int]] = {}
    while reader.has_next():
        topic, _data, t = reader.read_next()
        per_topic.setdefault(topic, []).append(int(t))
    return per_topic


def resolve_bag(path: Path) -> Tuple[str, str, List[Path]]:
    """Return (bag_uri for rosbag2, storage_id, sqlite file paths)."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix == ".db3":
            return str(path.parent), "sqlite3", [path]
        if path.suffix == ".mcap":
            return str(path.parent), "mcap", [path]
        raise ValueError(f"Unsupported bag file type: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"Bag path not found: {path}")

    storage_id, db_paths = _load_metadata(path)
    return str(path), storage_id, db_paths


def compute_all_stats(per_topic_ts: Dict[str, List[int]]) -> List[TopicStats]:
    stats: List[TopicStats] = []
    for name in sorted(per_topic_ts):
        ts = np.asarray(per_topic_ts[name], dtype=np.int64)
        count, mean_s, std_s = _interval_stats(ts)
        mean_hz = 1.0 / mean_s if mean_s > 0 and np.isfinite(mean_s) else float("nan")
        stats.append(TopicStats(name=name, count=count, mean_s=mean_s, std_s=std_s, mean_hz=mean_hz))
    return stats


def print_table(stats: List[TopicStats]) -> None:
    print(f"{'topic':<48} {'count':>8} {'mean_dt_ms':>12} {'std_dt_ms':>12} {'mean_hz':>10}")
    print("-" * 94)
    for s in stats:
        mean_ms = f"{s.mean_ms:.3f}" if np.isfinite(s.mean_ms) else "n/a"
        std_ms = f"{s.std_ms:.3f}" if np.isfinite(s.std_ms) else "n/a"
        mean_hz = f"{s.mean_hz:.2f}" if np.isfinite(s.mean_hz) else "n/a"
        print(f"{s.name:<48} {s.count:>8} {mean_ms:>12} {std_ms:>12} {mean_hz:>10}")


def plot_stats(stats: List[TopicStats], bag_label: str, save_path: Optional[Path], show: bool) -> None:
    valid = [s for s in stats if s.count >= 2 and np.isfinite(s.mean_s)]
    if not valid:
        print("No topic with enough messages to plot.", file=sys.stderr)
        return

    labels = [s.name for s in valid]
    short_labels = [lbl if len(lbl) <= 36 else "…" + lbl[-35:] for lbl in labels]
    mean_ms = [s.mean_ms for s in valid]
    std_ms = [s.std_ms for s in valid]
    x = np.arange(len(valid))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(valid) * 0.55), 5))
    fig.suptitle(f"Topic interval stats\n{bag_label}", fontsize=11)

    ax0 = axes[0]
    bars0 = ax0.bar(x, mean_ms, color="steelblue", alpha=0.85)
    ax0.errorbar(x, mean_ms, yerr=std_ms, fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax0.set_ylabel("interval mean ± std (ms)")
    ax0.set_title("Mean interval (error bar = std)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax0.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars0, mean_ms):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax1 = axes[1]
    ax1.bar(x, std_ms, color="coral", alpha=0.85)
    ax1.set_ylabel("interval std (ms)")
    ax1.set_title("Std of intervals (jitter)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-topic timestamp interval mean/std for a ROS 2 bag and plot."
    )
    p.add_argument("bag_path", type=Path, help="Path to rosbag2 directory or .db3 file")
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to this path (e.g. intervals.png)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot window",
    )
    p.add_argument(
        "--force-rosbag2",
        action="store_true",
        help="Read via rosbag2_py instead of direct sqlite3",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bag_uri, storage_id, db_paths = resolve_bag(args.bag_path)

    use_sqlite = (
        not args.force_rosbag2
        and storage_id == "sqlite3"
        and db_paths
        and all(p.suffix == ".db3" for p in db_paths)
    )

    if use_sqlite:
        print(f"Reading timestamps from sqlite3 ({len(db_paths)} file(s))...")
        per_topic = read_timestamps_sqlite(db_paths)
    else:
        print(f"Reading timestamps via rosbag2 (storage={storage_id})...")
        per_topic = read_timestamps_rosbag2(bag_uri, storage_id)

    stats = compute_all_stats(per_topic)
    print_table(stats)

    show = not args.no_show and args.save is None
    if args.save is not None or not args.no_show:
        plot_stats(stats, bag_label=args.bag_path.name, save_path=args.save, show=show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
