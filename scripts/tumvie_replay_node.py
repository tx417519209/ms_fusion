#!/usr/bin/env python3
"""Replay a prepared TUM-VIE sequence (output of tumvie_prepare_dataset.py) as
ROS2 topics for ms_fusion's offline build (MSF_USE_AXERA_IVPS=OFF).

Publishes:
  - sensor_msgs/CompressedImage on --cam0-topic / --cam1-topic (format="jpeg",
    raw jpg bytes from disk, no re-encode)
  - sensor_msgs/Imu on --imu-topic
  - nav_msgs/Odometry on --odom-topic when odom_data.txt exists

Timestamps: the dataset's raw "us" column is used directly as ROS Time seconds
(t_s = us * 1e-6). All three sources (imu/images/mocap) share this clock
domain already, so no re-synchronization is needed.

Pacing: ms_fusion_node has no use_sim_time support -- its internal windows/
timeouts (static_detector.runtime_window_s, init_timeout_s, keyframe interval,
...) run on wall-clock (MonotonicRawClock). This node therefore publishes at
real wall-clock pace matching the dataset's original inter-sample intervals
(like `ros2 bag play`), not as a burst.

Usage:
  source /opt/ros/humble/setup.bash
  python3 tumvie_replay_node.py /tmp/tumvie_skate \\
      --imu-topic /tumvie/imu --cam0-topic /tumvie/cam0 --cam1-topic /tumvie/cam1
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Imu


@dataclass
class ImuSample:
    t: float
    wx: float
    wy: float
    wz: float
    ax: float
    ay: float
    az: float


@dataclass
class ImagePair:
    t: float
    left_path: Path
    right_path: Path


@dataclass
class OdomSample:
    t: float
    vx: float
    vy: float
    vz: float
    wx: float
    wy: float
    wz: float


def load_imu(path: Path) -> List[ImuSample]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    return [ImuSample(t=row[0] * 1e-6, wx=row[1], wy=row[2], wz=row[3],
                       ax=row[4], ay=row[5], az=row[6]) for row in data]


def load_odom(path: Path) -> List[OdomSample]:
    if not path.exists():
        return []
    data = np.loadtxt(path, comments="#", ndmin=2)
    return [OdomSample(t=row[0] * 1e-6, vx=row[1], vy=row[2], vz=row[3],
                       wx=row[4], wy=row[5], wz=row[6]) for row in data]


def load_image_pairs(dataset_dir: Path) -> List[ImagePair]:
    ts_left = np.loadtxt(dataset_dir / "image_timestamps_left.txt", comments="#")
    left_files = sorted((dataset_dir / "left").glob("*.jpg"))
    right_files = sorted((dataset_dir / "right").glob("*.jpg"))
    n = min(len(ts_left), len(left_files), len(right_files))
    if len(ts_left) != len(left_files) or len(left_files) != len(right_files):
        print(f"[replay][WARN] count mismatch: timestamps={len(ts_left)} "
              f"left={len(left_files)} right={len(right_files)}, using min={n}")
    pairs = []
    for i in range(n):
        pairs.append(ImagePair(t=float(ts_left[i]) * 1e-6, left_path=left_files[i],
                                right_path=right_files[i]))
    return pairs


def to_ros_time(node: Node, t_s: float):
    sec = int(t_s)
    nanosec = int(round((t_s - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg_time = rclpy.time.Time(seconds=sec, nanoseconds=nanosec)
    return msg_time.to_msg()


def make_compressed_image_from_bytes(node: Node, t_s: float, jpg_data: "array",
                                     frame_id: str) -> CompressedImage:
    msg = CompressedImage()
    msg.header.stamp = to_ros_time(node, t_s)
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    # jpg_data 必须已经是 typecode='B' 的 array.array（不能是裸 bytes）：
    # rosidl_generator_py 生成的 data setter 对 array.array 只做 O(1) 的
    # typecode 断言，对别的序列类型（含 bytes）会退化成逐元素 Python 遍历校验
    # 每个字节是不是 int、是不是在 [0,256) 内——这个 O(n) 校验是发布单张
    # JPEG（几万字节）时的主要 CPU 开销来源，挪到这里用 array.array 传入可以
    # 避开这条路径，不用靠 `python3 -O` 整体关掉断言。
    msg.data = jpg_data
    return msg


def prefetch_worker(image_pairs: List[ImagePair], out_queue: "queue.Queue[Tuple[array, array]]") -> None:
    # 按 image_pairs 原始顺序读盘，主循环按同样顺序消费——out_queue 的 maxsize
    # 就是滑窗宽度：读到窗口打满会阻塞在 put()，天然把预读控制在窗口范围内，
    # 不会无限跑在消费者前面占内存。array.array 转换也放在这个后台线程里做，
    # 不占主循环（发布时机）的时间。
    for pair in image_pairs:
        left_data = array("B", pair.left_path.read_bytes())
        right_data = array("B", pair.right_path.read_bytes())
        out_queue.put((left_data, right_data))


def make_imu(node: Node, sample: ImuSample, frame_id: str) -> Imu:
    msg = Imu()
    msg.header.stamp = to_ros_time(node, sample.t)
    msg.header.frame_id = frame_id
    msg.angular_velocity.x = sample.wx
    msg.angular_velocity.y = sample.wy
    msg.angular_velocity.z = sample.wz
    msg.linear_acceleration.x = sample.ax
    msg.linear_acceleration.y = sample.ay
    msg.linear_acceleration.z = sample.az
    return msg


def make_odom(node: Node, sample: OdomSample, frame_id: str) -> Odometry:
    msg = Odometry()
    msg.header.stamp = to_ros_time(node, sample.t)
    msg.header.frame_id = frame_id
    msg.child_frame_id = "base_link"
    msg.pose.pose.orientation.w = 1.0
    msg.twist.twist.linear.x = sample.vx
    msg.twist.twist.linear.y = sample.vy
    msg.twist.twist.linear.z = sample.vz
    msg.twist.twist.angular.x = sample.wx
    msg.twist.twist.angular.y = sample.wy
    msg.twist.twist.angular.z = sample.wz
    return msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path,
                     help="output dir of tumvie_prepare_dataset.py (has left/, right/, imu_data.txt, "
                          "image_timestamps_left.txt, ...)")
    ap.add_argument("--imu-topic", default="/tumvie/imu")
    ap.add_argument("--cam0-topic", default="/tumvie/cam0")
    ap.add_argument("--cam1-topic", default="/tumvie/cam1")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--start-delay-s", type=float, default=2.0,
                     help="wait this long after node init before publishing (lets ms_fusion_node "
                          "finish subscribing)")
    ap.add_argument("--prefetch-window", type=int, default=64,
                     help="background thread reads this many image pairs ahead into memory, so "
                          "the publish loop never blocks on disk I/O at the scheduled instant")
    args = ap.parse_args()

    imu_samples = load_imu(args.dataset_dir / "imu_data.txt")
    odom_samples = load_odom(args.dataset_dir / "odom_data.txt")
    image_pairs = load_image_pairs(args.dataset_dir)
    print(f"[replay] loaded {len(imu_samples)} imu samples, {len(odom_samples)} odom samples, "
          f"{len(image_pairs)} image pairs")

    events = ([("imu", s.t, s) for s in imu_samples]
              + [("odom", s.t, s) for s in odom_samples]
              + [("img", p.t, p) for p in image_pairs])
    events.sort(key=lambda e: e[1])
    t0_data = events[0][1]

    rclpy.init()
    node = Node("tumvie_replay")
    qos = QoSProfile(depth=500, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                      durability=QoSDurabilityPolicy.VOLATILE)
    pub_imu = node.create_publisher(Imu, args.imu_topic, qos)
    pub_cam0 = node.create_publisher(CompressedImage, args.cam0_topic, qos)
    pub_cam1 = node.create_publisher(CompressedImage, args.cam1_topic, qos)
    pub_odom = node.create_publisher(Odometry, args.odom_topic, qos) if odom_samples else None

    img_bytes_queue: "queue.Queue[Tuple[array, array]]" = queue.Queue(maxsize=args.prefetch_window)
    prefetch_thread = threading.Thread(target=prefetch_worker, args=(image_pairs, img_bytes_queue),
                                       daemon=True)
    prefetch_thread.start()

    print(f"[replay] waiting {args.start_delay_s}s for subscribers to attach...")
    time.sleep(args.start_delay_s)

    t_wall0 = time.monotonic()
    n_imu = n_odom = n_img = 0
    last_print = t_wall0
    cur_lag = 0.0
    max_lag = 0.0
    max_lag_t_data = None
    n_behind = 0
    for kind, t_data, payload in events:
        target_wall = t_wall0 + (t_data - t0_data) / args.speed
        sleep_s = target_wall - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
            cur_lag = 0.0
        else:
            cur_lag = -sleep_s
            n_behind += 1
            if cur_lag > max_lag:
                max_lag = cur_lag
                max_lag_t_data = t_data
        if kind == "imu":
            pub_imu.publish(make_imu(node, payload, "imu"))
            n_imu += 1
        elif kind == "odom":
            assert pub_odom is not None
            pub_odom.publish(make_odom(node, payload, "odom"))
            n_odom += 1
        else:
            left_data, right_data = img_bytes_queue.get()
            pub_cam0.publish(make_compressed_image_from_bytes(node, payload.t, left_data, "cam0"))
            pub_cam1.publish(make_compressed_image_from_bytes(node, payload.t, right_data, "cam1"))
            n_img += 1
        now = time.monotonic()
        if now - last_print >= 0.2:
            print(f"\r[replay] {now - t_wall0:6.1f}s imu={n_imu}/{len(imu_samples)} "
                  f"odom={n_odom}/{len(odom_samples)} img={n_img}/{len(image_pairs)} "
                  f"lag={cur_lag:5.2f}s", end="", flush=True)
            last_print = now

    n_total = len(imu_samples) + len(odom_samples) + len(image_pairs)
    max_lag_info = (f"max_lag={max_lag:.2f}s@t_data={max_lag_t_data:.3f} "
                     if max_lag_t_data is not None else "max_lag=0.00s ")
    print(f"\r[replay] done: {time.monotonic() - t_wall0:.1f}s "
          f"imu={n_imu} odom={n_odom} img_pairs={n_img} {max_lag_info}"
          f"n_behind={n_behind}/{n_total}" + " " * 10)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
