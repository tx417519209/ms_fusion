# TUM VI `room1`：本地准备、板端回放

本流程测的是纯双目 VIO。`room1` 的 MoCap 真值覆盖全程，适合作为首次基线；
轮速、点云和回环均关闭。不要把它与仓库已有的 TUM-VIE `skate-hard`
流程或其标定混用。

## 本地准备

官方包和 512 标定已分别放在：

```bash
dataset/tumvi_room1/dataset-room1_512_16.tar
calib_files/tumvi_room1/pinhole-equi-512/camchain-imucam-imucalib.yaml
```

执行预处理（不编译，也不启动 ROS 节点）：

```bash
python3 scripts/tumvi_prepare_dataset.py \
  dataset/tumvi_room1/dataset-room1_512_16.tar \
  --calib calib_files/tumvi_room1/pinhole-equi-512/camchain-imucam-imucalib.yaml \
  --out tmp/tumvi_room1 \
  --profile-out calib_files/tumvi_room1/msf_robot.yaml
```

`T_virtual_imu_from_real_imu` 不从 room1 的任意静止窗口反解。官方校正版
数据已经完成 IMU scale/axis correction，并提供与相机处于同一 IMU frame 的
`T_cam_imu`；脚本用 cam0 的官方外参将 OpenCV camera（right/down/forward）
固定映射到 ms_fusion 的 virtual FRD（front/right/down）。该矩阵对整副刚性
rig 恒定，不依赖 room1 是否在起止阶段静止，也不需要下载 4.77 GB 的
`imu-static`。脚本会打印首 2 秒比力在该固定变换前后的值，仅作方向自检，
不把这段运动数据用于标定。

产物 `tmp/tumvi_room1/` 是板端回放输入：`left/`、`right/` 为 640×384 JPEG，
`imu_data.txt` 和 `image_timestamps_left.txt` 的时间单位均为微秒，
`mocap_data.txt` 是已经转换为 IMU 位姿约定的全程真值。脚本从官方 16-bit PNG
确定性映射至 8-bit，再做 equidistant fisheye 到 pinhole 的去畸变；生成 profile
会从官方 Kalibr `T_cam_imu` 取逆得到项目所需的 `T_imu_cam`。

预处理结束应至少检查：左右图与时间戳数相等、IMU 行数非零，以及脚本打印的
首 2 秒静止比力模长接近 9.81。

## 板端执行

将安装树、预处理数据、profile 和现有回放/评估脚本同步到实际板端工程目录。
profile 的相对模型路径依赖其位于 `install/ms_fusion/share/ms_fusion/config/`，
因此板端必须复制到该目录再启动：

```bash
cp calib_files/tumvi_room1/msf_robot.yaml \
  install/ms_fusion/share/ms_fusion/config/tumvi_room1.yaml

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ms_fusion msf_robot.launch.py \
  config_file:=$(pwd)/install/ms_fusion/share/ms_fusion/config/tumvi_room1.yaml \
  topic_cam0:=/tumvi/cam0 topic_cam1:=/tumvi/cam1 topic_imu:=/tumvi/imu \
  image_type:=compressed
```

另一终端录制，并在第三终端按原始时基回放：

```bash
python3 scripts/tumvie_eval_trajectory.py record \
  --topic /odometry_vio --out tmp/tumvi_room1_est.tum

python3 scripts/tumvie_replay_node.py tmp/tumvi_room1 \
  --imu-topic /tumvi/imu --cam0-topic /tumvi/cam0 --cam1-topic /tumvi/cam1
```

回放结束后停止录制，评估：

```bash
python3 scripts/tumvie_eval_trajectory.py eval \
  --gt tmp/tumvi_room1/mocap_data.txt --est tmp/tumvi_room1_est.tum \
  --out tmp/tumvi_room1_ate.txt
```

这项 ATE 是对齐后的纯 VIO 轨迹误差；它不验证板端实时性。另行保存启动日志与
回放期间的 tracking/queue timing，才能判断设备端是否跟得上 20 Hz 图像和 200 Hz IMU。
