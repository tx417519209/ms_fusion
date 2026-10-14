# MS Fusion

面向 Axera AX650 NPU 板端实时运行的 **双目 + IMU 视觉惯性里程计（VIO）**，ROS 2（Humble）工作空间，
可选轮速里程计、重定位和回环（位姿图）。

## 特点

- **学习型前端**：XFeat / SuperPoint 特征 + LightGlue 描述子匹配，推理跑在 AX650 NPU 上；
  左目跨帧（temporal）和左右目同帧（stereo）两路匹配，之后叠加运动范围门控、对极约束 / RANSAC 过滤外点。
- **IMU 滑窗优化**：IMU 预积分 + Ceres 滑窗（边缘化）联合优化，关键帧按间隔、视差、匹配数自适应触发。
- **静止检测与零速更新**：基于 IMU 统计量的静止判定，支持 ZUPT；初始化支持静止对准。
- **可选外部观测**：轮速里程计（`odom.enable`）、基于检测模型的场地关键点重定位（`reloc.enable`）、
  匹配点云输出（`matched_cloud.enable`）。
- **回环**：独立的 `ms_fusion_loop` 节点，DBoW3 词典检索 + Mutual NN / PnP 几何验证 + 4DoF 位姿图，
  对 VIO 输出做平滑校正（`/odometry`）；VIO 原始输出为 `/odometry_vio`。
- **配置驱动**：传感器外参、内参、噪声、前端/后端参数全部在一份 `msf_robot.yaml` 里，
  不同设备/数据集只换这一个文件。
- **离线回放评测**：同一套代码可编译成「订阅 ROS 话题图像」的离线版本，用于回放公开数据集（如 TUM VI）并计算 ATE。

与 SuperVINS 的前后端对比见 [`docs/SuperVINS_vs_ms_fusion_analysis_claude.md`](docs/SuperVINS_vs_ms_fusion_analysis_claude.md)。

## 为什么选择这个项目

适合你的情况：

- **目标平台就是 Axera AX650N 板**（或同类 Axera SoC），需要在板端实时跑学习型特征 VIO，
  而不是在 PC 上跑离线算法。特征提取、匹配、取图都已经接好 NPU / IVPS，并做了零拷贝取图和 NPU 亲和性绑定。
- 需要**学习型特征（XFeat / SuperPoint + LightGlue）**带来的弱纹理、光照变化鲁棒性，同时仍保持 IMU 紧耦合滑窗优化。
- 需要在 VIO 之上叠加**轮速、重定位、回环**这些工程化能力，且都能用同一份 yaml 开关。
- 想要一套**可复现的公开数据集评测流程**：TUM VI 的下载、预处理、回放、ATE 评估脚本齐全，
  `docs/` 里有 room1、outdoors4（回环）的板端评测记录。
- 需要对前后端细节做实验：参数全部配置化，评测脚本一键产出日志和轨迹，便于做对照。

不适合你的情况见下一节。

## 局限与不足

请在决定使用前确认：

1. **强依赖 Axera AX650N NPU 与其 SDK，不能在普通 PC 上直接运行。**
   - 前端只支持 `xfeat` / `superpoint` 两种检测器，FAST/KLT 路径已移除；它们的推理和 LightGlue 匹配只有 Axera NPU 实现
     （`.axmodel` 模型，运行库来自板端 `/soc/lib/libax_*.so`）。
   - 把 `MSF_USE_AXERA_SUPERPOINT` 设为 `OFF` 虽然能通过编译，但运行时检测器初始化会直接失败（`BUILD_DISABLED`），无法产生特征。
   - `scripts/run_colcon_build_onnx.sh`、`install_onnxruntime*.sh` 是早期 ONNX Runtime 后端的遗留脚本，
     当前 CMake 已不再提供对应的 `MSF_INFER_BACKEND` 选项，**不要依赖它们**。
   - 因此「离线回放评测」也要在 AX650N 板端（或能访问 NPU 的环境）上执行，离线构建只是把输入从 IVPS 换成了 ROS 话题。
2. **仅支持双目 + IMU**，不支持单目、RGB-D；相机模型仅 `pinhole-radtan4/5`，鱼眼数据需要先去畸变。
3. **使用门槛偏高**：板端交叉编译 / qemu 下编译耗时长（脚本里限制了并行度防止内存不足）；
   重定位依赖特定的场地检测模型和场地地图，换场景基本用不上；回环词典需要用目标场景图像自行训练。
4. **模型文件范围有限**：`third_party_axera/asset/models/` 只带运行默认流程所需的 3 个 `.axmodel`
   （XFeat top256、LightGlue top256、重定位检测模型）。SuperPoint 等其他模型未包含，
   要用 `tracking.extractor.feature_detector: superpoint` 时需自行准备模型并配置各 yaml 中的 `model_path`；
   配置里残留的 `superpoint_*.axmodel` 路径只在选择该检测器时才会被加载。
5. **文档与评测覆盖有限**：目前有 TUM VI 的 room1 / outdoors4 等序列的评测记录，没有覆盖更多数据集的系统性对比；
   单元测试很少（仅 `test_upright_p2_solver`），主要靠数据集回放验证。
6. **配置项很多、部分参数针对特定设备调过**：换设备或数据集时需要重新标定并按场景调参，
   TUM VI 各序列的 yaml 也是逐序列调出来的，不是一套参数通用。

## 快速上手（clone 之后怎么跑起来）

前提：一块能访问 NPU 的 AX650N 板（或板端容器），已装 ROS 2 Humble 与依赖库（见「编译 → 环境要求」）。

```bash
# 1. 获取代码（默认流程所需的 XFeat / LightGlue / 重定位模型已在仓库里）
git clone https://github.com/tx417519209/ms_fusion.git
cd ms_fusion
```

下面的命令需要在 colcon 工作空间根目录执行。若你把本仓库放在 `<ws>/src/ms_fusion/` 下，
就在 `<ws>/` 里运行，并把 `scripts/...` 写成 `src/ms_fusion/scripts/...`。

```bash
# 2. 构建离线回放版本（想接真实相机/IVPS 则用 build_ms_fusion.sh）
bash scripts/build_ms_fusion_offline.sh

# 3. 用 TUM VI room1 跑一遍：下载 → 预处理 → 回放 + 评估
bash dataset/download_tumvi_dataset.sh room1
# 把下载好的 tar 放到 dataset/tumvi_room1/ 后，按「测试 → TUM VI 回放评测」执行预处理
bash scripts/run_offline_eval.sh tumvi_room1
```

跑通的判断标准：`statistics/` 下生成 `*_vio_est.tum` 轨迹与日志，`tumvie_eval_trajectory.py eval` 能输出 ATE。
没有板子的话，到「编译」这一步之外都无法验证，请先确认硬件条件。

## 目录结构

| 目录 | 说明 |
|------|------|
| `msf_core/` | 核心库（共享库）：特征提取/匹配、跟踪、预积分、初始化、滑窗优化、边缘化、重定位等；自带 Sophus |
| `ros_wrapper/src/ms_fusion/` | ROS 2 节点 `ms_fusion_node`、launch、默认配置 |
| `ros_wrapper/src/ms_fusion_loop/` | 回环节点 `ms_fusion_loop_node`、DBoW3 词典与训练工具 |
| `third_party_axera/` | Axera NPU / IVPS 接口封装与头文件 |
| `tennis_court_map/` | 发布标准 ITF 网球场 `OccupancyGrid` 的辅助包 |
| `scripts/` | 构建、运行、数据集预处理、回放、评估脚本 |
| `calib_files/tumvi_*/` | TUM VI 各序列的配置（`msf_robot.yaml`） |
| `dataset/` | 数据集下载脚本（数据本身被 `.gitignore` 忽略） |
| `docs/` | 对比分析与 TUM VI 评测记录 |

## 编译

### 环境要求

- Ubuntu 22.04 + ROS 2 Humble、`colcon`、CMake ≥ 3.16、支持 C++20 的编译器
- 依赖库：Eigen3、OpenCV、Ceres、TBB、yaml-cpp、LAPACK/BLAS、fmt、spdlog
- 特征提取与匹配使用 Axera NPU（`MSF_USE_AXERA_SUPERPOINT`），需要 AX650 板端环境
  （`/soc/lib/libax_engine.so` 等）或对应交叉编译环境；CMake 在板端会自动探测并开启

### 构建

在工作空间根目录（即 `colcon build` 的目录）执行，**不要使用 `--symlink-install`**
（install 目录会链到 build 路径，拷到板端后失效）：

```bash
# 板端实时版本：IVPS 零拷贝取图
bash scripts/build_ms_fusion.sh

# 离线回放版本：图像来自 ROS 话题（cv::Mat）
bash scripts/build_ms_fusion_offline.sh
```

两个脚本都固定 `CMAKE_BUILD_TYPE=Release`，构建 `msf_core`、`ms_fusion`、`ms_fusion_loop` 三个包，
可在末尾追加额外的 `--cmake-args`。`PACK=1` 会额外打包 `install/`。

主要 CMake 选项：

| 选项 | 说明 |
|------|------|
| `MSF_USE_AXERA_SUPERPOINT` | 启用 Axera NPU 特征检测/匹配，板端默认 ON |
| `MSF_USE_AXERA_IVPS` | 启用 IVPS 零拷贝取图；**`msf_core`、`ms_fusion`、`ms_fusion_loop` 必须一致**（它改变 `ImageMessage` 的内存布局） |
| `MSF_BUILD_LOOP_TOOLS` | 构建词典采集/训练工具（见 `ros_wrapper/src/ms_fusion_loop/tools/README.md`） |
| `MSF_CORE_USE_PCH` / `MSF_USE_PCH` | 预编译头，默认 ON |

## 运行

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# 统一入口：使用 install/ms_fusion/share/ms_fusion/config/msf_robot.yaml
bash scripts/run_ms_fusion.sh
# 等价于
ros2 launch ms_fusion msf_robot.launch.py
```

- launch 会读取 yaml 中的 `loop.enable`，为 `true` 时同时拉起 `ms_fusion_loop_node`。
- 常用 launch 参数：`config_file`、`topic_imu`、`topic_cam0`、`topic_cam1`、`image_type`（`raw` / `compressed`）、`namespace`。
- 输入话题只使用 ROS 通用消息，**不依赖任何自定义消息包**：IMU 为 `sensor_msgs/Imu`，轮速（可选）为 `nav_msgs/Odometry`，
  双目图像为 `sensor_msgs/CompressedImage` 或 `sensor_msgs/Image`（离线构建）/ IVPS 取图（板端构建）；话题名在 yaml 中配置。
- 输出话题：`/odometry_vio`（VIO 原始位姿与 twist）、`/odometry`（回环校正后，需开启回环）。
- 运行日志写到启动目录下的 `runtime_log/`（带时间戳）。
- 实机使用的是 install 目录下的配置：把目标配置文件覆盖到
  `install/ms_fusion/share/ms_fusion/config/msf_robot.yaml` 再启动。

## 测试

项目的主要验证方式是 **公开数据集回放 + 轨迹误差（ATE）评估**，另有少量单元测试与 lint。

### 1. TUM VI 回放评测

以 `room1` 为例，详细步骤和注意事项见 [`docs/tumvi_room1_board_eval.md`](docs/tumvi_room1_board_eval.md)，
带回环的 `outdoors4` 见 [`docs/tumvi_outdoor4_loop_board_eval.md`](docs/tumvi_outdoor4_loop_board_eval.md)。

```bash
# 1) 下载官方 512x512 EuRoC 导出包（支持断点续传），放到 dataset/tumvi_<序列>/
bash dataset/download_tumvi_dataset.sh room1

# 2) 预处理：去畸变 + 转 8-bit JPEG，生成回放数据和 profile
python3 scripts/tumvi_prepare_dataset.py \
  dataset/tumvi_room1/dataset-room1_512_16.tar \
  --calib calib_files/tumvi_room1/pinhole-equi-512/camchain-imucam-imucalib.yaml \
  --template calib_files/tumvi_room1/msf_robot.yaml \
  --out tmp/tumvi_room1 \
  --profile-out calib_files/tumvi_room1/msf_robot.yaml

# 3) 一键回放并评估（部署 profile → 启动节点 → 录制轨迹 → 按原始时基回放）
bash scripts/run_offline_eval.sh tumvi_room1
```

`run_offline_eval.sh` 的常用选项：`--profile`（`calib_files/` 下的目录名，缺省按数据集名推断）、
`--out-dir`（缺省 `statistics/`）、`--speed`（评测请保持 1.0）、`--settle-s`、`--no-deploy`；
不带参数可列出 `tmp/` 下可用的数据集。产物为运行日志、回放日志和 TUM 格式轨迹
（`*_vio_est.tum`、开启回环时还有 `*_loop_est.tum`）。

也可以手动分步执行：

```bash
# 终端 1：启动节点（离线构建）
ros2 launch ms_fusion msf_robot.launch.py \
  config_file:=$(pwd)/install/ms_fusion/share/ms_fusion/config/tumvi_room1.yaml \
  topic_cam0:=/tumvi/cam0 topic_cam1:=/tumvi/cam1 topic_imu:=/tumvi/imu image_type:=compressed

# 终端 2：录制估计轨迹
python3 scripts/tumvie_eval_trajectory.py record --topic /odometry_vio --out tmp/est.tum

# 终端 3：按原始时基回放
python3 scripts/tumvie_replay_node.py tmp/tumvi_room1 \
  --imu-topic /tumvi/imu --cam0-topic /tumvi/cam0 --cam1-topic /tumvi/cam1

# 回放结束后评估（对齐后的 ATE）
python3 scripts/tumvie_eval_trajectory.py eval \
  --gt tmp/tumvi_room1/mocap_data.txt --est tmp/est.tum --out tmp/ate.txt
```

注意：ms_fusion 内部窗口与超时按墙钟时间运行，回放必须按真实时基进行（不要加速）；
ATE 只衡量轨迹精度，不代表板端实时性，实时性需另看日志中的 tracking / 队列耗时。

### 2. 单元测试与 lint

```bash
colcon build --packages-select msf_core
colcon test --packages-select msf_core ms_fusion ms_fusion_loop
colcon test-result --verbose
```

`msf_core` 带有 `test_upright_p2_solver`（纯 Eigen 的数学模块单测，不依赖厂商 SDK），
`ms_fusion` / `ms_fusion_loop` 只接入 `ament_lint_auto`。

## 自定义数据集

接入自己的双目 + IMU 数据集，需要两样东西：**回放数据**和**一份 profile（yaml 配置）**。

### 1. 数据格式

`scripts/tumvie_replay_node.py` 读取如下扁平目录：

```
<dataset_dir>/
  left/*.jpg                    左目图像，按文件名排序
  right/*.jpg                   右目图像，与左目一一对应
  image_timestamps_left.txt     每行一个时间戳（微秒），数量与图像一致
  imu_data.txt                  每行：t_us wx wy wz ax ay az   （角速度 rad/s，加速度 m/s²）
  odom_data.txt                 可选，轮速里程计
  mocap_data.txt                可选，真值，每行：t_us px py pz qx qy qz qw
```

- 三类数据的时间戳必须在同一时钟域；回放器直接把微秒值换算成 ROS 时间。
- 图像需要是 **针孔 + radtan** 模型（`pinhole-radtan4` / `pinhole-radtan5`），鱼眼图像要先去畸变，
  `scripts/tumvi_prepare_dataset.py` 里有 equidistant 鱼眼去畸变到 640×384 的完整示例。
- 已经是 ROS 2 bag 的数据需要自行转成上述目录（取出 `sensor_msgs/Imu`、双目图像并按微秒时间戳落盘），
  也可以直接在线发布同样的话题给节点使用（见下一节的话题说明）。

### 2. 配置 profile

复制一份现有配置（如 `calib_files/tumvi_room1/msf_robot.yaml`）到 `calib_files/<你的数据集>/msf_robot.yaml`，
至少修改下列部分：

| 配置段 | 要改的内容 |
|--------|-----------|
| `imu` | `topic`、噪声（`arw` / `vrw` / `gb_std` / `ab_std`）、`gravity`、`T_virtual_imu_from_real_imu`（真实 IMU 到 ms_fusion 虚拟 FRD 坐标系的固定旋转） |
| `camera.cam0 / cam1` | `offline.topic`、`image_type`、`resolution`、`intrinsic`（fx fy cx cy）、`distortion`、`T_imu_cam`、`td_cam_to_imu` |
| `odom` | 有轮速时 `enable: true` 并配置 `T_imu_from_odom` 与噪声；纯 VIO 设为 `false` |
| `tracking` / `optimizer` | 特征检测器（`xfeat` / `superpoint`）、窗口大小、关键帧触发门限，按场景调整 |
| `reloc` / `loop` | 不需要时保持 `enable: false`；开启回环需要与特征检测器匹配的 DBoW3 词典（`loop.voc_path`） |

外参约定：`T_imu_cam` 为相机到（虚拟）IMU 的变换，IMU 坐标系为 front/right/down；
如果标定来自 Kalibr 的 `T_cam_imu`，需要取逆，并先换算到虚拟 IMU 系。
`tumvi_prepare_dataset.py` 中的 `virtual_imu_from_cam0` 和 `write_profile` 给出了完整换算方式，可直接参考。

### 3. 回环词典（可选）

词典需与 `tracking.feature_detector` 一致，且最好用目标场景的图像训练：

```bash
# 板端采集灰度图 → 训练词典，详见 ros_wrapper/src/ms_fusion_loop/tools/README.md
ros2 run ms_fusion_loop collect_gray_images -- --output_dir /data/voc_images --interval_ms 200 --max_frames 2000
```

训练完成后放到 `ros_wrapper/src/ms_fusion_loop/voc/`，在 profile 中设置 `loop.voc_path`。

### 4. 跑起来

```bash
cp calib_files/<你的数据集>/msf_robot.yaml \
  install/ms_fusion/share/ms_fusion/config/<你的数据集>.yaml
bash scripts/run_offline_eval.sh <你的数据集目录> --profile <你的数据集>
```

## TUM VI 配置

| 目录 | 序列 |
|------|------|
| `calib_files/tumvi_corridor1/` | corridor1 |
| `calib_files/tumvi_outdoors4/` | outdoors4 |
| `calib_files/tumvi_room1/` | room1（另附 `pinhole-equi-512/` 官方标定结果） |
| `calib_files/tumvi_slides1/` | slides1 |

同一套 TUM VI 设备标定对所有序列通用，序列之间主要差别在 `tracking` / `optimizer` / `loop` 的调参。
