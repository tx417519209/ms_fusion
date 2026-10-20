# MS Fusion

面向 Axera AX650 NPU 板端实时运行的 **双目 + IMU 视觉惯性里程计（VIO）**，ROS 2（Humble）工作空间，
可选轮速里程计和回环（位姿图）。没有板子时也可以在 x86 PC 上用 ONNX Runtime（CPU）跑通整条流程，用于开发调试和精度评测（不保证实时）。

本文按当前源码与脚本核对（2026-10-20）。板端历史评测按构建批次记录，不代表当前工作区或 x86 后端已通过相同验证；
完整结果与已知退化见 [`benchmarks/tumvi/README.md`](benchmarks/tumvi/README.md)。

## 特点

- **学习型前端**：XFeat / SuperPoint 特征 + LightGlue 描述子匹配，推理跑在 AX650 NPU 上（`.axmodel`），或在 x86 CPU 上用 ONNX Runtime（目前仅 XFeat + LightGlue）；
  左目跨帧（temporal）和左右目同帧（stereo）两路匹配，之后叠加运动范围门控、对极约束 / RANSAC 过滤外点。
- **IMU 滑窗优化**：IMU 预积分 + Ceres 滑窗（边缘化）联合优化，关键帧按间隔、视差、匹配数自适应触发。
- **静止检测与零速更新**：基于 IMU 统计量的静止判定，支持 ZUPT；初始化支持静止对准。
- **可选轮速与可视化输出**：轮速里程计（`odom.enable`）参与融合，匹配点云（`matched_cloud.enable`）用于输出与可视化。
- **回环**：独立的 `ms_fusion_loop` 节点，DBoW3 词典检索 + Mutual NN / PnP 几何验证 + 4DoF 位姿图，
  对 VIO 输出做校正（`/odometry`），可通过 `loop.output_smooth_tau_s` 启用输出平滑；VIO 原始输出为 `/odometry_vio`，回环校正不回写其滑窗状态。
- **配置驱动**：传感器外参、内参、噪声、前端/后端参数全部在一份 `msf_robot.yaml` 里，
  模型与词典路径相对配置文件目录解析；换设备仍需提供匹配的标定、模型和词典。
- **离线回放评测**：同一套代码可编译成「订阅 ROS 话题图像」的离线版本，用于回放公开数据集（如 TUM VI）并计算 ATE。

与 SuperVINS 的前后端对比见 [`docs/SuperVINS_vs_ms_fusion_analysis_claude.md`](docs/SuperVINS_vs_ms_fusion_analysis_claude.md)。

## 为什么选择这个项目

适合你的情况：

- **主要目标平台是 Axera AX650N 板**（或同类 Axera SoC），需要在板端实时跑学习型特征 VIO。
  没有板子时可用 x86 + ONNX Runtime 路径做开发调试和公开数据集精度评测，但 CPU 推理不保证实时。特征提取、匹配、取图都已经接好 NPU / IVPS，并做了零拷贝取图和 NPU 亲和性绑定。
- 需要**学习型特征（XFeat / SuperPoint + LightGlue）**带来的弱纹理、光照变化鲁棒性，同时仍保持 IMU 紧耦合滑窗优化。
- 需要在 VIO 之上叠加**轮速、回环**这些工程化能力，且都能用同一份 yaml 开关。
- 想要一套**可复现的公开数据集评测流程**：TUM VI 的下载、预处理、回放、ATE 评估脚本齐全，
  `benchmarks/tumvi/` 有 28 条序列的新旧二进制纯 VIO 对比，`docs/` 另有 room1、outdoors4（回环）的历史板端记录。
- 需要对前后端细节做实验：参数全部配置化，评测脚本一键产出日志和轨迹，便于做对照。

不适合你的情况见下一节。

## 局限与不足

请在决定使用前确认：

1. **板端推理强依赖 Axera AX650N NPU 与其 SDK；PC 上只能走 x86 CPU（ONNX Runtime）路径。**
   - 前端只支持 `xfeat` / `superpoint` 两种检测器，FAST/KLT 路径已移除。NPU 路径使用 `.axmodel` 模型，
     运行库来自板端 `/soc/lib/libax_*.so`。
   - 在 x86 PC 上请用 `scripts/build_ms_fusion_x86.sh`（`USE_ONNX=ON`），见「编译 → x86 CPU（ONNX Runtime）构建」。
     该路径目前只实现了 **XFeat + LightGlue**，不支持 SuperPoint；CPU 推理比 NPU 慢，回放评测无法保证实时。
   - `scripts/run_colcon_build_onnx.sh`、`install_onnxruntime*.sh` 是早期 ONNX Runtime 后端的遗留脚本，
     当前 CMake 已不再提供对应的 `MSF_INFER_BACKEND` 选项，**不要依赖它们**；x86 请使用 `build_ms_fusion_x86.sh`。
   - 既不开 NPU（`MSF_USE_AXERA_SUPERPOINT=OFF`）也不开 `USE_ONNX` 时虽然能通过编译，但运行时检测器初始化会直接失败（`BUILD_DISABLED`），无法产生特征。
   - 板端「离线回放评测」要在 AX650N 板端（或能访问 NPU 的环境）上执行，离线构建只是把输入从 IVPS 换成了 ROS 话题。
2. **仅支持双目 + IMU**，不支持单目、RGB-D；相机模型仅 `pinhole-radtan4/5`，鱼眼数据需要先去畸变。
3. **使用门槛偏高**：板端交叉编译 / qemu 下编译耗时长（脚本里限制了并行度防止内存不足）；
   仓库提供了部分 XFeat 词典，但换场景需要验证检索与几何验证效果，必要时重新训练。
4. **模型文件范围有限**：`third_party/axera/asset/models/` 只带运行默认流程所需的 2 个 `.axmodel`
   （XFeat top256、LightGlue top256）。SuperPoint 等其他模型未包含，
   要用 `tracking.extractor.feature_detector: superpoint` 时需自行准备模型并配置各 yaml 中的 `model_path`；
   配置里残留的 `superpoint_*.axmodel` 路径只在选择该检测器时才会被加载。
5. **评测覆盖边界**：已有 TUM VI 28 条序列、每条 3 次的新旧二进制对比，但尚无更多数据集的系统性对比；
   单元测试只覆盖纯数学模块（旋转、相机模型、IMU 预积分、位姿流形、直立 2 点求解器），
   特征跟踪、回环、优化器整体等主流程没有单测，主要靠数据集回放验证。
6. **配置与默认值需要核对**：换设备或数据集时需要重新标定并验证参数。仓库保留了历史单序列配置，
   全序列基准则使用同一份 room1 模板、关闭回环，未逐序列调参。YAML 相同不保证跨版本有效参数相同：
   例如当前缺省 `imu.gyr_scale_std=0.005`，历史基线尚无该代理。复现实验应记录安装文件哈希、实际配置与启动日志。
7. **纯 VIO 会累计漂移**：GT 片段独立对齐的局部误差小，不代表长时间定位误差小；回环或外部约束的效果需单独验证，
   输出平滑本身不能消除定位误差。

## 快速上手（clone 之后怎么跑起来）

前提：已装 ROS 2 Humble 与依赖库（见「编译 → 环境要求」），并满足下面任一条件：

- 一块能访问 NPU 的 AX650N 板（或板端容器），按下面的步骤构建；
- 没有板子：在 x86_64 PC 上改用 `scripts/build_ms_fusion_x86.sh` 构建（ONNX Runtime CPU，见「编译 → x86 CPU（ONNX Runtime）构建」），
  再按下文准备数据并运行回放评测。

```bash
# 1. 获取代码（默认流程所需的 XFeat / LightGlue 模型已在仓库里）
git clone https://github.com/tx417519209/ms_fusion.git
cd ms_fusion
```

以下命令从仓库根目录执行。构建脚本会自行切换到仓库根目录，再调用 `colcon build`；
默认产物为本仓库的 `build/`、`install/`、`log/`。

```bash
# 2. 构建离线回放版本（想接真实相机/IVPS 则用 build_ms_fusion.sh）
bash scripts/build_ms_fusion_offline.sh

# 3. 按下文「TUM VI 回放评测」依次下载、预处理，再运行回放评测脚本
# run_offline_eval.sh 不会自动下载或预处理原始数据
```

跑通的判断标准：`statistics/` 下生成 `*_vio_est.tum` 轨迹与日志，`tumvie_eval_trajectory.py eval` 能输出 ATE。
x86 + ONNX 路径同样能产出轨迹并计算 ATE，但速度比 NPU 慢，回放评测无法保证实时。

## 目录结构

| 目录 | 说明 |
|------|------|
| `msf_core/` | 核心库（共享库）：特征提取/匹配、跟踪、预积分、初始化、滑窗优化、边缘化等 |
| `ros_wrapper/src/ms_fusion/` | ROS 2 节点 `ms_fusion_node`、launch、默认配置 |
| `ros_wrapper/src/ms_fusion_loop/` | 回环节点 `ms_fusion_loop_node`、DBoW3 词典与训练工具 |
| `third_party/axera/` | Axera NPU / IVPS 接口封装与头文件 |
| `third_party/Sophus/`、`third_party/ceres/`、`third_party/onnxruntime/` | 几何库、预编译 Ceres、ONNX Runtime 与 CPU 模型 |
| `tennis_court_map/` | 发布标准 ITF 网球场 `OccupancyGrid` 的辅助包 |
| `scripts/` | 构建、运行、数据集预处理、回放、评估脚本 |
| `calib_files/tumvi_*/` | TUM VI 各序列的配置（`msf_robot.yaml`） |
| `dataset/` | 数据集下载脚本（数据本身被 `.gitignore` 忽略） |
| `docs/` | 对比分析与 TUM VI 评测记录 |
| `benchmarks/tumvi/` | 全序列结果、逐次 CSV、新旧构建对比与轨迹图 |

## 编译

### 环境要求

- Ubuntu 22.04 + ROS 2 Humble、`colcon`、CMake ≥ 3.16、支持 C++20 的编译器
- 依赖库：Eigen3、OpenCV、Ceres（需 ≥ 2.1，代码使用 `ceres::Manifold`）、TBB、yaml-cpp、LAPACK/BLAS、fmt、spdlog
- 特征提取与匹配二选一：Axera NPU（`MSF_USE_AXERA_SUPERPOINT`），需要 AX650 板端环境
  （`/soc/lib/libax_engine.so` 等）或对应交叉编译环境，CMake 在板端会自动探测并开启；
  或 x86 CPU 的 ONNX Runtime（`USE_ONNX`），不需要任何 Axera SDK，见下文「x86 CPU（ONNX Runtime）构建」

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

### x86 CPU（ONNX Runtime）构建

没有 AX650N 板时，可以在 x86_64 PC 上用 CPU 推理 XFeat + LightGlue 跑通整条 VIO 流程（适合开发调试与精度对比，不保证实时）。

```bash
# 系统依赖（Ubuntu 22.04）。Ceres 与 ONNX Runtime 已随仓库放在 third_party/，不需要 apt 安装
sudo apt install -y libgoogle-glog-dev libgflags-dev libeigen3-dev libsuitesparse-dev \
    libopencv-dev libtbb-dev libyaml-cpp-dev libfmt-dev libspdlog-dev liblapack-dev

# 在工作空间根目录执行；不要使用 --symlink-install
bash scripts/build_ms_fusion_x86.sh            # 可在末尾追加额外的 --cmake-args
source install/setup.bash
```

- **第三方库**：`third_party/ceres/x86_64` 是预编译的 Ceres 2.2.0（Ubuntu 22.04 的 `libceres-dev` 只有 2.0，缺少 `ceres::Manifold`），
  `third_party/onnxruntime` 是 ONNX Runtime 1.23.2（CPU）及 XFeat / LightGlue 的 `.onnx` 模型。
  `libonnxruntime.so` 会随 `libmsf_core.so` 装进 `install/msf_core/lib`，无需额外设置 `LD_LIBRARY_PATH`。
- **配置沿用板端 yaml**：`tracking.extractor.xfeat.model_path`、`tracking.matcher.lightglue.model_path` 仍写 `.axmodel`，`USE_ONNX` 下会自动换成对应的 `.onnx`
  （先找 yaml 所在目录的 `../models/`，找不到再退回 `third_party/onnxruntime/models/`）。
- **LightGlue 约束**：ONNX 模型固定 256 个关键点，配置键为 `tracking.matcher.lightglue.max_num_features: 256`；
  初始化失败会回退到 MNN 匹配，应核对启动日志，避免把回退后的运行当作 LightGlue 结果。
- **回放评测**：`run_offline_eval.sh` 在本机复制 profile、启动节点和回放，没有远程板端部署步骤；
  x86 离线构建也可使用，或使用下文手动三终端方式。CPU 跟不上 1.0× 回放时需记录丢帧和处理延迟，不能直接视为同输入帧的 NPU 对照。
  同一网络里若有别的机器也在发 `/odometry_vio`，录到的轨迹会混入对方数据；在所有相关终端中设置
  `ROS_DOMAIN_ID` 为独占编号，并执行 `export ROS_LOCALHOST_ONLY=1`。
- 相关 CMake 选项：`USE_ONNX`（与 `MSF_USE_AXERA_SUPERPOINT` 互斥）、`MSF_USE_VENDORED_CERES`（默认 ON，且仅当 `third_party/ceres/<CPU架构>` 存在时生效；
  设为 OFF 则使用系统 Ceres，需 ≥ 2.1）。

主要 CMake 选项：

| 选项 | 说明 |
|------|------|
| `MSF_USE_AXERA_SUPERPOINT` | 启用 Axera NPU 特征检测/匹配，板端默认 ON |
| `USE_ONNX` | x86 CPU 路径：用 ONNX Runtime 推理 XFeat / LightGlue，默认 OFF；与 `MSF_USE_AXERA_SUPERPOINT` 互斥 |
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
- 未传 `config_file` 时使用 install 目录下的 `msf_robot.yaml`；也可显式传入其他配置路径。
  `run_offline_eval.sh` 则使用 install 目录下的 `<profile>.yaml`，默认会从 `calib_files/<profile>/msf_robot.yaml` 覆盖它。

## 测试

项目的主要验证方式是 **公开数据集回放 + 轨迹误差（ATE）评估**，另有少量单元测试与 lint。

### 1. TUM VI 回放评测

以 `room1` 为例，详细步骤和注意事项见 [`docs/tumvi_room1_board_eval.md`](docs/tumvi_room1_board_eval.md)，
带回环的 `outdoors4` 见 [`docs/tumvi_outdoor4_loop_board_eval.md`](docs/tumvi_outdoor4_loop_board_eval.md)。

```bash
# 1) 下载官方 512x512 EuRoC 导出包（支持断点续传），脚本自动保存到 dataset/tumvi_<序列>/
bash dataset/download_tumvi_dataset.sh room1

# 2) 预处理：去畸变 + 转 8-bit JPEG，生成回放数据和 profile
# --profile-out 会覆盖该配置；需要保留自定义配置时，先备份或选择另一个输出路径
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

### TUM VI 全序列评测结果

28 条序列 × 3 次的完整结果（ATE 三种对齐口径、RPE、初始化失败、前端 Hz、后端 P95、轨迹图，
含失败序列与调参对比）见 [`benchmarks/tumvi/README.md`](benchmarks/tumvi/README.md)。
以下两批板端实验均为同一 YAML 模板、回环关闭、未逐序列调参；新旧二进制的默认参数不完全相同。

| 构建批次 | 全序列结果与边界 |
|---|---|
| 2026-10-14 基线 | 84 次中 78 次通过发散筛查，6 次发散均为 outdoors3/5；旧版结果保留作对照 |
| 2026-10-19 新版 | 84 次均通过发散筛查，但 outdoors3 首段对齐 ATE 中位数仍约 105 m，outdoors5 约 8.84 m；通过筛查不等于准确定位 |

新版 room1–6 的 Global-SE3 ATE 中位数约 0.05–0.16 m；长序列仍有明显累计漂移。
magistrale1 的 Reference-SE3 ATE 从基线 1.44 / 1.95 / 2.35 m 变为 6.19 / 7.82 / 2.53 m。
同一新二进制补跑 `gyr_scale_std=0` 为 3.53 / 4.26 m，`0.002` 为 8.95 / 4.93 m：
关闭代理仍未恢复旧版水平，尚不能把退化单独归因于该参数，也不支持“调小就稳定改善”。

评测需区分 Global-SE3（全段共同对齐）、Reference-SE3（只用首个可评测 GT 段对齐）和 Own-SE3（各段独立对齐）。
TUM VI 长序列 GT 通常只覆盖部分区间；Own-SE3 会拟合掉段间累计偏差，不能代替长期漂移指标。
上述结果来自已归档二进制，确切构建 commit 未记录；不能据此声明当前源码或 x86 后端已达到同样结果。

### 2. 单元测试与 lint

```bash
colcon build --packages-select msf_core
colcon test --packages-select msf_core ms_fusion ms_fusion_loop
colcon test-result --verbose
```

`msf_core` 的单测只编译被测的纯数学源文件，不链接 `msf_core`，不依赖厂商 SDK 和 NPU：

| 测试 | 覆盖 |
|---|---|
| `test_rotation` | 旋转向量 / 四元数 / 欧拉角互转、`IntegratedRotation`、右雅可比与其逆 |
| `test_camera` | pinhole-radtan4 / radtan5 的投影与去畸变往返 |
| `test_preintegration` | IMU 预积分解析解、零残差自洽、四个雅可比与数值微分对拍、零偏一阶修正 |
| `test_preintegration_rot_sign` | 四元数符号分支下的预积分残差与解析雅可比一致性 |
| `test_pose_manifold` | 位姿流形 Plus / Minus 互逆 |
| `test_upright_p2_solver` | 直立 2 点绝对位姿解析解 |

注意：相机 `unproject` 用固定 10 次不动点迭代去畸变，畸变大的画面边角会残留约 0.03 px 的往返误差。

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
  odom_data.txt                 可选，每行：t_us vx vy vz wx wy wz（m/s、rad/s）
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
| `imu` | `topic`、噪声（`arw` / `vrw` / `gb_std` / `ab_std`）、`gravity`、`T_virtual_imu_from_real_imu`；显式记录 `gyr_scale_std`（当前默认 0.005，0 关闭；是姿态协方差代理，不是比例因子标定） |
| `camera.cam0 / cam1` | `offline.topic`、`image_type`、`resolution`、`intrinsic`（fx fy cx cy）、`distortion`、`T_imu_cam`、`td_cam_to_imu` |
| `odom` | 有轮速时 `enable: true` 并配置 `T_imu_from_odom` 与噪声；纯 VIO 设为 `false` |
| `tracking` / `optimizer` | 特征检测器（`xfeat` / `superpoint`）、窗口大小、关键帧触发门限，按场景调整 |
| `loop` | 开启回环需要与特征检测器匹配的 DBoW3 词典（`loop.voc_path`） |

外参约定：`T_imu_cam` 为相机到（虚拟）IMU 的变换，IMU 坐标系为 front/right/down；
如果标定来自 Kalibr 的 `T_cam_imu`，需要取逆，并先换算到虚拟 IMU 系。
`tumvi_prepare_dataset.py` 中的 `virtual_imu_from_cam0` 和 `write_profile` 给出了完整换算方式，可直接参考。

### 3. 回环词典（可选）

词典需与 `tracking.extractor.feature_detector` 及其描述子兼容，且最好用目标场景的图像训练：

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

这些目录包含历史单序列配置；完整 28 序列评测使用统一 room1 模板生成各序列 profile，不能将历史配置当作该评测的有效配置。
预处理同时读取相机链文件及同目录的 `imu-imucalib.yaml`，换数据集时应核对标定来源与坐标变换。

## 跟踪与诊断字段的含义

- 时序匹配使用后端接受的关键帧快照作为参考。只有被接受的关键帧观测才写入路标 `obs_`；普通帧匹配成功不等于进入滑窗优化。
- `long_track` 是当前输入中、已有至少 4 条关键帧观测的 ID 数，在本帧入库之前统计；不是平均轨迹长度，也不保证点已三角化或贡献视觉因子。
- `valid_match`、`valid_3d`、`vis_n` 分别表示参考快照二维连接、满足几何条件的三维连接、实际跨帧视觉残差块数，分母和含义不同。
- 锚帧边缘化会清除该点几何、保留可用二维观测；后续双目重初始化会换锚并截去更早观测。ID 存活不等于三维约束历史连续，已边缘化的信息则可能留在先验中。
- `vis_p50` / `vis_rmse` 是白化视觉残差块范数的统计量，不是像素误差；`dR_diff_deg` 小只说明优化结果与所用 IMU 约束接近，不是姿态真值证明。
- `new_outlier_obs` 是每次后端处理中新变为外点的观测数，在 timing 窗口中汇总；不是路标删除总数。队列为空也不代表没有丢帧。

逐关键帧求解与准入诊断见 [`docs/keyframe_diag.md`](docs/keyframe_diag.md)，预积分修复与协方差代理见
[`docs/preintegration_rot_jacobian_sign.md`](docs/preintegration_rot_jacobian_sign.md)、[`docs/gyro_scale_covariance_proxy.md`](docs/gyro_scale_covariance_proxy.md)。
