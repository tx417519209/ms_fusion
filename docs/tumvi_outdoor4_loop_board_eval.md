# TUM VI `outdoors4`：回环词典训练 + 板端回放评估

本流程测的是双目 VIO + 回环闭环（`loop.enable`）。跟 `docs/tumvi_room1_board_eval.md`
的纯 VIO 基线不同，`outdoors4` 的 MoCap 真值**只覆盖首尾两段**（起点后约 64s、
终点前约 63s，中间约 572s 完全没有真值），比 `corridor1`（中间 247s 无真值）
更长的无约束空窗——目的是验证：现有 `ms_fusion_loop` 位姿图能否在长时间视觉/
GT 双盲的空窗后，靠回到起点附近场景的回环检测把累计漂移纠正回来。

起点和终点 MoCap 位置只相差约 1.04m，是"走一圈回原地"的路线，具备回环的
几何前提。但注意：`loop.enable=true` 不保证一定触发回环——是否成功仍取决于
DBoW 检索分数、Mutual NN + PnP 几何验证能不能过门限（见
`ros_wrapper/src/ms_fusion_loop/docs/loop_closure_guide.html`）。

不要把这个流程跟 `room1` 纯 VIO 基线或 TUM-VIE `skate-hard` 流程/标定混用。

## 0. 已具备的前提

跟 `room1` 复用同一份设备级标定（`imu:`/`camera:` 两段跟
`calib_files/tumvi_room1/msf_robot.yaml` 逐字节相同，TUM-VI 同一录制装置
标定对所有序列通用），`outdoors4` 的本地准备已经做过：

```bash
dataset/tumvi_outdoors4/dataset-outdoors4_512_16.tar   # 官方包
tmp/tumvi_outdoors4/                                    # 预处理产物：left/ right/ imu_data.txt
                                                         # image_timestamps_left.txt mocap_data.txt
calib_files/tumvi_outdoors4/msf_robot.yaml          # profile
```

如需重新生成（比如 `tmp/tumvi_outdoors4` 被清理过），复现命令：

```bash
python3 scripts/tumvi_prepare_dataset.py \
  dataset/tumvi_outdoors4/dataset-outdoors4_512_16.tar \
  --calib calib_files/tumvi_room1/pinhole-equi-512/camchain-imucam-imucalib.yaml \
  --out tmp/tumvi_outdoors4 \
  --profile-out calib_files/tumvi_outdoors4/msf_robot.yaml
```

预处理结束应至少检查：左右图与时间戳数相等、IMU 行数非零，以及脚本打印的
首 2 秒静止比力模长接近 9.81。

## 1. 训练 outdoors4 专用回环词典（板端提特征 + 本地聚类，两台机器）

`calib_files/tumvi_outdoors4/msf_robot.yaml` 里 `loop.voc_path` 默认指向
`xfeat_voc_robot_0729.dbow3`——那是另一场景训的词典，跟 outdoors4 户外纹理
对不上，检索大概率长期拿不到候选。

这一步拆成两半：NPU 特征提取只能在板子上跑（没有 Python NPU 推理接口），但
DBoW3 层次 k-means 聚类是纯 CPU 工作，仓库里那份 DBoW3（带 quicklz 压缩）没
有迭代次数上限，实测在样本数逼近/小于分支因子 k 的深层节点上会反复震荡、
卡很久；所以聚类挪到本地开发机用 sklearn 的 KMeans 跑，快得多，也不占板子。

**板端（提特征，不聚类）：**

```bash
bash scripts/build_ms_fusion_offline.sh -DMSF_BUILD_LOOP_TOOLS=ON   # 只在还没编译过/没加 --features_out 之前编译过时需要
bash scripts/train_outdoor4_loop_voc.sh
```

产物默认在 `tmp/tumvi_outdoors4/xfeat_voc_outdoors4.msfv`（自定义的简单二进制
格式，只是每张图的原始描述子，不是 DBoW3 格式）。检查脚本打印的
`ok_rate`/`avg_kps`，`ok_rate` 明显低（比如 <90%）要先排查，不要直接拿去用。

**本地开发机（聚类，产出真正的 `.dbow3`）：**

```bash
pip install numpy scikit-learn   # 只需装一次
python3 scripts/dbow3_train_from_features.py \
  --features xfeat_voc_outdoors4.msfv \
  --output ros_wrapper/src/ms_fusion_loop/voc/xfeat_voc_outdoors4.dbow3 \
  --k 10 --L 4
```

`--L 4`（而不是 DBoW3 默认的 5）是刻意选的：`k^L` 是目标叶子词数，L=5 时
需要 100000 个叶子词，但 outdoors4 按 stride=8 抽帧后训练描述子量级是十几万
到几十万，平均每叶不到几个样本，深层节点的聚类会很不稳定；L=4（10000 词）
每叶平均样本数足够，检索能力也够用。脚本会在训练描述子明显不够时打印警告。

`dbow3_train_from_features.py` 写的是这份仓库 DBoW3 实现读取的**未压缩**二进
制格式（`Vocabulary::toStream` 的 `compressed=false` 分支），不涉及 quicklz，
跟运行时的 `Vocabulary::load()` 是同一套字节布局。产物再传回板子部署。

## 2. 打开回环，同步 profile 到板端安装目录

在 `calib_files/tumvi_outdoors4/msf_robot.yaml` 里改：

```yaml
loop:
  enable: true
  voc_path: ../models/xfeat_voc_outdoors4.dbow3
```

profile 的相对模型路径（`xfeat.model_path`/`loop.voc_path` 等）都依赖它位于
`install/ms_fusion/share/ms_fusion/config/`，因此板端必须复制到该目录再启动：

```bash
cp calib_files/tumvi_outdoors4/msf_robot.yaml \
  install/ms_fusion/share/ms_fusion/config/tumvi_outdoors4.yaml
```

## 3. 板端执行

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ms_fusion msf_robot.launch.py \
  config_file:=$(pwd)/install/ms_fusion/share/ms_fusion/config/tumvi_outdoors4.yaml \
  topic_cam0:=/tumvi/cam0 topic_cam1:=/tumvi/cam1 topic_imu:=/tumvi/imu \
  image_type:=compressed
```

`msf_robot.launch.py` 会读这份 yaml 的 `loop.enable`，为 true 时自动带起
`ms_fusion_loop_node`（同一份 config，见其 `_loop_enabled()`），不需要单独再启
一个回环节点。启动日志里确认能看到
`[msf_robot.launch] loop.enable=true → starting ms_fusion_loop_node`。

## 4. 同时录制原始 VIO 和回环校正后的轨迹

`/odometry_vio` 是 INS 高频原始递推输出（回环节点要去修正的上游源头），
`/odometry` 才是回环节点校正后的最终输出（见 `ms_fusion_loop` 文档 `vio_callback`）。
只录 `/odometry_vio` 会看不出回环有没有起作用，因此两路都录，另开两个终端：

```bash
python3 scripts/tumvie_eval_trajectory.py record \
  --topic /odometry_vio --out tmp/tumvi_outdoors4_vio_est.tum

python3 scripts/tumvie_eval_trajectory.py record \
  --topic /odometry --out tmp/tumvi_outdoors4_loop_est.tum
```

再开第三个终端按原始时基回放：

```bash
python3 scripts/tumvie_replay_node.py tmp/tumvi_outdoors4 \
  --imu-topic /tumvi/imu --cam0-topic /tumvi/cam0 --cam1-topic /tumvi/cam1
```

## 5. 评估

回放结束后停止两路录制，分别对 GT 评估（`mocap_data.txt` 只有首尾两段真值，
`tumvie_eval_trajectory.py` 会自动按 GT 覆盖窗口分段评估，中间约 572s 空窗
期间没有真值可比，属预期行为）：

```bash
python3 scripts/tumvie_eval_trajectory.py eval \
  --gt tmp/tumvi_outdoors4/mocap_data.txt --est tmp/tumvi_outdoors4_vio_est.tum \
  --out tmp/tumvi_outdoors4_vio_ate.txt

python3 scripts/tumvie_eval_trajectory.py eval \
  --gt tmp/tumvi_outdoors4/mocap_data.txt --est tmp/tumvi_outdoors4_loop_est.tum \
  --out tmp/tumvi_outdoors4_loop_ate.txt
```

把两份 `_ate.txt` 一起发出来对比：末尾 GT 段（空窗之后）如果 `loop_est` 相比
`vio_est` 的位置/yaw 误差明显收窄，说明回环确实在长空窗后把漂移纠正回来了；
如果两者几乎一样，去查回放期间 `ms_fusion_loop` 的日志（`loop.dbow.*`/
`loop.verify.*`/`loop.optimize.*`，见 `ros_wrapper/src/ms_fusion_loop/docs/
loop_closure_guide.html` §17.2 timing 对照表）是检索没给候选、几何验证没过、
还是压根没触发——不要只看 ATE 数字就下结论。

这项 ATE 是对齐后的纯轨迹误差，不验证板端实时性；另行保存启动日志与回放期间
的 tracking/backend/loop timing，才能判断设备端是否跟得上 20 Hz 图像、200 Hz
IMU 以及回环这一路额外的 DBoW/PnP/Ceres 开销。
