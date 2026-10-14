# SuperVINS 与 ms_fusion 前端/后端对比分析

## 结论

两者思路相同：**学习型局部特征 + LightGlue 描述子匹配 + IMU 滑窗 Ceres 优化**。SuperVINS 是"VINS-Fusion 换前端"的学术实现（ONNX Runtime、GPU/CPU 推理），ms_fusion 是面向 Axera AX650 NPU 板端实时部署、并叠加轮速里程计与重定位的工程化系统。两者算法思想可互相参考，但推理引擎、数据流和工程约束差异较大，不能直接照搬参数或代码。

---

## 一、前端

### 1.1 相同点

- 都用深度学习特征替代传统 Shi-Tomasi+KLT / ORB+BRIEF：SuperVINS 用 SuperPoint，ms_fusion 主用 XFeat（`INpuScoreMapDetector` 接口下也支持 SuperPoint）。
- 都用 LightGlue 做描述子匹配（而非光流跟踪），拆分为 temporal（左目跨帧）+ stereo（左右目同帧）两路匹配。
- 跟踪 ID 语义一致：左目为主 ID，右目观测挂靠左目 landmark ID；匹配成功继承旧 ID，失败/新出现的点补充分配新 ID。
- 匹配后都叠加几何验证过滤外点：F 矩阵 RANSAC（SuperVINS）/ 运动范围门控 + RANSAC 或对极约束（ms_fusion）。
- 都是双目 + IMU 配置（ms_fusion 额外支持可选轮速 odom）。

### 1.2 差异点

| 维度 | SuperVINS | ms_fusion |
|---|---|---|
| 推理引擎 | ONNX Runtime（CPU/GPU），`featureTracker/extractor_matcher_dpl.h/.cpp`，`Extractor_DPL`/`Matcher_DPL` | Axera AX650 NPU 硬件推理（`XFeatDetector`/`SuperPointDetector`），另有 MNN CPU 匹配 fallback（`FeatureMatcher`） |
| 权重/词典 | `weights_dpl/superpoint.onnx`、`superpoint_lightglue_fused.onnx` | 板端 NPU 模型 + `LightGlueRuntime`（NPU handle） |
| 特征规模 | SuperPoint 256 维描述子，典型 200~256 点/帧 | XFeat 稠密 score map + NMS，实测左目约 47~48 点/帧，规模明显小于 SuperVINS |
| 主流程入口 | `FeatureTrackerDPL::trackImage_dpl()`（`feature_tracker_dpl.cpp`），旧 KLT 版本 `feature_tracker.cpp` 仍在仓库但未被 `estimator.h` 使用 | `Tracking` 类（`msf_core/src/vision/tracking.cpp`），左右目独立 `MatchWorker` 线程并行编排（提交 `a0c7723`） |
| 左右目处理顺序 | 串行：左目提取 → temporal 匹配 → 补点定稿左目集合 → 右目提取 → 用最终左目点集与右目匹配（`feature_tracker_dpl.cpp:554-556, 735-746`） | 并行：左右目同时检测（`detectDense()` 双目并行推理，提交 `9495222`），temporal 与 stereo 两路 LightGlue/MNN 并行匹配，stereo 结果最后映射回 temporal 保留的左目最终轨迹（`tracking.cpp:351-389, 462-495`） |
| 退化帧处理 | 有专门放行逻辑：LightGlue 平均置信度高 + F-RANSAC 内点率明显低于历史值 → 判定为小运动导致 F 矩阵退化 → 放行全部 LightGlue 匹配（历史内点率窗口 `historical_inlier_rates`） | 无退化放行机制，temporal 路径固定为：LightGlue/MNN 匹配 → 最大运动距离限制 → RANSAC 过滤 → 失败点直接丢弃 → 后续网格补点分配新 ID（`feature_matcher.cpp:447-481`） |
| 补点方式 | 匹配失败区域用 mask 补充新 SuperPoint 特征 | 网格/最小间距补点（`FeatureExtractor::supplementLeftFeaturesFromDense`），补点后再关联右目观测 |
| 立体绑定 | 立体匹配后右目点直接绑定同一左目 `feature_id`（`feature_tracker_dpl.cpp:739-746`） | 语义相同，但右目仅作左目轨迹的附属观测，不维护独立右目长期轨迹（`tracking.cpp:504-524`）；最终接受条件主要是对极约束（`feature_matcher.cpp:663-709`） |
| 硬件零拷贝 | 无（标准内存 + ONNX 推理） | `IvpsFrameHandle`（提交 `8d3779d` 从 Tracking 拆出）用 `AX_SYS_MemAllocCached` 物理连续内存，`detect(IvpsFramePtr&, ...)` 走 `AX_IVPS_CropResizeVgp` 跳过整帧 memcpy 预处理 |
| 关键调参 | `matche_score_threshold`（LightGlue 置信度阈值）、`ransacReprojThreshold` | `score_threshold=0.005`、`nms_radius=0`（关闭冗余 CPU NMS）、`border=30`、`lightglue_score_threshold=0.9`（提交 `10b5c1d`/`6d7e623`） |
| 重定位 | 无独立重定位模块（依赖回环+位姿图） | 独立 `ReLoc` 类：YOLO-OBB 检测 + PnP RANSAC（`calcPose`/`optimizePose`/`checkPose`），`MsFusionManager::requestReloc`/`rtkCalibrate` 触发，结果经 `applyRelocMeasurement` 对齐滑窗状态；`tracking_sync_`/`reloc_sync_` 两个独立 `RawSynchronizer` 队列做左右目时间戳配对，容量已从 4 收紧到 2（提交 `4ca89fc`）以降低配对延迟 |

由于 ms_fusion 左目特征数（~47）远低于 SuperVINS（~200~256），匹配阈值、几何过滤和 ID 稳定性对少量点更敏感，日志中的"匹配成功率"在两边含义并不完全对等——SuperVINS 是相对固定分母下的比例，ms_fusion 因 dense 候选集每帧重新检测，分母本身会波动。

---

## 二、后端

### 2.1 相同点

都保留经典 VINS-Fusion 的滑窗+Ceres 结构：

- 固定大小滑动窗口
- IMU 预积分残差因子（中值积分）：SuperVINS 的 `IMUFactor : ceres::SizedCostFunction<15,7,9,7,9>`（`factor/imu_factor.h`）；ms_fusion 的 `PreintegrationFactor`（自研实现，非直接照搬代码，但残差维度和参数块结构同构）
- 视觉重投影因子（逆深度参数化）：SuperVINS 分三类——`projectionTwoFrameOneCamFactor`（单目跨帧）、`projectionOneFrameTwoCamFactor`（双目同帧）、`projectionTwoFrameTwoCamFactor`（双目跨帧）；ms_fusion 对应 `MonoVisionFactor`/`StereoVisionFactor`（`vision_factor.h`，手写解析 Jacobian）
- 边缘化因子：`MarginalizationFactor`/`MarginalizationInfo`，Schur 补，两边风格一致（源自 VINS-Mono）
- 特征管理：`FeatureManager` 维护 landmark×帧观测表，负责三角化和特征生命周期
- 初始化流程（IMU-视觉对齐、陀螺仪 bias 估计）沿用 VINS-Fusion 思路

### 2.2 关键差异：关键帧判定逻辑

这是后端最大的差异点。

**SuperVINS**：每帧观测先全部进入 `FeatureManager`，`addFeatureCheckParallax()` 综合以下条件判断是否为关键帧（`estimator/feature_manager.cpp:52-121`）：

- 长期跟踪点数量
- 新特征比例
- 平均视差

非关键帧仍会进入 estimator 参与优化，只是边缘化策略不同——关键帧触发 `MARGIN_OLD`（边缘化最老帧），非关键帧触发 `MARGIN_SECOND_NEW`（边缘化次新帧），见 `estimator/estimator.cpp:548-559`。

**ms_fusion**：`FeatureManager::tryAddKeyframe()` 主要判断时间门控：

```text
当前时间 - 上一个关键帧时间 >= keyframe_min_interval_s（默认约 0.099s）
```

未通过门控的帧不进入后续优化，直接标记为 backend rejected（`manager/ms_fusion_manager.cpp` 中记录拒绝日志）。因此日志中的 `accept_pct` 本质是"时间门控下被接受为关键帧的比例"，不是 LightGlue 匹配成功率，也不直接反映视差/几何质量。

### 2.3 其他差异

| 维度 | SuperVINS | ms_fusion |
|---|---|---|
| 里程计融合 | 无 | `OdomFactor`（轮速计，6 维残差，解析雅可比），可选启用 |
| 重定位因子 | 无 | `ReLocMonoReprojectionFactor`/`ReLocStereoReprojectionFactor`，把 reloc 结果整体对齐进滑窗 |
| 优化器 | 原版 VINS-Fusion `Estimator` 类 | 自研 `Optimizer`（`optimizer.h/.cpp`），`addImuFactors`/`addOdomFactors`/`addVisionFactors`/`addPriorFactors` 分模块组织 |
| 线程模型 | ROS1 topic 驱动，`trackThread`/`processThread`（`processMeasurements()`） | `MsFusionManager` 单例内 `runInsThread`（IMU 频率捷联惯导）+ `runFusionThread`（视觉触发滑窗优化），IMU/ODOM/Vision 三队列 + mutex/condition_variable 解耦 |
| ROS 版本/架构 | ROS1，三节点：`supervins_estimator`（前端+后端）、`supervins_loop_fusion`（回环）、`camera_models` | ROS2，`msf_core`（算法库）与 `ros_wrapper/ms_fusion`（硬件驱动/编排）、`ros_wrapper/ms_fusion_loop`（独立回环进程）分离 |

---

## 三、回环检测 / 位姿图优化

高度相似，都是"DBoW 词袋 + 学习特征描述子 + 独立位姿图优化节点"：

| 维度 | SuperVINS (`supervins_loop_fusion`) | ms_fusion (`ms_fusion_loop`) |
|---|---|---|
| 词袋库 | DBoW3（`ThirdParty/DBoW3`），用 SuperPoint 描述子专门训练词典 `superpoint1.yml.gz`；`pose_graph.h` 中 `Database dpl_db; Vocabulary dpl_voc;`，查询见 `pose_graph.cpp:389` | 同样 DBoW3，词典用 XFeat 描述子专门训练：`xfeat_voc_robot_0729.dbow3`、`xfeat_voc_stereoimu_0729.dbow3`（配套训练工具 `tools/train_xfeat_voc.cpp`，另有 `train_superpoint_voc.cpp`） |
| 几何验证 | 优先 LightGlue（`Matcher_DPL`），SuperPoint 数据不可用时 fallback 到传统 BRIEF（`computeBRIEFPoint()`/`searchByBRIEFDes()`） | `KeyFrame::findConnection` 用互近邻+Lowe ratio（`matchMutualNNLowe`）匹配 + `PnPRANSAC` 验证 |
| 位姿图优化 | 4DoF（`FourDOFError`，固定 pitch/roll，因重力可观）或 6DoF（`RelativeRTError`，视是否用 IMU），独立线程 `t_optimization` | 同源实现，`pose_graph.cpp`/`keyframe.cpp` 文件头仍带 HKUST VINS 版权声明，说明是直接继承 VINS-Fusion 回环模块而非重写 |

ms_fusion 的回环模块与 SuperVINS 相比，主要差异在于把词典训练对象从 SuperPoint 换成了 XFeat，几何验证阶段用互近邻+Lowe ratio 替代了 LightGlue/BRIEF 双通道方案，其余位姿图优化逻辑基本同源。

---

## 四、调试可视化的启示

提交 `a84da35`（`Drawer::drawStereoMatchDebug`，`vision/drawer.cpp`）把双目 MNN/LightGlue 匹配结果**按左图 track ID**（而非匹配结果数组下标）经 HSV hue 映射上色。原因：之前按下标伪随机上色，会因为每帧匹配成功数量变化导致颜色与物理特征点身份错位，看起来像"连线乱跳"；改为按 track ID 上色后，颜色与特征点身份绑定，才能跨帧目视核对同一点颜色是否稳定。这个经验对 SuperVINS 若做同类调试同样适用——SuperVINS 目前未见类似的按 ID 着色可视化。

---

## 五、值得借鉴 / 需谨慎的点

1. **关键帧策略**：可考虑把 SuperVINS 的"平均视差 + 跟踪点数量 + 新特征比例"判据叠加到当前的纯时间门控上，而非仅依赖 `keyframe_min_interval_s`。ms_fusion 左目特征数（~47）远低于 SuperVINS（~200~256），视差质量对三角化精度和后续优化收敛的影响更敏感，值得针对性验证。
2. **F-RANSAC 退化放行机制**：SuperVINS 在小运动场景下用历史内点率窗口判断 F 矩阵退化并放行高置信度 LightGlue 匹配，这个思路可以参考，但需要结合 ms_fusion 低特征数场景重新标定阈值，不能直接照搬 SuperVINS 的 0.5 匹配阈值或参数。
3. **按 ID 着色的调试可视化**经验可以反向输出给 SuperVINS 侧参考（如果需要在那边调试 LightGlue 匹配稳定性）。
4. **不建议**直接移植 SuperVINS 的 ONNX Runtime 推理链路或整套前端代码——两者硬件平台（GPU/CPU vs Axera AX650 NPU）和内存模型（标准 heap vs IVPS 零拷贝物理内存）差异太大，移植成本高于重新适配。

---

## 附：关键源码位置索引

**SuperVINS**
- 前端提取/匹配：`supervins_estimator/src/featureTracker/extractor_matcher_dpl.h/.cpp`、`feature_tracker_dpl.h/.cpp`
- IMU 因子：`supervins_estimator/src/factor/imu_factor.h`、`integration_base.h`
- 重投影因子：`supervins_estimator/src/factor/projectionTwoFrameOneCamFactor.h` 等三类
- 边缘化：`supervins_estimator/src/factor/marginalization_factor.h/.cpp`
- 特征管理：`supervins_estimator/src/estimator/feature_manager.cpp`
- 回环：`supervins_loop_fusion/src/pose_graph.h/.cpp`、`keyframe.cpp`

**ms_fusion**
- 前端检测：`msf_core/src/vision/xfeat_detector.h/.cpp`、`superpoint_detector.h/.cpp`、`npu_score_map_detector.h`、`feature_extractor.h/.cpp`
- 前端跟踪/匹配：`msf_core/src/vision/tracking.cpp`、`feature_matcher.h/.cpp`
- 硬件抽象：`msf_core/include/ms_fusion/common/ivps_frame_handle.h`、`ros_wrapper/src/ms_fusion/include/ms_fusion/raw_msf_frame.h`
- 重定位：`msf_core/src/vision/reloc.h/.cpp`
- 后端优化：`msf_core/src/optimization/optimizer.h/.cpp`、`msf_core/src/factors/vision_factor.h`
- 特征管理：`msf_core/src/manager/feature_manager.h/.cpp`
- 核心编排：`msf_core/src/manager/ms_fusion_manager.cpp`
- 回环：`ros_wrapper/ms_fusion_loop`（`pose_graph.cpp`/`keyframe.cpp`）
