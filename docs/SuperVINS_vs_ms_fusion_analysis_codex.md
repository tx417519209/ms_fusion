# SuperVINS 与 ms_fusion 前端/后端对比分析

## 结论

两者共同采用“学习型局部特征 + LightGlue + IMU 滑窗优化”的总体思路，但 SuperVINS 更接近经典 VINS-Fusion；当前 `ms_fusion` 是面向 AX650 实时部署的硬件化系统，在左目主轨迹、右目附属观测、轮速融合和关键帧策略上做了较大改造。

因此，两者可以互相借鉴算法思想，但不能直接照搬参数和数据流。

## 1. 前端共同点

| 项目 | SuperVINS | 当前 ms_fusion |
|---|---|---|
| 特征提取 | SuperPoint，代码也支持 DISK | 主要使用 XFeat，可选 SuperPoint |
| 特征匹配 | LightGlue，ONNX Runtime/CUDA | LightGlue AX650 NPU 模型，也有 MNN fallback |
| 时间匹配 | 上一帧左目到当前帧左目 | 上一帧左目到当前帧左目 |
| 立体匹配 | 当前左目到当前右目 | 当前左目到当前右目 |
| ID 管理 | 匹配成功继承左目 ID，新点分配新 ID | 同样是左目主 ID，右目挂在左目 ID 上 |
| 几何过滤 | LightGlue 后再做 F-RANSAC | temporal 使用运动门限和 RANSAC，stereo 使用对极约束 |
| 补点 | 匹配失败后通过 mask 补充新点 | 网格补点，补点后再做右目观测关联 |

SuperVINS 的 ID 继承逻辑见：

- `SuperVINS-main/supervins_estimator/src/featureTracker/feature_tracker_dpl.cpp:678-707`

当前项目对应逻辑见：

- `msf_core/src/vision/tracking.cpp:366-430`

## 2. 前端关键差异

### 2.1 特征数量和模型部署

SuperVINS 默认使用 SuperPoint，描述子维度为 256。其 LightGlue 代码会在特征数量不足 256 时进行 padding，以规避 GPU LightGlue 的小特征数问题。

当前 ms_fusion 使用 AX650 NPU 上的 XFeat，日志中每帧通常约有 47～48 个左目特征，使用适配 AX650 的 LightGlue 模型。当前方案更符合板端算力和实时性要求，但由于特征数量少，对匹配阈值、几何过滤和 ID 稳定性更敏感。

相关 SuperVINS 文件：

- `SuperVINS-main/supervins_estimator/src/featureTracker/extractor_matcher_dpl.cpp`

### 2.2 左右目处理顺序

SuperVINS 的流程基本是：

1. 提取左目特征；
2. 做左目 temporal 匹配；
3. 保留和补充左目点；
4. 提取右目特征；
5. 用最终左目点集和右目特征匹配。

对应代码：

- `SuperVINS-main/supervins_estimator/src/featureTracker/feature_tracker_dpl.cpp:554-556`
- `SuperVINS-main/supervins_estimator/src/featureTracker/feature_tracker_dpl.cpp:735-746`

当前 ms_fusion 会先检测左右目，再并行启动 temporal 和 stereo LightGlue，最后将 dense stereo 结果映射回左目最终轨迹：

- `msf_core/src/vision/tracking.cpp:351-389`
- `msf_core/src/vision/tracking.cpp:462-495`

当前方案吞吐更高，但 LightGlue 的 stereo 结果来自 dense 候选集，之后还要映射到 temporal 保留下来的左目集合。因此日志中的 `mutual`、`epi` 和最终 `stereo_matched` 不能简单使用同一个分母比较。

### 2.3 temporal 过滤策略

SuperVINS 的 F-RANSAC 有特殊的退化帧放行逻辑：

- LightGlue 平均置信度较高；
- 当前 F-RANSAC 内点率明显低于历史值；
- 认为可能是小运动导致 F 矩阵退化；
- 这时接受全部 LightGlue 匹配。

当前 ms_fusion 的 LightGlue temporal 路径是：

1. LightGlue 匹配；
2. 最大运动距离限制；
3. RANSAC 过滤；
4. 失败点丢弃；
5. 后续网格补点并分配新 ID。

对应当前代码：

- `msf_core/src/vision/feature_matcher.cpp:447-481`

所以两边的“匹配成功率”含义并不完全相同。SuperVINS 更偏向保留高置信度轨迹，当前项目更偏向保证几何一致性和实时稳定性。

## 3. 立体匹配差异

SuperVINS 立体匹配后，直接把右目点绑定到同一个左目 `feature_id`：

- `SuperVINS-main/supervins_estimator/src/featureTracker/feature_tracker_dpl.cpp:739-746`

当前 ms_fusion 也是这个语义，但右目只是左目轨迹的附属观测，不维护独立的右目长期轨迹：

- `msf_core/src/vision/tracking.cpp:504-524`

当前 stereo LightGlue 的最终接受条件主要是对极约束：

- `msf_core/src/vision/feature_matcher.cpp:663-709`

当前日志中常见：

```text
left features       ≈ 47
stereo matched      ≈ 4.5～5.5
```

因此可视化中立体连线频繁变化，通常不是右目 ID 独立漂移，而是以下因素叠加：

- dense 左右候选每帧重新检测；
- LightGlue 匹配结果可能替换；
- 只有通过对极约束的匹配才进入最终右目观测；
- 如果可视化按数组下标而不是 landmark ID 着色，匹配替换会被放大成明显的连线跳变。

## 4. 后端共同点

两者后端都具有典型 VINS-Fusion 结构：

- IMU 预积分；
- 特征管理器；
- 视觉重投影因子；
- Ceres 非线性优化；
- 固定滑窗；
- 特征三角化；
- 边缘化；
- 初始化和陀螺仪 bias 估计。

SuperVINS 在双目情况下优先使用同一时刻左右目观测三角化；没有立体观测时，再使用跨帧左目观测三角化：

- `SuperVINS-main/supervins_estimator/src/estimator/feature_manager.cpp:309-380`

当前 ms_fusion 也采用双目/跨帧三角化思想，但额外融合了：

- 轮速里程计预积分；
- wheel odometry factor；
- AX650 IVPS 共享内存图像输入；
- matched point cloud；
- ROS2 发布和实时调度。

因此当前后端不是纯 VIO，而是 VIO、轮速等功能的融合系统。

## 5. 后端最重要的差异：关键帧策略

### 5.1 SuperVINS：每帧进入后端，按视差决定关键帧

SuperVINS 的 `addFeatureCheckParallax()` 会先把当前帧的特征观测加入 `FeatureManager`，再根据以下条件判断关键帧：

- 长期跟踪点数量；
- 新特征比例；
- 平均视差。

对应代码：

- `SuperVINS-main/supervins_estimator/src/estimator/feature_manager.cpp:52-121`

非关键帧仍然进入 estimator，只是在滑窗时边缘化第二新的帧：

- `SuperVINS-main/supervins_estimator/src/estimator/estimator.cpp:548-559`

### 5.2 当前 ms_fusion：先用时间间隔过滤

当前 `FeatureManager::tryAddKeyframe()` 实际上主要判断：

```text
当前时间 - 上一个关键帧时间 >= keyframe_min_interval_s
```

对应代码：

- `msf_core/src/manager/feature_manager.cpp:12-58`

被拒绝的帧不会进入后续优化，而是直接记录为 backend rejected：

- `msf_core/src/manager/ms_fusion_manager.cpp:779-805`

因此当前日志中的：

```text
accept_pct=20%
```

本质表示“时间门控下被接受为关键帧的比例”，不是 LightGlue 匹配成功率，也不是 Ceres 优化成功率。

这也解释了为什么将 `keyframe_min_interval_s` 从约 `0.167` 调到 `0.1` 或 `0.125` 后，`accept_pct` 会明显变化。

## 6. 两种系统的优缺点

### SuperVINS 的优点

- 后端关键帧选择依据平均视差，更符合 VINS-Fusion 的几何语义；
- 每帧观测先进入 feature manager，非关键帧也不会完全丢失；
- SuperPoint 256 维特征通常具有较强描述能力；
- 右目观测与左目 ID 的绑定关系清晰；
- 有独立的 DBoW3 + SuperPoint/LightGlue 回环验证模块。

### SuperVINS 的不足

- 依赖 GPU/ONNX Runtime，不适合直接搬到 AX650；
- LightGlue 输入点数不足时需要 padding；
- 每帧存在较多日志和文件写入，不适合直接作为板端实时版本；
- 主要是 ROS1 + VINS-Fusion 风格，和当前 ROS2/IVPS/NPU 架构不同；
- F-RANSAC 放行策略可能提高轨迹连续性，但也可能保留错误匹配。

### 当前 ms_fusion 的优点

- 左目主跟踪、右目附属观测语义明确；
- XFeat/LightGlue 已适配 AX650 NPU；
- IVPS 共享内存和 NPU 推理链路适合实时运行；
- 具备轮速、激光点云等工程化能力；
- 当前日志显示前端约 30 FPS，后端关键帧约 6～7 Hz，队列基本没有严重堆积。

### 当前 ms_fusion 的主要风险

- 关键帧主要按时间，而不是按实际视差决定；
- 右目有效匹配数量约 5 个，双目深度约束偏弱；
- 左目特征数约 47 个，低于 SuperVINS 典型的 200～256 点规模；
- temporal/stereo 使用 dense 候选和最终轨迹两套集合，统计口径容易混淆；
- 可视化若按数组索引着色，会把正常的匹配替换放大成“连线乱跳”。

## 7. 最值得借鉴的部分

不建议直接移植 SuperVINS 的整套前端，优先考虑借鉴两个机制：

1. 使用“平均视差 + 最小时间间隔”联合决定关键帧，而不是只使用时间间隔；
2. 对高置信度 LightGlue 匹配，在 F/RANSAC 退化时增加退化检测和有限放行策略。

这两项都需要结合当前 AX650 的低特征数、轮速约束和实时预算验证，不能直接照搬 SuperVINS 的 `0.5` 匹配阈值或 `256` 点输入策略。
