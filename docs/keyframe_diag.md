# 逐关键帧优化诊断（optimizer.keyframe_diag）

只读诊断，默认关闭（`optimizer.keyframe_diag: false`），不改变估计结果。用来定位“求解停滞→大跳”
这类问题，例如旋转雅可比符号（见 `docs/preintegration_rot_jacobian_sign.md`）。

```yaml
optimizer:
  keyframe_diag: true
  keyframe_diag_start_s: 0   # 绝对传感器时间戳；<=0 不限
  keyframe_diag_end_s: 0
```

窗口用的是与 `timestamp_list` 同一时钟的绝对传感器时间戳，不是“从首条 IMU 起的秒数”。
TUM-VI 要看数据第 `S` 秒，就填 `t0 + S`（`t0` 为首条 IMU 时间）。整条序列都开时日志量约
每关键帧两行，几分钟的序列可接受。开启后日志输出耗时仍会进入关键帧间隔反馈，性能评测不要开。

## 日志

每个关键帧两行，按 `t_kf` 关联：

- `[kf-diag]`：求解前 / 求解后 / 一致性检查后三段 p/q/v，`solve_shift`、`consist_gauge_shift`
  （只是坐标对齐，不是修正）、`final_shift`；Ceres 终止细节（`cost` / `dcost` / 终止原因 /
  成功与失败步数 / 首末梯度 / 信赖域半径）；最新 IMU 区间的旋转残差与 σ 构成；各因子残差
  尝试/成功/失败/非有限计数；视觉 Huber 二次区占比；视觉准入计数（含单目降级原因、
  未三角化的分类）；`qsign_mismatch`：窗口中预积分区间的状态四元数符号与矩阵恢复四元数
  符号不一致的个数（求解前 / 后）。
- `[kf-diag-redo]`：INS 重传播前后同一 IMU 时刻的 p/q/v、`ts_match`、`redo_ok`、`dp / dv / dR`。

## 解析

日志按上面的字段逐行输出，可用 `grep '\[kf-diag'` 按 `t_kf` 关联两种行后自行统计。
判读要点：`steps_bad` 连续偏大且 `g_last` 很大是“冻住”；`final_shift` 里的大跳要看
`redo_ok / ts_match` 是否正常，再区分求解里产生还是坐标对齐。
