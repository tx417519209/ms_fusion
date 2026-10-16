#include "ms_fusion/common/ins_mech.h"
#include "ms_fusion/common/runtime_log.h"
#include "ms_fusion/common/rotation.h"

#include <iostream>

namespace ms_fusion {

size_t InsMech::getInsWindowIndex(
    const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& window, double time) {
    // 返回时间大于输入的第一个索引

    if (window.empty() || (window.front().first.timestamp > time)
        || (window.back().first.timestamp <= time)) {
        return 0;
    }

    size_t index = 0;
    size_t sta = 0, end = window.size();

    // 二分法搜索
    int counts = 0;
    while (true) {
        auto mid = (sta + end) / 2;

        auto first = window[mid - 1].first.timestamp;
        auto second = window[mid].first.timestamp;

        if ((first <= time) && (time < second)) {
            index = mid;
            break;
        } else if (first > time) {
            end = mid;
        } else if (second <= time) {
            sta = mid;
        }

        // 65536 / 200 = 327.68
        if (counts++ > 15) {
            MSF_VLOG << "Failed to get ins window index at " << std::fixed << time << std::endl;
            break;
        }
    }

    return index;
}

// 获取里程计数据对应的索引号
size_t InsMech::getDequeIndex(const Eigen::aligned_deque<OdomMeasurement>& odom_deque, double time) {
    if (odom_deque.size() <= 1 || (odom_deque.front().timestamp > time) || (odom_deque.back().timestamp < time)) {
        return 0;
    }

    // 定义索引号
    size_t index = 0;
    size_t sta = 0, end = odom_deque.size();

    int counts = 0;
    while (true) {
        //计算中值
        auto mid = (sta + end) / 2;

        // 拿到中值前面一个时间戳和中值对应的时间戳
        auto first  = odom_deque[mid - 1].timestamp;
        auto second = odom_deque[mid].timestamp;

        if (first <= time && time < second) {
            index = mid;
            break;
        } else if (first > time) {
            // 在前面
            end = mid;
        } else if (second <= time) {
            // 在后面
            sta = mid;
        }

        if (counts++ > 15) {
            MSF_VLOG << "Failed to get odom deque index at " << std::fixed << time << std::endl;
            break;
        }
    }

    return index;
}

// IMU进行单次的机械编排
void InsMech::insMechanization(const IntegrationConfiguration& config, 
                               const ImuMeasurement& imu_pre,
                               const ImuMeasurement& imu_cur, IntegrationState& state) {
    // IMU零偏补偿
    ImuMeasurement imu_cur2, imu_pre2;
    imu_cur2.dtheta = imu_cur.dtheta - imu_cur.dt * state.bg;
    imu_cur2.dvel = imu_cur.dvel - imu_cur.dt * state.ba;
    imu_pre2.dtheta = imu_pre.dtheta - imu_pre.dt * state.bg;
    imu_pre2.dvel = imu_pre.dvel - imu_pre.dt * state.ba;

    double dt = imu_cur.dt;
    state.timestamp = imu_cur.timestamp;

    // 双子样
    Eigen::Vector3d dvfb = imu_cur2.dvel + 0.5 * imu_cur2.dtheta.cross(imu_cur2.dvel) + 
        1.0 / 12.0 * (imu_pre2.dtheta.cross(imu_cur2.dvel) + imu_pre2.dvel.cross(imu_cur2.dtheta));
    Eigen::Vector3d dtheta = imu_cur2.dtheta + 1.0 / 12.0 * imu_pre2.dtheta.cross(imu_cur2.dtheta);

    // 速度变化量
    Eigen::Vector3d dvel = state.q.toRotationMatrix() * dvfb + config.gravity * dt;

    // 姿态
    state.q *= Rotation::rotvec2quaternion(dtheta);
    state.q.normalize();

    // 前后历元平均速度计算位置
    state.p += dt * state.v + 0.5 * dt * dvel;
    // 最后更新速度
    state.v += dvel;
}

// 优化以后从状态开始重新开始机械编排
bool InsMech::redoInsMechanization(
    const IntegrationConfiguration& config, const IntegrationState& updated_state,
    size_t reserved_ins_num,
    Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_window) {
    // 最新的优化过的位姿
    auto state = updated_state;
    size_t index = getInsWindowIndex(ins_window, state.timestamp);

    if (index == 0) {
        MSF_ELOG << "Failed to get right index in mechanization" << std::endl;
        return false;
    }

    // 拿到状态前后的IMU历元
    ImuMeasurement imu0 = ins_window[index - 1].first;
    ImuMeasurement imu1 = ins_window[index].first;

    // 判断是否需要内插
    int isneed = isNeedInterpolation(imu0, imu1, state.timestamp);
    if (isneed == -1) {
        // 前一时刻状态为状态量
        insMechanization(config, imu0, imu1, state);
        ins_window[index].second = state;
    } else if (isneed == 1) {
        // 当前时刻状态为状态量
        state.timestamp = imu1.timestamp;
        ins_window[index].second = state;
    } else if (isneed == 2) {
        imuInterpolation(imu1, imu0, imu1, state.timestamp);
        insMechanization(config, imu0, imu1, state);
        ins_window[index].second = state;
    }

    // 仅更新IMU时间点的状态
    for (size_t k = index + 1; k < ins_window.size(); k++) {
        imu0 = imu1;

        imu1 = ins_window[k].first;
        insMechanization(config, imu0, imu1, state);
        ins_window[k].second = state;
    }

    // 移除过期的IMU历元
    if (index < reserved_ins_num) {
        return true;
    }

    // 计算INS要删除的IMU历元
    size_t counts = index - reserved_ins_num;
    for (size_t k = 0; k < counts; k++) {
        ins_window.pop_front();
    }
    return true;
}

// 判断IMU是否需要插值
int InsMech::isNeedInterpolation(const ImuMeasurement& imu0, const ImuMeasurement& imu1, double mid) {
    double time = mid;

    // 靠近整秒
    if (imu0.timestamp <= time && imu1.timestamp > time) {
        double dt = time - imu0.timestamp;

        // 前一个历元接近
        if (dt < MINIMUM_TIME_INTERVAL) {
            return -1;
        }

        // 后一个历元接近
        dt = imu1.timestamp - time;
        if (dt < MINIMUM_TIME_INTERVAL) {
            return 1;
        }

        // 需内插
        return 2;
    }

    return 0;
}

// 判断ODOM是否需要插值
int InsMech::odomIsNeedInterpolation(const OdomMeasurement& odom0, const OdomMeasurement& odom1, double mid) {
    // 拿到时间戳
    double time = mid;

    //时间戳应该在两个观测值中间
    if (odom0.timestamp <= time && odom1.timestamp > time) {
        // 计算时间戳与前一个观测值的时间差
        double dt = time - odom0.timestamp;

        // 判断是否靠近前一个历元
        if (dt < MINIMUM_TIME_INTERVAL) {
            return -1;
        }

        // 计算时间戳与后一个观测的时间差
        dt = odom1.timestamp - time;
        if (dt < MINIMUM_TIME_INTERVAL) {
            return 1;
        }

        // 返回2，表明需要插值
        return 2;
    }

    return 0;
}

// IMU进行插值
void InsMech::imuInterpolation(const ImuMeasurement& imu01, ImuMeasurement& imu00,
                               ImuMeasurement& imu11, double mid) {
    double time = mid;

    double scale = (imu01.timestamp - time) / imu01.dt;
    ImuMeasurement buff = imu01;

    imu00.timestamp = time;
    imu00.dt = buff.dt - (buff.timestamp - time);
    imu00.dtheta = buff.dtheta * (1 - scale);
    imu00.dvel = buff.dvel * (1 - scale);
    // am/wm 是瞬时读数，不是随 dt 累积的增量，拆分前后两段时没有"按比例分"这回
    // 事——两段本来就是同一个原始采样的产物，直接原样传下去。不写的话 imu00/
    // imu11 是局部新构造的 ImuMeasurement，am/wm 默认零向量，比力读数会变成
    // [0,0,0]，用它算均值/静止判据会把比力模长拉低（getImuMeasurementSeries
    // 在关键帧边界插值时正是这么构造 imu00 的）。
    imu00.am = buff.am;
    imu00.wm = buff.wm;

    imu11.timestamp = buff.timestamp;
    imu11.dt = buff.timestamp - time;
    imu11.dtheta = buff.dtheta * scale;
    imu11.dvel = buff.dvel * scale;
    imu11.am = buff.am;
    imu11.wm = buff.wm;
}

//ODOM进行插值
void InsMech::odomInterpolation(const OdomMeasurement& odom0, const OdomMeasurement& odom1,
                                OdomMeasurement& odom01, double mid) {
    // 赋值时间戳
    double time = mid;
    // 拿到前一个观测值的时间戳
    const double& time0 = odom0.timestamp;
    // 拿到后一个观测值的时间戳
    const double& time1 = odom1.timestamp;

    // 先求尺度1，后面观测值所占比例
    double scalar_0_to_mid = (time - time0) / (time1 - time0);
    // 再求尺度0，前面面观测值所占比例
    double scalar_mid_to_1 = 1.0 - scalar_0_to_mid;

    // 赋值插值观测值的时间戳
    odom01.timestamp = time;

    // 插值线速度
    odom01.vm = scalar_mid_to_1 * odom0.vm + scalar_0_to_mid * odom1.vm;
    // 插值角速度
    odom01.wm = scalar_mid_to_1 * odom0.wm + scalar_0_to_mid * odom1.wm;

    // 插值旋转
    // 拿到前一个状态的旋转
    const Eigen::Quaterniond q0 = odom0.orientation;
    // 拿到后一个状态的旋转
    const Eigen::Quaterniond q1 = odom1.orientation;
    // 求旋转的增量(q_o1_from_o0)
    Eigen::Quaterniond dq = q1.inverse() * q0;
    // 把旋转增量转化为等效旋转矢量
    Eigen::Vector3d rvec = Rotation::quaternion2rotvec(dq);
    
    // 等效旋转矢量乘以比例，会得到增量
    rvec *= scalar_0_to_mid;
    // 求出旋转增量
    dq = Rotation::rotvec2quaternion(rvec);
    // 赋值旋转值
    odom01.orientation = (odom0.orientation * dq.inverse()).normalized();

    // 插值位移
    // 求位移的增量
    Eigen::Vector3d dp = odom1.position - odom0.position;
    // 赋值位移
    odom01.position = odom0.position + dp * scalar_0_to_mid;
}

// 获取IMU观测值的系列
void InsMech::getImuMeasurementSeries(
    const Eigen::aligned_deque<std::pair<ImuMeasurement, IntegrationState>>& ins_window,
    double start, double end,
    Eigen::aligned_vector<ImuMeasurement>& imu_measurement_series) {
    imu_measurement_series.clear();
    if (ins_window.empty()) {
        return;
    }

    // 拿到起点时间和终点时间对应的索引。getInsWindowIndex 找不到时统一返回 0
    // （见其函数头注释），不是合法的"命中下标 0"——0 在这里只可能是失败哨兵值。
    // 下面马上就要用 is-1 取前一个采样、用 ie-1 做循环上界，size_t 是无符号类型，
    // 只要 is 或 ie 单独一个是 0，减 1 就会下溢成一个天文数字，循环/取址直接越过
    // ins_window 的实际范围读到未定义内存——原来的 (is==0 && ie==0) 只挡住了两个
    // 都失败的情况，任意一个单独为 0 时完全没拦住，是能实际触发的越界，之前一直
    // 在悄悄读垃圾内存而不是崩溃或报错。
    //
    // 现场不能因为这个直接把进程崩掉（这里没有 try/catch，异常会直接终止线程），
    // 所以不抛异常：只记日志，然后把下标夹到离目标最近的合法边界继续算——end
    // 追上/超过窗口最新采样最常见（视觉线程偶尔跑到 IMU 采集前面），退化成用
    // 已有的最新数据；start 早于窗口最早采样同理夹到最早的合法位置。宁可这一小段
    // 预积分少了一点点，也不能读越界内存或者把线程异常终止掉。
    size_t is = getInsWindowIndex(ins_window, start);
    size_t ie = getInsWindowIndex(ins_window, end);

    bool clamped = false;
    if (is == 0) {
        is = 1;
        clamped = true;
    }
    if (ie == 0) {
        ie = ins_window.size() - 1;
        clamped = true;
    }
    if (clamped) {
        static double last_warn_timestamp = 0.0;
        if (end - last_warn_timestamp >= 1.0 || end < last_warn_timestamp) {
            last_warn_timestamp = end;
            MSF_ELOG << "[InsMech] getImuMeasurementSeries index lookup failed, clamped to nearest "
                        "bound (start="
                     << std::fixed << start << ", end=" << end
                     << ", window=[" << ins_window.front().first.timestamp << ", "
                     << ins_window.back().first.timestamp << "])" << std::endl;
        }
    }

    // 窗口太短（只有 1 个采样）时夹完可能出现 is > ie；此时至少回退成"just" 用
    // 窗口里最新的这一个采样撑住返回值非空，调用方后面会直接 .at(0)，绝不能让它
    // 拿到空 vector
    if (is > ie) {
        ImuMeasurement fallback = ins_window.back().first;
        fallback.timestamp = end;
        imu_measurement_series.push_back(fallback);
        return;
    }

    // 夹过的边界，start/end 不再必然落在 [imu0.timestamp, imu1.timestamp) 里，
    // isNeedInterpolation 会返回 0（不在任何一个分支里，什么都不 push），下面
    // 走的插值逻辑对这种情况不成立——直接原样搬 [is-1, ie] 区间的采样，不做插值，
    // 保证非空、不越界，代价只是边界那一小截时间对不齐，可以接受
    if (clamped) {
        imu_measurement_series.clear();
        for (size_t k = is - 1; k <= ie; k++) {
            imu_measurement_series.push_back(ins_window[k].first);
        }
        imu_measurement_series.back().timestamp = end;
        return;
    }

    // 定义IMU观测值变量
    ImuMeasurement imu0, imu1, imu;
    imu_measurement_series.clear();

    // 拿到开始时间对应的两个IMU
    imu0 = ins_window[is - 1].first;
    imu1 = ins_window[is].first;

    // 判断是否需要插值
    int isneed = isNeedInterpolation(imu0, imu1, start);
    if (isneed == -1) {
        imu_measurement_series.push_back(imu0);
        imu_measurement_series.push_back(imu1);
    } else if (isneed == 1) {
        imu_measurement_series.push_back(imu1);
    } else if (isneed == 2) {
        imuInterpolation(imu1, imu, imu1, start);
        imu_measurement_series.push_back(imu);
        imu_measurement_series.push_back(imu1);
    }

    // 中间部分直接保存数据
    for (size_t k = is + 1; k < ie - 1; k++) {
        imu_measurement_series.push_back(ins_window[k].first);
    }

    // 最后一帧
    imu0 = ins_window[ie - 1].first;
    imu1 = ins_window[ie].first;

    // 判断是否需要插值
    isneed = isNeedInterpolation(imu0, imu1, end);
    if (isneed == -1) {
        imu_measurement_series.push_back(imu0);
    } else if (isneed == 1) {
        imu_measurement_series.push_back(imu0);
        imu_measurement_series.push_back(imu1);
    } else if (isneed == 2) {
        imu_measurement_series.push_back(imu0);
        imuInterpolation(imu1, imu, imu1, end);
        imu_measurement_series.push_back(imu);
    }

    // 显式的替换时间
    imu_measurement_series.back().timestamp = end;
}

void InsMech::getOdomMeasurementSeries(Eigen::aligned_deque<OdomMeasurement>& odom_deque,
                                       double start, double end,
                                       Eigen::aligned_vector<OdomMeasurement>& odom_measurement_series) {
    // 拿到时间戳在容器中对应的索引号
    size_t is = getDequeIndex(odom_deque, start);
    size_t ie = getDequeIndex(odom_deque, end);
    
    // 异常数据处理
    if ((is == 0) || (ie == 0)) {
        //抛出异常，要求程序不崩溃
        throw std::runtime_error("Failed to get right index in getDequeIndex");
        
        return;
    }

    // 定义里程计观测值
    OdomMeasurement odom0, odom1, odom;
    // 清理容器
    odom_measurement_series.clear();

    // 拿到开始时间对应的里程计观测值
    odom0 = odom_deque[is - 1];
    odom1 = odom_deque[is];

    // 判断是否需要插值
    int is_need = odomIsNeedInterpolation(odom0, odom1, start);
    if (is_need == -1) {
        // 靠近前一个历元，两个观测值都需要保存
        odom_measurement_series.push_back(odom0);
        odom_measurement_series.push_back(odom1);
    } else if (is_need == 1) {
        // 靠近后面一个历元，只保存后面一个观测值
        odom_measurement_series.push_back(odom1);
    } else if (is_need == 2) {
        // 轮速计进行插值
        odomInterpolation(odom0, odom1, odom, start);
        odom_measurement_series.push_back(odom);
        odom_measurement_series.push_back(odom1);
    }

    // 保存中间部分的数据
    for (size_t k = is + 1; k < ie - 1; k++) {
        odom_measurement_series.push_back(odom_deque[k]);
    }

    // 拿到最后的前面一帧和最后一帧的数据
    odom0 = odom_deque[ie - 1];
    odom1 = odom_deque[ie];

    // 判断是否需要插值
    is_need = odomIsNeedInterpolation(odom0, odom1, end);
    if (is_need == -1) {
        // 靠近前一个历元
        odom_measurement_series.push_back(odom0);
    } else if (is_need == 1) {
        // 靠近后一个历元
        odom_measurement_series.push_back(odom0);
        odom_measurement_series.push_back(odom1);
    } else if (is_need == 2) {
        // 需要插值
        odom_measurement_series.push_back(odom0);
        // 进行插值
        odomInterpolation(odom0, odom1, odom, end);
        // 保存插值的数据
        odom_measurement_series.push_back(odom);
    }

    // 显示的替换时间
    odom_measurement_series.back().timestamp = end;

    // 删除对应的里程计数据
    size_t counts = 0;
    if (is > 2) {
        counts = is - 2;
    }
    for (size_t k = 0; k < counts; k++) {
        odom_deque.pop_front();
    }
}

size_t InsMech::getStateDataIndex(const std::deque<double>& timelist, double time, double interval) {
    size_t index = 0;

    size_t sta = 0, end = timelist.size();
    int counts = 0;
    while (true) {
        auto mid = (sta + end) / 2;
        auto mid_time = timelist[mid];

        if (isTheSameTimeNode(mid_time, time, interval)) {
            index = mid;
            break;
        } else if (mid_time > time) {
            end = mid;
        } else if (mid_time < time) {
            sta = mid;
        }

        if (counts++ > 10) {
            MSF_ELOG << "Failed to get state data index at " << std::fixed << time << std::endl;
            break;
        }
    }

    return index;
}

bool InsMech::isTheSameTimeNode(double time0, double time1, double interval) {
    return fabs(time0 - time1) < interval;
}

Pose InsMech::stateToPoseWithExtrinsic(const IntegrationState& state, const Pose& T_imu_from_cam) {
    // 定义一个变换矩阵，用于
    Pose T_w_from_cam = Pose();

    // 拿到状态量
    Eigen::Matrix3d R_w_from_imu = state.q.toRotationMatrix();
    Eigen::Vector3d t_w_from_imu = state.p;

    // 拿到相机的外参
    const Eigen::Matrix3d& R_imu_from_cam = T_imu_from_cam.R;
    const Eigen::Vector3d& t_imu_from_cam = T_imu_from_cam.t;
    
    // 赋值
    T_w_from_cam.R = R_w_from_imu * R_imu_from_cam;
    T_w_from_cam.t = R_w_from_imu * t_imu_from_cam + t_w_from_imu;

    return T_w_from_cam;
}

} // namespace ms_fusion