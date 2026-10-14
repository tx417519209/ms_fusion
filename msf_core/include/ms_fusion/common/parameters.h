#ifndef MS_FUSION_PARAMETERS_H
#define MS_FUSION_PARAMETERS_H

#include "eigen_utils.hpp"
#include "config_params.h"

#include <opencv2/core/mat.hpp>

#include <mutex>

namespace ms_fusion {

// ################################## data struct ##################################
// 定义全局变量
static constexpr int NUM_LINES_CLASS = 15;

// 位姿结构体
struct Pose {
    // 定义构造函数
    Pose() = default;
    Pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) : R(R), t(t) {}

    // 求位姿的逆
    Pose inverse() const {
        Pose pose;
        pose.R = R.transpose();
        pose.t = -R.transpose() * t;
        return pose;
    }

    // 求位姿的乘积
    Pose operator*(const Pose& other) const {
        Pose pose;
        pose.R = R * other.R;
        pose.t = R * other.t + t;
        return pose;
    }

    // 旋转矩阵
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    // 平移向量
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 相机外参
struct CameraExtrinsicParams {
    // 左相机到右相机的变化矩阵
    Pose T_cam1_from_cam0;

    // 右相机到左相机的变化矩阵
    Pose T_cam0_from_cam1;

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 相机和IMU的外参
struct CameraImuExtrinsicParams {
    // 左相机到IMU的变换矩阵
    Pose T_imu_from_cam0;

    // 右相机到IMU的变换矩阵
    Pose T_imu_from_cam1;

    // IMU到左相机的变换矩阵
    Pose T_cam0_from_imu;

    // IMU到右相机的变换矩阵
    Pose T_cam1_from_imu;

    // EIGEN的数据对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 框转线的数据结构体
struct ObbLine {
    // 检测线的得分
    float score = 0.f;

    // 检测线的类别
    int class_id = -1;

    // 检测线的起始坐标
    cv::Point2f start_point = cv::Point2f();
    cv::Point2f end_point = cv::Point2f();
};

// 线表达式的结构体
struct Line {
    // 线的ID号
    int class_id = -1;

    // 线的表达式
    float a = 0.f;
    float b = 0.f;
    float c = 0.f;
};

// 线交点信息的结构体
struct IntersectionMapping {
    int cat_a;  // 第一条线的类别索引
    int cat_b;  // 第二条线的类别索引
    int kp_idx; // 关键点索引 (0-11对应KP1-KP12)
};

// 线交点的映射
const IntersectionMapping INTERSECTION_LINES2KEYPOINTS[] = {
    // {1, 5, 0},       // KP0
    {1, 10, 0},      // KP0
    // {1, 6, 1},       // KP1
    {1, 11, 1},      // KP1
    // {1, 7, 2},       // KP2
    {1, 12, 2},      // KP2
    // {1, 8, 3},       // KP3
    {1, 13, 3},      // KP3
    // {1, 9, 4},       // KP4
    {1, 14, 4},      // KP4
    {3, 6, 5},       // KP5
    {3, 7, 6},       // KP6
    {3, 8, 7},       // KP7
    {4, 5, 8},       // KP8
    {4, 6, 9},       // KP9
    {4, 8, 10},      // KP10
    {4, 9, 11},      // KP11
};

// 求出所有线交点的数量
const int NUM_INTERSECTION_MAPPINGS = 
    sizeof(INTERSECTION_LINES2KEYPOINTS) / sizeof(IntersectionMapping);

// ################################## 设置全局参数 ##################################
// 图片的高和宽
extern int image_height;
extern int image_width;
// 相机的焦距
extern double focal_length;
// 真实IMU坐标系到虚拟IMU坐标系（前-右-下）的变换矩阵
extern Pose T_virtual_imu_from_real_imu;
// 左相机和右相机的外参
extern CameraExtrinsicParams camera_extrinsic_params;
// 左相机和右相机到IMU的外参
extern CameraImuExtrinsicParams camera_imu_extrinsic_params;
// 设置互斥锁
extern std::mutex camera_imu_extrinsic_params_mutex;

// ################################## 定义一个静态类，用于管理参数 #####################
class ParametersManager {
public:
    // 设置全局参数
    static void readParameters(const MsFusionConfigParams& config_params);

    // 更新相机和IMU的外参
    static void updateCameraImuExtrinsicParams(const Pose& T_imu_from_cam0);
};


} // namespace ms_fusion

#endif // MS_FUSION_PARAMETERS_H