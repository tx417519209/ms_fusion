#include "config_params.h"
#include "eigen_utils.hpp"
#include "ms_fusion/common/runtime_log.h"

#include <bitset>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace ms_fusion {

namespace {

void resolvePathRelativeToConfigFile(std::string& path, const std::string& config_file) {
    if (path.empty() || path.front() == '/') {
        return;
    }
    const std::filesystem::path base =
        std::filesystem::path(config_file).parent_path();
    const std::filesystem::path resolved = base / path;
    if (!std::filesystem::exists(resolved)) {
        throw std::runtime_error(
            "Path not found (relative to config dir " + base.string() + "): " + path +
            ". If this is a reloc/tracking model, run colcon build --packages-select ms_fusion "
            "so .axmodel files are installed to share/ms_fusion/models/.");
    }
    path = std::filesystem::absolute(resolved).lexically_normal().string();
}

std::uint32_t parseNpuAffinity(const YAML::Node& node, const std::string& config_path) {
    const std::uint32_t npu_set = node.as<std::uint32_t>();
    if (npu_set == 0 || (npu_set & ~0b111u) != 0) {
        throw std::runtime_error(config_path + " must be an NPU bitmask in range 1..7");
    }
    return npu_set;
}

} // namespace

namespace {

std::string normalizeImageEnum(std::string value) {
    for (auto& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (value == "compressed" || value == "raw") return value;
    throw std::runtime_error("base.image_type must be compressed|raw, got: " + value);
}

} // namespace

std::string normalizeImageType(const std::string& value) {
    return normalizeImageEnum(value);
}

// 读取系统的配置参数配置参数
void MsFusionConfigParams::loadConfigFromYAML(const std::string config_file) {
    // 定义一个节点变量
    YAML::Node node = YAML::Node();

    // 读取配置参数
    try {
        // 读取主节点
        node = YAML::LoadFile(config_file);

        // 读取基础的配置参数
        if (node["base"]) {
            // 读取base节点
            YAML::Node base_node = node["base"];

            // 读取base节点对应的配置参数
            base_config_params_.loadConfigFromYAML(base_node);
            RuntimeLog::configure(base_config_params_.runtimeLogConfig());
        } else {
            throw std::runtime_error("base section not found in config file");
        }

        // 读取IMU的配置参数
        if (node["imu"]) {
            // 读取imu节点
            YAML::Node imu_node = node["imu"];

            // 读取imu节点对应的配置参数
            imu_config_params_.loadConfigFromYAML(imu_node);
        } else {
            throw std::runtime_error("imu section not found in config file");
        }

        // 读取ODOM的配置参数
        if (node["odom"]) {
            // 读取odom节点
            YAML::Node odom_node = node["odom"];

            // 读取odom节点对应的配置参数
            odom_config_params_.loadConfigFromYaml(odom_node);
        } else {
            throw std::runtime_error("odom section not found in config file");
        }

        // 读取特征点提取跟踪的配置参数
        if (node["tracking"]) {
            // 读取tracking节点
            YAML::Node tracking_node = node["tracking"];

            // 读取tracking节点对应的配置参数
            tracking_config_params_.loadConfigFromYAML(tracking_node);
        } else {
            throw std::runtime_error("tracking section not found in config file");
        }

        // 读取相机内外参的配置参数
        if (node["camera"]) {
            // 读取camera节点
            YAML::Node camera_node = node["camera"];

            // 读取camera节点对应的配置参数
            camera_config_params_.loadConfigFromYAML(camera_node);
        } else {
            throw std::runtime_error("camera section not found in config file");
        }

        // 读取优化器的配置参数
        if (node["optimizer"]) {
            // 读取optimizer节点
            YAML::Node optimizer_node = node["optimizer"];

            // 读取optimizer节点对应的配置参数
            optimizer_config_params_.loadConfigFromYAML(optimizer_node);
        } else {
            throw std::runtime_error("optimizer section not found in config file");
        }

        if (tracking_config_params_.feature_detector == FeatureDetectorType::SUPERPOINT) {
            resolvePathRelativeToConfigFile(
                tracking_config_params_.superpoint_config_params.model_path, config_file);
        }
        if (tracking_config_params_.feature_detector == FeatureDetectorType::XFEAT) {
            resolvePathRelativeToConfigFile(
                tracking_config_params_.xfeat_config_params.model_path, config_file);
        }
        if (tracking_config_params_.use_lightglue &&
            !tracking_config_params_.lightglue_model_path.empty()) {
            resolvePathRelativeToConfigFile(
                tracking_config_params_.lightglue_model_path, config_file);
        }

        // 读取静止检测的配置参数（整段可选，不存在就用代码默认值）
        if (node["static_detector"]) {
            static_detector_config_params_.loadConfigFromYAML(node["static_detector"]);
        }

        // 读取 Initializer 的配置参数（整段可选，不存在就用代码默认值）
        if (node["initializer"]) {
            initializer_config_params_.loadConfigFromYAML(node["initializer"]);
        }

        // 读取重定位的配置参数
        if (node["reloc"]) {
            // 读取reloc节点
            YAML::Node reloc_node = node["reloc"];

            // 读取reloc节点对应的配置参数
            reloc_config_params_.loadConfigFromYAML(reloc_node);
            resolvePathRelativeToConfigFile(reloc_config_params_.model_path, config_file);
        } else {
            throw std::runtime_error("reloc section not found in config file");
        }

        // loop.enable：决定周期重定位触发后，绝对修正落到 Loop 输出层还是
        // 直接对齐融合后端（见 MsFusionManager::applyRelocCorrectionLocked）
        if (node["loop"] && node["loop"]["enable"]) {
            loop_enabled_ = node["loop"]["enable"].as<bool>();
        }

        // loop.max_correction_position_m/max_correction_yaw_deg：直接硬对齐路径
        // （use_loop_route==false）的输出跳变验收上限，跟 ms_fusion_loop 侧
        // PoseGraph::evaluateRelocJump 复用同一份配置（语义等价：两边比较的都是
        // "没有额外未回写修正"的当前最终状态），不在 reloc 段单独配置一份。
        if (node["loop"] && node["loop"]["max_correction_position_m"]) {
            reloc_config_params_.max_correction_position_m =
                node["loop"]["max_correction_position_m"].as<double>();
        }
        if (!std::isfinite(reloc_config_params_.max_correction_position_m)) {
            throw std::runtime_error("loop.max_correction_position_m must be finite");
        }
        if (node["loop"] && node["loop"]["max_correction_yaw_deg"]) {
            reloc_config_params_.max_correction_yaw_deg =
                node["loop"]["max_correction_yaw_deg"].as<double>();
        }
        if (!std::isfinite(reloc_config_params_.max_correction_yaw_deg)) {
            throw std::runtime_error("loop.max_correction_yaw_deg must be finite");
        }

        // loop.enable=false 时没有周期 reloc 的稳态落地点（原滑窗软约束已
        // 删除），Manager 会忽略 periodic_reloc_period_s，不派发周期推理。
        // 保留启动告警，避免现场误以为配置了正周期就会持续纠偏。
        if (!loop_enabled_ && reloc_config_params_.enable &&
            reloc_config_params_.periodic_reloc_period_s > 0.0) {
            MSF_ILOG << "[WARN] loop.enable=false 且 reloc.periodic_reloc_period_s > 0："
                     << "没有 Loop 输出层且旧的滑窗软约束已删除，周期重定位将被忽略，"
                     << "不会派发 YOLO/PnP；初始化和手动 /reloc 不受影响。"
                     << std::endl;
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load config from YAML file: ") + e.what());
    }
};

// 打印对应的配置参数
void MsFusionConfigParams::printConfig() const {
    if (!RuntimeLog::printConfigOnStartup()) {
        return;
    }

    // 打印基础的配置参数
    base_config_params_.printConfig();

    // 打印IMU的配置参数
    imu_config_params_.printConfig();

    // 打印Odom的配置参数
    odom_config_params_.printConfig();

    // 打印特征点提取跟踪的配置参数
    tracking_config_params_.printConfig();

    // 打印相机内外参的配置参数
    camera_config_params_.printConfig();

    // 打印优化器的配置参数
    optimizer_config_params_.printConfig();

    // 打印静止检测的配置参数
    static_detector_config_params_.printConfig();

    // 打印 Initializer 的配置参数
    initializer_config_params_.printConfig();

    // 打印重定位的配置参数
    reloc_config_params_.printConfig();

    MSF_ILOG << "loop.enable: " << (loop_enabled_ ? "true" : "false") << std::endl;
}

// 读取基础的配置参数
void BaseConfigParams::loadConfigFromYAML(const YAML::Node& base_node) {
    try {
        // [2] 读取vision_realtime_force
        if (base_node["vision_realtime_force"]) {
            vision_realtime_force = base_node["vision_realtime_force"].as<bool>();
        } else {
            throw std::runtime_error("vision_realtime_force not found in config file");
        }

        image_type = base_node["image_type"]
                         ? normalizeImageType(base_node["image_type"].as<std::string>())
                         : "compressed";

        log_profile = LogProfile::RELEASE;
        verbose_runtime_log = false;
        enable_timing_stats = false;
        print_config_on_startup = false;
        if (base_node["log_profile"]) {
            log_profile = parseLogProfile(base_node["log_profile"].as<std::string>());
        }
        if (log_profile == LogProfile::DEBUG) {
            verbose_runtime_log = true;
            enable_timing_stats = true;
            print_config_on_startup = true;
        } else {
            enable_timing_stats = true;
        }
        if (base_node["verbose_runtime_log"]) {
            verbose_runtime_log = base_node["verbose_runtime_log"].as<bool>();
        }
        if (base_node["enable_timing_stats"]) {
            enable_timing_stats = base_node["enable_timing_stats"].as<bool>();
        }
        if (base_node["print_config_on_startup"]) {
            print_config_on_startup = base_node["print_config_on_startup"].as<bool>();
        }

        odom_publish_stride = 1;
        if (base_node["odom_publish_stride"]) {
            odom_publish_stride = base_node["odom_publish_stride"].as<int>();
        }
        if (odom_publish_stride < 1) {
            odom_publish_stride = 1;
        }
        odometry_topic = base_node["odometry_topic"]
                             ? base_node["odometry_topic"].as<std::string>()
                             : "/odometry_vio";
        if (odometry_topic.empty()) {
            throw std::runtime_error("base.odometry_topic must not be empty");
        }

        auto normalize_qos = [](std::string value, const char* key) {
            for (auto& c : value) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (value != "best_effort" && value != "reliable") {
                throw std::runtime_error(std::string("base.") + key +
                                         " must be 'best_effort' or 'reliable', got: " + value);
            }
            return value;
        };

        qos = "best_effort";
        if (base_node["qos"]) {
            qos = normalize_qos(base_node["qos"].as<std::string>(), "qos");
        }
        odometry_qos = qos;
        if (base_node["odometry_qos"]) {
            odometry_qos = normalize_qos(base_node["odometry_qos"].as<std::string>(), "odometry_qos");
        }
        odometry_qos_depth = base_node["odometry_qos_depth"]
                                 ? base_node["odometry_qos_depth"].as<int>()
                                 : 10;
        if (odometry_qos_depth < 1) {
            throw std::runtime_error("base.odometry_qos_depth must be >= 1");
        }
        tf_qos = qos;
        if (base_node["tf_qos"]) {
            tf_qos = normalize_qos(base_node["tf_qos"].as<std::string>(), "tf_qos");
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load base config from YAML file: ") + e.what());
    }
}

RuntimeLogConfig BaseConfigParams::runtimeLogConfig() const {
    RuntimeLogConfig config;
    config.profile = log_profile;
    config.verbose_runtime_log = verbose_runtime_log;
    config.enable_timing_stats = enable_timing_stats;
    config.print_config_on_startup = print_config_on_startup;
    return config;
}

// 打印对应的配置参数
void BaseConfigParams::printConfig() const {

    // 打印vision_realtime_force
    MSF_ILOG << "vision_realtime_force: " << (vision_realtime_force ? "true" : "false") << std::endl;

    MSF_ILOG << "image_type: " << image_type
             << " (compressed|raw; MSF_USE_AXERA_IVPS=OFF only)" << std::endl;

    MSF_ILOG << "odom_publish_stride: " << odom_publish_stride
             << " (publish odometry every N IMU frames)" << std::endl;
    MSF_ILOG << "odometry_topic: " << odometry_topic << std::endl;
    MSF_ILOG << "qos: " << qos << " (default pub reliability)" << std::endl;
    MSF_ILOG << "odometry_qos: " << odometry_qos
             << " depth=" << odometry_qos_depth << std::endl;
    MSF_ILOG << "tf_qos: " << tf_qos << std::endl;

    MSF_ILOG << "log_profile: "
             << (log_profile == LogProfile::DEBUG ? "debug" : "release") << std::endl;
    MSF_ILOG << "verbose_runtime_log: " << (verbose_runtime_log ? "true" : "false") << std::endl;
    MSF_ILOG << "enable_timing_stats: " << (enable_timing_stats ? "true" : "false") << std::endl;
    MSF_ILOG << "print_config_on_startup: "
             << (print_config_on_startup ? "true" : "false") << std::endl;
}

// 读取IMU的配置参数
void ImuConfigParams::loadConfigFromYAML(const YAML::Node& imu_node) {
    try {
        // [2] 读取角度随机游走参数(deg/sqrt(hr))
        if (imu_node["arw"]) {
            arw = imu_node["arw"].as<double>();
        } else {
            throw std::runtime_error("arw not found in config file");
        }
    
        // [3] 读取速度随机游走参数(m/s/sqrt(hr))
        if (imu_node["vrw"]) {
            vrw = imu_node["vrw"].as<double>();
        } else {
            throw std::runtime_error("vrw not found in config file");
        }
    
        // 读取陀螺仪bias的标准差(deg/hr)
        if (imu_node["gb_std"]) {
            gb_std = imu_node["gb_std"].as<double>();
        } else {
            throw std::runtime_error("gb_std not found in config file");
        }
    
        // 读取加速度计bias的标准差(mGal)
        if (imu_node["ab_std"]) {
            ab_std = imu_node["ab_std"].as<double>();
        } else {
            throw std::runtime_error("ab_std not found in config file");
        }
    
        // 读取时间相关性(hr)
        if (imu_node["corr_time"]) {
            corr_time = imu_node["corr_time"].as<double>();
        } else {
            throw std::runtime_error("corr_time not found in config file");
        }   

        // [4] 读取当地重力
        if (imu_node["gravity"]) {
            gravity = imu_node["gravity"].as<double>();
        } else {
            throw std::runtime_error("gravity not found in config file");
        }

        // [6] 读取use_median
        if (imu_node["use_median"]) {
            use_median = imu_node["use_median"].as<bool>();
        } else {    
            throw std::runtime_error("use_median not found in config file");
        }
    
        // 读取真实IMU坐标系到虚拟IMU坐标系（前-右-下）的变换矩阵
        if (imu_node["T_virtual_imu_from_real_imu"]) {
            T_virtual_imu_from_real_imu = 
                ConfigUtils::readEigenMatrix4dFromYAML(imu_node["T_virtual_imu_from_real_imu"]);
        } else {
            throw std::runtime_error("T_virtual_imu_from_real_imu not found in config file");
        }

        // base_link → 真实 IMU（发布 world←base_link 用；缺省保留结构体内默认值）
        if (imu_node["T_imu_from_base"]) {
            T_imu_from_base =
                ConfigUtils::readEigenMatrix4dFromYAML(imu_node["T_imu_from_base"]);
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load imu config from YAML file: ") + e.what());
    }
}

// 打印对应的配置参数
void ImuConfigParams::printConfig() const {
    // 打印arw
    MSF_ILOG << "arw: " << arw << " (deg/sqrt(hr))" << std::endl;

    // 打印vrw
    MSF_ILOG << "vrw: " << vrw << " (m/s/sqrt(hr))" << std::endl;

    // 打印gb_std
    MSF_ILOG << "gb_std: " << gb_std << " (deg/hr)" << std::endl;

    // 打印ab_std
    MSF_ILOG << "ab_std: " << ab_std << " (mGal)" << std::endl;

    // 打印corr_time
    MSF_ILOG << "corr_time: " << corr_time << " (hr)" << std::endl;

    // 打印gravity
    MSF_ILOG << "gravity: " << gravity << " (m/s^2)" << std::endl;

    // 打印use_median
    MSF_ILOG << "use_median: " << (use_median ? "true" : "false") << std::endl;

    // 打印T_virtual_imu_from_real_imu
    MSF_ILOG << "T_virtual_imu_from_real_imu: \n" << T_virtual_imu_from_real_imu << std::endl;

    // 打印T_imu_from_base
    MSF_ILOG << "T_imu_from_base: \n" << T_imu_from_base << std::endl;
}

// 读取里程计相关的参数
void OdomConfigParams::loadConfigFromYaml(const YAML::Node& odom_node) {
    try {
        // 读取轮速计开关（缺省为 true，兼容旧配置）
        if (odom_node["enable"]) {
            enable = odom_node["enable"].as<bool>();
        }

        // 读取x轴角速度的标准差
        if (odom_node["w_x_std"]) {
            w_x_std = odom_node["w_x_std"].as<double>();
        } else {
            throw std::runtime_error("w_x_std not found in config file");
        }

        // 读取y轴角速度的标准差
        if (odom_node["w_y_std"]) {
            w_y_std = odom_node["w_y_std"].as<double>();
        } else {
            throw std::runtime_error("w_y_std not found in config file");
        }

        // 读取z轴角速度的标准差
        if (odom_node["w_z_std"]) {
            w_z_std = odom_node["w_z_std"].as<double>();
        } else {
            throw std::runtime_error("w_z_std not found in config file");
        }

        // 读取x轴线速度的标准差
        if (odom_node["v_x_std"]) {
            v_x_std = odom_node["v_x_std"].as<double>();
        } else {
            throw std::runtime_error("v_x_std not found in config file");
        }

        // 读取x轴线速度的标准差
        if (odom_node["v_y_std"]) {
            v_y_std = odom_node["v_y_std"].as<double>();
        } else {
            throw std::runtime_error("v_y_std not found in config file");
        }

        // 读取x轴线速度的标准差
        if (odom_node["v_z_std"]) {
            v_z_std = odom_node["v_z_std"].as<double>();
        } else {
            throw std::runtime_error("v_z_std not found in config file");
        }

        // 读取外参
        if (odom_node["T_imu_from_odom"]) {
            T_imu_from_odom = ConfigUtils::readEigenMatrix4dFromYAML(odom_node["T_imu_from_odom"]);
        } else {
            throw std::runtime_error("T_imu_from_odom not found in config file");
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load odom config from YAML file: ") + e.what());
    }
}

void OdomConfigParams::printConfig() const {
    MSF_ILOG << "enable: " << (enable ? "true" : "false") << std::endl;
    // 打印角速度标准差的值
    MSF_ILOG << "w_std: " << w_x_std << ", " << w_y_std << ", " << w_z_std << std::endl;
    // 打印线速度标准差的值
    MSF_ILOG << "v_std: " << v_x_std << ", " << v_y_std << ", " << v_z_std << std::endl;
    // 打印外参
    MSF_ILOG <<"T_imu_from_odom: \n" << T_imu_from_odom << std::endl;
}

// 读取相机的参数
void CameraConfigParams::loadConfigFromYAML(const YAML::Node& camera_node) {
    try {
        // 遍历相机节点（现在考虑的是双目相机）
        for (int i = 0; i < 2; i++) {
            // 获取相机节点的名称，通过索引构造(cam0, cam1)
            std::string cam_name = "cam" + std::to_string(i);

            // 获取相机节点的配置参数
            YAML::Node cam_node = camera_node[cam_name];

            // 判断节点是否存在
            if (cam_node) {
                // 读取相机的模型
                if (cam_node["model"]) {
                    std::string model = cam_node["model"].as<std::string>();
                    cam_models.push_back(model);
                } else {
                    throw std::runtime_error("model not found in config file for camera " + cam_name);
                }

                // 读取相机的分辨率
                if (cam_node["resolution"]) {
                    std::vector<int> resolution = cam_node["resolution"].as<std::vector<int>>();
                    cam_resolutions.push_back(resolution);
                } else {
                    throw std::runtime_error("resolution not found in config file for camera " + cam_name);
                }

                // 读取相机的内参
                if (cam_node["intrinsic"]) {
                    std::vector<double> intrinsic = cam_node["intrinsic"].as<std::vector<double>>();
                    cam_intrinsics.push_back(intrinsic);
                } else {
                    throw std::runtime_error("intrinsic not found in config file for camera " + cam_name);
                }

                // 读取相机的畸变系数
                if (cam_node["distortion"]) {
                    std::vector<double> distortion = cam_node["distortion"].as<std::vector<double>>();
                    cam_distortions.push_back(distortion);
                } else {
                    throw std::runtime_error("distortion not found in config file for camera " + cam_name);
                }

                // 读取相机的时间延迟
                if (cam_node["td_cam_to_imu"]) {
                    double cam_to_imu_time_offset = cam_node["td_cam_to_imu"].as<double>();
                    cams_to_imu_time_offsets.push_back(cam_to_imu_time_offset);
                } else {
                    throw std::runtime_error("td_cam_to_imu not found in config file for camera " + cam_name);
                }

                // 读取相机的外参
                if (cam_node["T_imu_cam"]) {
                    Eigen::Matrix4d T_imu_from_cam = 
                        ConfigUtils::readEigenMatrix4dFromYAML(cam_node["T_imu_cam"]); 
                    T_imu_from_cams.push_back(T_imu_from_cam);
                } else {
                    throw std::runtime_error("T_imu_cam not found in config file for camera " + cam_name);
                }

            } else {
                throw std::runtime_error("camera section not found in config file");
            }
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load camera config from YAML file: ") + e.what());
    }
}

// 打印对应的配置参数
void CameraConfigParams::printConfig() const {
    // 打印左右相机的模型
    MSF_ILOG << "camera models: " << std::endl;
    for (int i = 0; i < 2; i++) {
        MSF_ILOG << "camera " << i << " model: " << cam_models.at(i) << std::endl;
    }

    // 打印左右相机的分辨率
    MSF_ILOG << "camera resolutions: " << std::endl;
    for (int i = 0; i < 2; i++) {
        MSF_ILOG << "camera " << i << " resolution: [" 
                 << cam_resolutions.at(i)[0] << ", " 
                 << cam_resolutions.at(i)[1] << "] " 
                 << std::endl;
    }

    // 打印左右相机的内参
    MSF_ILOG << "camera intrinsics: " << std::endl;
    for (int i = 0; i < 2; i++) {
        MSF_ILOG << "camera " << i << " intrinsic: [" 
                 << cam_intrinsics.at(i)[0] << ", "
                 << cam_intrinsics.at(i)[1] << ", "
                 << cam_intrinsics.at(i)[2] << ", "
                 << cam_intrinsics.at(i)[3] << "] " 
                 << std::endl;
    }

    // 打印左右相机的畸变系数
    MSF_ILOG << "camera distortions: " << std::endl;
    for (int i = 0; i < 2; i++) {
        MSF_ILOG << "camera " << i << " distortion: [" 
                 << cam_distortions.at(i)[0] << ", "
                 << cam_distortions.at(i)[1] << ", "
                 << cam_distortions.at(i)[2] << ", "
                 << cam_distortions.at(i)[3] << "] " 
                 << std::endl;
    }

    // 打印左右相机的时间延迟
    MSF_ILOG << "td_cam_to_imu: " << std::endl;
    for (int i = 0; i < 2; i++) {
        MSF_ILOG << "camera " << i << " to imu time offset: " 
                 << cams_to_imu_time_offsets.at(i) << " (s) " 
                 << std::endl;
    }

    // 打印左右相机的外参
    MSF_ILOG << "T_imu_from_cam: " << std::endl;
    for (int i = 0; i < 2; i++) {
        MSF_ILOG << "imu " << " from cam" << i << ": \n" 
                 << T_imu_from_cams.at(i) << std::endl;
    }

    const Eigen::Matrix3d& R_imu_from_cam0 = T_imu_from_cams.at(0).block<3, 3>(0, 0);
    const Eigen::Vector3d& t_imu_from_cam0 = T_imu_from_cams.at(0).block<3, 1>(0, 3);

    const Eigen::Matrix3d& R_imu_from_cam1 = T_imu_from_cams.at(1).block<3, 3>(0, 0);
    const Eigen::Vector3d& t_imu_from_cam1 = T_imu_from_cams.at(1).block<3, 1>(0, 3);

    // T_cam0_to_cam1: p_cam1 = R * p_cam0 + t（真实 IMU 系下由 T_imu_cam 推导）
    const Eigen::Matrix3d R_cam0_to_cam1 = R_imu_from_cam1.transpose() * R_imu_from_cam0;
    const Eigen::Vector3d t_cam0_to_cam1 =
        R_imu_from_cam1.transpose() * (t_imu_from_cam0 - t_imu_from_cam1);

    Eigen::Matrix4d T_cam0_to_cam1 = Eigen::Matrix4d::Identity();
    T_cam0_to_cam1.block<3, 3>(0, 0) = R_cam0_to_cam1;
    T_cam0_to_cam1.block<3, 1>(0, 3) = t_cam0_to_cam1;

    MSF_ILOG << std::fixed << std::setprecision(8);
    MSF_ILOG << "T_cam0_to_cam1 (real IMU frame, from T_imu_cam):\n"
             << T_cam0_to_cam1 << std::endl;
    MSF_ILOG << "  baseline |t|: " << t_cam0_to_cam1.norm() << " m" << std::endl;
}

FeatureDetectorType parseFeatureDetectorType(const std::string& type_str) {
    if (type_str == "fast" || type_str == "FAST") {
        return FeatureDetectorType::FAST;
    }
    if (type_str == "superpoint" || type_str == "SUPERPOINT") {
        return FeatureDetectorType::SUPERPOINT;
    }
    if (type_str == "xfeat" || type_str == "XFEAT") {
        return FeatureDetectorType::XFEAT;
    }
    throw std::runtime_error("Unsupported feature_detector: " + type_str +
                             ", expected 'fast', 'superpoint' or 'xfeat'");
}

const char* featureDetectorTypeToString(FeatureDetectorType type) {
    switch (type) {
    case FeatureDetectorType::FAST:
        return "fast";
    case FeatureDetectorType::SUPERPOINT:
        return "superpoint";
    case FeatureDetectorType::XFEAT:
        return "xfeat";
    }
    return "unknown";
}

void XFeatConfigParams::loadConfigFromYAML(const YAML::Node& xfeat_node) {
    try {
        if (xfeat_node["model_path"]) {
            model_path = xfeat_node["model_path"].as<std::string>();
        } else {
            throw std::runtime_error("xfeat.model_path not found in config file");
        }

        if (xfeat_node["score_threshold"]) {
            score_threshold = xfeat_node["score_threshold"].as<float>();
        }

        if (xfeat_node["border"]) {
            border = xfeat_node["border"].as<int>();
        }

        if (xfeat_node["enable_hardware_preprocess"]) {
            enable_hardware_preprocess =
                xfeat_node["enable_hardware_preprocess"].as<bool>();
        }

        if (xfeat_node["enable_preprocess_timing"]) {
            enable_preprocess_timing =
                xfeat_node["enable_preprocess_timing"].as<bool>();
        }

        if (xfeat_node["npu_affinity_left"]) {
            npu_affinity_left =
                parseNpuAffinity(xfeat_node["npu_affinity_left"],
                                 "tracking.extractor.xfeat.npu_affinity_left");
        }

        if (xfeat_node["npu_affinity_right"]) {
            npu_affinity_right =
                parseNpuAffinity(xfeat_node["npu_affinity_right"],
                                 "tracking.extractor.xfeat.npu_affinity_right");
        }

        if (xfeat_node["publish_keypoints_image"]) {
            publish_keypoints_image = xfeat_node["publish_keypoints_image"].as<bool>();
        }

        if (xfeat_node["keypoints_image_topic"]) {
            keypoints_image_topic = xfeat_node["keypoints_image_topic"].as<std::string>();
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load xfeat config from YAML file: ") + e.what());
    }
}

void XFeatConfigParams::printConfig() const {
    MSF_ILOG << "  xfeat.model_path: " << model_path << std::endl;
    MSF_ILOG << "  xfeat.score_threshold: " << score_threshold << std::endl;
    MSF_ILOG << "  xfeat.border: " << border << std::endl;
    MSF_ILOG << "  xfeat.enable_hardware_preprocess: "
             << enable_hardware_preprocess << std::endl;
    MSF_ILOG << "  xfeat.enable_preprocess_timing: "
             << enable_preprocess_timing << std::endl;
    MSF_ILOG << "  xfeat.npu_affinity_left: 0b"
             << std::bitset<3>(npu_affinity_left) << std::endl;
    MSF_ILOG << "  xfeat.npu_affinity_right: 0b"
             << std::bitset<3>(npu_affinity_right) << std::endl;
    MSF_ILOG << "  xfeat.publish_keypoints_image: "
             << publish_keypoints_image << std::endl;
    MSF_ILOG << "  xfeat.keypoints_image_topic: "
             << keypoints_image_topic << std::endl;
}

void SuperPointConfigParams::loadConfigFromYAML(const YAML::Node& superpoint_node) {
    try {
        if (superpoint_node["model_path"]) {
            model_path = superpoint_node["model_path"].as<std::string>();
        } else {
            throw std::runtime_error("superpoint.model_path not found in config file");
        }

        if (superpoint_node["score_threshold"]) {
            score_threshold = superpoint_node["score_threshold"].as<float>();
        }

        if (superpoint_node["nms_radius"]) {
            nms_radius = superpoint_node["nms_radius"].as<int>();
        }

        if (superpoint_node["border"]) {
            border = superpoint_node["border"].as<int>();
        }

        if (superpoint_node["enable_hardware_preprocess"]) {
            enable_hardware_preprocess =
                superpoint_node["enable_hardware_preprocess"].as<bool>();
        }

        if (superpoint_node["enable_preprocess_timing"]) {
            enable_preprocess_timing =
                superpoint_node["enable_preprocess_timing"].as<bool>();
        }

        if (superpoint_node["publish_debug_image"]) {
            publish_debug_image = superpoint_node["publish_debug_image"].as<bool>();
        }

        if (superpoint_node["debug_image_topic"]) {
            debug_image_topic = superpoint_node["debug_image_topic"].as<std::string>();
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load superpoint config from YAML file: ") + e.what());
    }
}

void SuperPointConfigParams::printConfig() const {
    MSF_ILOG << "  superpoint.model_path: " << model_path << std::endl;
    MSF_ILOG << "  superpoint.score_threshold: " << score_threshold << std::endl;
    MSF_ILOG << "  superpoint.nms_radius: " << nms_radius << std::endl;
    MSF_ILOG << "  superpoint.border: " << border << std::endl;
    MSF_ILOG << "  superpoint.enable_hardware_preprocess: "
             << enable_hardware_preprocess << std::endl;
    MSF_ILOG << "  superpoint.enable_preprocess_timing: "
             << enable_preprocess_timing << std::endl;
    MSF_ILOG << "  superpoint.publish_debug_image: "
             << publish_debug_image << std::endl;
    MSF_ILOG << "  superpoint.debug_image_topic: "
             << debug_image_topic << std::endl;
}

// 读取特征点提取跟踪的配置参数
void TrackingConfigParams::loadConfigFromYAML(const YAML::Node& tracking_node) {
    try {
        const bool structured = tracking_node["extractor"] || tracking_node["matcher"] ||
                                tracking_node["tracking"];
        if (structured &&
            (!tracking_node["extractor"] || !tracking_node["matcher"] ||
             !tracking_node["tracking"])) {
            throw std::runtime_error(
                "tracking must contain extractor, matcher and tracking sections");
        }

        const YAML::Node extractor_node =
            structured ? tracking_node["extractor"] : tracking_node;
        const YAML::Node matcher_node = structured ? tracking_node["matcher"] : tracking_node;
        const YAML::Node runtime_node = structured ? tracking_node["tracking"] : tracking_node;

        if (extractor_node["feature_detector"]) {
            feature_detector =
                parseFeatureDetectorType(extractor_node["feature_detector"].as<std::string>());
        } else {
            feature_detector = FeatureDetectorType::FAST;
        }

        if (feature_detector == FeatureDetectorType::SUPERPOINT) {
            if (!extractor_node["superpoint"]) {
                throw std::runtime_error(
                    "tracking.extractor.superpoint is required when feature_detector is "
                    "superpoint");
            }
            superpoint_config_params.loadConfigFromYAML(extractor_node["superpoint"]);
        }

        if (feature_detector == FeatureDetectorType::XFEAT) {
            if (!extractor_node["xfeat"]) {
                throw std::runtime_error(
                    "tracking.extractor.xfeat is required when feature_detector is xfeat");
            }
            xfeat_config_params.loadConfigFromYAML(extractor_node["xfeat"]);
        }

        // 读取最小特征点距离
        if (extractor_node["min_distance"]) {
            min_distance = extractor_node["min_distance"].as<double>();
        } else {
            throw std::runtime_error("tracking.extractor.min_distance not found in config file");
        }

        // 读取网格x方向的格子数
        if (extractor_node["grid_x"]) {
            grid_x = extractor_node["grid_x"].as<int>();
        } else {
            throw std::runtime_error("tracking.extractor.grid_x not found in config file");
        }

        // 读取网格y方向的格子数
        if (extractor_node["grid_y"]) {
            grid_y = extractor_node["grid_y"].as<int>();
        } else {
            throw std::runtime_error("tracking.extractor.grid_y not found in config file");
        }

        // 读取最大特征点数量。这个上限本质是 LightGlue 模型固定 I/O 维度决定的
        // （必须和模型导出时的 top-K 数量一致，否则要么截断真实候选、要么喂给
        // LightGlue 的 buffer 装不下直接被拒绝），extractor 侧检测数量只是复用
        // 同一个值，所以 structured 格式下放进 matcher.lightglue 组，不放在
        // extractor 里；flat 格式没有分组，沿用原有的 tracking 顶层位置。
        if (structured) {
            if (matcher_node["lightglue"] && matcher_node["lightglue"]["max_num_features"]) {
                max_num_features = matcher_node["lightglue"]["max_num_features"].as<int>();
            } else {
                throw std::runtime_error(
                    "tracking.matcher.lightglue.max_num_features not found in config file");
            }
        } else {
            if (matcher_node["max_num_features"]) {
                max_num_features = matcher_node["max_num_features"].as<int>();
            } else {
                throw std::runtime_error("tracking.max_num_features not found in config file");
            }
        }

        // 读取双目匹配极线距离阈值（像素）
        if (matcher_node["epipolar_error_threshold_px"]) {
            epipolar_error_threshold_px = matcher_node["epipolar_error_threshold_px"].as<double>();
        } else {
            throw std::runtime_error(
                "tracking.matcher.epipolar_error_threshold_px not found in config file");
        }

        if (structured && matcher_node["mnn"] && matcher_node["mnn"]["ratio_threshold"]) {
            descriptor_match_ratio_threshold = matcher_node["mnn"]["ratio_threshold"].as<double>();
        } else if (matcher_node["descriptor_match_ratio_threshold"]) {
            descriptor_match_ratio_threshold =
                matcher_node["descriptor_match_ratio_threshold"].as<double>();
        }
        if (structured && matcher_node["max_motion_px"]) {
            descriptor_match_max_motion_px = matcher_node["max_motion_px"].as<double>();
        } else if (matcher_node["descriptor_match_max_motion_px"]) {
            descriptor_match_max_motion_px =
                matcher_node["descriptor_match_max_motion_px"].as<double>();
        }

        if (structured) {
            if (!matcher_node["method"]) {
                throw std::runtime_error("tracking.matcher.method not found in config file");
            }
            const std::string method = matcher_node["method"].as<std::string>();
            if (method == "lightglue" || method == "LIGHTGLUE") {
                use_lightglue = true;
            } else if (method == "mnn" || method == "MNN") {
                use_lightglue = false;
            } else {
                throw std::runtime_error(
                    "tracking.matcher.method must be 'lightglue' or 'mnn'");
            }

            if (matcher_node["lightglue"]) {
                const YAML::Node lightglue_node = matcher_node["lightglue"];
                if (lightglue_node["model_path"]) {
                    lightglue_model_path = lightglue_node["model_path"].as<std::string>();
                }
                if (lightglue_node["score_threshold"]) {
                    lightglue_score_threshold = lightglue_node["score_threshold"].as<float>();
                }
                if (lightglue_node["npu_affinity_temporal"]) {
                    lightglue_temporal_npu_affinity = parseNpuAffinity(
                        lightglue_node["npu_affinity_temporal"],
                        "tracking.matcher.lightglue.npu_affinity_temporal");
                }
                if (lightglue_node["npu_affinity_stereo"]) {
                    lightglue_stereo_npu_affinity = parseNpuAffinity(
                        lightglue_node["npu_affinity_stereo"],
                        "tracking.matcher.lightglue.npu_affinity_stereo");
                }
            }
            if (use_lightglue && lightglue_model_path.empty()) {
                throw std::runtime_error(
                    "tracking.matcher.lightglue.model_path is required for lightglue");
            }
        } else {
            if (matcher_node["use_lightglue"]) {
                use_lightglue = matcher_node["use_lightglue"].as<bool>();
            }
            if (matcher_node["lightglue_model_path"]) {
                lightglue_model_path = matcher_node["lightglue_model_path"].as<std::string>();
            }
            if (matcher_node["lightglue_score_threshold"]) {
                lightglue_score_threshold =
                    matcher_node["lightglue_score_threshold"].as<float>();
            }
        }

        if (matcher_node["ransac_max_iters"]) {
            ransac_max_iters = matcher_node["ransac_max_iters"].as<int>();
        }
        if (matcher_node["homography_precheck_max_iters"]) {
            homography_precheck_max_iters =
                matcher_node["homography_precheck_max_iters"].as<int>();
        }
        if (matcher_node["homography_min_inlier_ratio"]) {
            homography_min_inlier_ratio =
                matcher_node["homography_min_inlier_ratio"].as<double>();
        }
        if (matcher_node["homography_min_inliers"]) {
            homography_min_inliers = matcher_node["homography_min_inliers"].as<int>();
        }
        if (matcher_node["ransac_threshold_min_px"]) {
            ransac_threshold_min_px = matcher_node["ransac_threshold_min_px"].as<double>();
        }
        if (matcher_node["ransac_threshold_max_px"]) {
            ransac_threshold_max_px = matcher_node["ransac_threshold_max_px"].as<double>();
        }
        if (matcher_node["ransac_threshold_margin"]) {
            ransac_threshold_margin = matcher_node["ransac_threshold_margin"].as<double>();
        }
        if (matcher_node["ransac_threshold_ema_alpha"]) {
            ransac_threshold_ema_alpha = matcher_node["ransac_threshold_ema_alpha"].as<double>();
        }
        if (matcher_node["ransac_threshold_min_inliers"]) {
            ransac_threshold_min_inliers = matcher_node["ransac_threshold_min_inliers"].as<int>();
        }
        if (matcher_node["ransac_threshold_min_inlier_ratio"]) {
            ransac_threshold_min_inlier_ratio =
                matcher_node["ransac_threshold_min_inlier_ratio"].as<double>();
        }
        if (runtime_node["publish_temporal_tracking_debug"]) {
            publish_temporal_tracking_debug =
                runtime_node["publish_temporal_tracking_debug"].as<bool>();
        }
        if (runtime_node["temporal_tracking_debug_topic"]) {
            temporal_tracking_debug_topic =
                runtime_node["temporal_tracking_debug_topic"].as<std::string>();
        }
        if (runtime_node["publish_stereo_match_debug"]) {
            publish_stereo_match_debug =
                runtime_node["publish_stereo_match_debug"].as<bool>();
        }
        if (runtime_node["stereo_match_debug_topic"]) {
            stereo_match_debug_topic =
                runtime_node["stereo_match_debug_topic"].as<std::string>();
        }

        // 读取相机降采样
        if (runtime_node["downsize_camera"]) {
            downsize_camera = runtime_node["downsize_camera"].as<bool>();
        } else {
            throw std::runtime_error(
                "tracking.tracking.downsize_camera not found in config file");
        }

        // 读取相机降采样因子
        if (runtime_node["downsize_factor"]) {
            downsize_factor = runtime_node["downsize_factor"].as<int>();
        } else {
            throw std::runtime_error(
                "tracking.tracking.downsize_factor not found in config file");
        }
        if (downsize_factor <= 0) {
            throw std::runtime_error("tracking.downsize_factor must be > 0");
        }

        if (runtime_node["drop_bottom_feature_rows"]) {
            drop_bottom_feature_rows = runtime_node["drop_bottom_feature_rows"].as<int>();
        }
        if (drop_bottom_feature_rows < 0) {
            throw std::runtime_error("tracking.drop_bottom_feature_rows must be >= 0");
        }

        if (runtime_node["ivps_left_grp"]) {
            ivps_left_grp = runtime_node["ivps_left_grp"].as<int>();
        }
        if (runtime_node["ivps_right_grp"]) {
            ivps_right_grp = runtime_node["ivps_right_grp"].as<int>();
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load tracking config from YAML file: ") + e.what());
    }
}

// 打印对应的配置参数
void TrackingConfigParams::printConfig() const {
    MSF_ILOG << "feature_detector: " << featureDetectorTypeToString(feature_detector)
             << std::endl;
    if (feature_detector == FeatureDetectorType::SUPERPOINT) {
        superpoint_config_params.printConfig();
    }
    if (feature_detector == FeatureDetectorType::XFEAT) {
        xfeat_config_params.printConfig();
    }
    MSF_ILOG << "min_distance: " << min_distance << std::endl;
    MSF_ILOG << "grid_x: " << grid_x << std::endl;
    MSF_ILOG << "grid_y: " << grid_y << std::endl;
    MSF_ILOG << "max_num_features: " << max_num_features << std::endl;
    MSF_ILOG << "epipolar_error_threshold_px: " << epipolar_error_threshold_px << std::endl;
    MSF_ILOG << "descriptor_match_ratio_threshold: " << descriptor_match_ratio_threshold
             << std::endl;
    MSF_ILOG << "descriptor_match_max_motion_px: " << descriptor_match_max_motion_px
             << std::endl;
    MSF_ILOG << "matcher.method: " << (use_lightglue ? "lightglue" : "mnn") << std::endl;
    MSF_ILOG << "lightglue_model_path: " << lightglue_model_path << std::endl;
    MSF_ILOG << "lightglue_score_threshold: " << lightglue_score_threshold << std::endl;
    MSF_ILOG << "lightglue_npu_affinity_temporal: 0b"
             << std::bitset<3>(lightglue_temporal_npu_affinity) << std::endl;
    MSF_ILOG << "lightglue_npu_affinity_stereo: 0b"
             << std::bitset<3>(lightglue_stereo_npu_affinity) << std::endl;
    MSF_ILOG << "ransac_max_iters: " << ransac_max_iters << std::endl;
    MSF_ILOG << "homography_precheck_max_iters: " << homography_precheck_max_iters << std::endl;
    MSF_ILOG << "homography_min_inlier_ratio: " << homography_min_inlier_ratio << std::endl;
    MSF_ILOG << "ransac_threshold_min_px: " << ransac_threshold_min_px << std::endl;
    MSF_ILOG << "ransac_threshold_max_px: " << ransac_threshold_max_px << std::endl;
    MSF_ILOG << "ransac_threshold_margin: " << ransac_threshold_margin << std::endl;
    MSF_ILOG << "ransac_threshold_ema_alpha: " << ransac_threshold_ema_alpha << std::endl;
    MSF_ILOG << "ransac_threshold_min_inliers: " << ransac_threshold_min_inliers << std::endl;
    MSF_ILOG << "ransac_threshold_min_inlier_ratio: " << ransac_threshold_min_inlier_ratio
             << std::endl;
    MSF_ILOG << "homography_min_inliers: " << homography_min_inliers << std::endl;
    MSF_ILOG << "publish_temporal_tracking_debug: "
             << (publish_temporal_tracking_debug ? "true" : "false") << std::endl;
    MSF_ILOG << "temporal_tracking_debug_topic: " << temporal_tracking_debug_topic << std::endl;
    MSF_ILOG << "publish_stereo_match_debug: "
             << (publish_stereo_match_debug ? "true" : "false") << std::endl;
    MSF_ILOG << "stereo_match_debug_topic: " << stereo_match_debug_topic << std::endl;
    MSF_ILOG << "downsize_camera: " << (downsize_camera ? "true" : "false") << std::endl;
    MSF_ILOG << "downsize_factor: " << downsize_factor << std::endl;
    MSF_ILOG << "drop_bottom_feature_rows: " << drop_bottom_feature_rows << std::endl;
    MSF_ILOG << "ivps_left_grp: " << ivps_left_grp << std::endl;
    MSF_ILOG << "ivps_right_grp: " << ivps_right_grp << std::endl;
}

// 读取优化器的配置参数
void OptimizerConfigParams::loadConfigFromYAML(const YAML::Node& optimizer_node) {
    try {
        // 读取滑窗的大小
        if (optimizer_node["window_size"]) {
            window_size = optimizer_node["window_size"].as<int>();
        } else {
            throw std::runtime_error("window_size not found in config file");
        }

        // 读取最大迭代次数
        if (optimizer_node["max_iterations"]) {
            max_iterations = optimizer_node["max_iterations"].as<int>();
        } else {
            throw std::runtime_error("max_iterations not found in config file");
        }

        // 读取视觉的标准差
        if (optimizer_node["optimize_reprojection_std"]) {
            optimize_reprojection_std = optimizer_node["optimize_reprojection_std"].as<double>();
        } else {
            throw std::runtime_error("optimize_reprojection_std not found in config file");
        }

        // 读取粗差观测门限（optimize_reprojection_std 的倍数）；缺省 3.0
        if (optimizer_node["outlier_reprojection_sigma"]) {
            outlier_reprojection_sigma =
                optimizer_node["outlier_reprojection_sigma"].as<double>();
        }

        if (optimizer_node["triangulation_max_reprojection_px"]) {
            triangulation_max_reprojection_px = optimizer_node["triangulation_max_reprojection_px"].as<double>();
        }
        if (optimizer_node["triangulation_min_angle_deg"]) {
            triangulation_min_angle_deg = optimizer_node["triangulation_min_angle_deg"].as<double>();
        }
        if (optimizer_node["triangulation_min_depth_baseline_ratio"]) {
            triangulation_min_depth_baseline_ratio =
                optimizer_node["triangulation_min_depth_baseline_ratio"].as<double>();
        }
        if (!std::isfinite(triangulation_min_depth_baseline_ratio) ||
            triangulation_min_depth_baseline_ratio <= 0.0) {
            throw std::runtime_error("triangulation_min_depth_baseline_ratio must be finite and positive");
        }
        if (!std::isfinite(triangulation_max_reprojection_px) || triangulation_max_reprojection_px <= 0.0 ||
            !std::isfinite(triangulation_min_angle_deg) || triangulation_min_angle_deg <= 0.0 ||
            triangulation_min_angle_deg >= 90.0) {
            throw std::runtime_error("Invalid triangulation quality thresholds");
        }

        if (optimizer_node["keyframe_min_interval_s"]) {
            keyframe_min_interval_s = optimizer_node["keyframe_min_interval_s"].as<double>();
        }

        if (optimizer_node["keyframe_min_parallax_px"]) {
            keyframe_min_parallax_px = optimizer_node["keyframe_min_parallax_px"].as<double>();
        }

        if (optimizer_node["keyframe_min_parallax_track_num"]) {
            keyframe_min_parallax_track_num =
                optimizer_node["keyframe_min_parallax_track_num"].as<int>();
        }

        // 关键帧间隔连续调节参数，全部可选，缺省用头文件里的默认值
        if (optimizer_node["keyframe_min_valid_matches"]) {
            keyframe_min_valid_matches = optimizer_node["keyframe_min_valid_matches"].as<int>();
        }
        if (optimizer_node["keyframe_min_match_ratio"]) {
            keyframe_min_match_ratio = optimizer_node["keyframe_min_match_ratio"].as<double>();
        }
        if (optimizer_node["keyframe_min_connected_matches"]) {
            keyframe_min_connected_matches = optimizer_node["keyframe_min_connected_matches"].as<int>();
        }
        if (optimizer_node["keyframe_min_triangulated_matches"]) {
            keyframe_min_triangulated_matches = optimizer_node["keyframe_min_triangulated_matches"].as<int>();
        }
        if (keyframe_min_connected_matches < 4 ||
            keyframe_min_valid_matches < keyframe_min_connected_matches ||
            !std::isfinite(keyframe_min_match_ratio) ||
            keyframe_min_match_ratio < 0.0 || keyframe_min_match_ratio > 1.0) {
            throw std::runtime_error("invalid keyframe match thresholds: require valid >= connected >= 4 and ratio in [0,1]");
        }
        if (optimizer_node["keyframe_adaptive_floor_s"]) {
            keyframe_adaptive_floor_s = optimizer_node["keyframe_adaptive_floor_s"].as<double>();
        }
        if (optimizer_node["keyframe_adaptive_backend_margin"]) {
            keyframe_adaptive_backend_margin = optimizer_node["keyframe_adaptive_backend_margin"].as<double>();
        }
        if (optimizer_node["keyframe_adaptive_blocks_low"]) {
            keyframe_adaptive_blocks_low = optimizer_node["keyframe_adaptive_blocks_low"].as<double>();
        }
        if (optimizer_node["keyframe_adaptive_blocks_high"]) {
            keyframe_adaptive_blocks_high = optimizer_node["keyframe_adaptive_blocks_high"].as<double>();
        }
        if (optimizer_node["keyframe_adaptive_disparity_low_px"]) {
            keyframe_adaptive_disparity_low_px = optimizer_node["keyframe_adaptive_disparity_low_px"].as<double>();
        }
        if (optimizer_node["keyframe_adaptive_disparity_high_px"]) {
            keyframe_adaptive_disparity_high_px = optimizer_node["keyframe_adaptive_disparity_high_px"].as<double>();
        }
        if (optimizer_node["keyframe_adaptive_relax_alpha"]) {
            keyframe_adaptive_relax_alpha = optimizer_node["keyframe_adaptive_relax_alpha"].as<double>();
        }
        if (optimizer_node["vision_sigma_min_px"]) {
            vision_sigma_min_px = optimizer_node["vision_sigma_min_px"].as<double>();
        }
        if (optimizer_node["vision_sigma_margin"]) {
            vision_sigma_margin = optimizer_node["vision_sigma_margin"].as<double>();
        }
        if (optimizer_node["vision_sigma_ema_alpha"]) {
            vision_sigma_ema_alpha = optimizer_node["vision_sigma_ema_alpha"].as<double>();
        }
        if (optimizer_node["vision_sigma_min_blocks"]) {
            vision_sigma_min_blocks = optimizer_node["vision_sigma_min_blocks"].as<int>();
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load optimizer config from YAML file: ") + e.what());
    }
}

// 打印对应的配置参数
void OptimizerConfigParams::printConfig() const {
    MSF_ILOG << "triangulation_max_reprojection_px: " << triangulation_max_reprojection_px << std::endl;
    MSF_ILOG << "triangulation_min_angle_deg: " << triangulation_min_angle_deg << std::endl;
    MSF_ILOG << "triangulation_min_depth_baseline_ratio: "
             << triangulation_min_depth_baseline_ratio << std::endl;
    MSF_ILOG << "window_size: " << window_size << std::endl;
    MSF_ILOG << "max_iterations: " << max_iterations << std::endl;
    MSF_ILOG << "optimize_reprojection_std: " << optimize_reprojection_std << std::endl;
    MSF_ILOG << "outlier_reprojection_sigma: " << outlier_reprojection_sigma << " (x std)"
             << std::endl;
    MSF_ILOG << "keyframe_min_interval_s: " << keyframe_min_interval_s << " (s)" << std::endl;
    MSF_ILOG << "keyframe_min_parallax_px: " << keyframe_min_parallax_px << " (px)" << std::endl;
    MSF_ILOG << "keyframe_min_parallax_track_num: " << keyframe_min_parallax_track_num
             << std::endl;
    MSF_ILOG << "keyframe_min_valid_matches: " << keyframe_min_valid_matches << std::endl;
    MSF_ILOG << "keyframe_min_match_ratio: " << keyframe_min_match_ratio << std::endl;
    MSF_ILOG << "keyframe_min_connected_matches: " << keyframe_min_connected_matches << std::endl;
    MSF_ILOG << "keyframe_min_triangulated_matches: " << keyframe_min_triangulated_matches << std::endl;
    MSF_ILOG << "keyframe_adaptive_floor_s: " << keyframe_adaptive_floor_s << std::endl;
    MSF_ILOG << "keyframe_adaptive_backend_margin: " << keyframe_adaptive_backend_margin << std::endl;
    MSF_ILOG << "keyframe_adaptive_blocks_low: " << keyframe_adaptive_blocks_low << std::endl;
    MSF_ILOG << "keyframe_adaptive_blocks_high: " << keyframe_adaptive_blocks_high << std::endl;
    MSF_ILOG << "keyframe_adaptive_disparity_low_px: " << keyframe_adaptive_disparity_low_px << std::endl;
    MSF_ILOG << "keyframe_adaptive_disparity_high_px: " << keyframe_adaptive_disparity_high_px << std::endl;
    MSF_ILOG << "keyframe_adaptive_relax_alpha: " << keyframe_adaptive_relax_alpha << std::endl;
    MSF_ILOG << "vision_sigma_min_px: " << vision_sigma_min_px << std::endl;
    MSF_ILOG << "vision_sigma_margin: " << vision_sigma_margin << std::endl;
    MSF_ILOG << "vision_sigma_ema_alpha: " << vision_sigma_ema_alpha << std::endl;
    MSF_ILOG << "vision_sigma_min_blocks: " << vision_sigma_min_blocks << std::endl;
}

// 读取静止检测的配置参数
void StaticDetectorConfigParams::loadConfigFromYAML(const YAML::Node& static_detector_node) {
    try {
        if (static_detector_node["enable_zupt"]) {
            enable_zupt = static_detector_node["enable_zupt"].as<bool>();
        }
        if (static_detector_node["gyr_std_threshold"]) {
            gyr_std_threshold = static_detector_node["gyr_std_threshold"].as<double>();
        }
        if (static_detector_node["acc_xy_std_threshold"]) {
            acc_xy_std_threshold = static_detector_node["acc_xy_std_threshold"].as<double>();
        }
        if (static_detector_node["acc_z_std_threshold"]) {
            acc_z_std_threshold = static_detector_node["acc_z_std_threshold"].as<double>();
        }
        if (static_detector_node["gravity_tolerance"]) {
            gravity_tolerance = static_detector_node["gravity_tolerance"].as<double>();
        }
        if (static_detector_node["gyr_mean_norm_threshold"]) {
            gyr_mean_norm_threshold =
                static_detector_node["gyr_mean_norm_threshold"].as<double>();
        }
        if (static_detector_node["confirm_windows"]) {
            confirm_windows = static_detector_node["confirm_windows"].as<int>();
        }
        if (confirm_windows < 1) {
            throw std::runtime_error("static_detector.confirm_windows must be >= 1");
        }
        if (static_detector_node["zupt_velocity_std"]) {
            zupt_velocity_std = static_detector_node["zupt_velocity_std"].as<double>();
        }
        if (zupt_velocity_std <= 0.0) {
            throw std::runtime_error("static_detector.zupt_velocity_std must be > 0");
        }
        if (static_detector_node["odom_vel_static_threshold"]) {
            odom_vel_static_threshold =
                static_detector_node["odom_vel_static_threshold"].as<double>();
        }
        if (odom_vel_static_threshold <= 0.0) {
            throw std::runtime_error("static_detector.odom_vel_static_threshold must be > 0");
        }
        if (static_detector_node["runtime_window_s"]) {
            runtime_window_s = static_detector_node["runtime_window_s"].as<double>();
        }
        if (runtime_window_s <= 0.0) {
            throw std::runtime_error("static_detector.runtime_window_s must be > 0");
        }
        if (static_detector_node["std_smoothing_window"]) {
            std_smoothing_window = static_detector_node["std_smoothing_window"].as<int>();
        }
        if (std_smoothing_window < 1) {
            throw std::runtime_error("static_detector.std_smoothing_window must be >= 1");
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load static_detector config from YAML file: ") + e.what());
    }
}

// 打印静止检测的配置参数
void StaticDetectorConfigParams::printConfig() const {
    MSF_ILOG << "enable_zupt: " << (enable_zupt ? "true" : "false") << std::endl;
    MSF_ILOG << "gyr_std_threshold: " << gyr_std_threshold << " (rad/s)" << std::endl;
    MSF_ILOG << "acc_xy_std_threshold: " << acc_xy_std_threshold << " (m/s^2)" << std::endl;
    MSF_ILOG << "acc_z_std_threshold: " << acc_z_std_threshold << " (m/s^2)" << std::endl;
    MSF_ILOG << "gravity_tolerance: " << gravity_tolerance << " (m/s^2)" << std::endl;
    MSF_ILOG << "gyr_mean_norm_threshold: " << gyr_mean_norm_threshold << " (rad/s)" << std::endl;
    MSF_ILOG << "confirm_windows: " << confirm_windows << std::endl;
    MSF_ILOG << "zupt_velocity_std: " << zupt_velocity_std << " (m/s)" << std::endl;
    MSF_ILOG << "odom_vel_static_threshold: " << odom_vel_static_threshold << " (m/s)" << std::endl;
    MSF_ILOG << "runtime_window_s: " << runtime_window_s << " (s)" << std::endl;
    MSF_ILOG << "std_smoothing_window: " << std_smoothing_window << " (samples)" << std::endl;
}

// 读取 Initializer 的配置参数
void InitializerConfigParams::loadConfigFromYAML(const YAML::Node& initializer_node) {
    try {
        if (initializer_node["enable"]) {
            enable = initializer_node["enable"].as<bool>();
        }
        if (initializer_node["gyr_std_threshold"]) {
            gyr_std_threshold = initializer_node["gyr_std_threshold"].as<double>();
        }
        if (initializer_node["acc_xy_std_threshold"]) {
            acc_xy_std_threshold = initializer_node["acc_xy_std_threshold"].as<double>();
        }
        if (initializer_node["acc_z_std_threshold"]) {
            acc_z_std_threshold = initializer_node["acc_z_std_threshold"].as<double>();
        }
        if (initializer_node["gravity_tolerance"]) {
            gravity_tolerance = initializer_node["gravity_tolerance"].as<double>();
        }
        if (initializer_node["gyr_mean_norm_threshold"]) {
            gyr_mean_norm_threshold = initializer_node["gyr_mean_norm_threshold"].as<double>();
        }
        if (initializer_node["confirm_windows"]) {
            confirm_windows = initializer_node["confirm_windows"].as<int>();
        }
        if (confirm_windows < 1) {
            throw std::runtime_error("initializer.confirm_windows must be >= 1");
        }
        if (initializer_node["std_smoothing_window"]) {
            std_smoothing_window = initializer_node["std_smoothing_window"].as<int>();
        }
        if (std_smoothing_window < 1) {
            throw std::runtime_error("initializer.std_smoothing_window must be >= 1");
        }
        if (initializer_node["init_window_s"]) {
            init_window_s = initializer_node["init_window_s"].as<double>();
        }
        if (init_window_s <= 0.0) {
            throw std::runtime_error("initializer.init_window_s must be > 0");
        }
        if (initializer_node["init_timeout_s"]) {
            init_timeout_s = initializer_node["init_timeout_s"].as<double>();
        }
        if (!std::isfinite(init_timeout_s)) {
            throw std::runtime_error("initializer.init_timeout_s must be finite");
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load initializer config from YAML file: ") + e.what());
    }
}

// 打印 Initializer 的配置参数
void InitializerConfigParams::printConfig() const {
    MSF_ILOG << "enable: " << (enable ? "true" : "false") << std::endl;
    MSF_ILOG << "gyr_std_threshold: " << gyr_std_threshold << " (rad/s)" << std::endl;
    MSF_ILOG << "acc_xy_std_threshold: " << acc_xy_std_threshold << " (m/s^2)" << std::endl;
    MSF_ILOG << "acc_z_std_threshold: " << acc_z_std_threshold << " (m/s^2)" << std::endl;
    MSF_ILOG << "gravity_tolerance: " << gravity_tolerance << " (m/s^2)" << std::endl;
    MSF_ILOG << "gyr_mean_norm_threshold: " << gyr_mean_norm_threshold << " (rad/s)" << std::endl;
    MSF_ILOG << "confirm_windows: " << confirm_windows << std::endl;
    MSF_ILOG << "std_smoothing_window: " << std_smoothing_window << " (samples)" << std::endl;
    MSF_ILOG << "init_window_s: " << init_window_s << " (s)" << std::endl;
    MSF_ILOG << "init_timeout_s: " << init_timeout_s << " (s, <=0 disables)" << std::endl;
}

namespace {

bool isSupportedRelocResolution(int width, int height) {
    return (width == 640 && height == 384) || (width == 640 && height == 640) ||
           (width == 512 && height == 512);
}

}  // namespace

// 读取重定位的配置参数
void RelocConfigParams::loadConfigFromYAML(const YAML::Node& reloc_node) {
    try {
        if (reloc_node["enable"]) {
            enable = reloc_node["enable"].as<bool>();
        }

        // 读取模型的路径
        if (reloc_node["model_path"]) {
            model_path = reloc_node["model_path"].as<std::string>();
        } else {
            throw std::runtime_error("reloc.model_path not found in config file");
        }

        // 读取重定位输入分辨率
        if (reloc_node["resolution"]) {
            const auto resolution = reloc_node["resolution"].as<std::vector<int>>();
            if (resolution.size() != 2) {
                throw std::runtime_error("reloc.resolution must be [width, height]");
            }
            resolution_width = resolution[0];
            resolution_height = resolution[1];
            if (!isSupportedRelocResolution(resolution_width, resolution_height)) {
                throw std::runtime_error(
                    "reloc.resolution must be one of [640, 384], [640, 640], [512, 512]");
            }
        } else {
            throw std::runtime_error("reloc.resolution not found in config file");
        }

        if (reloc_node["enable_undistort"]) {
            enable_undistort = reloc_node["enable_undistort"].as<bool>();
        }
        if (reloc_node["undistort_alpha"]) {
            undistort_alpha = reloc_node["undistort_alpha"].as<double>();
        }
        if (!std::isfinite(undistort_alpha) || undistort_alpha < 0.0 ||
            undistort_alpha > 1.0) {
            throw std::runtime_error("reloc.undistort_alpha must be in [0, 1]");
        }
        if (reloc_node["publish_debug_image"]) {
            publish_debug_image = reloc_node["publish_debug_image"].as<bool>();
        }
        if (reloc_node["debug_image_topic"]) {
            debug_image_topic = reloc_node["debug_image_topic"].as<std::string>();
        }
        if (debug_image_topic.empty()) {
            throw std::runtime_error("reloc.debug_image_topic must not be empty");
        }

        // 读取置信度阈值
        if (reloc_node["conf_threshold"]) {
            conf_threshold = reloc_node["conf_threshold"].as<float>();
        } else {
            throw std::runtime_error("reloc.conf_threshold not found in config file");
        }

        // 读取NMS阈值
        if (reloc_node["nms_threshold"]) {
            nms_threshold = reloc_node["nms_threshold"].as<float>();
        } else {
            throw std::runtime_error("reloc.nms_threshold not found in config file");
        }

        // 读取类别数
        if (reloc_node["num_classes"]) {
            num_classes = reloc_node["num_classes"].as<int>();
        } else {
            throw std::runtime_error("reloc.num_classes not found in config file");
        }
        
        // 读取最大检测数
        if (reloc_node["max_det"]) {
            max_det = reloc_node["max_det"].as<int>();
        } else {
            throw std::runtime_error("reloc.max_det not found in config file");
        }

        if (reloc_node["left_npu_affinity"]) {
            left_npu_affinity =
                parseNpuAffinity(reloc_node["left_npu_affinity"], "reloc.left_npu_affinity");
        }
        if (reloc_node["right_npu_affinity"]) {
            right_npu_affinity =
                parseNpuAffinity(reloc_node["right_npu_affinity"], "reloc.right_npu_affinity");
        }

        // 周期触发节奏（原 reloc_backend 字段）
        if (reloc_node["periodic_reloc_period_s"]) {
            periodic_reloc_period_s = reloc_node["periodic_reloc_period_s"].as<double>();
        }
        if (!std::isfinite(periodic_reloc_period_s)) {
            throw std::runtime_error("reloc.periodic_reloc_period_s must be finite");
        }
        if (reloc_node["periodic_reloc_retry_interval_s"]) {
            periodic_reloc_retry_interval_s =
                reloc_node["periodic_reloc_retry_interval_s"].as<double>();
        }
        if (!std::isfinite(periodic_reloc_retry_interval_s) ||
            periodic_reloc_retry_interval_s <= 0.0) {
            throw std::runtime_error(
                "reloc.periodic_reloc_retry_interval_s must be finite and > 0");
        }
        if (reloc_node["reloc_max_apply_delay_s"]) {
            reloc_max_apply_delay_s = reloc_node["reloc_max_apply_delay_s"].as<double>();
        }
        if (!std::isfinite(reloc_max_apply_delay_s)) {
            throw std::runtime_error("reloc.reloc_max_apply_delay_s must be finite");
        }
        // max_correction_position_m/max_correction_yaw_deg 不在这里解析：它们
        // 复用 loop.max_correction_position_m/max_correction_yaw_deg，见
        // MsFusionConfigParams::loadConfigFromYAML 里 loop.enable 旁边的解析。

        // checkPose() 位姿验收门槛
        if (reloc_node["max_height_m"]) {
            max_height_m = reloc_node["max_height_m"].as<double>();
        }
        if (!std::isfinite(max_height_m) || max_height_m <= 0.0) {
            throw std::runtime_error("reloc.max_height_m must be finite and > 0");
        }
        if (reloc_node["max_reprojection_error_px"]) {
            max_reprojection_error_px = reloc_node["max_reprojection_error_px"].as<double>();
        }
        if (!std::isfinite(max_reprojection_error_px) || max_reprojection_error_px <= 0.0) {
            throw std::runtime_error("reloc.max_reprojection_error_px must be finite and > 0");
        }
        if (reloc_node["min_offplane_stereo_pairs"]) {
            min_offplane_stereo_pairs = reloc_node["min_offplane_stereo_pairs"].as<int>();
        }
        if (min_offplane_stereo_pairs < 0) {
            throw std::runtime_error("reloc.min_offplane_stereo_pairs must be >= 0");
        }

        // 2 点直立解假设评分参数
        if (reloc_node["inlier_pixel_threshold_px"]) {
            inlier_pixel_threshold_px = reloc_node["inlier_pixel_threshold_px"].as<double>();
        }
        if (!std::isfinite(inlier_pixel_threshold_px) || inlier_pixel_threshold_px <= 0.0) {
            throw std::runtime_error("reloc.inlier_pixel_threshold_px must be finite and > 0");
        }
        if (reloc_node["min_support"]) {
            min_support = reloc_node["min_support"].as<int>();
        }
        if (min_support < 4) {
            throw std::runtime_error("reloc.min_support must be >= 4");
        }
        if (reloc_node["periodic_multi_frame_confirm"]) {
            periodic_multi_frame_confirm = reloc_node["periodic_multi_frame_confirm"].as<bool>();
        }
        if (reloc_node["confirm_frames"]) {
            confirm_frames = reloc_node["confirm_frames"].as<int>();
        }
        if (confirm_frames < 2) {
            throw std::runtime_error("reloc.confirm_frames must be >= 2");
        }
        if (reloc_node["confirm_max_position_m"]) {
            confirm_max_position_m = reloc_node["confirm_max_position_m"].as<double>();
        }
        if (reloc_node["confirm_max_yaw_deg"]) {
            confirm_max_yaw_deg = reloc_node["confirm_max_yaw_deg"].as<double>();
        }
        if (reloc_node["confirm_max_gap_s"]) {
            confirm_max_gap_s = reloc_node["confirm_max_gap_s"].as<double>();
        }
        if (!std::isfinite(confirm_max_position_m) || confirm_max_position_m <= 0.0 ||
            !std::isfinite(confirm_max_yaw_deg) || confirm_max_yaw_deg <= 0.0 ||
            !std::isfinite(confirm_max_gap_s) || confirm_max_gap_s <= 0.0) {
            throw std::runtime_error("reloc.confirm_max_* must be finite and > 0");
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to load reloc config from YAML file: ") + e.what());
    }
}

// 打印对应的配置参数
void RelocConfigParams::printConfig() const {
    MSF_ILOG << "reloc.enable: " << (enable ? "true" : "false") << std::endl;
    MSF_ILOG << "reloc.model_path: " << model_path << std::endl;
    MSF_ILOG << "reloc.resolution: [" << resolution_width << ", " << resolution_height << "]"
             << std::endl;
    MSF_ILOG << "reloc.enable_undistort: " << (enable_undistort ? "true" : "false")
             << std::endl;
    MSF_ILOG << "reloc.undistort_alpha: " << undistort_alpha << std::endl;
    MSF_ILOG << "reloc.publish_debug_image: " << (publish_debug_image ? "true" : "false")
             << std::endl;
    MSF_ILOG << "reloc.debug_image_topic: " << debug_image_topic << std::endl;
    MSF_ILOG << "reloc.conf_threshold: " << conf_threshold << std::endl;
    MSF_ILOG << "reloc.nms_threshold: " << nms_threshold << std::endl;
    MSF_ILOG << "reloc.num_classes: " << num_classes << std::endl;
    MSF_ILOG << "reloc.max_det: " << max_det << std::endl;
    MSF_ILOG << "reloc.left_npu_affinity: 0b" << std::bitset<3>(left_npu_affinity) << std::endl;
    MSF_ILOG << "reloc.right_npu_affinity: 0b" << std::bitset<3>(right_npu_affinity) << std::endl;
    MSF_ILOG << "reloc.periodic_reloc_period_s: " << periodic_reloc_period_s << " (s)"
             << (periodic_reloc_period_s > 0.0 ? "" : " (disabled)") << std::endl;
    MSF_ILOG << "reloc.periodic_reloc_retry_interval_s: " << periodic_reloc_retry_interval_s
             << " (s, loop.enable=true only)" << std::endl;
    MSF_ILOG << "reloc.reloc_max_apply_delay_s: " << reloc_max_apply_delay_s
             << (reloc_max_apply_delay_s > 0.0 ? " (s)" : " (disabled)") << std::endl;
    MSF_ILOG << "reloc.max_correction_position_m (from loop.*): " << max_correction_position_m
             << (max_correction_position_m > 0.0 ? " (m)" : " (disabled)") << std::endl;
    MSF_ILOG << "reloc.max_correction_yaw_deg (from loop.*): " << max_correction_yaw_deg
             << (max_correction_yaw_deg > 0.0 ? " (deg)" : " (disabled)") << std::endl;
    MSF_ILOG << "reloc.max_height_m: " << max_height_m << " (m)" << std::endl;
    MSF_ILOG << "reloc.max_reprojection_error_px: " << max_reprojection_error_px << " (px)"
             << std::endl;
    MSF_ILOG << "reloc.min_offplane_stereo_pairs: " << min_offplane_stereo_pairs
             << (min_offplane_stereo_pairs > 0 ? "" : " (disabled)") << std::endl;
    MSF_ILOG << "reloc.inlier_pixel_threshold_px: " << inlier_pixel_threshold_px << " (px)"
             << std::endl;
    MSF_ILOG << "reloc.min_support: " << min_support << std::endl;
    MSF_ILOG << "reloc.periodic_multi_frame_confirm: " << (periodic_multi_frame_confirm ? "true" : "false")
             << " confirm_frames=" << confirm_frames
             << " confirm_max_position_m=" << confirm_max_position_m
             << " confirm_max_yaw_deg=" << confirm_max_yaw_deg
             << " confirm_max_gap_s=" << confirm_max_gap_s << std::endl;
}

// 读取EIGEN的4X4矩阵
Eigen::Matrix4d ConfigUtils::readEigenMatrix4dFromYAML(const YAML::Node& matrix_node) {
    if (!matrix_node.IsSequence() || matrix_node.size() != 4) {
        throw std::runtime_error("Matrix must be a 4x4 matrix");
    }

    // 定义矩阵的初值
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero();

    // 按照行读取矩阵的值
    for (int i = 0; i < 4; i++) {
        const YAML::Node row = matrix_node[i];
        if (!row.IsSequence() || row.size() != 4) {
            throw std::runtime_error("Matrix row must be a sequence of 4 elements");
        }

        // 读取行中的4个元素
        for (int j = 0; j < 4; j++) {
            matrix(i, j) = row[j].as<double>();
        }
    }

    return matrix;
}

} // namespace ms_fusion
