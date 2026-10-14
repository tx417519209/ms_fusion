#include "ms_fusion/ms_fusion_node.h"
#include "ms_fusion_interface.h"
#include "ms_fusion/common/runtime_log.h"

#include <glog/logging.h>

int main(int argc, char **argv) {
    // glog 只用于崩溃时打印堆栈；日志输出走 MSF_*LOG（spdlog，控制台 + 本次运行目录的滚动文件）
    FLAGS_logtostderr = true;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    // 不依赖配置文件，进程启动即在当前工作目录的 runtime_log/ 下新建以时间戳命名的日志文件
    ms_fusion::RuntimeLog::startFileLogging();

    // 初始化ROS2
    rclcpp::init(argc, argv);

    // 设置ROS2的参数
    auto options = rclcpp::NodeOptions();
    options.automatically_declare_parameters_from_overrides(true);
    options.allow_undeclared_parameters(true);

    // 创建ROS2节点
    auto ros_node = std::make_shared<rclcpp::Node>("ms_fusion_node", options);
    
    // 创建多传感器融合节点
    auto ms_fusion_node = std::make_shared<ms_fusion::MsFusionNode>(ros_node);

    // 设置ROS2节点多线程运行
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(ros_node);
    executor.spin();

    // 停止融合系统的工作线程。必须在 ms_fusion_node 析构之前：线程里调用的发布回调绑定在
    // ms_fusion_node 上，节点先析构的话，正在处理的那一帧发布时就是 use-after-free
    ms_fusion::stopMsFusion();

    // 关闭ROS2
    rclcpp::shutdown();

    // flush 并关闭本地日志文件
    ms_fusion::RuntimeLog::stopFileLogging();

    // 返回
    return EXIT_SUCCESS;
}