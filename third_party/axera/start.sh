#!/bin/bash

# 设置LD_LIBRARY_PATH环境变量
export LD_LIBRARY_PATH=/soc/lib/

# 加载ROS环境配置（可通过 ROS_SETUP 覆盖）
if [ -n "${ROS_SETUP:-}" ] && [ -f "${ROS_SETUP}" ]; then
    source "${ROS_SETUP}"
elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "Error: ROS setup.bash not found" >&2
    exit 1
fi

# 进入build目录
cd build || { echo "错误: 无法进入build目录"; exit 1; }

# 运行superpoint_msg_app
./superpoint_ivps_app ../asset/models/superpoint_v6_fp32_640384.axmodel -s 0.3 -n 5 -k 150 -v
