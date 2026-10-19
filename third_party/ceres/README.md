# third_party/ceres

x86_64 预编译 Ceres Solver 2.2.0（静态库，Release，`BUILD_TESTING/EXAMPLES/BENCHMARKS=OFF`），
用于 `scripts/build_ms_fusion_x86.sh`。工程用到 `ceres::Manifold`，需要 Ceres >= 2.1，而
Ubuntu 22.04 的 `libceres-dev` 是 2.0。

- 安装前缀：`third_party/ceres/x86_64`（`lib/cmake/Ceres/CeresConfig.cmake`）
- `msf_core/CMakeLists.txt` 在该目录存在时自动使用；`-DMSF_USE_VENDORED_CERES=OFF` 改用系统 Ceres
- 系统仍需安装其依赖：`libgoogle-glog-dev libgflags-dev libeigen3-dev libsuitesparse-dev`
- 重新生成：`git clone -b 2.2.0 https://github.com/ceres-solver/ceres-solver.git && cmake -S . -B b -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_INSTALL_PREFIX=<repo>/third_party/ceres/x86_64 && cmake --build b -j && cmake --install b`
