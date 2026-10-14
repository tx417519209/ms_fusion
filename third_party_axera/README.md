# third_party_axera

基于 Axera AX650 NPU 的双目在线抓图与 YOLO-OBB、rf-detr、yolo26-pose ... 等 推理封装库。

---

## 构建

```bash
./build.sh                        
```

产物：`build/myapp`（离线单图推理）、`build/online_app`（在线双目推理）

---

## 公共接口

### grab_images_api.h

| 函数 | 签名 | 说明 |
|------|------|------|
| `grab_sync_images` | `(AX_VIDEO_FRAME_T& left, AX_VIDEO_FRAME_T& right) -> bool` | 从 IVPS 同步抓取左右相机帧，校验序列号一致后返回；失败时不持有任何帧资源 |
| `release_sync_images` | `(AX_VIDEO_FRAME_T& left, AX_VIDEO_FRAME_T& right)` | 释放一对同步帧（含 Munmap + ReleaseChnFrame），抓图成功后必须调用 |
| `get_images_addr` | `(AX_VIDEO_FRAME_T& frame) -> void*` | 返回帧的 CPU 可读虚拟地址；仅有物理地址时自动 MmapCache 并写回结构体 |
| `alloc_resized_rgb_frame` | `(AX_VIDEO_FRAME_T& dst, int h, int w, int c) -> bool` | 在 CMM Cached 物理内存中分配 RGB888 帧缓冲，用作存储缩放/裁剪后的图像 |
| `free_resized_frame` | `(AX_VIDEO_FRAME_T& dst)` | 释放 `alloc_resized_rgb_frame` 分配的帧缓冲并清零结构体 |
| `resize_image_ivps` | `(const AX_VIDEO_FRAME_T& src, AX_VIDEO_FRAME_T& dst, int h, int w) -> bool` | 用 IVPS VPP 硬件将 src 拉伸缩放到 dst 指定尺寸，完成后自动 invalidate cache |

### infer_api.h

| 函数 | 签名 | 说明 |
|------|------|------|
| `init_model` | `(std::string model_file) -> int` | 初始化 AX SYS / NPU 引擎，加载 `.axmodel`，分配输出缓冲；返回 0 成功 |
| `deinit_model` | `()` | 释放输出缓冲、销毁引擎句柄、反初始化 SYS；未初始化时为 no-op |
| `set_config` | `(const infer_config& cfg)` | 设置后处理参数：置信度阈值、NMS 阈值、类别数、最大检测数、坐标还原分辨率 |
| `bind_external_inputs` | `(AX_ENGINE_IO_BUFFER_T* inputs, AX_U32 size)` | 绑定外部输入缓冲（库不分配、不释放输入内存） |
| `check_input_binding` | `(bool verbose=true) -> bool` | 校验已绑定输入的数量、指针和缓冲大小是否符合模型要求 |
| `run_model` | `()` | 执行同步推理（AX_ENGINE_RunSync），完成后自动调用后处理 |
| `get_last_detections` | `() -> const vector<obb_detection>&` | 返回上次推理的 OBB 检测结果（NMS 后，按置信度降序） |

<!-- | `debug_info` | `()` | 打印模型初始化状态及所有张量的名称、数据类型、颜色空间、shape |
| `get_input_info` | `() -> map<string, vector<int>>` | 返回所有输入张量的名字与 shape，用于外部确定缩放尺寸和 buffer 大小 |
| `get_output_info` | `() -> map<string, vector<int>>` | 返回所有输出张量的名字与 shape，供调试验证 |
| `get_config` | `() -> infer_config` | 返回当前后处理配置的副本 | -->

`obb_detection` 结构体字段：`cx, cy, w, h, angle`（弧度）, `score`, `class_id`

---

## 在自己的项目中使用本库

本库为**纯头文件库**，无需编译为独立 `.so`，直接将 `include/` 目录加入项目即可。

### 目录结构（交付物）

```
third_party_axera/
├── include/
│   ├── infer_api.h          # 推理 API
│   ├── grab_images_api.h    # 抓图 API
│   ├── ax_sys_api.h         # Axera SYS（头文件）
│   ├── ax_engine_api.h      # Axera NPU Engine（头文件）
│   ├── ax_ivps_api.h        # Axera IVPS（头文件）
│   └── utilities/
│       └── file.hpp
└── README.md
```

运行时依赖的 `.so` 由设备 `/soc/lib/` 提供，无需随库分发。

### CMake 接入（推荐）

在你的 `CMakeLists.txt` 中添加：

```cmake
# 1. 引入头文件目录
include_directories(/path/to/third_party_axera/include)

# 2. 链接 Axera 运行时库（仅推理场景）
target_link_libraries(your_target
    /soc/lib/libax_sys.so
    /soc/lib/libax_engine.so
    /soc/lib/libax_interpreter.so
)

# 3. 若使用在线抓图 API，额外链接 IVPS
target_link_libraries(your_target
    /soc/lib/libax_ivps.so
)

# 4. 设置运行时库搜索路径（可选，避免手动设置 LD_LIBRARY_PATH）
set_target_properties(your_target PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib"
)
```

### 代码中引入

```cpp
#include "infer_api.h"        // 推理（必选）
#include "grab_images_api.h"  // 在线抓图（可选，仅在线场景）
```

### 运行时要求

| 依赖 | 路径 | 说明 |
|------|------|------|
| `libax_sys.so` | `/soc/lib/` | Axera 系统基础库，始终需要 |
| `libax_engine.so` | `/soc/lib/` | NPU 推理引擎 |
| `libax_interpreter.so` | `/soc/lib/` | NPU 模型解析器 |
| `libax_ivps.so` | `/soc/lib/` | IVPS 图像处理，仅在线抓图场景需要 |
| OpenCV | 系统安装 | `可选，用于可视化` |
| `.axmodel` 文件 | 自行提供 | 编译好的 Axera 模型文件 |

---

## 典型调用流程

```
init_model → get_input_info → set_config
  └─[循环] bind_external_inputs → check_input_binding → run_model → get_last_detections
→ deinit_model
```

输入 buffer 准备方式因场景而异：

- **离线**：`AX_SYS_MemAllocCached` + `memcpy` + `AX_SYS_MflushCache`
- **在线**：`alloc_resized_rgb_frame` + `grab_sync_images` + `resize_image_ivps`

详见 [API_GUIDE.txt](API_GUIDE.txt) 和 `example/` 目录。
