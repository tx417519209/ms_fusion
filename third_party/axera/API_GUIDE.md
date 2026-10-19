================================================================================
  third_party/axera 库使用说明
  （抓图 API + 推理 API）
================================================================================

--------------------------------------------------------------------------------
一、公共接口列表
--------------------------------------------------------------------------------

【grab_images_api.h — 双目在线抓图接口】

  1. grab_sync_images(frame_left, frame_right) -> bool
     从 IVPS 通道同步抓取左右相机帧（硬件对齐序列号）。
     - 左路：GROUP=1, CHN=0；右路：GROUP=0, CHN=0（宏定义固定）
     - 超时 200ms/次，最多重试 10 次 (MAX_SYNC_RETRIES)
     - 返回 true 时两帧序列号保证相同；返回 false 时不持有任何帧资源

  2. get_images_addr(frame_src) -> void*
     返回帧的虚拟地址指针（CPU 可读）。
     - 若帧本身已有虚拟地址则直接返回
     - 若只有物理地址则调用 AX_SYS_MmapCache 映射，并写回 frame_src.u64VirAddr[0]
     - 返回 nullptr 表示映射失败

  3. release_sync_images(frame_left, frame_right)
     释放一对同步帧：先 Munmap（若有虚拟地址映射），再 ReleaseChnFrame，最后清零结构体。
     对抓图成功的帧必须调用，否则 IVPS 帧池耗尽。

  4. alloc_resized_rgb_frame(frame_dst, input_h, input_w, input_c) -> bool
     在物理内存（CMM Cached）分配一块 RGB 帧缓冲，用作 IVPS 缩放的目标。
     - 格式固定为 AX_FORMAT_RGB888
     - stride = input_w * input_c；frame_size = input_h * stride

  5. free_resized_frame(frame_dst)
     释放 alloc_resized_rgb_frame 分配的帧缓冲并清零结构体。

  6. resize_image_ivps(frame_src, frame_dst, input_h, input_w) -> bool
     用 IVPS VPP 硬件将 frame_src 拉伸缩放到 frame_dst 指定尺寸（STRETCH 模式）。
     - frame_dst 必须已由 alloc_resized_rgb_frame 分配好
     - 完成后自动调用 AX_SYS_MinvalidateCache 使 CPU 可见新数据


--------------------------------------------------------------------------------
三、注意事项（容易出错的点）
--------------------------------------------------------------------------------

【内存管理】

  1. 原始帧必须在同一次 grab 内释放。
     grab_sync_images 成功后，不论后续步骤是否出错，都必须调用
     release_sync_images。漏释放将导致 IVPS 帧池耗尽，后续抓图超时失败。

  2. get_images_addr 会修改帧结构体。
     若 frame 只有物理地址，get_images_addr 会调用 MmapCache 并写回
     u64VirAddr[0]。release_sync_images 会识别该字段并自动 Munmap，
     因此不需要手动 unmap，但也不能在 release 后继续使用该指针。

  3. resize 目标帧必须预先分配。
     resize_image_ivps 要求 frame_dst 的物理/虚拟地址、宽高、格式、stride、
     framesize 全部已正确填写，否则会在内部校验时失败返回 false。
     推荐统一使用 alloc_resized_rgb_frame 分配以确保字段一致。

  4. 离线场景 CPU 写数据后必须 flush cache 才能送推理。
     memcpy 写完后必须调用 AX_SYS_MflushCache，否则 NPU 读到的是旧数据。
     在线场景 resize_image_ivps 内部已调用 MinvalidateCache，无需重复操作。

【初始化与资源】

  9. init_model 内部调用 AX_SYS_Init 和 AX_ENGINE_Init。
     若进程中其他模块已调用过这两个接口，需确认引用计数语义，避免重复初始化
     与提前反初始化冲突。同理 deinit_model 会调用 AX_SYS_Deinit，可能影响
     进程中其他使用 AX SYS 的代码。

  10. init_model 不可重复调用。
      对同一实例调用两次 init_model 会直接返回 -1（_initialized 保护），
      若需重新加载模型，必须先 deinit_model。

【抓图同步】

  11. 序列号同步最多重试 10 次 (MAX_SYNC_RETRIES)。
      若双目相机帧率差异持续存在或 IVPS 管道积压严重，可能达到上限并返回 false，
      调用方需做错误处理（如跳过本帧重试）而非直接崩溃。

  12. grab_sync_images 的 200ms 超时是单次 GetChnFrame 的等待上限。
      在重试循环内最坏情况下总等待时间可达约 200ms × (2 × MAX_SYNC_RETRIES)
      = 4 秒，对实时系统需评估是否可接受。

================================================================================
