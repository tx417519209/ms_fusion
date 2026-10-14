================================================================================
  third_party_axera 库使用说明
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


【infer_api.h — YOLO-OBB 推理接口（ax_yolo_obb_infer_node）】

    继承自 ax_infer_base，禁用拷贝。

  1. init_model(model_file) -> int
     初始化 NPU 引擎并加载 .axmodel 文件；内部依次调用：
       AX_SYS_Init → AX_ENGINE_Init → 加载模型 → CreateHandle → CreateContext
       → GetIOInfo → 分配输出缓冲（输入缓冲由外部提供，不在此分配）
     - 返回 0 表示成功；非 0 或重复调用返回 -1

  2. deinit_model()
     释放输出缓冲、销毁 Handle、反初始化引擎和 SYS。
     - 未初始化时调用为 no-op

  3. get_input_info() -> map<string, vector<int>>
     返回模型每个输入张量的名字和 shape（用于外部确定缩放尺寸和 buffer 大小）。

  4. get_output_info() -> map<string, vector<int>>
     返回模型每个输出张量的名字和 shape（调试/验证用）。

  5. set_config(cfg)
     设置后处理参数（infer_config 结构体）：
       - conf_thresh    : 置信度阈值（默认 0.25）
       - nms_thresh     : NMS IoU 阈值（默认 0.45）
       - num_classes    : 类别数（默认 80，需按实际模型设置）
       - max_det        : 最大检测框数（默认 300）
       - restore_width  : 原图宽（>0 时将检测框坐标从模型尺寸映射回原图尺寸）
       - restore_height : 原图高（同上）

  6. get_config() -> infer_config
     返回当前配置副本。

  7. bind_external_inputs(inputs, input_size)
     将外部分配的 AX_ENGINE_IO_BUFFER_T 数组绑定为模型输入。
     - 输入内存的生命周期由调用方管理，库不持有所有权

  8. check_input_binding(verbose=true) -> bool
     验证已绑定输入的数量、指针、缓冲大小是否与模型要求一致。
     推荐在 run_model 前显式调用。

  9. run_model()
     执行同步推理（AX_ENGINE_RunSync），推理完成后自动调用 post_process()。

  10. get_last_detections() -> const vector<obb_detection>&
      返回上次推理的 OBB 检测结果（NMS 后，按 score 降序排列）。
      obb_detection 字段：cx, cy, w, h, angle(rad), score, class_id

  11. debug_info()
      打印模型初始化状态及所有输入输出张量的名称、数据类型、颜色空间、shape。


--------------------------------------------------------------------------------
二、API 调用序列（最简顺序）
--------------------------------------------------------------------------------

【场景 A：离线单图推理（test_infer_once.cpp）】

  -- 初始化（一次性）--
  1.  infer.init_model(model_path)
  2.  infer.get_input_info()              // 获取 input_h / input_w / input_c
  3.  infer.set_config(cfg)               // 含 restore_width/height = 原图尺寸

  -- 准备输入（一次性，此场景单帧）--
  4.  AX_SYS_MemAllocCached(...)          // 手动分配物理输入 buffer
  5.  memcpy(ext_input.pVirAddr, img_data, input_bytes)
  6.  AX_SYS_MflushCache(...)            // CPU 写完后必须 flush

  -- 推理 --
  7.  infer.bind_external_inputs(&ext_input, 1)
  8.  infer.check_input_binding()         // 可选，推荐
  9.  infer.run_model()
  10. infer.get_last_detections()         // 结果坐标已还原到原图分辨率

  -- 清理 --
  11. AX_SYS_MemFree(ext_input.phyAddr, ext_input.pVirAddr)
  12. infer.deinit_model()


【场景 B：在线双目推理（test_infer_from_ivps.cpp）】

  -- 初始化（一次性）--
  1.  infer.init_model(model_path)
  2.  infer.get_input_info()              // 获取 input_h / input_w / input_c
  3.  alloc_resized_rgb_frame(frame_left_resized,  input_h, input_w, input_c)
  4.  alloc_resized_rgb_frame(frame_right_resized, input_h, input_w, input_c)

  -- 推理循环 --
  5.  grab_sync_images(frame_left, frame_right)
  6.  infer.set_config(cfg)               // 首次：补填 restore_width/height
  7.  resize_image_ivps(frame_left,  frame_left_resized,  input_h, input_w)
  8.  resize_image_ivps(frame_right, frame_right_resized, input_h, input_w)
      // 对左、右各做一次：
  9.    infer.bind_external_inputs(&ext_input[view], 1)
  10.   infer.check_input_binding()
  11.   infer.run_model()
  12.   infer.get_last_detections()        // 结果坐标已还原到原图分辨率
      // （可选）可视化：
  13.   get_images_addr(frame_left/right)  // 获取原图 CPU 地址用于绘图
  14. release_sync_images(frame_left, frame_right)   // 循环末尾必须释放
      // 回到步骤 5

  -- 清理 --
  15. free_resized_frame(frame_left_resized)
  16. free_resized_frame(frame_right_resized)
  17. infer.deinit_model()


  两场景的 infer_api 调用路径完全相同：
    init_model → get_input_info → set_config
    → [bind_external_inputs → check_input_binding → run_model → get_last_detections] × N
    → deinit_model
  区别仅在于输入 buffer 由谁分配、如何填充数据（CPU memcpy vs. IVPS 硬件缩放）。


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

【推理配置】

  5. num_classes 必须与模型一致。
     后处理依赖 num_classes 识别分类输出分支（按最后一维 == num_classes 匹配）。
     若设置错误会导致 bbox/cls/angle 索引识别混乱，输出错误结果但不报错。

  6. set_config 必须在 init_model 之后调用。
     update_restore_scale 需要读取 _io_info 中的模型输入 HW，若在 init 前调用
     则还原比例不会生效（scale 保持 1.0，不做坐标映射）。

  7. restore_width/restore_height 设置一次即可复用。
     只要输入分辨率不变，只需在第一帧后设置一次，循环内无需重复调用 set_config。

  8. 模型实例不可共享给多线程并发调用。
     _io_data、_last_detections 是实例内部状态，多线程并发 run_model 会产生数据竞争。
     多线程场景请各线程独立持有实例。

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
