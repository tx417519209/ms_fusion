#pragma once
// sys dependency
#include <stdio.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cmath>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
// axera dependency
#include <ax_sys_api.h>
#include <ax_engine_api.h>

// user dependency
#include "utilities/file.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define AX_CMM_ALIGN_SIZE 128
static constexpr const char* AX_CMM_SESSION_NAME = "ax-samples-cmm";
typedef enum
{
    AX_ENGINE_ABST_DEFAULT = 0,
    AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> INPUT_OUTPUT_ALLOC_STRATEGY;

namespace ax_infer_log {
inline bool& sync_run_timing_enabled() {
    static bool enabled = false;
    return enabled;
}
}  // namespace ax_infer_log


class ax_infer_base {
    private:
        void free_io_index(AX_ENGINE_IO_BUFFER_T* io_buf, size_t index)
        {
            for (int i = 0; i < (int)index; ++i)
            {
                AX_ENGINE_IO_BUFFER_T* pBuf = io_buf + i;
                AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
            }
        }    
        int prepare_input_buffer(AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T* io_data, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T strategy)
        {
            auto ret = 0;
            io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
            memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);
            io_data->nInputSize = info->nInputSize;
            for (int i = 0; i < (int)info->nInputSize; ++i)
            {
                auto meta = info->pInputs[i];
                auto buffer = &io_data->pInputs[i];
                if (strategy == AX_ENGINE_ABST_CACHED)
                {
                    ret = AX_SYS_MemAllocCached((AX_U64*)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8*)(AX_CMM_SESSION_NAME));
                }
                else
                {
                    ret = AX_SYS_MemAlloc((AX_U64*)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8*)(AX_CMM_SESSION_NAME));
                }
                if (ret != 0)
                {
                    free_io_index(io_data->pInputs, i);
                    delete[] io_data->pInputs;
                    io_data->pInputs = nullptr;
                    io_data->nInputSize = 0;
                    fprintf(stderr, "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
                    return ret;
                }
                // fprintf(stderr, "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes }. \n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
            }
            return 0;
        }
        int prepare_output_buffer(AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T* io_data, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T strategy)
        {
            auto ret = 0;
            memset(io_data, 0, sizeof(*io_data));
            io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
            memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
            io_data->nOutputSize = info->nOutputSize;
            for (int i = 0; i < (int)info->nOutputSize; ++i)
            {
                auto meta = info->pOutputs[i];
                auto buffer = &io_data->pOutputs[i];
                buffer->nSize = meta.nSize;
                if (strategy == AX_ENGINE_ABST_CACHED)
                {
                    ret = AX_SYS_MemAllocCached((AX_U64*)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8*)(AX_CMM_SESSION_NAME));
                }
                else
                {
                    ret = AX_SYS_MemAlloc((AX_U64*)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8*)(AX_CMM_SESSION_NAME));
                }
                if (ret != 0)
                {
                    fprintf(stderr, "Allocate output{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
                    free_io_index(io_data->pOutputs, i);
                    delete[] io_data->pOutputs;
                    io_data->pOutputs = nullptr;
                    io_data->nOutputSize = 0;
                    return ret;
                }
            }
            return 0;
        }
        int free_input_buffer(AX_ENGINE_IO_T* io)
        {
            for (size_t j = 0; j < io->nInputSize; ++j)
            {
                AX_ENGINE_IO_BUFFER_T* pBuf = io->pInputs + j;
                AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
            }
            delete[] io->pInputs;
            return 0;
        }
        int free_output_buffer(AX_ENGINE_IO_T* io)
        {
            if (!io || !io->pOutputs)
            {
                return 0;
            }
            for (size_t j = 0; j < io->nOutputSize; ++j)
            {
                AX_ENGINE_IO_BUFFER_T* pBuf = io->pOutputs + j;
                AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
            }
            delete[] io->pOutputs;
            io->pOutputs = nullptr;
            io->nOutputSize = 0;
            return 0;
        }
        bool validate_input_buffers(const AX_ENGINE_IO_T* io, bool verbose = true) const
        {
            if (!_io_info)
            {
                if (verbose) fprintf(stderr, "Input validation failed: model io info is null.\n");
                return false;
            }
            if (!io || !io->pInputs)
            {
                if (verbose) fprintf(stderr, "Input validation failed: input buffers are null.\n");
                return false;
            }
            if (io->nInputSize != _io_info->nInputSize)
            {
                if (verbose) fprintf(stderr, "Input validation failed: input size mismatch, got %u expect %u.\n", io->nInputSize, _io_info->nInputSize);
                return false;
            }
            for (AX_U32 i = 0; i < io->nInputSize; ++i)
            {
                const AX_ENGINE_IO_BUFFER_T* buf = io->pInputs + i;
                const AX_ENGINE_IOMETA_T& meta = _io_info->pInputs[i];
                if (!buf->pVirAddr)
                {
                    if (verbose) fprintf(stderr, "Input validation failed: input[%u] vir addr is null.\n", i);
                    return false;
                }
                if (buf->nSize < meta.nSize)
                {
                    if (verbose) fprintf(stderr, "Input validation failed: input[%u] size too small, got %u expect >= %u.\n", i, buf->nSize, meta.nSize);
                    return false;
                }
            }
            return true;
        }
        static void print_io_info(AX_ENGINE_IO_INFO_T* io_info)
        {
            static std::map<AX_ENGINE_DATA_TYPE_T, const char*> data_type = {
                {AX_ENGINE_DT_UNKNOWN, "UNKNOWN"},
                {AX_ENGINE_DT_UINT8, "UINT8"},
                {AX_ENGINE_DT_UINT16, "UINT16"},
                {AX_ENGINE_DT_FLOAT32, "FLOAT32"},
                {AX_ENGINE_DT_SINT16, "SINT16"},
                {AX_ENGINE_DT_SINT8, "SINT8"},
                {AX_ENGINE_DT_SINT32, "SINT32"},
                {AX_ENGINE_DT_UINT32, "UINT32"},
                {AX_ENGINE_DT_FLOAT64, "FLOAT64"},
                {AX_ENGINE_DT_UINT10_PACKED, "UINT10_PACKED"},
                {AX_ENGINE_DT_UINT12_PACKED, "UINT12_PACKED"},
                {AX_ENGINE_DT_UINT14_PACKED, "UINT14_PACKED"},
                {AX_ENGINE_DT_UINT16_PACKED, "UINT16_PACKED"},
            };
    
            static std::map<AX_ENGINE_COLOR_SPACE_T, const char*> color_type = {
                {AX_ENGINE_CS_FEATUREMAP, "FEATUREMAP"},
                {AX_ENGINE_CS_RAW8, "RAW8"},
                {AX_ENGINE_CS_RAW10, "RAW10"},
                {AX_ENGINE_CS_RAW12, "RAW12"},
                {AX_ENGINE_CS_RAW14, "RAW14"},
                {AX_ENGINE_CS_RAW16, "RAW16"},
                {AX_ENGINE_CS_NV12, "NV12"},
                {AX_ENGINE_CS_NV21, "NV21"},
                {AX_ENGINE_CS_RGB, "RGB"},
                {AX_ENGINE_CS_BGR, "BGR"},
                {AX_ENGINE_CS_RGBA, "RGBA"},
                {AX_ENGINE_CS_GRAY, "GRAY"},
                {AX_ENGINE_CS_YUV444, "YUV444"},
            };
            printf("\ninput size: %d\n", io_info->nInputSize);
            for (uint32_t i = 0; i < io_info->nInputSize; ++i)
            {
                // print shape info,like [batchsize x channel x height x width]
                auto& info = io_info->pInputs[i];
                printf("    name: \033[1;32m%8s", info.pName);
    
                std::string dt = "unknown";
                if (data_type.find(info.eDataType) != data_type.end())
                {
                    dt = data_type[info.eDataType];
                    printf(" \033[1;34m[%s] ", dt.c_str());
                }
                else
                {
                    printf(" \033[1;31m[%s] ", dt.c_str());
                }
    
                std::string ct = "unknown";
                if (info.pExtraMeta && color_type.find(info.pExtraMeta->eColorSpace) != color_type.end())
                {
                    ct = color_type[info.pExtraMeta->eColorSpace];
                    printf("\033[1;34m[%s]", ct.c_str());
                }
                else
                {
                    printf("\033[1;31m[%s]", ct.c_str());
                }
                printf(" \n        \033[1;31m");
    
                for (AX_U8 s = 0; s < info.nShapeSize; s++)
                {
                    printf("%d", info.pShape[s]);
                    if (s != info.nShapeSize - 1)
                    {
                        printf(" x ");
                    }
                }
                printf("\033[0m\n\n");
            }
    
            printf("\noutput size: %d\n", io_info->nOutputSize);
            for (uint32_t i = 0; i < io_info->nOutputSize; ++i)
            {
                // print shape info,like [batchsize x channel x height x width]
                auto& info = io_info->pOutputs[i];
                printf("    name: \033[1;32m%8s \033[1;34m[%s]\033[0m\n        \033[1;31m", info.pName, data_type[info.eDataType]);
                for (AX_U8 s = 0; s < info.nShapeSize; s++)
                {
                    printf("%d", info.pShape[s]);
                    if (s != info.nShapeSize - 1)
                    {
                        printf(" x ");
                    }
                }
                printf("\033[0m\n\n");
            }
        }

    protected:
        AX_ENGINE_HANDLE _handle;
        AX_ENGINE_IO_T _io_data;
        AX_ENGINE_IO_INFO_T *_io_info;
        int _ret;
        bool _initialized;

    public:
        // 禁用拷贝构造和赋值
        ax_infer_base(const ax_infer_base&) = delete;
        ax_infer_base& operator=(const ax_infer_base&) = delete;
        ax_infer_base() : _handle(0), _io_info(0), _ret(0), _initialized(false)
        {
            memset(&_io_data, 0, sizeof(_io_data));
        }
        virtual ~ax_infer_base() = default;


        //// 外部开放接口----------------------------------------------------
        // 绑定外部输入内存（base 不分配、不释放输入内存）
        virtual void bind_external_inputs(AX_ENGINE_IO_BUFFER_T* inputs, AX_U32 input_size)
        {
            _io_data.pInputs = inputs;
            _io_data.nInputSize = input_size;
        }
        // 运行前可调用该函数检查外部输入绑定是否合法
        bool check_input_binding(bool verbose = true) const
        {
            return validate_input_buffers(&_io_data, verbose);
        }
        // 获取输入 tensor 信息（名字 -> 元素个数），用于外部动态构造输入
        std::map<std::string, std::vector<int>> get_input_info() const
        {
            std::map<std::string, std::vector<int>> info;
            if (!_io_info) return info;
            for (AX_U32 i = 0; i < _io_info->nInputSize; ++i)
            {
                std::string name(_io_info->pInputs[i].pName);
                std::vector<int> shape(_io_info->pInputs[i].pShape,
                                    _io_info->pInputs[i].pShape + _io_info->pInputs[i].nShapeSize);
                info[name] = shape;
            }
            return info;
        }

        // 获取输出 tensor 信息（名字 -> shape）
        std::map<std::string, std::vector<int>> get_output_info() const
        {
            std::map<std::string, std::vector<int>> info;
            if (!_io_info) return info;
            for (AX_U32 i = 0; i < _io_info->nOutputSize; ++i)
            {
                std::string name(_io_info->pOutputs[i].pName);
                std::vector<int> shape(_io_info->pOutputs[i].pShape,
                                    _io_info->pOutputs[i].pShape + _io_info->pOutputs[i].nShapeSize);
                info[name] = shape;
            }
            return info;
        }

        // 打印模型的信息
        void debug_info() const
        {
            if (!ax_infer_log::sync_run_timing_enabled()) {
                return;
            }
            if(_ret != 0)
            {
                std::cerr << "Model initialization failed, ret: " << _ret << "\n";
                return;
            }
            else{
                std::cout << "Model initialization success, ret: " << _ret << "\n";
            }
            if (!_io_info) {
                std::cerr << "Model not initialized.\n";
                return;
            }
            print_io_info(_io_info);
        }

        int init_model(std::string model_file){
            if (!utilities::file_exist(model_file))
            {
                std::cerr << "Model not found.\n";
                return -1;
            }
            if(_initialized)
            {
                std::cerr << "Model already initialized.\n";
                return -1;
            }
            AX_SYS_Init();
            // 1. init engine
            AX_ENGINE_NPU_ATTR_T npu_attr;
            memset(&npu_attr, 0, sizeof(npu_attr));
            npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_STD;
            _ret = AX_ENGINE_Init(&npu_attr);
            if (0 != _ret)
            {
                return _ret;
            }
            
            // 2. load model
            std::vector<char> model_buffer;
            if (!utilities::read_file(model_file, model_buffer))
            {
                fprintf(stderr, "Read Run-Joint model(%s) file failed.\n", model_file.c_str());
                return -1;
            }    
            // 3. create handle
            
            _ret = AX_ENGINE_CreateHandle(&_handle, model_buffer.data(), model_buffer.size());
            if (0 != _ret || _handle == 0)                               
            {                                          
                return AX_ENGINE_DestroyHandle(_handle); 
            }
            (void)AX_ENGINE_SetAffinity(_handle, 0b010);
            // 4. create context
            
            _ret = AX_ENGINE_CreateContext(_handle);
            if (0 != _ret|| _handle == 0)                               
            {                                          
                return AX_ENGINE_DestroyHandle(_handle); 
            }  
            // 5. set io
            
            _ret = AX_ENGINE_GetIOInfo(_handle, &_io_info);
            if (0 != _ret || _io_info == 0)                               
            {                                          
                return AX_ENGINE_DestroyHandle(_handle); 
            }  

            // 6. alloc io
      

            // ## cached 内存：非 cached 内存下 CPU 侧 memcpy（scoresToCPU/descriptorsToCPU）
            // ## 明显更慢，切到 cached 后单帧可省约 11ms；各检测器自己拷贝前已经调用
            // ## AX_SYS_MinvalidateCache，不需要额外挪动 invalidate 时机
            _ret = prepare_output_buffer(_io_info, &_io_data, AX_ENGINE_ABST_CACHED);
            if (0 != _ret)                               
            {                                       
                return AX_ENGINE_DestroyHandle(_handle); 
            }     

            // _ret = prepare_input_buffer(_io_info, &_io_data, AX_ENGINE_ABST_DEFAULT);
            // if (0 != _ret)                               
            // {       
            //     free_output_buffer(&_io_data);                                      
            //     return AX_ENGINE_DestroyHandle(_handle); 
            // }   

            _initialized = true;
            return 0;
        }

        // 单输入多输出推理接口（派生类按模型实现）
        virtual void run_model() = 0;

        // 输出后处理（派生类按模型实现）
        virtual void post_process() = 0;

        virtual void deinit_model(){
            if (!_initialized)
            {
                return;
            }
            // free_input_buffer(&_io_data);
            free_output_buffer(&_io_data);
            AX_ENGINE_DestroyHandle(_handle);        
            _handle = nullptr;
            _io_info = nullptr;
            memset(&_io_data,0,sizeof(AX_ENGINE_IO_T));
            AX_ENGINE_Deinit();
            AX_SYS_Deinit();
            _initialized = false;
        }

};

class ax_yolo_obb_infer_node : public ax_infer_base
{
    public:
        struct obb_detection
        {
            float cx = 0.0f;
            float cy = 0.0f;
            float w = 0.0f;
            float h = 0.0f;
            float angle = 0.0f;
            float score = 0.0f;
            int class_id = -1;
        };

        struct infer_config
        {
            float conf_thresh = 0.25f;
            float nms_thresh = 0.45f;
            int num_classes = 80;
            int max_det = 300;
            int nms_pre_topk = 1000;
            // 目标分辨率（通常为原图分辨率）。<=0 表示不做分辨率恢复。
            int restore_width = 0;
            int restore_height = 0;
        };

        ax_yolo_obb_infer_node() : ax_infer_base() {
            _cfg.conf_thresh = 0.25f;
            _cfg.nms_thresh = 0.45f;
            _cfg.num_classes = 80;
            _cfg.max_det = 300;
            _cfg.nms_pre_topk = 1000;
        }
        ~ax_yolo_obb_infer_node() override
        {
        }

        void set_config(const infer_config& cfg)
        {
            _cfg = cfg;
            update_restore_scale();
        }

        infer_config get_config() const
        {
            return _cfg;
        }

        const std::vector<obb_detection>& get_last_detections() const
        {
            return _last_detections;
        }

        void run_model() override
        {
            _last_detections.clear();

            if (!_initialized || !_io_info)
            {
                std::cerr << "YOLO-OBB run failed: model is not initialized.\n";
                return;
            }
            if (!check_input_binding(true))
            {
                return;
            }
            if (_handle == 0)
            {
                std::cerr << "YOLO-OBB run failed: invalid engine handle.\n";
                return;
            }

            _ret = AX_ENGINE_RunSync(_handle, &_io_data);
            if (_ret != 0)
            {
                std::cerr << "YOLO-OBB run failed: AX_ENGINE_RunSync ret=" << _ret << "\n";
                return;
            }
            post_process();
        }

        void post_process() override
        {
            process_outputs(_io_info,
                            _io_data,
                            _cfg,
                            _enable_restore,
                            _restore_scale_x,
                            _restore_scale_y,
                            _last_detections);
        }

    protected:
        void get_restore_params(bool& enable_restore, float& scale_x, float& scale_y) const
        {
            enable_restore = _enable_restore;
            scale_x = _restore_scale_x;
            scale_y = _restore_scale_y;
        }
        void set_last_detections_compat(const std::vector<obb_detection>& dets)
        {
            _last_detections = dets;
        }

        static bool process_outputs(const AX_ENGINE_IO_INFO_T* io_info,
                                    const AX_ENGINE_IO_T& io_data,
                                    const infer_config& cfg,
                                    bool enable_restore,
                                    float restore_scale_x,
                                    float restore_scale_y,
                                    std::vector<obb_detection>& out_detections)
        {
            out_detections.clear();
            int bbox_idx = -1;
            int cls_idx = -1;
            int angle_idx = -1;
            if (!find_obb_output_indices(io_info, cfg, bbox_idx, cls_idx, angle_idx))
            {
                std::cerr << "YOLO-OBB post process failed: can not identify bbox/cls/angle outputs.\n";
                return false;
            }

            const float* output_bbox = reinterpret_cast<const float*>(io_data.pOutputs[bbox_idx].pVirAddr);
            const float* output_cls = reinterpret_cast<const float*>(io_data.pOutputs[cls_idx].pVirAddr);
            const float* output_angle = reinterpret_cast<const float*>(io_data.pOutputs[angle_idx].pVirAddr);
            if (!output_bbox || !output_cls || !output_angle)
            {
                std::cerr << "YOLO-OBB post process failed: output tensor pointer is null.\n";
                return false;
            }

            const AX_U32 bbox_float_count = io_info->pOutputs[bbox_idx].nSize / sizeof(float);
            const AX_U32 cls_float_count = io_info->pOutputs[cls_idx].nSize / sizeof(float);
            const AX_U32 angle_float_count = io_info->pOutputs[angle_idx].nSize / sizeof(float);
            const AX_U32 num_points = bbox_float_count / 4;
            if (num_points == 0 || cls_float_count == 0 || angle_float_count == 0)
            {
                std::cerr << "YOLO-OBB post process failed: invalid output size.\n";
                return false;
            }

            const AX_U32 cls_dim = get_last_dim(io_info->pOutputs[cls_idx]);
            if (cls_dim == 0)
            {
                std::cerr << "YOLO-OBB post process failed: invalid cls output shape.\n";
                return false;
            }
            const AX_U32 cls_num = std::min<AX_U32>(cls_dim, static_cast<AX_U32>(cfg.num_classes));
            const AX_U32 valid_points = std::min<AX_U32>(num_points, std::min<AX_U32>(cls_float_count / cls_dim, angle_float_count));
            for (AX_U32 i = 0; i < valid_points; ++i)
            {
                int class_index = -1;
                float class_score = std::numeric_limits<float>::lowest();
                const AX_U32 cls_base = i * cls_dim;
                for (AX_U32 c = 0; c < cls_num; ++c)
                {
                    const float score = output_cls[cls_base + c];
                    if (score > class_score)
                    {
                        class_score = score;
                        class_index = static_cast<int>(c);
                    }
                }
                if (class_score < cfg.conf_thresh || class_index < 0)
                {
                    continue;
                }

                obb_detection det;
                det.cx = output_bbox[i * 4 + 0];
                det.cy = output_bbox[i * 4 + 1];
                det.w = output_bbox[i * 4 + 2];
                det.h = output_bbox[i * 4 + 3];
                det.angle = output_angle[i];
                det.score = class_score;
                det.class_id = class_index;
                out_detections.emplace_back(det);
            }

            const int pre_topk = cfg.nms_pre_topk > 0 ? cfg.nms_pre_topk : cfg.max_det;
            if (pre_topk > 0 && static_cast<int>(out_detections.size()) > pre_topk)
            {
                std::nth_element(
                    out_detections.begin(),
                    out_detections.begin() + pre_topk,
                    out_detections.end(),
                    [](const obb_detection& a, const obb_detection& b) { return a.score > b.score; });
                out_detections.resize(pre_topk);
            }

            apply_nms(out_detections, cfg.nms_thresh);

            if (enable_restore)
            {
                for (auto& det : out_detections)
                {
                    restore_det_stretch(det, restore_scale_x, restore_scale_y);
                }
            }

            if (cfg.max_det > 0 && static_cast<int>(out_detections.size()) > cfg.max_det)
            {
                out_detections.resize(cfg.max_det);
            }
            return true;
        }

    private:
        static bool parse_hw_from_iometa(const AX_ENGINE_IOMETA_T& item, AX_U32& h, AX_U32& w)
        {
            if (item.nShapeSize < 3 || !item.pShape)
            {
                return false;
            }
            // 兼容 NCHW / NHWC / HWC
            if (item.nShapeSize == 4)
            {
                if (item.pShape[1] == 1 || item.pShape[1] == 3)
                {
                    h = static_cast<AX_U32>(item.pShape[2]);
                    w = static_cast<AX_U32>(item.pShape[3]);
                    return true;
                }
                h = static_cast<AX_U32>(item.pShape[1]);
                w = static_cast<AX_U32>(item.pShape[2]);
                return true;
            }
            h = static_cast<AX_U32>(item.pShape[0]);
            w = static_cast<AX_U32>(item.pShape[1]);
            return true;
        }

        bool get_model_input_hw(AX_U32& h, AX_U32& w) const
        {
            h = 0;
            w = 0;
            if (!_io_info || _io_info->nInputSize == 0 || !_io_info->pInputs)
            {
                return false;
            }
            return parse_hw_from_iometa(_io_info->pInputs[0], h, w);
        }

        void update_restore_scale()
        {
            _enable_restore = false;
            _restore_scale_x = 1.0f;
            _restore_scale_y = 1.0f;
            if (_cfg.restore_width <= 0 || _cfg.restore_height <= 0)
            {
                return;
            }

            AX_U32 model_h = 0;
            AX_U32 model_w = 0;
            if (!get_model_input_hw(model_h, model_w) || model_w == 0 || model_h == 0)
            {
                return;
            }

            _restore_scale_x = static_cast<float>(_cfg.restore_width) / static_cast<float>(model_w);
            _restore_scale_y = static_cast<float>(_cfg.restore_height) / static_cast<float>(model_h);
            _enable_restore = (_restore_scale_x != 1.0f || _restore_scale_y != 1.0f);
        }

        static void restore_det_stretch(obb_detection& det, float scale_x, float scale_y)
        {
            const float hw = std::max(det.w, 1.0f) * 0.5f;
            const float hh = std::max(det.h, 1.0f) * 0.5f;
            const float cos_a = std::cos(det.angle);
            const float sin_a = std::sin(det.angle);

            // 先在模型输入分辨率下恢复四点
            cv::Point2f pts[4];
            pts[0] = cv::Point2f(det.cx - hw * cos_a + hh * sin_a, det.cy - hw * sin_a - hh * cos_a);
            pts[1] = cv::Point2f(det.cx + hw * cos_a + hh * sin_a, det.cy + hw * sin_a - hh * cos_a);
            pts[2] = cv::Point2f(det.cx + hw * cos_a - hh * sin_a, det.cy + hw * sin_a + hh * cos_a);
            pts[3] = cv::Point2f(det.cx - hw * cos_a - hh * sin_a, det.cy - hw * sin_a + hh * cos_a);

            // 点坐标按目标分辨率恢复
            for (auto& p : pts)
            {
                p.x *= scale_x;
                p.y *= scale_y;
            }

            // 由恢复后的四点重建旋转框
            const cv::RotatedRect rr_restored = cv::minAreaRect(std::vector<cv::Point2f>(pts, pts + 4));
            det.cx = rr_restored.center.x;
            det.cy = rr_restored.center.y;
            det.w = rr_restored.size.width;
            det.h = rr_restored.size.height;
            det.angle = rr_restored.angle * static_cast<float>(CV_PI) / 180.0f;
        }

        static AX_U32 get_last_dim(const AX_ENGINE_IOMETA_T& item)
        {
            if (item.nShapeSize == 0 || !item.pShape)
            {
                return 0;
            }
            return static_cast<AX_U32>(item.pShape[item.nShapeSize - 1]);
        }

        static bool find_obb_output_indices(const AX_ENGINE_IO_INFO_T* io_info,
                                            const infer_config& cfg,
                                            int& bbox_idx,
                                            int& cls_idx,
                                            int& angle_idx)
        {
            bbox_idx = -1;
            cls_idx = -1;
            angle_idx = -1;
            if (!io_info || !io_info->pOutputs)
            {
                return false;
            }
            for (AX_U32 i = 0; i < io_info->nOutputSize; ++i)
            {
                const AX_ENGINE_IOMETA_T& info = io_info->pOutputs[i];
                const AX_U32 last_dim = get_last_dim(info);
                if (last_dim == 4)
                {
                    bbox_idx = static_cast<int>(i);
                }
                else if (last_dim == 1)
                {
                    angle_idx = static_cast<int>(i);
                }
                else if (last_dim > 1 && cls_idx < 0)
                {
                    // 优先按配置类别数识别；若不匹配，退化为“剩余的大于1维输出”
                    cls_idx = static_cast<int>(i);
                }
                if (last_dim == static_cast<AX_U32>(cfg.num_classes))
                {
                    cls_idx = static_cast<int>(i);
                }
            }

            return bbox_idx >= 0 && cls_idx >= 0 && angle_idx >= 0;
        }

        static float iou_aabb(const obb_detection& a, const obb_detection& b)
        {
            const float ax1 = a.cx - a.w * 0.5f;
            const float ay1 = a.cy - a.h * 0.5f;
            const float ax2 = a.cx + a.w * 0.5f;
            const float ay2 = a.cy + a.h * 0.5f;
            const float bx1 = b.cx - b.w * 0.5f;
            const float by1 = b.cy - b.h * 0.5f;
            const float bx2 = b.cx + b.w * 0.5f;
            const float by2 = b.cy + b.h * 0.5f;

            const float inter_x1 = std::max(ax1, bx1);
            const float inter_y1 = std::max(ay1, by1);
            const float inter_x2 = std::min(ax2, bx2);
            const float inter_y2 = std::min(ay2, by2);
            const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
            const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
            const float inter_area = inter_w * inter_h;

            const float area_a = std::max(0.0f, (ax2 - ax1)) * std::max(0.0f, (ay2 - ay1));
            const float area_b = std::max(0.0f, (bx2 - bx1)) * std::max(0.0f, (by2 - by1));
            const float denom = area_a + area_b - inter_area;
            if (denom <= 1e-6f)
            {
                return 0.0f;
            }
            return inter_area / denom;
        }

        static void apply_nms(std::vector<obb_detection>& dets, float nms_thresh)
        {
            if (dets.empty() || nms_thresh <= 0.0f)
            {
                return;
            }

            std::sort(dets.begin(), dets.end(), [](const obb_detection& a, const obb_detection& b) {
                return a.score > b.score;
            });

            std::vector<obb_detection> kept;
            kept.reserve(dets.size());
            for (const auto& det : dets)
            {
                bool suppressed = false;
                for (const auto& k : kept)
                {
                    if (det.class_id != k.class_id)
                    {
                        continue;
                    }
                    if (iou_aabb(det, k) > nms_thresh)
                    {
                        suppressed = true;
                        break;
                    }
                }
                if (!suppressed)
                {
                    kept.emplace_back(det);
                }
            }
            dets.swap(kept);
        }

    private:
        infer_config _cfg;
        std::vector<obb_detection> _last_detections;
        float _restore_scale_x {1.0f};
        float _restore_scale_y {1.0f};
        bool _enable_restore {false};
};

class ax_yolo_obb_sync_infer_node : public ax_yolo_obb_infer_node
{
    public:
        ax_yolo_obb_sync_infer_node() = default;
        ~ax_yolo_obb_sync_infer_node() override
        {
            deinit_model();
        }

        // 必须在 init_model() 之前调用：亲和性是在 init_model() 里通过
        // AX_ENGINE_SetAffinity 落到左目主 handle（_handle）和右目 worker
        // （_right_worker）上的，init 之后再改不会生效。
        void set_npu_affinity(AX_ENGINE_NPU_SET_T left_npu_set, AX_ENGINE_NPU_SET_T right_npu_set)
        {
            _left_npu_set = left_npu_set;
            _right_npu_set = right_npu_set;
        }

        void bind_external_inputs(AX_ENGINE_IO_BUFFER_T* inputs, AX_U32 input_size) override
        {
            if (!inputs || input_size == 0)
            {
                _pending_inputs.clear();
                _pending_bind_invalid = true;
                _dual_mode_requested = false;
                ax_yolo_obb_infer_node::bind_external_inputs(inputs, input_size);
                return;
            }

            _pending_bind_invalid = false;
            if (input_size == 1)
            {
                // 显式单模型模式：仅使用第 1 个输入。
                _pending_inputs.assign(inputs, inputs + 1);
                _dual_mode_requested = false;
                ax_yolo_obb_infer_node::bind_external_inputs(&inputs[0], 1);
                return;
            }

            if (input_size == 2)
            {
                // 显式双模型模式：要求两个输入位都有效。
                if (!is_valid_bound_input(inputs[0]) || !is_valid_bound_input(inputs[1]))
                {
                    std::cerr << "sync infer bind failed: input_size=2 but one input is invalid.\n";
                    _pending_inputs.clear();
                    _pending_bind_invalid = true;
                    _dual_mode_requested = true;
                    ax_yolo_obb_infer_node::bind_external_inputs(inputs, 2);
                    return;
                }
                _pending_inputs.assign(inputs, inputs + 2);
                _dual_mode_requested = true;
                ax_yolo_obb_infer_node::bind_external_inputs(&inputs[0], 1);
                return;
            }

            _pending_inputs.clear();
            _pending_bind_invalid = true;
            _dual_mode_requested = false;
            ax_yolo_obb_infer_node::bind_external_inputs(inputs, input_size);
        }

        int init_model(const std::string& model_file)
        {
            _last_pair_detections = {};
            _pending_inputs.clear();
            _pending_bind_invalid = false;
            _dual_mode_requested = false;
            const int ret = ax_yolo_obb_infer_node::init_model(model_file);
            if (ret != 0)
            {
                return ret;
            }

            std::vector<char> model_buffer;
            if (!utilities::read_file(model_file, model_buffer))
            {
                std::cerr << "sync infer init failed: read model file failed.\n";
                ax_yolo_obb_infer_node::deinit_model();
                return -1;
            }

            (void)AX_ENGINE_SetAffinity(_handle, _left_npu_set);
            if (!init_worker(_right_worker, model_buffer, _right_npu_set))
            {
                cleanup_worker(_right_worker);
                ax_yolo_obb_infer_node::deinit_model();
                return -1;
            }
            if (!start_runtime_workers())
            {
                cleanup_worker(_right_worker);
                ax_yolo_obb_infer_node::deinit_model();
                return -1;
            }
            _dual_ready = true;
            return 0;
        }

        void deinit_model() override
        {
            stop_runtime_workers();
            cleanup_worker(_right_worker);
            _dual_ready = false;
            _pending_inputs.clear();
            _dual_mode_requested = false;
            _last_pair_detections = {};
            ax_yolo_obb_infer_node::deinit_model();
        }

        void run_model() override
        {
            const auto run_t0 = std::chrono::steady_clock::now();
            auto print_elapsed = [&run_t0](const char* mode) {
                if (!ax_infer_log::sync_run_timing_enabled()) {
                    return;
                }
                const auto run_t1 = std::chrono::steady_clock::now();
                const auto run_cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(run_t1 - run_t0).count();
                std::cout << "sync run_model elapsed: " << run_cost_ms << " ms, mode=" << mode << "\n";
            };

            _last_pair_detections = {};
            if (!can_run_dual_mode())
            {
                ax_yolo_obb_infer_node::run_model();
                _pending_inputs.clear();
                _dual_mode_requested = false;
                print_elapsed("single_fallback");
                return;
            }

            AX_ENGINE_IO_T parent_io_backup = _io_data;
            AX_ENGINE_IO_BUFFER_T* left_input = &_pending_inputs[0];
            AX_ENGINE_IO_BUFFER_T* right_input = &_pending_inputs[1];
            ax_yolo_obb_infer_node::bind_external_inputs(left_input, 1);
            _right_worker.io_data.pInputs = right_input;
            _right_worker.io_data.nInputSize = 1;

            if (!check_input_binding(true))
            {
                _io_data = parent_io_backup;
                ax_yolo_obb_infer_node::run_model();
                _pending_inputs.clear();
                _dual_mode_requested = false;
                print_elapsed("single_fallback_invalid_left_input");
                return;
            }
            if (!validate_single_input(_right_worker.io_info, _right_worker.io_data, true))
            {
                _io_data = parent_io_backup;
                ax_yolo_obb_infer_node::run_model();
                _pending_inputs.clear();
                _dual_mode_requested = false;
                print_elapsed("single_fallback_invalid_right_input");
                return;
            }
            bool enable_restore = false;
            float restore_scale_x = 1.0f;
            float restore_scale_y = 1.0f;
            get_restore_params(enable_restore, restore_scale_x, restore_scale_y);
            const infer_config cfg_snapshot = get_config();
            AX_ENGINE_IO_T left_io = parent_io_backup;
            left_io.pInputs = left_input;
            left_io.nInputSize = 1;
            AX_ENGINE_IO_T right_io = _right_worker.io_data;
            right_io.pInputs = right_input;
            right_io.nInputSize = 1;

            submit_task(_left_runtime, _handle, _io_info, left_io, cfg_snapshot, enable_restore, restore_scale_x, restore_scale_y);
            submit_task(_right_runtime, _right_worker.handle, _right_worker.io_info, right_io, cfg_snapshot, enable_restore, restore_scale_x, restore_scale_y);

            wait_task_done(_left_runtime);
            wait_task_done(_right_runtime);

            if (_left_runtime.ret != 0 || _right_runtime.ret != 0)
            {
                std::cerr << "YOLO-OBB sync run failed: left ret=" << _left_runtime.ret
                          << ", right ret=" << _right_runtime.ret << "\n";
                _io_data = parent_io_backup;
                ax_yolo_obb_infer_node::run_model();
                _pending_inputs.clear();
                _dual_mode_requested = false;
                print_elapsed("single_fallback_worker_error");
                return;
            }

            _last_pair_detections.first.swap(_left_runtime.detections);
            _last_pair_detections.second.swap(_right_runtime.detections);
            // 双模型模式下固定镜像左路，保持旧接口可用且避免重复后处理。
            set_last_detections_compat(_last_pair_detections.first);

            _pending_inputs.clear();
            _dual_mode_requested = false;
            print_elapsed("dual_parallel");
        }

        const std::pair<std::vector<obb_detection>, std::vector<obb_detection>>& get_last_pair_detections() const
        {
            return _last_pair_detections;
        }

    private:
        struct worker_context
        {
            AX_ENGINE_HANDLE handle = 0;
            AX_ENGINE_IO_T io_data {};
            AX_ENGINE_IO_INFO_T* io_info = nullptr;
        };

        struct runtime_worker
        {
            std::thread thread;
            std::mutex mtx;
            std::condition_variable cv;
            AX_ENGINE_HANDLE handle = 0;
            AX_ENGINE_IO_INFO_T* io_info = nullptr;
            AX_ENGINE_IO_T io {};
            infer_config cfg {};
            bool enable_restore = false;
            float restore_scale_x = 1.0f;
            float restore_scale_y = 1.0f;
            bool has_task = false;
            bool done = false;
            bool stop = false;
            int ret = 0;
            std::vector<obb_detection> detections;
        };

        static int allocate_output_buffers(AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T* io_data)
        {
            if (!info || !io_data)
            {
                return -1;
            }
            io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
            memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
            io_data->nOutputSize = info->nOutputSize;
            for (AX_U32 i = 0; i < info->nOutputSize; ++i)
            {
                AX_ENGINE_IO_BUFFER_T* buf = io_data->pOutputs + i;
                buf->nSize = info->pOutputs[i].nSize;
                const int ret = AX_SYS_MemAlloc(reinterpret_cast<AX_U64*>(&buf->phyAddr),
                                                &buf->pVirAddr,
                                                buf->nSize,
                                                AX_CMM_ALIGN_SIZE,
                                                reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
                if (ret != 0)
                {
                    for (AX_U32 j = 0; j < i; ++j)
                    {
                        AX_ENGINE_IO_BUFFER_T* old = io_data->pOutputs + j;
                        AX_SYS_MemFree(old->phyAddr, old->pVirAddr);
                    }
                    delete[] io_data->pOutputs;
                    io_data->pOutputs = nullptr;
                    io_data->nOutputSize = 0;
                    return ret;
                }
            }
            return 0;
        }

        static void free_output_buffers(AX_ENGINE_IO_T* io_data)
        {
            if (!io_data || !io_data->pOutputs)
            {
                return;
            }
            for (AX_U32 i = 0; i < io_data->nOutputSize; ++i)
            {
                AX_ENGINE_IO_BUFFER_T* buf = io_data->pOutputs + i;
                AX_SYS_MemFree(buf->phyAddr, buf->pVirAddr);
            }
            delete[] io_data->pOutputs;
            io_data->pOutputs = nullptr;
            io_data->nOutputSize = 0;
        }

        static bool validate_single_input(const AX_ENGINE_IO_INFO_T* info, const AX_ENGINE_IO_T& io, bool verbose)
        {
            if (!info || !info->pInputs)
            {
                if (verbose) std::cerr << "sync infer validate failed: io info is null.\n";
                return false;
            }
            if (!io.pInputs || io.nInputSize != info->nInputSize)
            {
                if (verbose) std::cerr << "sync infer validate failed: input size mismatch.\n";
                return false;
            }
            for (AX_U32 i = 0; i < io.nInputSize; ++i)
            {
                const AX_ENGINE_IO_BUFFER_T& in = io.pInputs[i];
                const AX_ENGINE_IOMETA_T& meta = info->pInputs[i];
                if (!in.pVirAddr || in.nSize < meta.nSize)
                {
                    if (verbose) std::cerr << "sync infer validate failed: input buffer invalid.\n";
                    return false;
                }
            }
            return true;
        }

        static bool is_valid_bound_input(const AX_ENGINE_IO_BUFFER_T& in)
        {
            return in.pVirAddr != nullptr && in.nSize > 0;
        }

        bool start_runtime_workers()
        {
            _left_runtime.thread = std::thread([this]() { worker_loop(_left_runtime); });
            _right_runtime.thread = std::thread([this]() { worker_loop(_right_runtime); });
            return _left_runtime.thread.joinable() && _right_runtime.thread.joinable();
        }

        void stop_runtime_workers()
        {
            stop_runtime_worker(_left_runtime);
            stop_runtime_worker(_right_runtime);
        }

        static void stop_runtime_worker(runtime_worker& worker)
        {
            {
                std::lock_guard<std::mutex> lock(worker.mtx);
                worker.stop = true;
                worker.cv.notify_one();
            }
            if (worker.thread.joinable())
            {
                worker.thread.join();
            }
        }

        static void submit_task(runtime_worker& worker,
                                AX_ENGINE_HANDLE handle,
                                AX_ENGINE_IO_INFO_T* io_info,
                                const AX_ENGINE_IO_T& io,
                                const infer_config& cfg,
                                bool enable_restore,
                                float restore_scale_x,
                                float restore_scale_y)
        {
            std::lock_guard<std::mutex> lock(worker.mtx);
            worker.handle = handle;
            worker.io_info = io_info;
            worker.io = io;
            worker.cfg = cfg;
            worker.enable_restore = enable_restore;
            worker.restore_scale_x = restore_scale_x;
            worker.restore_scale_y = restore_scale_y;
            worker.done = false;
            worker.has_task = true;
            worker.cv.notify_one();
        }

        static void wait_task_done(runtime_worker& worker)
        {
            std::unique_lock<std::mutex> lock(worker.mtx);
            worker.cv.wait(lock, [&worker]() { return worker.done; });
        }

        static void worker_loop(runtime_worker& worker)
        {
            while (true)
            {
                AX_ENGINE_HANDLE handle = 0;
                AX_ENGINE_IO_INFO_T* io_info = nullptr;
                AX_ENGINE_IO_T io {};
                infer_config cfg {};
                bool enable_restore = false;
                float restore_scale_x = 1.0f;
                float restore_scale_y = 1.0f;
                {
                    std::unique_lock<std::mutex> lock(worker.mtx);
                    worker.cv.wait(lock, [&worker]() { return worker.stop || worker.has_task; });
                    if (worker.stop)
                    {
                        return;
                    }
                    handle = worker.handle;
                    io_info = worker.io_info;
                    io = worker.io;
                    cfg = worker.cfg;
                    enable_restore = worker.enable_restore;
                    restore_scale_x = worker.restore_scale_x;
                    restore_scale_y = worker.restore_scale_y;
                    worker.has_task = false;
                }

                int ret = AX_ENGINE_RunSync(handle, &io);
                std::vector<obb_detection> detections;
                if (ret == 0)
                {
                    (void)process_outputs(io_info, io, cfg, enable_restore, restore_scale_x, restore_scale_y, detections);
                }

                {
                    std::lock_guard<std::mutex> lock(worker.mtx);
                    worker.ret = ret;
                    worker.detections.swap(detections);
                    worker.done = true;
                    worker.cv.notify_one();
                }
            }
        }

        bool can_run_dual_mode() const
        {
            if (!_dual_ready || _pending_bind_invalid || !_dual_mode_requested)
            {
                return false;
            }
            if (_pending_inputs.size() != 2 || !_io_info || _io_info->nInputSize != 1)
            {
                return false;
            }
            const AX_ENGINE_IO_BUFFER_T& left = _pending_inputs[0];
            const AX_ENGINE_IO_BUFFER_T& right = _pending_inputs[1];
            if (left.pVirAddr == nullptr || right.pVirAddr == nullptr)
            {
                return false;
            }
            const bool same_phy = (left.phyAddr != 0 && right.phyAddr != 0 && left.phyAddr == right.phyAddr);
            const bool same_vir = (left.pVirAddr == right.pVirAddr);
            if (same_phy || same_vir)
            {
                std::cerr << "sync infer dual mode rejected: two bound inputs are identical.\n";
                return false;
            }
            if (left.nSize < _io_info->pInputs[0].nSize || right.nSize < _io_info->pInputs[0].nSize)
            {
                std::cerr << "sync infer dual mode rejected: input size is too small.\n";
                return false;
            }
            return true;
        }

        bool init_worker(worker_context& worker, const std::vector<char>& model_buffer, AX_ENGINE_NPU_SET_T npu_set)
        {
            worker = {};
            int ret = AX_ENGINE_CreateHandle(&worker.handle, model_buffer.data(), model_buffer.size());
            if (ret != 0 || worker.handle == 0)
            {
                return false;
            }
            (void)AX_ENGINE_SetAffinity(worker.handle, npu_set);
            ret = AX_ENGINE_CreateContext(worker.handle);
            if (ret != 0)
            {
                AX_ENGINE_DestroyHandle(worker.handle);
                worker.handle = 0;
                return false;
            }
            ret = AX_ENGINE_GetIOInfo(worker.handle, &worker.io_info);
            if (ret != 0 || !worker.io_info)
            {
                AX_ENGINE_DestroyHandle(worker.handle);
                worker.handle = 0;
                return false;
            }
            ret = allocate_output_buffers(worker.io_info, &worker.io_data);
            if (ret != 0)
            {
                AX_ENGINE_DestroyHandle(worker.handle);
                worker.handle = 0;
                worker.io_info = nullptr;
                return false;
            }
            return true;
        }

        void cleanup_worker(worker_context& worker)
        {
            free_output_buffers(&worker.io_data);
            if (worker.handle)
            {
                AX_ENGINE_DestroyHandle(worker.handle);
            }
            worker = {};
        }

    private:
        worker_context _right_worker;
        std::vector<AX_ENGINE_IO_BUFFER_T> _pending_inputs;
        std::pair<std::vector<obb_detection>, std::vector<obb_detection>> _last_pair_detections;
        AX_ENGINE_NPU_SET_T _left_npu_set {0b010};
        AX_ENGINE_NPU_SET_T _right_npu_set {0b010};
        bool _pending_bind_invalid {false};
        bool _dual_ready {false};
        bool _dual_mode_requested {false};
        runtime_worker _left_runtime;
        runtime_worker _right_runtime;
};

class ax_rfdetr_sync_infer_node : public ax_yolo_obb_sync_infer_node
{
    public:
        struct rfdetr_detection
        {
            float x1 = 0.0f;
            float y1 = 0.0f;
            float x2 = 0.0f;
            float y2 = 0.0f;
            int class_id = -1;
            float score = 0.0f;
            int mask_h = 192;
            int mask_w = 192;
            std::vector<float> mask_logits;
        };

        struct infer_config
        {
            float conf_thresh = 0.25f;
            int num_classes = 11;
            int max_det = 300;
            int restore_width = 0;
            int restore_height = 0;
            bool exclude_last_class_as_all = true;
        };

        ax_rfdetr_sync_infer_node() = default;
        ~ax_rfdetr_sync_infer_node() override
        {
            deinit_model();
        }

        void set_config(const infer_config& cfg)
        {
            _cfg = cfg;
            _cfg.conf_thresh = std::max(0.0f, std::min(1.0f, _cfg.conf_thresh));
            _cfg.num_classes = std::max(1, _cfg.num_classes);
            _cfg.max_det = std::max(1, _cfg.max_det);
        }

        infer_config get_config() const
        {
            return _cfg;
        }

        void bind_external_inputs(AX_ENGINE_IO_BUFFER_T* inputs, AX_U32 input_size) override
        {
            if (!inputs || input_size == 0)
            {
                _pending_inputs.clear();
                _pending_bind_invalid = true;
                _dual_mode_requested = false;
                ax_infer_base::bind_external_inputs(inputs, input_size);
                return;
            }

            _pending_bind_invalid = false;
            if (input_size == 1)
            {
                _pending_inputs.assign(inputs, inputs + 1);
                _dual_mode_requested = false;
                ax_infer_base::bind_external_inputs(&inputs[0], 1);
                return;
            }

            if (input_size == 2)
            {
                if (!is_valid_bound_input(inputs[0]) || !is_valid_bound_input(inputs[1]))
                {
                    std::cerr << "rfdetr sync bind failed: input_size=2 but one input is invalid.\n";
                    _pending_inputs.clear();
                    _pending_bind_invalid = true;
                    _dual_mode_requested = true;
                    ax_infer_base::bind_external_inputs(inputs, 2);
                    return;
                }
                _pending_inputs.assign(inputs, inputs + 2);
                _dual_mode_requested = true;
                ax_infer_base::bind_external_inputs(&inputs[0], 1);
                return;
            }

            _pending_inputs.clear();
            _pending_bind_invalid = true;
            _dual_mode_requested = false;
            ax_infer_base::bind_external_inputs(inputs, input_size);
        }

        int init_model(const std::string& model_file)
        {
            _last_pair_detections = {};
            _last_detections.clear();
            _pending_inputs.clear();
            _pending_bind_invalid = false;
            _dual_mode_requested = false;
            const int ret = ax_infer_base::init_model(model_file);
            if (ret != 0)
            {
                return ret;
            }

            std::vector<char> model_buffer;
            if (!utilities::read_file(model_file, model_buffer))
            {
                std::cerr << "rfdetr sync init failed: read model file failed.\n";
                ax_infer_base::deinit_model();
                return -1;
            }

            (void)AX_ENGINE_SetAffinity(_handle, _left_npu_set);
            if (!init_worker(_right_worker, model_buffer, _right_npu_set))
            {
                cleanup_worker(_right_worker);
                ax_infer_base::deinit_model();
                return -1;
            }
            if (!start_runtime_workers())
            {
                cleanup_worker(_right_worker);
                ax_infer_base::deinit_model();
                return -1;
            }
            _dual_ready = true;
            return 0;
        }

        void deinit_model() override
        {
            stop_runtime_workers();
            cleanup_worker(_right_worker);
            _dual_ready = false;
            _pending_inputs.clear();
            _dual_mode_requested = false;
            _last_pair_detections = {};
            _last_detections.clear();
            ax_infer_base::deinit_model();
        }

        void run_model() override
        {
            _last_pair_detections = {};
            _last_detections.clear();
            if (!can_run_dual_mode())
            {
                if (!check_input_binding(true))
                {
                    _pending_inputs.clear();
                    _dual_mode_requested = false;
                    return;
                }
                const int ret = AX_ENGINE_RunSync(_handle, &_io_data);
                if (ret != 0)
                {
                    std::cerr << "RFDETR single run failed: ret=" << ret << "\n";
                    _pending_inputs.clear();
                    _dual_mode_requested = false;
                    return;
                }
                post_process();
                _last_pair_detections.first = _last_detections;
                _pending_inputs.clear();
                _dual_mode_requested = false;
                return;
            }

            AX_ENGINE_IO_T parent_io_backup = _io_data;
            AX_ENGINE_IO_BUFFER_T* left_input = &_pending_inputs[0];
            AX_ENGINE_IO_BUFFER_T* right_input = &_pending_inputs[1];
            ax_infer_base::bind_external_inputs(left_input, 1);
            _right_worker.io_data.pInputs = right_input;
            _right_worker.io_data.nInputSize = 1;

            if (!check_input_binding(true) || !validate_single_input(_right_worker.io_info, _right_worker.io_data, true))
            {
                _io_data = parent_io_backup;
                _pending_inputs.clear();
                _dual_mode_requested = false;
                return;
            }

            const infer_config cfg_snapshot = get_config();
            AX_ENGINE_IO_T left_io = parent_io_backup;
            left_io.pInputs = left_input;
            left_io.nInputSize = 1;
            AX_ENGINE_IO_T right_io = _right_worker.io_data;
            right_io.pInputs = right_input;
            right_io.nInputSize = 1;

            submit_task(_left_runtime, _handle, _io_info, left_io, cfg_snapshot);
            submit_task(_right_runtime, _right_worker.handle, _right_worker.io_info, right_io, cfg_snapshot);

            wait_task_done(_left_runtime);
            wait_task_done(_right_runtime);

            if (_left_runtime.ret != 0 || _right_runtime.ret != 0)
            {
                std::cerr << "RFDETR sync run failed: left ret=" << _left_runtime.ret
                          << ", right ret=" << _right_runtime.ret << "\n";
                _io_data = parent_io_backup;
                _pending_inputs.clear();
                _dual_mode_requested = false;
                return;
            }

            _last_pair_detections.first.swap(_left_runtime.detections);
            _last_pair_detections.second.swap(_right_runtime.detections);
            _last_detections = _last_pair_detections.first;
            _pending_inputs.clear();
            _dual_mode_requested = false;
        }

        void post_process() override
        {
            (void)process_outputs(_io_info, _io_data, _cfg, _last_detections);
        }

        const std::vector<rfdetr_detection>& get_last_detections() const
        {
            return _last_detections;
        }

        const std::pair<std::vector<rfdetr_detection>, std::vector<rfdetr_detection>>& get_last_pair_detections() const
        {
            return _last_pair_detections;
        }

    private:
        struct worker_context
        {
            AX_ENGINE_HANDLE handle = 0;
            AX_ENGINE_IO_T io_data {};
            AX_ENGINE_IO_INFO_T* io_info = nullptr;
        };

        struct runtime_worker
        {
            std::thread thread;
            std::mutex mtx;
            std::condition_variable cv;
            AX_ENGINE_HANDLE handle = 0;
            AX_ENGINE_IO_INFO_T* io_info = nullptr;
            AX_ENGINE_IO_T io {};
            infer_config cfg {};
            bool has_task = false;
            bool done = false;
            bool stop = false;
            int ret = 0;
            std::vector<rfdetr_detection> detections;
        };

        static bool process_outputs(const AX_ENGINE_IO_INFO_T* io_info,
                                    const AX_ENGINE_IO_T& io_data,
                                    const infer_config& cfg,
                                    std::vector<rfdetr_detection>& out_detections)
        {
            out_detections.clear();
            int box_idx = -1;
            int cls_idx = -1;
            int mask_idx = -1;
            if (!find_output_indices(io_info, box_idx, cls_idx, mask_idx))
            {
                std::cerr << "RFDETR post process failed: can not identify box/class/mask outputs.\n";
                return false;
            }

            const AX_ENGINE_IOMETA_T& box_meta = io_info->pOutputs[box_idx];
            const AX_ENGINE_IOMETA_T& cls_meta = io_info->pOutputs[cls_idx];
            const AX_ENGINE_IOMETA_T& mask_meta = io_info->pOutputs[mask_idx];
            const float* box_ptr = reinterpret_cast<const float*>(io_data.pOutputs[box_idx].pVirAddr);
            const float* cls_ptr = reinterpret_cast<const float*>(io_data.pOutputs[cls_idx].pVirAddr);
            const float* mask_ptr = reinterpret_cast<const float*>(io_data.pOutputs[mask_idx].pVirAddr);
            if (!box_ptr || !cls_ptr || !mask_ptr)
            {
                std::cerr << "RFDETR post process failed: output ptr is null.\n";
                return false;
            }

            const int query_num = get_dim(box_meta, 1);
            const int box_dim = get_dim(box_meta, 2);
            const int cls_num = get_dim(cls_meta, 2);
            const int mask_h = get_dim(mask_meta, 2);
            const int mask_w = get_dim(mask_meta, 3);
            if (query_num <= 0 || box_dim != 4 || cls_num <= 0 || mask_h <= 0 || mask_w <= 0)
            {
                std::cerr << "RFDETR post process failed: invalid output shape.\n";
                return false;
            }
            if (query_num != get_dim(cls_meta, 1) || query_num != get_dim(mask_meta, 1))
            {
                std::cerr << "RFDETR post process failed: query dim mismatch.\n";
                return false;
            }

            struct score_item
            {
                int index = -1;
                int class_id = -1;
                float score = 0.0f;
            };
            std::vector<score_item> scored;
            scored.reserve(static_cast<size_t>(query_num));
            for (int i = 0; i < query_num; ++i)
            {
                const float* cls_base = cls_ptr + static_cast<size_t>(i) * static_cast<size_t>(cls_num);
                int best_id = -1;
                float best_score = std::numeric_limits<float>::lowest();
                int valid_cls_num = std::min(cls_num, cfg.num_classes);
                if (cfg.exclude_last_class_as_all && valid_cls_num > 1)
                {
                    // RFDETR 的最后一维为 "all" 汇总项，不应作为具体类别参与 argmax。
                    --valid_cls_num;
                }
                for (int c = 0; c < valid_cls_num; ++c)
                {
                    if (cls_base[c] > best_score)
                    {
                        best_score = cls_base[c];
                        best_id = c;
                    }
                }
                if (best_id < 0)
                {
                    continue;
                }
                score_item item;
                item.index = i;
                item.class_id = best_id;
                item.score = sigmoid(best_score);
                scored.emplace_back(item);
            }

            std::sort(scored.begin(), scored.end(), [](const score_item& a, const score_item& b) {
                return a.score > b.score;
            });
            if (cfg.max_det > 0 && static_cast<int>(scored.size()) > cfg.max_det)
            {
                scored.resize(cfg.max_det);
            }

            out_detections.clear();
            out_detections.reserve(scored.size());
            const bool restore_to_origin = (cfg.restore_width > 0 && cfg.restore_height > 0);
            const size_t mask_size = static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);
            for (const auto& item : scored)
            {
                if (item.score < cfg.conf_thresh)
                {
                    continue;
                }
                const float* box_base = box_ptr + static_cast<size_t>(item.index) * 4;
                const float cx = box_base[0];
                const float cy = box_base[1];
                const float w = box_base[2];
                const float h = box_base[3];
                float x1 = cx - w * 0.5f;
                float y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f;
                float y2 = cy + h * 0.5f;
                if (restore_to_origin)
                {
                    x1 *= static_cast<float>(cfg.restore_width);
                    x2 *= static_cast<float>(cfg.restore_width);
                    y1 *= static_cast<float>(cfg.restore_height);
                    y2 *= static_cast<float>(cfg.restore_height);
                }

                rfdetr_detection det;
                det.x1 = x1;
                det.y1 = y1;
                det.x2 = x2;
                det.y2 = y2;
                if (det.x1 > det.x2) std::swap(det.x1, det.x2);
                if (det.y1 > det.y2) std::swap(det.y1, det.y2);
                det.class_id = item.class_id;
                det.score = item.score;
                det.mask_h = mask_h;
                det.mask_w = mask_w;
                const float* mask_base = mask_ptr + static_cast<size_t>(item.index) * mask_size;
                det.mask_logits.assign(mask_base, mask_base + mask_size);
                out_detections.emplace_back(std::move(det));
            }
            return true;
        }

        static float sigmoid(float x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        }

        static int get_dim(const AX_ENGINE_IOMETA_T& meta, int idx)
        {
            if (!meta.pShape || idx < 0 || idx >= meta.nShapeSize) return -1;
            return meta.pShape[idx];
        }

        static bool is_shape_1x100x4(const AX_ENGINE_IOMETA_T& meta)
        {
            return meta.nShapeSize == 3 && get_dim(meta, 0) == 1 && get_dim(meta, 1) == 100 && get_dim(meta, 2) == 4;
        }

        static bool is_shape_1x100x11(const AX_ENGINE_IOMETA_T& meta)
        {
            return meta.nShapeSize == 3 && get_dim(meta, 0) == 1 && get_dim(meta, 1) == 100 && get_dim(meta, 2) == 11;
        }

        static bool is_shape_1x100x252x252(const AX_ENGINE_IOMETA_T& meta)
        {
            return meta.nShapeSize == 4 && get_dim(meta, 0) == 1 && get_dim(meta, 1) == 100 &&
                   get_dim(meta, 2) == 252 && get_dim(meta, 3) == 252;
        }

        static bool is_shape_1x100x192x192(const AX_ENGINE_IOMETA_T& meta)
        {
            return meta.nShapeSize == 4 && get_dim(meta, 0) == 1 && get_dim(meta, 1) == 100 &&
                   get_dim(meta, 2) == 192 && get_dim(meta, 3) == 192;
        }

        static bool find_output_indices(const AX_ENGINE_IO_INFO_T* io_info, int& box_idx, int& cls_idx, int& mask_idx)
        {
            box_idx = -1;
            cls_idx = -1;
            mask_idx = -1;
            if (!io_info || !io_info->pOutputs) return false;
            for (AX_U32 i = 0; i < io_info->nOutputSize; ++i)
            {
                const AX_ENGINE_IOMETA_T& meta = io_info->pOutputs[i];
                if (is_shape_1x100x4(meta)) box_idx = static_cast<int>(i);
                if (is_shape_1x100x11(meta)) cls_idx = static_cast<int>(i);
                if (is_shape_1x100x192x192(meta)) mask_idx = static_cast<int>(i);
            }
            return box_idx >= 0 && cls_idx >= 0 && mask_idx >= 0;
        }

        static int allocate_output_buffers(AX_ENGINE_IO_INFO_T* info, AX_ENGINE_IO_T* io_data)
        {
            if (!info || !io_data) return -1;
            io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
            memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
            io_data->nOutputSize = info->nOutputSize;
            for (AX_U32 i = 0; i < info->nOutputSize; ++i)
            {
                AX_ENGINE_IO_BUFFER_T* buf = io_data->pOutputs + i;
                buf->nSize = info->pOutputs[i].nSize;
                const int ret = AX_SYS_MemAlloc(reinterpret_cast<AX_U64*>(&buf->phyAddr),
                                                &buf->pVirAddr,
                                                buf->nSize,
                                                AX_CMM_ALIGN_SIZE,
                                                reinterpret_cast<const AX_S8*>(AX_CMM_SESSION_NAME));
                if (ret != 0)
                {
                    for (AX_U32 j = 0; j < i; ++j)
                    {
                        AX_ENGINE_IO_BUFFER_T* old = io_data->pOutputs + j;
                        AX_SYS_MemFree(old->phyAddr, old->pVirAddr);
                    }
                    delete[] io_data->pOutputs;
                    io_data->pOutputs = nullptr;
                    io_data->nOutputSize = 0;
                    return ret;
                }
            }
            return 0;
        }

        static void free_output_buffers(AX_ENGINE_IO_T* io_data)
        {
            if (!io_data || !io_data->pOutputs) return;
            for (AX_U32 i = 0; i < io_data->nOutputSize; ++i)
            {
                AX_ENGINE_IO_BUFFER_T* buf = io_data->pOutputs + i;
                AX_SYS_MemFree(buf->phyAddr, buf->pVirAddr);
            }
            delete[] io_data->pOutputs;
            io_data->pOutputs = nullptr;
            io_data->nOutputSize = 0;
        }

        static bool validate_single_input(const AX_ENGINE_IO_INFO_T* info, const AX_ENGINE_IO_T& io, bool verbose)
        {
            if (!info || !info->pInputs)
            {
                if (verbose) std::cerr << "rfdetr sync validate failed: io info is null.\n";
                return false;
            }
            if (!io.pInputs || io.nInputSize != info->nInputSize)
            {
                if (verbose) std::cerr << "rfdetr sync validate failed: input size mismatch.\n";
                return false;
            }
            for (AX_U32 i = 0; i < io.nInputSize; ++i)
            {
                const AX_ENGINE_IO_BUFFER_T& in = io.pInputs[i];
                const AX_ENGINE_IOMETA_T& meta = info->pInputs[i];
                if (!in.pVirAddr || in.nSize < meta.nSize)
                {
                    if (verbose) std::cerr << "rfdetr sync validate failed: input buffer invalid.\n";
                    return false;
                }
            }
            return true;
        }

        static bool is_valid_bound_input(const AX_ENGINE_IO_BUFFER_T& in)
        {
            return in.pVirAddr != nullptr && in.nSize > 0;
        }

        bool start_runtime_workers()
        {
            _left_runtime.thread = std::thread([this]() { worker_loop(_left_runtime); });
            _right_runtime.thread = std::thread([this]() { worker_loop(_right_runtime); });
            return _left_runtime.thread.joinable() && _right_runtime.thread.joinable();
        }

        void stop_runtime_workers()
        {
            stop_runtime_worker(_left_runtime);
            stop_runtime_worker(_right_runtime);
        }

        static void stop_runtime_worker(runtime_worker& worker)
        {
            {
                std::lock_guard<std::mutex> lock(worker.mtx);
                worker.stop = true;
                worker.cv.notify_one();
            }
            if (worker.thread.joinable()) worker.thread.join();
        }

        static void submit_task(runtime_worker& worker,
                                AX_ENGINE_HANDLE handle,
                                AX_ENGINE_IO_INFO_T* io_info,
                                const AX_ENGINE_IO_T& io,
                                const infer_config& cfg)
        {
            std::lock_guard<std::mutex> lock(worker.mtx);
            worker.handle = handle;
            worker.io_info = io_info;
            worker.io = io;
            worker.cfg = cfg;
            worker.done = false;
            worker.has_task = true;
            worker.cv.notify_one();
        }

        static void wait_task_done(runtime_worker& worker)
        {
            std::unique_lock<std::mutex> lock(worker.mtx);
            worker.cv.wait(lock, [&worker]() { return worker.done; });
        }

        static void worker_loop(runtime_worker& worker)
        {
            while (true)
            {
                AX_ENGINE_HANDLE handle = 0;
                AX_ENGINE_IO_INFO_T* io_info = nullptr;
                AX_ENGINE_IO_T io {};
                infer_config cfg {};
                {
                    std::unique_lock<std::mutex> lock(worker.mtx);
                    worker.cv.wait(lock, [&worker]() { return worker.stop || worker.has_task; });
                    if (worker.stop) return;
                    handle = worker.handle;
                    io_info = worker.io_info;
                    io = worker.io;
                    cfg = worker.cfg;
                    worker.has_task = false;
                }
                int ret = AX_ENGINE_RunSync(handle, &io);
                std::vector<rfdetr_detection> dets;
                if (ret == 0)
                {
                    (void)process_outputs(io_info, io, cfg, dets);
                }
                {
                    std::lock_guard<std::mutex> lock(worker.mtx);
                    worker.ret = ret;
                    worker.detections.swap(dets);
                    worker.done = true;
                    worker.cv.notify_one();
                }
            }
        }

        bool can_run_dual_mode() const
        {
            if (!_dual_ready || _pending_bind_invalid || !_dual_mode_requested) return false;
            if (_pending_inputs.size() != 2 || !_io_info || _io_info->nInputSize != 1) return false;
            const AX_ENGINE_IO_BUFFER_T& left = _pending_inputs[0];
            const AX_ENGINE_IO_BUFFER_T& right = _pending_inputs[1];
            if (!left.pVirAddr || !right.pVirAddr) return false;
            const bool same_phy = (left.phyAddr != 0 && right.phyAddr != 0 && left.phyAddr == right.phyAddr);
            const bool same_vir = (left.pVirAddr == right.pVirAddr);
            if (same_phy || same_vir) return false;
            if (left.nSize < _io_info->pInputs[0].nSize || right.nSize < _io_info->pInputs[0].nSize) return false;
            return true;
        }

        bool init_worker(worker_context& worker, const std::vector<char>& model_buffer, AX_ENGINE_NPU_SET_T npu_set)
        {
            worker = {};
            int ret = AX_ENGINE_CreateHandle(&worker.handle, model_buffer.data(), model_buffer.size());
            if (ret != 0 || worker.handle == 0) return false;
            (void)AX_ENGINE_SetAffinity(worker.handle, npu_set);
            ret = AX_ENGINE_CreateContext(worker.handle);
            if (ret != 0)
            {
                AX_ENGINE_DestroyHandle(worker.handle);
                worker.handle = 0;
                return false;
            }
            ret = AX_ENGINE_GetIOInfo(worker.handle, &worker.io_info);
            if (ret != 0 || !worker.io_info)
            {
                AX_ENGINE_DestroyHandle(worker.handle);
                worker.handle = 0;
                return false;
            }
            ret = allocate_output_buffers(worker.io_info, &worker.io_data);
            if (ret != 0)
            {
                AX_ENGINE_DestroyHandle(worker.handle);
                worker.handle = 0;
                worker.io_info = nullptr;
                return false;
            }
            return true;
        }

        void cleanup_worker(worker_context& worker)
        {
            free_output_buffers(&worker.io_data);
            if (worker.handle) AX_ENGINE_DestroyHandle(worker.handle);
            worker = {};
        }

    private:
        infer_config _cfg {};
        worker_context _right_worker;
        std::vector<AX_ENGINE_IO_BUFFER_T> _pending_inputs;
        std::vector<rfdetr_detection> _last_detections;
        std::pair<std::vector<rfdetr_detection>, std::vector<rfdetr_detection>> _last_pair_detections;
        AX_ENGINE_NPU_SET_T _left_npu_set {0};
        AX_ENGINE_NPU_SET_T _right_npu_set {1};
        bool _pending_bind_invalid {false};
        bool _dual_ready {false};
        bool _dual_mode_requested {false};
        runtime_worker _left_runtime;
        runtime_worker _right_runtime;
};

class ax_yolo_pose_infer_node : public ax_infer_base
{
    public:
        struct pose_keypoint
        {
            float x = 0.0f;
            float y = 0.0f;
            float score = 0.0f;
        };

        struct pose_detection
        {
            float x1 = 0.0f;
            float y1 = 0.0f;
            float x2 = 0.0f;
            float y2 = 0.0f;
            float score = 0.0f;
            int class_id = -1;
            std::vector<pose_keypoint> keypoints;
        };

        struct infer_config
        {
            float score_thres = 0.25f;
            float nms_thres = 0.45f;
            int num_classes = 1;
            int max_det = 300;
            int kpt_num = 17;
            int kpt_dims = 3;
            int restore_width = 0;
            int restore_height = 0;
        };

        ax_yolo_pose_infer_node() : ax_infer_base()
        {
        }

        ~ax_yolo_pose_infer_node() override = default;

        void set_config(const infer_config& cfg)
        {
            _cfg = cfg;
            _cfg.score_thres = std::max(1e-6f, std::min(0.999999f, _cfg.score_thres));
            _cfg.nms_thres = std::max(0.0f, std::min(0.999999f, _cfg.nms_thres));
            _cfg.kpt_num = std::max(1, _cfg.kpt_num);
            _cfg.kpt_dims = std::max(2, _cfg.kpt_dims);
            update_restore_scale();
        }

        infer_config get_config() const
        {
            return _cfg;
        }

        const std::vector<pose_detection>& get_last_detections() const
        {
            return _last_detections;
        }

        void run_model() override
        {
            _last_detections.clear();
            if (!_initialized || !_io_info)
            {
                std::cerr << "YOLO-Pose run failed: model is not initialized.\n";
                return;
            }
            if (!check_input_binding(true))
            {
                return;
            }
            if (_handle == 0)
            {
                std::cerr << "YOLO-Pose run failed: invalid engine handle.\n";
                return;
            }

            _ret = AX_ENGINE_RunSync(_handle, &_io_data);
            if (_ret != 0)
            {
                std::cerr << "YOLO-Pose run failed: AX_ENGINE_RunSync ret=" << _ret << "\n";
                return;
            }
            post_process();
        }

        void post_process() override
        {
            _last_detections.clear();
            if (!_io_info || !_io_data.pOutputs || _io_info->nOutputSize < 3)
            {
                std::cerr << "YOLO-Pose post process failed: invalid output buffers.\n";
                return;
            }

            static const std::array<int, 3> strides = {8, 16, 32};
            const float conf_raw = -std::log(1.0f / _cfg.score_thres - 1.0f);
            AX_U32 model_h = 0;
            AX_U32 model_w = 0;
            const bool has_input_hw = get_model_input_hw(model_h, model_w);

            std::vector<candidate> candidates;

            const AX_U32 out_size = _io_info->nOutputSize;
            for (size_t scale_idx = 0; scale_idx < strides.size(); ++scale_idx)
            {
                const AX_U32 box_idx = static_cast<AX_U32>(scale_idx * 3);
                const AX_U32 cls_idx = static_cast<AX_U32>(scale_idx * 3 + 1);
                const AX_U32 kpt_idx = static_cast<AX_U32>(scale_idx * 3 + 2);
                if (box_idx >= out_size || cls_idx >= out_size || kpt_idx >= out_size)
                {
                    continue;
                }

                AX_U32 expect_h = 0;
                AX_U32 expect_w = 0;
                if (has_input_hw)
                {
                    expect_h = model_h / static_cast<AX_U32>(strides[scale_idx]);
                    expect_w = model_w / static_cast<AX_U32>(strides[scale_idx]);
                }

                output_layout box_l, cls_l, kpt_l;
                if (!resolve_output_layout(_io_info->pOutputs[box_idx], expect_h, expect_w, box_l) ||
                    !resolve_output_layout(_io_info->pOutputs[cls_idx], expect_h, expect_w, cls_l) ||
                    !resolve_output_layout(_io_info->pOutputs[kpt_idx], expect_h, expect_w, kpt_l))
                {
                    continue;
                }
                const AX_U32 h = box_l.h;
                const AX_U32 w = box_l.w;
                const AX_U32 box_c = box_l.c;
                const AX_U32 cls_c = cls_l.c;
                const AX_U32 kpt_c = kpt_l.c;
                if (h == 0 || w == 0 || box_c == 0 || cls_c == 0 || kpt_c == 0 ||
                    cls_l.h != h || cls_l.w != w || kpt_l.h != h || kpt_l.w != w)
                {
                    continue;
                }

                const float* box_ptr = reinterpret_cast<const float*>(_io_data.pOutputs[box_idx].pVirAddr);
                const float* cls_ptr = reinterpret_cast<const float*>(_io_data.pOutputs[cls_idx].pVirAddr);
                const float* kpt_ptr = reinterpret_cast<const float*>(_io_data.pOutputs[kpt_idx].pVirAddr);
                if (!box_ptr || !cls_ptr || !kpt_ptr)
                {
                    continue;
                }

                int reg_max = 0;
                if (box_c > 4 && (box_c % 4) == 0)
                {
                    reg_max = static_cast<int>(box_c / 4);
                }

                const AX_U32 num_cells = h * w;
                std::vector<float> anchor_xy(num_cells * 2, 0.0f);
                for (AX_U32 yy = 0; yy < h; ++yy)
                {
                    for (AX_U32 xx = 0; xx < w; ++xx)
                    {
                        const AX_U32 idx = yy * w + xx;
                        anchor_xy[idx * 2 + 0] = static_cast<float>(xx) + 0.5f;
                        anchor_xy[idx * 2 + 1] = static_cast<float>(yy) + 0.5f;
                    }
                }

                for (AX_U32 cell = 0; cell < num_cells; ++cell)
                {
                    float raw_best = read_output_value(cls_ptr, cls_l, cell, 0);
                    int cls_id = 0;
                    if (cls_c > 1)
                    {
                        for (AX_U32 c = 1; c < cls_c; ++c)
                        {
                            const float cls_val = read_output_value(cls_ptr, cls_l, cell, c);
                            if (cls_val > raw_best)
                            {
                                raw_best = cls_val;
                                cls_id = static_cast<int>(c);
                            }
                        }
                    }
                    if (raw_best < conf_raw)
                    {
                        continue;
                    }

                    const float prob = sigmoid(raw_best);
                    std::vector<float> box_vec(box_c, 0.0f);
                    for (AX_U32 c = 0; c < box_c; ++c)
                    {
                        box_vec[c] = read_output_value(box_ptr, box_l, cell, c);
                    }
                    std::array<float, 4> box = decode_bbox(
                        box_vec.data(),
                        box_c,
                        anchor_xy[cell * 2 + 0],
                        anchor_xy[cell * 2 + 1],
                        strides[scale_idx],
                        reg_max);

                    std::vector<float> kpt_vec(kpt_c, 0.0f);
                    for (AX_U32 c = 0; c < kpt_c; ++c)
                    {
                        kpt_vec[c] = read_output_value(kpt_ptr, kpt_l, cell, c);
                    }
                    std::vector<pose_keypoint> kpts;
                    decode_keypoints(
                        kpt_vec.data(),
                        kpt_c,
                        anchor_xy[cell * 2 + 0],
                        anchor_xy[cell * 2 + 1],
                        strides[scale_idx],
                        _cfg.kpt_num,
                        _cfg.kpt_dims,
                        kpts);

                    candidate det;
                    det.x1 = box[0];
                    det.y1 = box[1];
                    det.x2 = box[2];
                    det.y2 = box[3];
                    det.score = prob;
                    det.class_id = cls_id;
                    det.kpts = std::move(kpts);
                    candidates.emplace_back(std::move(det));
                }
            }

            if (candidates.empty())
            {
                return;
            }

            apply_nms(candidates);

            for (auto& c : candidates)
            {
                pose_detection det;
                det.x1 = c.x1;
                det.y1 = c.y1;
                det.x2 = c.x2;
                det.y2 = c.y2;
                det.score = c.score;
                det.class_id = c.class_id;
                det.keypoints = std::move(c.kpts);
                if (_enable_restore)
                {
                    scale_box_lefttop(det, _restore_scale_x, _restore_scale_y);
                    scale_keypoints_lefttop(det.keypoints, _restore_scale_x, _restore_scale_y);
                }
                _last_detections.emplace_back(std::move(det));
            }

            if (_cfg.max_det > 0 && static_cast<int>(_last_detections.size()) > _cfg.max_det)
            {
                _last_detections.resize(_cfg.max_det);
            }
        }

    private:
        struct candidate
        {
            float x1 = 0.0f;
            float y1 = 0.0f;
            float x2 = 0.0f;
            float y2 = 0.0f;
            float score = 0.0f;
            int class_id = -1;
            std::vector<pose_keypoint> kpts;
        };

        struct nms_box
        {
            float x;
            float y;
            float w;
            float h;
            float score;
            int cls_id;
            size_t src_idx;
        };

        struct output_layout
        {
            AX_U32 h = 0;
            AX_U32 w = 0;
            AX_U32 c = 0;
            bool nchw = false;
        };

        static float sigmoid(float x)
        {
            if (x >= 0.0f)
            {
                const float z = std::exp(-x);
                return 1.0f / (1.0f + z);
            }
            const float z = std::exp(x);
            return z / (1.0f + z);
        }

        static bool resolve_output_layout(
            const AX_ENGINE_IOMETA_T& item,
            AX_U32 expect_h,
            AX_U32 expect_w,
            output_layout& out)
        {
            out = output_layout{};
            if (!item.pShape || item.nShapeSize < 3)
            {
                return false;
            }
            if (item.nShapeSize == 4)
            {
                const AX_U32 s1 = static_cast<AX_U32>(item.pShape[1]);
                const AX_U32 s2 = static_cast<AX_U32>(item.pShape[2]);
                const AX_U32 s3 = static_cast<AX_U32>(item.pShape[3]);
                if (expect_h > 0 && expect_w > 0)
                {
                    if (s1 == expect_h && s2 == expect_w)
                    {
                        out.h = s1;
                        out.w = s2;
                        out.c = s3;
                        out.nchw = false;
                        return true;
                    }
                    if (s2 == expect_h && s3 == expect_w)
                    {
                        out.h = s2;
                        out.w = s3;
                        out.c = s1;
                        out.nchw = true;
                        return true;
                    }
                }
                // 回退启发式：通常网格维度明显大于通道维度
                if (s1 >= s3 && s2 >= s3)
                {
                    out.h = s1;
                    out.w = s2;
                    out.c = s3;
                    out.nchw = false;
                }
                else
                {
                    out.h = s2;
                    out.w = s3;
                    out.c = s1;
                    out.nchw = true;
                }
                return true;
            }
            const AX_U32 s0 = static_cast<AX_U32>(item.pShape[0]);
            const AX_U32 s1 = static_cast<AX_U32>(item.pShape[1]);
            const AX_U32 s2 = static_cast<AX_U32>(item.pShape[2]);
            if (expect_h > 0 && expect_w > 0)
            {
                if (s0 == expect_h && s1 == expect_w)
                {
                    out.h = s0;
                    out.w = s1;
                    out.c = s2;
                    out.nchw = false;
                    return true;
                }
                if (s1 == expect_h && s2 == expect_w)
                {
                    out.h = s1;
                    out.w = s2;
                    out.c = s0;
                    out.nchw = true;
                    return true;
                }
            }
            if (s0 >= s2 && s1 >= s2)
            {
                out.h = s0;
                out.w = s1;
                out.c = s2;
                out.nchw = false;
            }
            else
            {
                out.h = s1;
                out.w = s2;
                out.c = s0;
                out.nchw = true;
            }
            return true;
        }

        static float read_output_value(const float* base, const output_layout& l, AX_U32 cell, AX_U32 ch)
        {
            const AX_U32 y = cell / l.w;
            const AX_U32 x = cell % l.w;
            if (!l.nchw)
            {
                return base[cell * l.c + ch];
            }
            const size_t idx = (static_cast<size_t>(ch) * l.h + y) * l.w + x;
            return base[idx];
        }

        bool get_model_input_hw(AX_U32& h, AX_U32& w) const
        {
            h = 0;
            w = 0;
            if (!_io_info || _io_info->nInputSize == 0 || !_io_info->pInputs)
            {
                return false;
            }
            const AX_ENGINE_IOMETA_T& in = _io_info->pInputs[0];
            if (!in.pShape || in.nShapeSize < 3)
            {
                return false;
            }
            if (in.nShapeSize == 4)
            {
                if (in.pShape[1] == 1 || in.pShape[1] == 3)
                {
                    h = static_cast<AX_U32>(in.pShape[2]);
                    w = static_cast<AX_U32>(in.pShape[3]);
                }
                else
                {
                    h = static_cast<AX_U32>(in.pShape[1]);
                    w = static_cast<AX_U32>(in.pShape[2]);
                }
                return true;
            }
            h = static_cast<AX_U32>(in.pShape[0]);
            w = static_cast<AX_U32>(in.pShape[1]);
            return true;
        }

        void update_restore_scale()
        {
            _enable_restore = false;
            _restore_scale_x = 1.0f;
            _restore_scale_y = 1.0f;
            if (_cfg.restore_width <= 0 || _cfg.restore_height <= 0)
            {
                return;
            }
            AX_U32 model_h = 0;
            AX_U32 model_w = 0;
            if (!get_model_input_hw(model_h, model_w) || model_h == 0 || model_w == 0)
            {
                return;
            }
            _restore_scale_x = static_cast<float>(_cfg.restore_width) / static_cast<float>(model_w);
            _restore_scale_y = static_cast<float>(_cfg.restore_height) / static_cast<float>(model_h);
            _enable_restore = (_restore_scale_x != 1.0f || _restore_scale_y != 1.0f);
        }

        static std::array<float, 4> decode_bbox(
            const float* box_pred,
            AX_U32 box_c,
            float anchor_x,
            float anchor_y,
            int stride,
            int reg_max)
        {
            std::array<float, 4> ltrb = {0.0f, 0.0f, 0.0f, 0.0f};
            if (reg_max > 0 && static_cast<AX_U32>(4 * reg_max) == box_c)
            {
                for (int side = 0; side < 4; ++side)
                {
                    const float* logits = box_pred + side * reg_max;
                    float max_logit = logits[0];
                    for (int i = 1; i < reg_max; ++i)
                    {
                        max_logit = std::max(max_logit, logits[i]);
                    }
                    float sum = 0.0f;
                    float exp_sum = 0.0f;
                    for (int i = 0; i < reg_max; ++i)
                    {
                        const float e = std::exp(logits[i] - max_logit);
                        exp_sum += e;
                        sum += e * static_cast<float>(i);
                    }
                    ltrb[side] = exp_sum > 1e-6f ? (sum / exp_sum) : 0.0f;
                }
            }
            else
            {
                if (box_c < 4)
                {
                    return {0.0f, 0.0f, 0.0f, 0.0f};
                }
                ltrb[0] = box_pred[0];
                ltrb[1] = box_pred[1];
                ltrb[2] = box_pred[2];
                ltrb[3] = box_pred[3];
            }

            const float x1 = (anchor_x - ltrb[0]) * static_cast<float>(stride);
            const float y1 = (anchor_y - ltrb[1]) * static_cast<float>(stride);
            const float x2 = (anchor_x + ltrb[2]) * static_cast<float>(stride);
            const float y2 = (anchor_y + ltrb[3]) * static_cast<float>(stride);
            return {x1, y1, x2, y2};
        }

        static void decode_keypoints(
            const float* kpt_pred,
            AX_U32 kpt_channels,
            float anchor_x,
            float anchor_y,
            int stride,
            int kpt_num,
            int kpt_dims,
            std::vector<pose_keypoint>& out)
        {
            out.clear();
            const AX_U32 need = static_cast<AX_U32>(kpt_num * kpt_dims);
            if (kpt_channels < static_cast<AX_U32>(kpt_dims))
            {
                return;
            }
            out.reserve(kpt_num);
            for (int k = 0; k < kpt_num; ++k)
            {
                const AX_U32 base = static_cast<AX_U32>(k * kpt_dims);
                if (base + 1 >= kpt_channels || base + 1 >= need)
                {
                    break;
                }
                pose_keypoint p;
                p.x = (kpt_pred[base + 0] + anchor_x) * static_cast<float>(stride);
                p.y = (kpt_pred[base + 1] + anchor_y) * static_cast<float>(stride);
                if (kpt_dims >= 3 && base + 2 < kpt_channels)
                {
                    p.score = sigmoid(kpt_pred[base + 2]);
                }
                else
                {
                    p.score = 1.0f;
                }
                out.emplace_back(p);
            }
        }

        static float iou_xyxy(const nms_box& a, const nms_box& b)
        {
            const float ax1 = a.x;
            const float ay1 = a.y;
            const float ax2 = a.x + a.w;
            const float ay2 = a.y + a.h;
            const float bx1 = b.x;
            const float by1 = b.y;
            const float bx2 = b.x + b.w;
            const float by2 = b.y + b.h;

            const float inter_x1 = std::max(ax1, bx1);
            const float inter_y1 = std::max(ay1, by1);
            const float inter_x2 = std::min(ax2, bx2);
            const float inter_y2 = std::min(ay2, by2);
            const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
            const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
            const float inter_area = inter_w * inter_h;
            const float area_a = std::max(0.0f, a.w) * std::max(0.0f, a.h);
            const float area_b = std::max(0.0f, b.w) * std::max(0.0f, b.h);
            const float denom = area_a + area_b - inter_area;
            if (denom <= 1e-6f)
            {
                return 0.0f;
            }
            return inter_area / denom;
        }

        void apply_nms(std::vector<candidate>& candidates) const
        {
            std::vector<nms_box> boxes;
            boxes.reserve(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                const auto& c = candidates[i];
                nms_box b;
                b.x = c.x1;
                b.y = c.y1;
                b.w = std::max(0.0f, c.x2 - c.x1);
                b.h = std::max(0.0f, c.y2 - c.y1);
                b.score = c.score;
                b.cls_id = c.class_id;
                b.src_idx = i;
                boxes.emplace_back(b);
            }

            std::sort(boxes.begin(), boxes.end(), [](const nms_box& a, const nms_box& b) {
                return a.score > b.score;
            });

            std::vector<size_t> keep;
            keep.reserve(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                const nms_box& b = boxes[i];
                bool suppressed = false;
                for (const size_t kept_idx : keep)
                {
                    const nms_box& k = boxes[kept_idx];
                    if (k.cls_id != b.cls_id)
                    {
                        continue;
                    }
                    if (iou_xyxy(k, b) > _cfg.nms_thres)
                    {
                        suppressed = true;
                        break;
                    }
                }
                if (!suppressed)
                {
                    keep.emplace_back(i);
                }
            }

            std::vector<candidate> filtered;
            filtered.reserve(keep.size());
            for (const size_t idx : keep)
            {
                filtered.emplace_back(std::move(candidates[boxes[idx].src_idx]));
            }
            candidates.swap(filtered);
        }

        static void scale_box_lefttop(pose_detection& det, float sx, float sy)
        {
            det.x1 *= sx;
            det.x2 *= sx;
            det.y1 *= sy;
            det.y2 *= sy;
        }

        static void scale_keypoints_lefttop(std::vector<pose_keypoint>& kpts, float sx, float sy)
        {
            for (auto& kp : kpts)
            {
                kp.x *= sx;
                kp.y *= sy;
            }
        }

    private:
        infer_config _cfg;
        std::vector<pose_detection> _last_detections;
        float _restore_scale_x {1.0f};
        float _restore_scale_y {1.0f};
        bool _enable_restore {false};
};
