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
