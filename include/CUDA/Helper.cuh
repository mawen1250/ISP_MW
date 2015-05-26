#ifndef HELPER_CUH_
#define HELPER_CUH_


#include <iostream>
#include <xutility>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Type.h"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
    case cudaSuccess:
        return "cudaSuccess";

    case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";

    case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";

    case cudaErrorInitializationError:
        return "cudaErrorInitializationError";

    case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";

    case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";

    case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";

    case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";

    case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";

    case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";

    case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";

    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";

    case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";

    case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";

    case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";

    case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";

    case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";

    case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";

    case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";

    case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";

    case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";

    case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";

    case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";

    case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";

    case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";

    case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";

    case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";

    case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";

    case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";

    case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";

    case cudaErrorUnknown:
        return "cudaErrorUnknown";

    case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";

    case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";

    case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";

    case cudaErrorNotReady:
        return "cudaErrorNotReady";

    case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";

    case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";

    case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";

    case cudaErrorNoDevice:
        return "cudaErrorNoDevice";

    case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";

    case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";

    case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";

    case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";

    case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";

    case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";

    case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";

    case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";

    case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";

    case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";

    case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";

    case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";

    case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";

    case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";

    case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";

    case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";

    case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";

    case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
    case cudaErrorAssert:
        return "cudaErrorAssert";

    case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";

    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";

    case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
    case cudaErrorOperatingSystem:
        return "cudaErrorOperatingSystem";

    case cudaErrorPeerAccessUnsupported:
        return "cudaErrorPeerAccessUnsupported";

    case cudaErrorLaunchMaxDepthExceeded:
        return "cudaErrorLaunchMaxDepthExceeded";

    case cudaErrorLaunchFileScopedTex:
        return "cudaErrorLaunchFileScopedTex";

    case cudaErrorLaunchFileScopedSurf:
        return "cudaErrorLaunchFileScopedSurf";

    case cudaErrorSyncDepthExceeded:
        return "cudaErrorSyncDepthExceeded";

    case cudaErrorLaunchPendingCountExceeded:
        return "cudaErrorLaunchPendingCountExceeded";

    case cudaErrorNotPermitted:
        return "cudaErrorNotPermitted";

    case cudaErrorNotSupported:
        return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
    case cudaErrorHardwareStackError:
        return "cudaErrorHardwareStackError";

    case cudaErrorIllegalInstruction:
        return "cudaErrorIllegalInstruction";

    case cudaErrorMisalignedAddress:
        return "cudaErrorMisalignedAddress";

    case cudaErrorInvalidAddressSpace:
        return "cudaErrorInvalidAddressSpace";

    case cudaErrorInvalidPc:
        return "cudaErrorInvalidPc";

    case cudaErrorIllegalAddress:
        return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
    case cudaErrorInvalidPtx:
        return "cudaErrorInvalidPtx";

    case cudaErrorInvalidGraphicsContext:
        return "cudaErrorInvalidGraphicsContext";

    case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";

    case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}
#endif

#ifdef __cuda_cuda_h__
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error)
{
    switch (error)
    {
    case CUDA_SUCCESS:
        return "CUDA_SUCCESS";

    case CUDA_ERROR_INVALID_VALUE:
        return "CUDA_ERROR_INVALID_VALUE";

    case CUDA_ERROR_OUT_OF_MEMORY:
        return "CUDA_ERROR_OUT_OF_MEMORY";

    case CUDA_ERROR_NOT_INITIALIZED:
        return "CUDA_ERROR_NOT_INITIALIZED";

    case CUDA_ERROR_DEINITIALIZED:
        return "CUDA_ERROR_DEINITIALIZED";

    case CUDA_ERROR_PROFILER_DISABLED:
        return "CUDA_ERROR_PROFILER_DISABLED";

    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

    case CUDA_ERROR_NO_DEVICE:
        return "CUDA_ERROR_NO_DEVICE";

    case CUDA_ERROR_INVALID_DEVICE:
        return "CUDA_ERROR_INVALID_DEVICE";

    case CUDA_ERROR_INVALID_IMAGE:
        return "CUDA_ERROR_INVALID_IMAGE";

    case CUDA_ERROR_INVALID_CONTEXT:
        return "CUDA_ERROR_INVALID_CONTEXT";

    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

    case CUDA_ERROR_MAP_FAILED:
        return "CUDA_ERROR_MAP_FAILED";

    case CUDA_ERROR_UNMAP_FAILED:
        return "CUDA_ERROR_UNMAP_FAILED";

    case CUDA_ERROR_ARRAY_IS_MAPPED:
        return "CUDA_ERROR_ARRAY_IS_MAPPED";

    case CUDA_ERROR_ALREADY_MAPPED:
        return "CUDA_ERROR_ALREADY_MAPPED";

    case CUDA_ERROR_NO_BINARY_FOR_GPU:
        return "CUDA_ERROR_NO_BINARY_FOR_GPU";

    case CUDA_ERROR_ALREADY_ACQUIRED:
        return "CUDA_ERROR_ALREADY_ACQUIRED";

    case CUDA_ERROR_NOT_MAPPED:
        return "CUDA_ERROR_NOT_MAPPED";

    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
        return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
        return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

    case CUDA_ERROR_ECC_UNCORRECTABLE:
        return "CUDA_ERROR_ECC_UNCORRECTABLE";

    case CUDA_ERROR_UNSUPPORTED_LIMIT:
        return "CUDA_ERROR_UNSUPPORTED_LIMIT";

    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

    case CUDA_ERROR_INVALID_SOURCE:
        return "CUDA_ERROR_INVALID_SOURCE";

    case CUDA_ERROR_FILE_NOT_FOUND:
        return "CUDA_ERROR_FILE_NOT_FOUND";

    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
        return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

    case CUDA_ERROR_OPERATING_SYSTEM:
        return "CUDA_ERROR_OPERATING_SYSTEM";

    case CUDA_ERROR_INVALID_HANDLE:
        return "CUDA_ERROR_INVALID_HANDLE";

    case CUDA_ERROR_NOT_FOUND:
        return "CUDA_ERROR_NOT_FOUND";

    case CUDA_ERROR_NOT_READY:
        return "CUDA_ERROR_NOT_READY";

    case CUDA_ERROR_LAUNCH_FAILED:
        return "CUDA_ERROR_LAUNCH_FAILED";

    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

    case CUDA_ERROR_LAUNCH_TIMEOUT:
        return "CUDA_ERROR_LAUNCH_TIMEOUT";

    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
        return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
        return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
        return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

    case CUDA_ERROR_ASSERT:
        return "CUDA_ERROR_ASSERT";

    case CUDA_ERROR_TOO_MANY_PEERS:
        return "CUDA_ERROR_TOO_MANY_PEERS";

    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
        return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

    case CUDA_ERROR_UNKNOWN:
        return "CUDA_ERROR_UNKNOWN";
    }

    return "<unknown>";
}
#endif


#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
#ifdef _DEBUG
        __debugbreak();
        _STD _DEBUG_ERROR("check: CUDA debug error!");
#else
        // Make sure we call CUDA Device Reset before exiting
        DEVICE_RESET;
        exit(EXIT_FAILURE);
#endif
    }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, cudaGetErrorString(err));
#ifdef _DEBUG
        __debugbreak();
        _STD _DEBUG_ERROR("getLastCudaError: CUDA debug error!");
#else
        // Make sure we call CUDA Device Reset before exiting
        DEVICE_RESET;
        exit(EXIT_FAILURE);
#endif
    }
}
#endif


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef unsigned int cuIdx;


enum class CudaMemMode
{
    Device = 0,
    PageLocked,
    Host2Device,
    Host2PageLocked
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _Ty >
void CUDA_Malloc(_Ty *&devPtr, size_t count = 1)
{
    checkCudaErrors(cudaMalloc(&devPtr, sizeof(_Ty) * count));
}

template < typename _Ty >
void CUDA_Free(_Ty *&devPtr)
{
    checkCudaErrors(cudaFree(devPtr));
}

template < typename _Ty >
void CUDA_Memset(_Ty *data, size_t count = 1, int value = 0)
{
    checkCudaErrors(cudaMemset(data, value, sizeof(_Ty) * count));
}

template < typename _Dt1, typename _St1 >
void CUDA_Memcpy(const _Dt1 *dst, const _St1 *src, size_t count, cudaMemcpyKind kind)
{
    checkCudaErrors(cudaMemcpy(const_cast<_Dt1 *>(dst), src, sizeof(_St1) * count, kind));
}

template < typename _Dt1, typename _St1 >
void CUDA_MemcpyH2H(_Dt1 *dst, const _St1 *src, size_t count = 1)
{
    CUDA_Memcpy(dst, src, count, cudaMemcpyHostToHost);
}

template < typename _Dt1, typename _St1 >
void CUDA_MemcpyH2D(_Dt1 *dst, const _St1 *src, size_t count = 1)
{
    CUDA_Memcpy(dst, src, count, cudaMemcpyHostToDevice);
}

template < typename _Dt1, typename _St1 >
void CUDA_MemcpyD2H(_Dt1 *dst, const _St1 *src, size_t count = 1)
{
    CUDA_Memcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

template < typename _Dt1, typename _St1 >
void CUDA_MemcpyD2D(_Dt1 *dst, const _St1 *src, size_t count = 1)
{
    CUDA_Memcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

template < typename _Ty >
void CUDA_HostRegister(const _Ty *ptr, size_t count = 1, cuIdx flags = CU_MEMHOSTALLOC_DEVICEMAP)
{
    checkCudaErrors(cudaHostRegister(const_cast<_Ty *>(ptr), sizeof(_Ty) * count, flags));
}

template < typename _Ty >
void CUDA_HostUnregister(const _Ty *ptr)
{
    checkCudaErrors(cudaHostUnregister(const_cast<_Ty *>(ptr)));
}

template < typename _Dt1, typename _St1 >
void CUDA_HostGetDevicePointer(_Dt1 *&pDevice, const _St1 *pHost, cuIdx flags = 0)
{
    void *pDeviceTemp = nullptr;
    cudaHostGetDevicePointer(&pDeviceTemp, const_cast<_St1 *>(pHost), flags);
    pDevice = reinterpret_cast<_Dt1 *>(pDeviceTemp);
}


inline cuIdx CudaGridDim(cuIdx _thread_count, cuIdx _block_dim)
{
    return (_thread_count + _block_dim - 1) / _block_dim;
}

#define CudaGlobalCall(_func, _grid_dim, _block_dim) _func<<<_grid_dim, _block_dim>>>

inline void CudaGlobalCheck()
{
    // Check for any errors launching the kernel
    getLastCudaError("Kernel execution failed");

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch
    checkCudaErrors(cudaDeviceSynchronize());
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct CUDA_FilterMode
{
    typedef CUDA_FilterMode _Myt;

    CudaMemMode mem_mode;
    cuIdx block_dim;

    CUDA_FilterMode(CudaMemMode _mem_mode = CudaMemMode::Host2Device, cuIdx _block_dim = 256)
        : mem_mode(_mem_mode), block_dim(_block_dim)
    {}

    CUDA_FilterMode(const _Myt &_right)
        : _Myt(_right.mem_mode, _right.block_dim)
    {}

    CUDA_FilterMode(const _Myt &&_right)
        : _Myt(_right.mem_mode, _right.block_dim)
    {}
};

extern const CUDA_FilterMode cuFM_Default;


template < typename _Ty = FLType >
struct CUDA_FilterData
    : public CUDA_FilterMode
{
    typedef CUDA_FilterData _Myt;
    typedef CUDA_FilterMode _Mybase;

    int dst_mode;
    int temp_mode;

    PCType height = 0;
    PCType width = 0;
    PCType stride = 0;
    size_t count = 0;

    _Ty *dst_data = nullptr;
    const _Ty *src_data = nullptr;
    _Ty *dst_dev = nullptr;
    const _Ty *src_dev = nullptr;
    _Ty *temp_dev = nullptr;

    CUDA_FilterData(CudaMemMode _mem_mode = CudaMemMode::Host2Device, cuIdx _block_dim = 256, int _dst_mode = 0, int _temp_mode = 0)
        : _Mybase(_mem_mode, _block_dim), dst_mode(_dst_mode), temp_mode(_temp_mode)
    {}

    CUDA_FilterData(const _Mybase &_right)
        : _Mybase(_right)
    {}

    CUDA_FilterData(const _Mybase &&_right)
        : _Mybase(_right)
    {}

    CUDA_FilterData(const _Myt &_right)
        : _Mybase(_right), height(_right.height), width(_right.width), stride(_right.stride), count(_right.count),
        dst_data(_right.dst_data), src_data(_right.src_data), dst_dev(_right.dst_dev), src_dev(_right.src_dev)
    {}

    CUDA_FilterData(const _Myt &&_right)
        : _Mybase(_right), height(_right.height), width(_right.width), stride(_right.stride), count(_right.count),
        dst_data(_right.dst_data), src_data(_right.src_data), dst_dev(_right.dst_dev), src_dev(_right.src_dev)
    {}

    template < typename _St1 >
    void Init(_St1 &dst, const _St1 &src)
    {
        if (typeid(_Ty) != typeid(typename _St1::value_type))
        {
            std::cerr << "CUDA_FilterData::Init: _Ty and _St1::value_type don't match!\n";
            exit(EXIT_FAILURE);
        }

        InitSize(dst);

        dst_data = dst.data();
        src_data = src.data();

        InitMemory();
    }

    void Init(_Ty *dst, const _Ty *src, PCType _height, PCType _width, PCType _stride)
    {
        InitSize(_height, _width, _stride);

        dst_data = dst;
        src_data = src;

        InitMemory();
    }

    void End()
    {
        EndMemory();

        dst_dev = nullptr;
        src_dev = nullptr;
        dst_data = nullptr;
        src_data = nullptr;

        EndSize();
    }

protected:
    template < typename _St1 >
    void InitSize(const _St1 &dst)
    {
        height = dst.Height();
        width = dst.Width();
        stride = dst.Stride();
        count = dst.size();
    }

    void InitSize(PCType _height, PCType _width, PCType _stride)
    {
        height = _height;
        width = _width;
        stride = _stride;
        count = _height * _stride;
    }

    void EndSize()
    {
        height = 0;
        width = 0;
        stride = 0;
        count = 0;
    }

    void InitMemory()
    {
        _Ty *src_data_temp = nullptr;

        switch (mem_mode)
        {
        case CudaMemMode::Host2Device:
            CUDA_Malloc(temp_dev, count);
            CUDA_MemcpyH2D(temp_dev, src_data, count);
            src_dev = temp_dev;
            if (dst_mode == 1) dst_dev = temp_dev;
            else CUDA_Malloc(dst_dev, count);
            temp_dev = nullptr;
            break;
        case CudaMemMode::Device:
            dst_dev = dst_data;
            src_dev = src_data;
            break;
        case CudaMemMode::Host2PageLocked:
            AlignedMalloc(src_data_temp, count);
            memcpy(src_data_temp, src_data, sizeof(_Ty) * count);
            src_data = src_data_temp;
            if (dst_data != src_data) CUDA_HostRegister(dst_data, count);
            CUDA_HostRegister(src_data, count);
        case CudaMemMode::PageLocked:
            CUDA_HostGetDevicePointer(dst_dev, dst_data);
            CUDA_HostGetDevicePointer(src_dev, src_data);
            break;
        default:
            std::cerr << "CUDA_FilterData::InitMemory: Unsupported mem_mode!\n";
            exit(EXIT_FAILURE);
        }

        switch (temp_mode)
        {
        case 1:
            CUDA_Malloc(temp_dev, count);
            break;
        case 2:
            temp_dev = const_cast<_Ty *>(src_dev);
            break;
        default:
            break;
        }
    }

    void EndMemory()
    {
        _Ty *src_data_temp = nullptr;

        switch (temp_mode)
        {
        case 1:
            CUDA_Free(temp_dev);
            break;
        case 2:
            temp_dev = nullptr;
            break;
        default:
            break;
        }

        switch (mem_mode)
        {
        case CudaMemMode::Host2Device:
            CUDA_MemcpyD2H(dst_data, dst_dev, count);
            if (dst_mode == 1) dst_dev = nullptr;
            else CUDA_Free(dst_dev);
            temp_dev = const_cast<_Ty *>(src_dev);
            src_dev = nullptr;
            CUDA_Free(temp_dev);
            break;
        case CudaMemMode::Device:
            break;
        case CudaMemMode::Host2PageLocked:
            if (dst_data != src_data) CUDA_HostUnregister(dst_data);
            CUDA_HostUnregister(src_data);
            src_data_temp = const_cast<_Ty *>(src_data);
            AlignedFree(src_data_temp);
        case CudaMemMode::PageLocked:
            break;
        default:
            std::cerr << "CUDA_FilterData::EndMemory: Unsupported mem_mode!\n";
            exit(EXIT_FAILURE);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Max Min
template < typename T >
static __inline__ __device__ T cuMax(T a, T b)
{
    return a < b ? b : a;
}

template < typename T >
static __inline__ __device__ T cuMin(T a, T b)
{
    return a > b ? b : a;
}

template < typename T >
static __inline__ __device__ T cuClip(T input, T lower, T upper)
{
    return input >= upper ? upper : input <= lower ? lower : input;
}


// Abs
template < typename T >
static __inline__ __device__ T cuAbs(T input)
{
    return input < 0 ? -input : input;
}

template < typename T >
static __inline__ __device__ T cuAbsSub(T a, T b)
{
    return a >= b ? a - b : b - a;
}


// Initialization
template < typename _Ty >
__global__ void CUDA_Set_Kernel(_Ty *dst, const cuIdx count, _Ty value)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count)
    {
        dst[idx] = value;
    }
}

template < typename _Ty >
inline void CUDA_Set(_Ty *dst, const cuIdx count, _Ty value = 0, cuIdx _block_dim = 256)
{
    if (count < _block_dim) _block_dim = count;
    CudaGlobalCall(CUDA_Set_Kernel, CudaGridDim(count, _block_dim), _block_dim)(dst, count, value);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif
