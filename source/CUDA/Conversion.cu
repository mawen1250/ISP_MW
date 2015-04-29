#include "Conversion.cuh"
#include "HelperCU.cuh"
#include "Histogram.cuh"


static const cuIdx BLOCKS = 256;
//static const cuIdx BLOCKS4 = BLOCKS / 4;
static const cuIdx BLOCKS8 = BLOCKS / 8;
static const cuIdx THREADS = 256;
static const cuIdx THREADS4 = THREADS / 4;


static __global__ void CUDA_GetMinMax_Kernel(const FLType *src, const cuIdx pcount, FLType *min, FLType *max)
{
    __shared__ FLType min_block[THREADS4];
    __shared__ FLType max_block[THREADS4];

    FLType *min_thread = min_block + threadIdx.x;
    FLType *max_thread = max_block + threadIdx.x;

    const cuIdx lower = blockIdx.x * THREADS4 + threadIdx.x;
    const cuIdx step = BLOCKS8 * THREADS4;

    *max_thread = *min_thread = src[lower];

    for (cuIdx i = lower + step; i < pcount; i += step)
    {
        const FLType value = src[i];
        if (value < *min_thread) *min_thread = value;
        if (value > *max_thread) *max_thread = value;
    }

    for (cuIdx stride = THREADS4 / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < stride)
        {
            if (min_thread[stride] < *min_thread) *min_thread = min_thread[stride];
            if (max_thread[stride] > *max_thread) *max_thread = max_thread[stride];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicMinFloat(min, min_block[0]);
        atomicMaxFloat(max, max_block[0]);
    }
}


void CUDA_GetMinMax(const FLType *src, const cuIdx pcount, FLType &min, FLType &max)
{
    FLType *min_dev, *max_dev;
    CUDA_Malloc(min_dev, 1);
    CUDA_Malloc(max_dev, 1);
    CUDA_Set(min_dev, 1, FLType_MAX);
    CUDA_Set(max_dev, 1, -FLType_MAX);

    CudaGlobalCall(CUDA_GetMinMax_Kernel, BLOCKS8, THREADS4)(src, pcount, min_dev, max_dev);

    CUDA_MemcpyD2H(&min, min_dev, 1);
    CUDA_MemcpyD2H(&max, max_dev, 1);
    CUDA_Free(min_dev);
    CUDA_Free(max_dev);
}

void CUDA_GetMinMax(const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount, FLType &min, FLType &max)
{
    FLType *min_dev, *max_dev;
    CUDA_Malloc(min_dev, 1);
    CUDA_Malloc(max_dev, 1);
    CUDA_Set(min_dev, 1, FLType_MAX);
    CUDA_Set(max_dev, 1, -FLType_MAX);

    CudaGlobalCall(CUDA_GetMinMax_Kernel, BLOCKS8, THREADS4)(srcR, pcount, min_dev, max_dev);
    CudaGlobalCall(CUDA_GetMinMax_Kernel, BLOCKS8, THREADS4)(srcG, pcount, min_dev, max_dev);
    CudaGlobalCall(CUDA_GetMinMax_Kernel, BLOCKS8, THREADS4)(srcB, pcount, min_dev, max_dev);

    CUDA_MemcpyD2H(&min, min_dev, 1);
    CUDA_MemcpyD2H(&max, max_dev, 1);
    CUDA_Free(min_dev);
    CUDA_Free(max_dev);
}


void CUDA_ValidRange(const FLType *src, const cuIdx pcount, FLType &min, FLType &max,
    double lower_thr, double upper_thr, bool protect, FLType Floor, FLType Ceil)
{
    CUDA_GetMinMax(src, pcount, min, max);

    if (protect && max <= min)
    {
        min = Floor;
        max = Ceil;
    }
    else if (lower_thr > 0 || upper_thr > 0)
    {
        const FLType _min = min;
        const FLType _max = max;

        CUDA_Histogram<> hist;
        hist.Generate(src, pcount, _min, _max);

        if (lower_thr > 0) min = hist.Min(lower_thr, _min, _max);
        if (upper_thr > 0) max = hist.Max(upper_thr, _min, _max);
    }
}

void CUDA_ValidRange(const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount, FLType &min, FLType &max,
    double lower_thr, double upper_thr, bool protect, FLType Floor, FLType Ceil)
{
    CUDA_GetMinMax(srcR, srcG, srcB, pcount, min, max);

    if (protect && max <= min)
    {
        min = Floor;
        max = Ceil;
    }
    else if (lower_thr > 0 || upper_thr > 0)
    {
        const FLType _min = min;
        const FLType _max = max;

        CUDA_Histogram<> hist;
        hist.Generate(srcR, pcount, _min, _max);
        hist.Add(srcG, pcount, _min, _max);
        hist.Add(srcB, pcount, _min, _max);

        if (lower_thr > 0) min = hist.Min(lower_thr, _min, _max);
        if (upper_thr > 0) max = hist.Max(upper_thr, _min, _max);
    }
}


static __global__ void CUDA_ConvertToY_Kernel_Average(FLType *dst, const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount)
{
    const cuIdx lower = blockIdx.x * THREADS + threadIdx.x;
    const cuIdx step = BLOCKS * THREADS;

    const FLType gain = FLType(1. / 3.);

    for (cuIdx i = lower; i < pcount; i += step)
    {
        dst[i] = (srcR[i] + srcG[i] + srcB[i]) * gain;
    }
}

static __global__ void CUDA_ConvertToY_Kernel_Minimum(FLType *dst, const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount)
{
    const cuIdx lower = blockIdx.x * THREADS + threadIdx.x;
    const cuIdx step = BLOCKS * THREADS;

    for (cuIdx i = lower; i < pcount; i += step)
    {
        dst[i] = cuMin(srcR[i], cuMin(srcG[i], srcB[i]));
    }
}

static __global__ void CUDA_ConvertToY_Kernel_Maximum(FLType *dst, const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount)
{
    const cuIdx lower = blockIdx.x * THREADS + threadIdx.x;
    const cuIdx step = BLOCKS * THREADS;

    for (cuIdx i = lower; i < pcount; i += step)
    {
        dst[i] = cuMin(srcR[i], cuMax(srcG[i], srcB[i]));
    }
}

static __global__ void CUDA_ConvertToY_Kernel_Common(FLType *dst, const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount,
    const FLType Kr, const FLType Kg, const FLType Kb)
{
    const cuIdx lower = blockIdx.x * THREADS + threadIdx.x;
    const cuIdx step = BLOCKS * THREADS;

    for (cuIdx i = lower; i < pcount; i += step)
    {
        dst[i] = srcR[i] * Kr + srcG[i] * Kg + srcB[i] * Kb;
    }
}

void CUDA_ConvertToY(FLType *dst, const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount, ColorMatrix dstColorMatrix)
{
    if (dstColorMatrix == ColorMatrix::Average)
    {
        CudaGlobalCall(CUDA_ConvertToY_Kernel_Average, BLOCKS, THREADS)(dst, srcR, srcG, srcB, pcount);
    }
    else if (dstColorMatrix == ColorMatrix::Minimum)
    {
        CudaGlobalCall(CUDA_ConvertToY_Kernel_Minimum, BLOCKS, THREADS)(dst, srcR, srcG, srcB, pcount);
    }
    else if (dstColorMatrix == ColorMatrix::Maximum)
    {
        CudaGlobalCall(CUDA_ConvertToY_Kernel_Maximum, BLOCKS, THREADS)(dst, srcR, srcG, srcB, pcount);
    }
    else
    {
        FLType Kr, Kg, Kb;
        ColorMatrix_Parameter(dstColorMatrix, Kr, Kg, Kb);
        CudaGlobalCall(CUDA_ConvertToY_Kernel_Common, BLOCKS, THREADS)(dst, srcR, srcG, srcB, pcount, Kr, Kg, Kb);
    }
}
