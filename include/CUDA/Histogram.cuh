#ifndef HISTOGRAM_CUH_
#define HISTOGRAM_CUH_


#include "Helper.cuh"


template < cuIdx HIST_BINS, cuIdx HIST_BLOCKS, cuIdx HIST_THREADS, typename _Ty >
__global__ void CUDA_Histogram_Generate_Kernel1(int *pHist, const _Ty *src, const cuIdx pcount, const FLType gain, const FLType offset)
{
    const cuIdx HIST_SHARED_BINS = HIST_BINS * HIST_THREADS;

    __shared__ int subHist[HIST_SHARED_BINS];

    // Initialize subHist
    for (cuIdx i = threadIdx.x; i < HIST_SHARED_BINS; i += HIST_THREADS)
    {
        subHist[i] = 0;
    }

    __syncthreads();

    // Add up pixel counts to corresponding bins in subHist to current thread
    const cuIdx lower = blockIdx.x * HIST_THREADS + threadIdx.x;
    int *subHistT = subHist + threadIdx.x * HIST_BINS;

    for (cuIdx pos = lower; pos < pcount; pos += HIST_BLOCKS * HIST_THREADS)
    {
        const cuIdx hist_level = static_cast<cuIdx>(
            cuClip(static_cast<FLType>(src[pos]) * gain + offset, FLType(0), FLType(HIST_BINS - 1)));
        ++subHistT[hist_level];
    }

    __syncthreads();

    // Add up pixel counts to corresponding bins in pHist to currect block
    int *pHistT = pHist + blockIdx.x * HIST_BINS;

    for (cuIdx bin = threadIdx.x; bin < HIST_BINS; bin += HIST_THREADS)
    {
        int sum = 0;

        for (cuIdx i = bin; i < HIST_SHARED_BINS; i += HIST_BINS)
        {
            sum += subHist[i];
        }

        pHistT[bin] = sum;
    }
}

template < cuIdx HIST_BINS, cuIdx HIST_BLOCKS >
__global__ void CUDA_Histogram_Generate_Kernel2(int *hist, const int *pHist)
{
    const cuIdx bin = blockIdx.x;

    __shared__ int data[HIST_BLOCKS];

    int *thread_data = data + threadIdx.x;
    *thread_data = pHist[bin + threadIdx.x * HIST_BINS];

    for (cuIdx stride = HIST_BLOCKS / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < stride)
        {
            *thread_data += thread_data[stride];
        }
    }

    if (threadIdx.x == 0)
    {
        hist[bin] = data[0];
    }
}


template < cuIdx HIST_BINS, cuIdx HIST_BLOCKS >
__global__ void CUDA_Histogram_Add_Kernel2(int *hist, const int *pHist)
{
    const cuIdx bin = blockIdx.x;

    __shared__ int data[HIST_BLOCKS];

    int *thread_data = data + threadIdx.x;
    *thread_data = pHist[bin + threadIdx.x * HIST_BINS];

    for (cuIdx stride = HIST_BLOCKS / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < stride)
        {
            *thread_data += thread_data[stride];
        }
    }

    if (threadIdx.x == 0)
    {
        hist[bin] += data[0];
    }
}


template < cuIdx HIST_BINS >
__global__ void CUDA_Histogram_Min_Kernel(cuIdx *level, const int *hist, const cuIdx count_thr)
{
    int sum = 0;

    for (int i = 0; i < HIST_BINS; ++i)
    {
        sum += hist[i];

        if (sum > count_thr)
        {
            *level = i;
            return;
        }
    }
}

template < cuIdx HIST_BINS >
__global__ void CUDA_Histogram_Max_Kernel(cuIdx *level, const int *hist, const cuIdx count_thr)
{
    int sum = 0;

    for (int i = HIST_BINS; i >= 0; --i)
    {
        sum += hist[i];

        if (sum > count_thr)
        {
            *level = i;
            return;
        }
    }
}


template < cuIdx HIST_BINS = 256, cuIdx HIST_BLOCKS = 64, cuIdx HIST_THREADS = 3 >
class CUDA_Histogram
{
public:
    typedef CUDA_Histogram<HIST_BINS, HIST_BLOCKS, HIST_THREADS> _Myt;

    static const cuIdx HIST_PARTIAL_BINS = HIST_BINS * HIST_BLOCKS;

private:
    int *hist = nullptr;
    cuIdx pcount = 0;

public:
    CUDA_Histogram()
    {
        CUDA_Malloc(hist, HIST_BINS);
    }

    ~CUDA_Histogram()
    {
        CUDA_Free(hist);
    }

    void ToHost(int *hist_host) const
    {
        CUDA_MemcpyD2H(hist_host, hist, HIST_BINS);
    }

    void FromHost(const int *hist_host)
    {
        CUDA_MemcpyH2D(hist, hist_host, HIST_BINS);
    }

    template < typename _Ty >
    void Generate(const _Ty *src, const cuIdx _pcount, const _Ty Floor = 0, const _Ty Ceil = 1)
    {
        pcount = _pcount;

        const ldbl _gain = ldbl(HIST_BINS - 1) / static_cast<ldbl>(Ceil - Floor);
        const ldbl _offset = -Floor * _gain + 0.5L;

        const FLType gain = static_cast<FLType>(_gain);
        const FLType offset = static_cast<FLType>(_offset);

        int *pHist = nullptr;
        CUDA_Malloc(pHist, HIST_PARTIAL_BINS);

        CUDA_Histogram_Generate_Kernel1<HIST_BINS, HIST_BLOCKS, HIST_THREADS, _Ty>
            <<<HIST_BLOCKS, HIST_THREADS>>>(pHist, src, _pcount, gain, offset);
        CUDA_Histogram_Generate_Kernel2<HIST_BINS, HIST_BLOCKS>
            <<<HIST_BINS, HIST_BLOCKS>>>(hist, pHist);

        CUDA_Free(pHist);
    }

    template < typename _Ty >
    void Add(const _Ty *src, const cuIdx _pcount, const _Ty Floor = 0, const _Ty Ceil = 1)
    {
        pcount += _pcount;

        const ldbl _gain = ldbl(HIST_BINS - 1) / static_cast<ldbl>(Ceil - Floor);
        const ldbl _offset = -Floor * _gain + 0.5L;

        const FLType gain = static_cast<FLType>(_gain);
        const FLType offset = static_cast<FLType>(_offset);

        int *pHist = nullptr;
        CUDA_Malloc(pHist, HIST_PARTIAL_BINS);

        CUDA_Histogram_Generate_Kernel1<HIST_BINS, HIST_BLOCKS, HIST_THREADS, _Ty>
            <<<HIST_BLOCKS, HIST_THREADS>>>(pHist, src, _pcount, gain, offset);
        CUDA_Histogram_Add_Kernel2<HIST_BINS, HIST_BLOCKS>
            <<<HIST_BINS, HIST_BLOCKS>>>(hist, pHist);
        
        CUDA_Free(pHist);
    }

    template < typename _Ty >
    _Ty Min(double ratio = 0, const _Ty Floor = 0, const _Ty Ceil = 1) const
    {
        const cuIdx count_thr = static_cast<cuIdx>(pcount * ratio + 0.5);

        cuIdx level, *level_dev;
        CUDA_Malloc(level_dev, 1);

        CUDA_Histogram_Min_Kernel<HIST_BINS>
            <<<1, 1>>>(level_dev, hist, count_thr);

        CUDA_MemcpyD2H(&level, level_dev, 1);
        CUDA_Free(level_dev);

        return static_cast<_Ty>(level / static_cast<double>(HIST_BINS - 1) * (Ceil - Floor) + Floor);
    }

    template < typename _Ty >
    _Ty Max(double ratio = 0, const _Ty Floor = 0, const _Ty Ceil = 1) const
    {
        const cuIdx count_thr = static_cast<cuIdx>(pcount * ratio + 0.5);

        cuIdx level, *level_dev;
        CUDA_Malloc(level_dev, 1);

        CUDA_Histogram_Max_Kernel<HIST_BINS>
            <<<1, 1>>>(level_dev, hist, count_thr);

        CUDA_MemcpyD2H(&level, level_dev, 1);
        CUDA_Free(level_dev);

        return static_cast<_Ty>(level / static_cast<double>(HIST_BINS - 1) * (Ceil - Floor) + Floor);
    }
};


#endif
