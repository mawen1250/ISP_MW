#include "Haze_Removal.cuh"
#include "Gaussian.cuh"
#include "HelperCU.cuh"
#include "Specification.cuh"
#include "Histogram.cuh"
#include "Conversion.cuh"


const CUDA_Haze_Removal_Para CUDA_Haze_Removal_Default;


// Functions for class CUDA_Haze_Removal_Retinex
void CUDA_Haze_Removal_Retinex::Init(const Frame &src)
{
    height = src.Height();
    width = src.Width();
    stride = src.Stride();

    pcount = src.R().PixelCount();
    sFloor = src.R().Floor();
    sCeil = src.R().Ceil();
    GRID_DIM = CudaGridDim(pcount, BLOCK_DIM);

    CUDA_Malloc(RdevInt, pcount);
    CUDA_Malloc(GdevInt, pcount);
    CUDA_Malloc(BdevInt, pcount);

    CUDA_MemcpyH2D(RdevInt, src.R().data(), pcount);
    CUDA_MemcpyH2D(GdevInt, src.G().data(), pcount);
    CUDA_MemcpyH2D(BdevInt, src.B().data(), pcount);
}

void CUDA_Haze_Removal_Retinex::End()
{
    CUDA_Free(Rdev);
    CUDA_Free(Gdev);
    CUDA_Free(Bdev);
}


static __global__ void IntToLinear_Kernel(FLType *dstR, FLType *dstG, FLType *dstB,
    const DType *srcR, const DType *srcG, const DType *srcB, const cuIdx pcount, FLType gain, DType floor,
    CUDA_TransferChar_Conv<FLType> transfer)
{
    const cuIdx lower = blockIdx.x * blockDim.x + threadIdx.x;
    const cuIdx step = gridDim.x * blockDim.x;

    for (cuIdx i = lower; i < pcount; i += step)
    {
        dstR[i] = transfer((srcR[i] - floor) * gain);
        dstG[i] = transfer((srcG[i] - floor) * gain);
        dstB[i] = transfer((srcB[i] - floor) * gain);
    }
}

void CUDA_Haze_Removal_Retinex::IntToLinear()
{
    CUDA_Malloc(Rdev, pcount);
    CUDA_Malloc(Gdev, pcount);
    CUDA_Malloc(Bdev, pcount);

    FLType gain = FLType(1) / static_cast<FLType>(sCeil - sFloor);
    CUDA_TransferChar_Conv<FLType> transfer(TransferChar::linear, para.TransferChar_);

    const cuIdx BLOCKS = 256;
    const cuIdx THREADS = 64;
    CudaGlobalCall(IntToLinear_Kernel, BLOCKS, THREADS)(Rdev, Gdev, Bdev, RdevInt, GdevInt, BdevInt, pcount, gain, sFloor, transfer);

    CUDA_Free(RdevInt);
    CUDA_Free(GdevInt);
    CUDA_Free(BdevInt);
}


// Main process flow
Frame &CUDA_Haze_Removal_Retinex::process_Frame(Frame &dst, const Frame &src)
{
    height = dst.Height();
    width = dst.Width();
    stride = dst.Stride();

    if (src.isYUV() || dst.isYUV())
    {
        const char *FunctionName = "Haze_Removal::process";
        std::cerr << FunctionName << ": YUV input/output is not supported.\n";
        exit(EXIT_FAILURE);
    }
    else
    {
        Init(src);
        IntToLinear();
        CUDA_Malloc(tMapInv_dev, pcount);
        GetTMapInv();
        GetAtmosLight();
        RemoveHaze();
        CUDA_Free(tMapInv_dev);
        StoreResult(dst);
        End();
    }

    return dst;
}


// Generate the Inverted Transmission Map from intensity image
static __global__ void GetTMapInv_Kernel2(FLType *dst, const FLType *src0, const FLType *src1, const cuIdx pcount)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pcount)
    {
        dst[idx] = sqrt(src0[idx] * src1[idx]);
    }
}

static __global__ void GetTMapInv_Kernel3_1(FLType *dst, const FLType *src, const cuIdx pcount)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pcount)
    {
        dst[idx] *= src[idx];
    }
}

static __global__ void GetTMapInv_Kernel3_2(FLType *data, const cuIdx pcount, const FLType exponent)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pcount)
    {
        data[idx] = pow(data[idx], exponent);
    }
}

void CUDA_Haze_Removal_Retinex::GetTMapInv()
{
    FLType *refY = nullptr;
    CUDA_Malloc(refY, pcount);
    CUDA_ConvertToY(refY, Rdev, Gdev, Bdev, pcount, para.Ymode == 1 ? ColorMatrix::Minimum
        : para.Ymode == 2 ? ColorMatrix::Maximum : ColorMatrix::Average);

    const size_t scount = para.sigmaVector.size();
    size_t s;

    // Use refY as tMapInv if no Gaussian need to be applied
    for (s = 0; s < scount; ++s)
    {
        if (para.sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        CUDA_Free(tMapInv_dev);
        tMapInv_dev = refY;
        refY = nullptr;
        return;
    }

    if (scount == 1 && para.sigmaVector[0] > 0) // single-scale Gaussian filter
    {
        CUDA_RecursiveGaussian GFilter(para.sigmaVector[0], false, CudaMemMode::Device);
        GFilter.Filter(tMapInv_dev, refY, height, width, stride);
    }
    else if (scount == 2 && para.sigmaVector[0] > 0 && para.sigmaVector[1] > 0) // double-scale Gaussian filter
    {
        FLType *gauss0 = nullptr;
        FLType *gauss1 = nullptr;

        CUDA_Malloc(gauss0, pcount);
        CUDA_Malloc(gauss1, pcount);

        CUDA_RecursiveGaussian GFilter0(para.sigmaVector[0], false, CudaMemMode::Device);
        GFilter0.Filter(gauss0, refY, height, width, stride);

        CUDA_RecursiveGaussian GFilter1(para.sigmaVector[1], false, CudaMemMode::Device);
        GFilter1.Filter(gauss1, refY, height, width, stride);

        CudaGlobalCall(GetTMapInv_Kernel2, GRID_DIM, BLOCK_DIM)(tMapInv_dev, gauss0, gauss1, pcount);

        CUDA_Free(gauss0);
        CUDA_Free(gauss1);
    }
    else // multi-scale Gaussian filter
    {
        FLType *gauss = nullptr;
        CUDA_Malloc(gauss, pcount);
        CUDA_Set(tMapInv_dev, pcount, FLType(1), BLOCK_DIM);
        
        for (s = 0; s < scount; ++s)
        {
            if (para.sigmaVector[s] > 0)
            {
                CUDA_RecursiveGaussian GFilter(para.sigmaVector[s], false, CudaMemMode::Device);
                GFilter.Filter(gauss, refY, height, width, stride);

                CudaGlobalCall(GetTMapInv_Kernel3_1, GRID_DIM, BLOCK_DIM)(tMapInv_dev, gauss, pcount);
            }
            else
            {
                CudaGlobalCall(GetTMapInv_Kernel3_1, GRID_DIM, BLOCK_DIM)(tMapInv_dev, refY, pcount);
            }
        }

        // Calculate geometric mean of multiple scales
        const FLType scountRec = FLType(1) / static_cast<FLType>(scount);

        CudaGlobalCall(GetTMapInv_Kernel3_2, GRID_DIM, BLOCK_DIM)(tMapInv_dev, pcount, scountRec);

        CUDA_Free(gauss);
    }

    CUDA_Free(refY);
}


// Get the Global Atmospheric Light from Inverted Transmission Map and src
const cuIdx GetAtmosLight_BLOCKS = 32;
const cuIdx GetAtmosLight_THREADS = 64;

static __global__ void GetAtmosLight_Kernel(int *count, FLType *AL_Rsum, FLType *AL_Gsum, FLType *AL_Bsum,
    const FLType *tMapInv, const FLType *srcR, const FLType *srcG, const FLType *srcB,
    const cuIdx pcount, const FLType tMapLowerThr)
{
    __shared__ int count_block[GetAtmosLight_THREADS];
    __shared__ FLType AL_Rsum_block[GetAtmosLight_THREADS];
    __shared__ FLType AL_Gsum_block[GetAtmosLight_THREADS];
    __shared__ FLType AL_Bsum_block[GetAtmosLight_THREADS];

    int *count_thread = count_block + threadIdx.x;
    FLType *AL_Rsum_thread = AL_Rsum_block + threadIdx.x;
    FLType *AL_Gsum_thread = AL_Gsum_block + threadIdx.x;
    FLType *AL_Bsum_thread = AL_Bsum_block + threadIdx.x;

    *count_thread = 0;
    *AL_Rsum_thread = 0;
    *AL_Gsum_thread = 0;
    *AL_Bsum_thread = 0;

    const cuIdx lower = blockIdx.x * GetAtmosLight_THREADS + threadIdx.x;
    const cuIdx step = GetAtmosLight_BLOCKS * GetAtmosLight_THREADS;

    for (cuIdx i = lower; i < pcount; i += step)
    {
        if (tMapInv[i] >= tMapLowerThr)
        {
            ++(*count_thread);
            *AL_Rsum_thread += srcR[i];
            *AL_Gsum_thread += srcG[i];
            *AL_Bsum_thread += srcB[i];
        }
    }

    for (cuIdx stride = GetAtmosLight_THREADS / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < stride)
        {
            *count_thread += count_thread[stride];
            *AL_Rsum_thread += AL_Rsum_thread[stride];
            *AL_Gsum_thread += AL_Gsum_thread[stride];
            *AL_Bsum_thread += AL_Bsum_thread[stride];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(count, count_block[0]);
        atomicAddFloat(AL_Rsum, AL_Rsum_block[0]);
        atomicAddFloat(AL_Gsum, AL_Gsum_block[0]);
        atomicAddFloat(AL_Bsum, AL_Bsum_block[0]);
    }
}

void CUDA_Haze_Removal_Retinex::GetAtmosLight()
{
    CUDA_Histogram<> hist;
    hist.Generate(tMapInv_dev, pcount, FLType(0), FLType(1));
    const FLType tMapLowerThr = hist.Max(para.tMap_thr, FLType(0), FLType(1));

    int count, *count_dev;
    FLType AL_Rsum, *AL_Rsum_dev;
    FLType AL_Gsum, *AL_Gsum_dev;
    FLType AL_Bsum, *AL_Bsum_dev;

    CUDA_Malloc(count_dev, 1);
    CUDA_Malloc(AL_Rsum_dev, 1);
    CUDA_Malloc(AL_Gsum_dev, 1);
    CUDA_Malloc(AL_Bsum_dev, 1);

    CUDA_Memset(count_dev, 1, 0);
    CUDA_Set(AL_Rsum_dev, 1, FLType(0));
    CUDA_Set(AL_Gsum_dev, 1, FLType(0));
    CUDA_Set(AL_Bsum_dev, 1, FLType(0));

    CudaGlobalCall(GetAtmosLight_Kernel, GetAtmosLight_BLOCKS, GetAtmosLight_THREADS)
        (count_dev, AL_Rsum_dev, AL_Gsum_dev, AL_Bsum_dev,
        tMapInv_dev, Rdev, Gdev, Bdev, pcount, tMapLowerThr);

    CUDA_MemcpyD2H(&count, count_dev, 1);
    CUDA_MemcpyD2H(&AL_Rsum, AL_Rsum_dev, 1);
    CUDA_MemcpyD2H(&AL_Gsum, AL_Gsum_dev, 1);
    CUDA_MemcpyD2H(&AL_Bsum, AL_Bsum_dev, 1);

    AL_R = Min(para.ALmax, AL_Rsum / static_cast<FLType>(count));
    AL_G = Min(para.ALmax, AL_Gsum / static_cast<FLType>(count));
    AL_B = Min(para.ALmax, AL_Bsum / static_cast<FLType>(count));

    if (para.debug > 0)
    {
        std::cout << "tMapLowerThr = " << tMapLowerThr << std::endl;
        std::cout << "count = " << count << ", AL_sum (R, G, B) = (" << AL_Rsum << ", " << AL_Gsum << ", " << AL_Gsum << ")\n";
        std::cout << "Global Atmospheric Light (R, G, B) = (" << AL_R << ", " << AL_G << ", " << AL_B << ")\n";
    }

    CUDA_Free(count_dev);
    CUDA_Free(AL_Rsum_dev);
    CUDA_Free(AL_Gsum_dev);
    CUDA_Free(AL_Bsum_dev);
}


static __global__ void RemoveHaze_ShowTMap_Kernel(FLType *dstR, FLType *dstG, FLType *dstB, const FLType *tMapInv, const cuIdx pcount,
    const FLType tMapMin, const FLType tMapMax, const FLType mulR, const FLType mulG, const FLType mulB)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pcount)
    {
        dstR[idx] = cuMax(tMapMin, tMapMax - tMapInv[idx] * mulR);
        dstG[idx] = cuMax(tMapMin, tMapMax - tMapInv[idx] * mulG);
        dstB[idx] = cuMax(tMapMin, tMapMax - tMapInv[idx] * mulB);
    }
}

static __global__ void RemoveHaze_Kernel(FLType *dstR, FLType *dstG, FLType *dstB, const FLType *tMapInv, const cuIdx pcount,
    const FLType tMapMin, const FLType tMapMax, const FLType mulR, const FLType mulG, const FLType mulB,
    const FLType AL_R, const FLType AL_G, const FLType AL_B)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pcount)
    {
         const FLType divR = cuMax(tMapMin, tMapMax - tMapInv[idx] * mulR);
         const FLType divG = cuMax(tMapMin, tMapMax - tMapInv[idx] * mulG);
         const FLType divB = cuMax(tMapMin, tMapMax - tMapInv[idx] * mulB);
         dstR[idx] = (dstR[idx] - AL_R) / divR + AL_R;
         dstG[idx] = (dstG[idx] - AL_G) / divG + AL_G;
         dstB[idx] = (dstB[idx] - AL_B) / divB + AL_B;
    }
}

void CUDA_Haze_Removal_Retinex::RemoveHaze()
{
    const FLType mulR = para.strength / AL_R;
    const FLType mulG = para.strength / AL_G;
    const FLType mulB = para.strength / AL_B;

    if (para.debug == 2)
    {
        CUDA_MemcpyD2D(Rdev, tMapInv_dev, pcount);
        CUDA_MemcpyD2D(Gdev, tMapInv_dev, pcount);
        CUDA_MemcpyD2D(Bdev, tMapInv_dev, pcount);
    }
    else if (para.debug == 3)
    {
        CudaGlobalCall(RemoveHaze_ShowTMap_Kernel, GRID_DIM, BLOCK_DIM)
            (Rdev, Gdev, Bdev, tMapInv_dev, pcount,
            para.tMapMin, para.tMapMax, mulR, mulG, mulB);
    }
    else
    {
        CudaGlobalCall(RemoveHaze_Kernel, GRID_DIM, BLOCK_DIM)
            (Rdev, Gdev, Bdev, tMapInv_dev, pcount,
            para.tMapMin, para.tMapMax, mulR, mulG, mulB,
            AL_R, AL_G, AL_B);
    }
}


static __global__ void StoreResult_Kernel(DType *dstR, DType *dstG, DType *dstB,
    const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount,
    const FLType gain1, const FLType offset1, CUDA_TransferChar_Conv<FLType> transfer,
    const FLType Floor, const FLType Ceil)
{
    const cuIdx lower = blockIdx.x * blockDim.x + threadIdx.x;
    const cuIdx step = gridDim.x * blockDim.x;

    const FLType gain2 = Ceil - Floor;
    const FLType offset2 = Floor + FLType(0.5);

    for (cuIdx i = lower; i < pcount; i += step)
    {
        dstR[i] = static_cast<DType>(cuClip(transfer(srcR[i] * gain1 + offset1) * gain2 + offset2, Floor, Ceil));
        dstG[i] = static_cast<DType>(cuClip(transfer(srcG[i] * gain1 + offset1) * gain2 + offset2, Floor, Ceil));
        dstB[i] = static_cast<DType>(cuClip(transfer(srcB[i] * gain1 + offset1) * gain2 + offset2, Floor, Ceil));
    }
}

void CUDA_Haze_Removal_Retinex::StoreResult(Frame &dst)
{
    Plane &dstR = dst.R();
    Plane &dstG = dst.G();
    Plane &dstB = dst.B();

    const FLType dFloor = static_cast<FLType>(dst.R().Floor());
    const FLType dCeil = static_cast<FLType>(dst.R().Ceil());

    FLType min, max;
    FLType gain1, offset1;

    if (para.debug >= 2 || para.ppmode <= 0) // No range scaling
    {
        min = FLType(0);
        max = FLType(1);
    }
    else if (para.ppmode == 1) // Canonical range scaling
    {
        min = FLType(0.13); // Experimental lower limit
        max = (AL_R + AL_G + AL_B) / FLType(3); // Take average Global Atmospheric Light as upper limit
    }
    else if (para.ppmode == 2) // Gaussian filtered average channel for range detection
    {
        FLType *temp = nullptr;
        CUDA_Malloc(temp, pcount);
        CUDA_RecursiveGaussian GFilter(para.pp_sigma, true, CudaMemMode::Device);

        CUDA_ConvertToY(temp, Rdev, Gdev, Bdev, pcount, ColorMatrix::Average);

        GFilter.Filter(temp, temp, height, width, stride);
        CUDA_GetMinMax(temp, pcount, min, max);

        if (max <= min)
        {
            min = FLType(0);
            max = FLType(1);
        }

        CUDA_Free(temp);
    }
    else // Simpliest Color Balance with all channels combined for range detection
    {
        CUDA_ValidRange(Rdev, Gdev, Bdev, pcount, min, max, para.lower_thr, para.upper_thr, true, FLType(0), FLType(1));
    }

    if (para.debug > 0)
    {
        std::cout << "StoreResult: min = " << min << ", max = " << max << std::endl;
    }

    gain1 = FLType(1) / (max - min);
    offset1 = -min * gain1;
    CUDA_TransferChar_Conv<FLType> transfer(para.TransferChar_, TransferChar::linear);

    CUDA_Malloc(RdevInt, pcount);
    CUDA_Malloc(GdevInt, pcount);
    CUDA_Malloc(BdevInt, pcount);

    const cuIdx BLOCKS = 256;
    const cuIdx THREADS = 64;
    CudaGlobalCall(StoreResult_Kernel, BLOCKS, THREADS)
        (RdevInt, GdevInt, BdevInt, Rdev, Gdev, Bdev, pcount,
        gain1, offset1, transfer, dFloor, dCeil);

    CUDA_MemcpyD2H(dstR.data(), RdevInt, pcount);
    CUDA_MemcpyD2H(dstG.data(), GdevInt, pcount);
    CUDA_MemcpyD2H(dstB.data(), BdevInt, pcount);

    CUDA_Free(RdevInt);
    CUDA_Free(GdevInt);
    CUDA_Free(BdevInt);
}
