#define ENABLE_PPL


#include "Gaussian.cuh"
#include "Conversion.hpp"


const CUDA_FilterMode CUDA_Gaussian2D_Default(CudaMemMode::Host2Device, 64);
const CUDA_FilterMode CUDA_RecursiveGaussian_Default(CudaMemMode::Host2Device, 64);


// Public functions of class CUDA_Gaussian2D
Plane &CUDA_Gaussian2D::process_Plane(Plane &dst, const Plane &src)
{
    if (para.sigma <= 0)
    {
        dst = src;
        return dst;
    }

    Plane_FL data(src);
    CUDA_RecursiveGaussian GFilter(para.sigma, cuFM);

    GFilter.Filter(data);
    RangeConvert(dst, data, true);

    return dst;
}


// CUDA global functions
__global__ void CUDA_RecursiveGaussian_FilterV_Kernel(FLType *dst, const FLType *src, const CUDA_RecursiveGaussian f)
{
    // CUDA index
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Parameters
    const FLType &B = f.B;
    const FLType &B1 = f.B1;
    const FLType &B2 = f.B2;
    const FLType &B3 = f.B3;

    const PCType &height = f.D().height;
    const PCType &width = f.D().width;
    const PCType &stride = f.D().stride;

    const PCType lower = idx;
    const PCType upper = lower + (height - 1) * stride;

    // Skip processing if index out of range
    if (idx >= width) return;

    // Process
    PCType i = lower;
    FLType P0, P1, P2, P3;
    P3 = P2 = P1 = src[i];
    dst[i] = src[i];

    for (; i < upper;)
    {
        i += stride;
        P0 = B * src[i] + B1 * P1 + B2 * P2 + B3 * P3;
        P3 = P2;
        P2 = P1;
        P1 = P0;
        dst[i] = P0;
    }

    P3 = P2 = P1 = dst[i];

    for (; i > lower;)
    {
        i -= stride;
        P0 = B * dst[i] + B1 * P1 + B2 * P2 + B3 * P3;
        P3 = P2;
        P2 = P1;
        P1 = P0;
        dst[i] = P0;
    }
}

__global__ void CUDA_RecursiveGaussian_FilterH_Kernel(FLType *dst, const FLType *src, const CUDA_RecursiveGaussian f)
{
    // CUDA index
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Parameters
    const FLType &B = f.B;
    const FLType &B1 = f.B1;
    const FLType &B2 = f.B2;
    const FLType &B3 = f.B3;

    const PCType &height = f.D().height;
    const PCType &width = f.D().width;
    const PCType &stride = f.D().stride;

    const PCType lower = idx * stride;
    const PCType upper = lower + width - 1;

    // Skip processing if index out of range
    if (idx >= height) return;

    // Process
    PCType i = lower;
    FLType P0, P1, P2, P3;
    P3 = P2 = P1 = src[i];
    dst[i] = src[i];

    for (; i < upper;)
    {
        ++i;
        P0 = B * src[i] + B1 * P1 + B2 * P2 + B3 * P3;
        P3 = P2;
        P2 = P1;
        P1 = P0;
        dst[i] = P0;
    }

    P3 = P2 = P1 = dst[i];

    for (; i > lower;)
    {
        --i;
        P0 = B * dst[i] + B1 * P1 + B2 * P2 + B3 * P3;
        P3 = P2;
        P2 = P1;
        P1 = P0;
        dst[i] = P0;
    }
}


// Public functions of class CUDA_RecursiveGaussian
void CUDA_RecursiveGaussian::FilterV(Plane_FL &dst, const Plane_FL &src)
{
    d.Init(dst, src);

    CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.src_dev, *this);

    d.End();
}

void CUDA_RecursiveGaussian::FilterH(Plane_FL &dst, const Plane_FL &src)
{
    d.Init(dst, src);

    CudaGlobalCall(CUDA_RecursiveGaussian_FilterH_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.dst_dev, d.src_dev, *this);

    d.End();
}

void CUDA_RecursiveGaussian::Filter(Plane_FL &dst, const Plane_FL &src)
{
    d.Init(dst, src);

    CudaGlobalCall(CUDA_RecursiveGaussian_FilterH_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.dst_dev, d.src_dev, *this);
    CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.dst_dev, *this);

    d.End();
}

void CUDA_RecursiveGaussian::Filter(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    d.Init(dst, src, height, width, stride);

    CudaGlobalCall(CUDA_RecursiveGaussian_FilterH_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.dst_dev, d.src_dev, *this);
    CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.dst_dev, *this);

    d.End();
}
