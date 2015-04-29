#define ENABLE_PPL


#include "Gaussian.cuh"
#include "Transform.cuh"
#include "Conversion.hpp"


// Public functions of class CUDA_Gaussian2D
Plane &CUDA_Gaussian2D::process_Plane(Plane &dst, const Plane &src)
{
    if (para.sigma <= 0)
    {
        dst = src;
        return dst;
    }

    Plane_FL data(src);
    CUDA_RecursiveGaussian GFilter(para.sigma, true, mem_mode);

    GFilter.Filter(data);
    RangeConvert(dst, data, true);

    return dst;
}


// CUDA global functions
static __global__ void CUDA_RecursiveGaussian_FilterV_Kernel(FLType *dst, const FLType *src,
    const PCType height, const PCType width, const PCType stride, const RecursiveGaussian f)
{
    // CUDA index
    const PCType lower = blockIdx.x * blockDim.x + threadIdx.x;
    const PCType upper = lower + (height - 1) * stride;

    // Parameters
    const FLType &B = f.B;
    const FLType &B1 = f.B1;
    const FLType &B2 = f.B2;
    const FLType &B3 = f.B3;

    // Skip processing if index out of range
    if (lower >= width) return;

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
        if (f.allow_negative || P0 >= 0) dst[i] = P0;
        else dst[i] = 0;
    }
}

/*static __global__ void CUDA_RecursiveGaussian_FilterH_Kernel(FLType *dst, const FLType *src,
    const PCType height, const PCType width, const PCType stride, const RecursiveGaussian f)
{
    // CUDA index
    const PCType lower = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    const PCType upper = lower + width - 1;

    // Parameters
    const FLType &B = f.B;
    const FLType &B1 = f.B1;
    const FLType &B2 = f.B2;
    const FLType &B3 = f.B3;

    // Skip processing if index out of range
    if (lower >= height * stride) return;

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
        if (f.allow_negative || P0 >= 0) dst[i] = P0;
        else dst[i] = 0;
    }
}*/


// Public functions of class CUDA_RecursiveGaussian
void CUDA_RecursiveGaussian::FilterV(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    d.dst_mode = 1;
    d.temp_mode = 0;
    d.Init(dst, src, height, width, stride);

    CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.src_dev, d.height, d.width, d.width, *this);

    for (int i = 1; i < iter; ++i)
    {
        CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.dst_dev, d.height, d.width, d.width, *this);
    }

    d.End();
}

void CUDA_RecursiveGaussian::FilterH(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    if (dst == src)
    {
        d.dst_mode = 1;
        d.temp_mode = 1;
        d.Init(dst, src, height, width, stride);

        CUDA_Transpose(d.temp_dev, d.src_dev, d.height, d.width, CudaMemMode::Device);

        for (int i = 0; i < iter; ++i)
        {
            CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.temp_dev, d.temp_dev, d.width, d.height, d.height, *this);
        }

        CUDA_Transpose(d.dst_dev, d.temp_dev, d.width, d.height, CudaMemMode::Device);
    }
    else
    {
        d.dst_mode = 0;
        d.temp_mode = 2;
        d.Init(dst, src, height, width, stride);

        CUDA_Transpose(d.dst_dev, d.src_dev, d.height, d.width, CudaMemMode::Device);

        CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.temp_dev, d.dst_dev, d.width, d.height, d.height, *this);

        for (int i = 1; i < iter; ++i)
        {
            CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.temp_dev, d.temp_dev, d.width, d.height, d.height, *this);
        }

        CUDA_Transpose(d.dst_dev, d.temp_dev, d.width, d.height, CudaMemMode::Device);

        d.End();
    }
}

void CUDA_RecursiveGaussian::Filter(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    if (dst == src)
    {
        d.dst_mode = 1;
        d.temp_mode = 1;
        d.Init(dst, src, height, width, stride);

        CUDA_Transpose(d.temp_dev, d.src_dev, d.height, d.width, CudaMemMode::Device);

        for (int i = 0; i < iter; ++i)
        {
            CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.temp_dev, d.temp_dev, d.width, d.height, d.height, *this);
        }

        CUDA_Transpose(d.dst_dev, d.temp_dev, d.width, d.height, CudaMemMode::Device);

        for (int i = 0; i < iter; ++i)
        {
            CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.dst_dev, d.height, d.width, d.width, *this);
        }
    }
    else
    {
        d.dst_mode = 0;
        d.temp_mode = 2;
        d.Init(dst, src, height, width, stride);

        CUDA_Transpose(d.dst_dev, d.src_dev, d.height, d.width, CudaMemMode::Device);

        for (int i = 0; i < iter; ++i)
        {
            CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.height, d.block_dim), d.block_dim)(d.dst_dev, d.dst_dev, d.width, d.height, d.height, *this);
        }

        CUDA_Transpose(d.temp_dev, d.dst_dev, d.width, d.height, CudaMemMode::Device);

        CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.temp_dev, d.height, d.width, d.width, *this);

        for (int i = 1; i < iter; ++i)
        {
            CudaGlobalCall(CUDA_RecursiveGaussian_FilterV_Kernel, CudaGridDim(d.width, d.block_dim), d.block_dim)(d.dst_dev, d.dst_dev, d.height, d.width, d.width, *this);
        }
    }

    d.End();
}


void CUDA_RecursiveGaussian::FilterV(Plane_FL &dst, const Plane_FL &src)
{
    FilterV(dst.data(), src.data(), dst.Height(), dst.Width(), dst.Stride());
}

void CUDA_RecursiveGaussian::FilterH(Plane_FL &dst, const Plane_FL &src)
{
    FilterH(dst.data(), src.data(), dst.Height(), dst.Width(), dst.Stride());
}

void CUDA_RecursiveGaussian::Filter(Plane_FL &dst, const Plane_FL &src)
{
    Filter(dst.data(), src.data(), dst.Height(), dst.Width(), dst.Stride());
}
