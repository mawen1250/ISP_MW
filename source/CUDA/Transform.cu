#define ENABLE_PPL


#include "Transform.cuh"


const cuIdx TransposeBlockDim = 16;


template < typename _Ty >
static __global__ void CUDA_Transpose_Kernel(_Ty *dst, const _Ty *src,
    const PCType src_height, const PCType src_width)
{
    __shared__ _Ty block[TransposeBlockDim][TransposeBlockDim + 1];

    // read the matrix tile into shared memory
    cuIdx xIdx = blockIdx.x * TransposeBlockDim + threadIdx.x;
    cuIdx yIdx = blockIdx.y * TransposeBlockDim + threadIdx.y;

    if (xIdx < src_width && yIdx < src_height)
    {
        block[threadIdx.y][threadIdx.x] = src[yIdx * src_width + xIdx];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIdx = blockIdx.y * TransposeBlockDim + threadIdx.x;
    yIdx = blockIdx.x * TransposeBlockDim + threadIdx.y;

    if (xIdx < src_height && yIdx < src_width)
    {
        dst[yIdx * src_height + xIdx] = block[threadIdx.x][threadIdx.y];
    }
}

template < typename _Ty >
static __global__ void CUDA_Transpose_Kernel_Align(_Ty *dst, const _Ty *src,
    const PCType src_height, const PCType src_width)
{
    __shared__ _Ty block[TransposeBlockDim][TransposeBlockDim + 1];

    // read the matrix tile into shared memory
    cuIdx xIdx = blockIdx.x * TransposeBlockDim + threadIdx.x;
    cuIdx yIdx = blockIdx.y * TransposeBlockDim + threadIdx.y;

    block[threadIdx.y][threadIdx.x] = src[yIdx * src_width + xIdx];

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIdx = blockIdx.y * TransposeBlockDim + threadIdx.x;
    yIdx = blockIdx.x * TransposeBlockDim + threadIdx.y;

    dst[yIdx * src_height + xIdx] = block[threadIdx.x][threadIdx.y];
}


template < typename _Ty >
void CUDA_Transpose(_Ty *dst, const _Ty *src, const PCType src_height, const PCType src_width,
    CudaMemMode mem_mode = CudaMemMode::Host2Device)
{
    CUDA_FilterData<_Ty> d(mem_mode);

    d.Init(dst, src, src_width, src_height, src_height);

    if (d.dst_dev == d.src_dev)
    {
        std::cerr << "CUDA_Transpose: dst_dev and src_dev should not be the same!\n";
        exit(EXIT_FAILURE);
    }

    dim3 grid_dim(CudaGridDim(src_width, TransposeBlockDim), CudaGridDim(src_height, TransposeBlockDim));
    dim3 block_dim(TransposeBlockDim, TransposeBlockDim);

    if (src_width % TransposeBlockDim == 0 && src_height % TransposeBlockDim == 0)
    {
        CudaGlobalCall(CUDA_Transpose_Kernel_Align, grid_dim, block_dim)(d.dst_dev, d.src_dev, src_height, src_width);
    }
    else
    {
        CudaGlobalCall(CUDA_Transpose_Kernel, grid_dim, block_dim)(d.dst_dev, d.src_dev, src_height, src_width);
    }

    d.End();
}

void CUDA_Transpose(DType *dst, const DType *src, const PCType src_height, const PCType src_width, CudaMemMode mem_mode)
{
    CUDA_Transpose<DType>(dst, src, src_height, src_width, mem_mode);
}

void CUDA_Transpose(FLType *dst, const FLType *src, const PCType src_height, const PCType src_width, CudaMemMode mem_mode)
{
    CUDA_Transpose<FLType>(dst, src, src_height, src_width, mem_mode);
}


template < typename _St1 >
_St1 &CUDA_Transpose(_St1 &dst, const _St1 &src, CudaMemMode mem_mode = CudaMemMode::Host2Device)
{
    const PCType src_width = src.Width();
    const PCType src_height = src.Height();

    // Change Plane info
    dst.ReSize(src_height, src_width);

    CUDA_Transpose(dst.data(), src.data(), src_height, src_width, mem_mode);

    return dst;
}

Plane &CUDA_Transpose(Plane &dst, const Plane &src, CudaMemMode mem_mode)
{
    return CUDA_Transpose<Plane>(dst, src, mem_mode);
}

Plane_FL &CUDA_Transpose(Plane_FL &dst, const Plane_FL &src, CudaMemMode mem_mode)
{
    return CUDA_Transpose<Plane_FL>(dst, src, mem_mode);
}
