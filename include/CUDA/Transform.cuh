#ifndef TRANSFORM_CUH_
#define TRANSFORM_CUH_


#include "Helper.cuh"
#include "Image_Type.h"


void CUDA_Transpose(DType *dst, const DType *src, const PCType src_height, const PCType src_width,
    CudaMemMode mem_mode = CudaMemMode::Host2Device);
void CUDA_Transpose(FLType *dst, const FLType *src, const PCType src_height, const PCType src_width,
    CudaMemMode mem_mode = CudaMemMode::Host2Device);
Plane &CUDA_Transpose(Plane &dst, const Plane &src, CudaMemMode mem_mode = CudaMemMode::Host2Device);
Plane_FL &CUDA_Transpose(Plane_FL &dst, const Plane_FL &src, CudaMemMode mem_mode = CudaMemMode::Host2Device);


inline Frame &CUDA_Transpose(Frame &dst, const Frame &src)
{
    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
        CUDA_Transpose(dst.P(i), src.P(i));
    return dst;
}

inline Plane &CUDA_Transpose(Plane &src)
{
    Plane temp(src);
    return CUDA_Transpose(src, temp);
}

inline Plane_FL &CUDA_Transpose(Plane_FL &src)
{
    Plane_FL temp(src);
    return CUDA_Transpose(src, temp);
}

inline Frame &CUDA_Transpose(Frame &src)
{
    Frame temp(src);
    return CUDA_Transpose(src, temp);
}


#endif
