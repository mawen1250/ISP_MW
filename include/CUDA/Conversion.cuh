#ifndef CONVERSION_CUH_
#define CONVERSION_CUH_


#include "Helper.cuh"
#include "Specification.h"


void CUDA_GetMinMax(const FLType *src, const cuIdx pcount, FLType &min, FLType &max);
void CUDA_GetMinMax(const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount, FLType &min, FLType &max);

void CUDA_ValidRange(const FLType *src, const cuIdx pcount, FLType &min, FLType &max,
    double lower_thr = 0., double upper_thr = 0., bool protect = false, FLType Floor = 0, FLType Ceil = 1);
void CUDA_ValidRange(const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount, FLType &min, FLType &max,
    double lower_thr = 0., double upper_thr = 0., bool protect = false, FLType Floor = 0, FLType Ceil = 1);

void CUDA_ConvertToY(FLType *dst, const FLType *srcR, const FLType *srcG, const FLType *srcB, const cuIdx pcount, ColorMatrix dstColorMatrix = ColorMatrix::Average);


#endif
