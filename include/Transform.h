#ifndef TRANSFORM_H_
#define TRANSFORM_H_


#include "Image_Type.h"


Plane &Transpose(Plane &dst, const Plane &src);
Plane_FL &Transpose(Plane_FL &dst, const Plane_FL &src);
inline Frame &Transpose(Frame &dst, const Frame &src)
{
    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
        Transpose(dst.P(i), src.P(i));
    return dst;
}

inline Plane &Transpose(Plane &src)
{
    Plane temp(src);
    return Transpose(src, temp);
}

inline Plane_FL &Transpose(Plane_FL &src)
{
    Plane_FL temp(src);
    return Transpose(src, temp);
}

inline Frame &Transpose(Frame &src)
{
    Frame temp(src);
    return Transpose(src, temp);
}


template < typename _Ty >
void Transpose(_Ty *dst, const _Ty *src, const PCType src_width,
    const PCType src_height, const PCType src_stride, const PCType dst_stride)
{
    const PCType &dst_width = src_height;
    const PCType &dst_height = src_width;

    // Apply transpose
    for (PCType j = 0; j < dst_height; ++j)
    {
        PCType di = j * dst_stride;
        PCType si = j;

        for (const PCType upper = di + dst_width; di < upper; ++di, si += src_stride)
        {
            dst[di] = src[si];
        }
    }
}


#endif
