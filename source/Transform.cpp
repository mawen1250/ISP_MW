#include "Transform.h"


template < typename _St1 >
_St1 &TransposeT(_St1 &dst, const _St1 &src)
{
    const PCType src_width = src.Width();
    const PCType src_height = src.Height();

    // Change Plane info
    dst.ReSize(src_height, src_width);

    const PCType src_stride = src.Stride();
    const PCType dst_stride = dst.Stride();
    
    // Apply transpose
    Transpose(dst.data(), src.data(), src_width, src_height, src_stride, dst_stride);

    // Output
    return dst;
}


Plane &Transpose(Plane &dst, const Plane &src)
{
    return TransposeT(dst, src);
}


Plane_FL &Transpose(Plane_FL &dst, const Plane_FL &src)
{
    return TransposeT(dst, src);
}
