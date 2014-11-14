#ifndef IMAGE_TYPE_HPP_
#define IMAGE_TYPE_HPP_


#include <algorithm>
#include <ppl.h>


#define ENABLE_PPL
#define ENABLE_AMP


#define FOR_EACH _For_each
#define TRANSFORM _Transform
#define CONVOLUTE _Convolute

#ifdef ENABLE_PPL
#define FOR_EACH_PPL _For_each_PPL
#define TRANSFORM_PPL _Transform_PPL
#define CONVOLUTE_PPL _Convolute_PPL
#else
#define FOR_EACH_PPL FOR_EACH
#define TRANSFORM_PPL TRANSFORM
#define CONVOLUTE_PPL CONVOLUTE
#endif

#ifdef ENABLE_AMP
#define FOR_EACH_AMP _For_each_AMP
#define TRANSFORM_AMP _Transform_AMP
#define CONVOLUTE_AMP _Convolute_AMP
#else
#define FOR_EACH_AMP FOR_EACH
#define TRANSFORM_AMP TRANSFORM
#define CONVOLUTE_AMP CONVOLUTE
#endif


// Template functions of algorithm
template < typename _St1, typename _Fn1 > inline
void _For_each(_St1& data, _Fn1& _Func)
{
    for (PCType j = 0; j < data.Height(); ++j)
    {
        auto datap = data.Data() + data.Stride() * j;

        for (auto upper = datap + data.Width(); datap != upper; ++datap)
        {
            _Func(*datap);
        }
    }
}

template < typename _St1, typename _Fn1 > inline
void _Transform(_St1& data, _Fn1& _Func)
{
    for (PCType j = 0; j < data.Height(); ++j)
    {
        auto datap = data.Data() + data.Stride() * j;

        for (auto upper = datap + data.Width(); datap != upper; ++datap)
        {
            *datap = _Func(*datap);
        }
    }
}

template < typename _Dt1, typename _St1, typename _Fn1 > inline
void _Transform(_Dt1& dst, const _St1& src, _Fn1& _Func)
{
    const char * FunctionName = "_Transform";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    for (PCType j = 0; j < dst.Height(); ++j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto srcp = src.Data() + src.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++srcp)
        {
            *dstp = _Func(*srcp);
        }
    }
}

template < typename _Dt1, typename _St1, typename _St2, typename _Fn1 > inline
void _Transform(_Dt1& dst, const _St1& src1, const _St2& src2, _Fn1& _Func)
{
    const char * FunctionName = "_Transform";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height() || dst.Width() != src2.Width() || dst.Height() != src2.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1 and src2 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    for (PCType j = 0; j < dst.Height(); ++j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto src1p = src1.Data() + src1.Stride() * j;
        auto src2p = src2.Data() + src2.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++src1p, ++src2p)
        {
            *dstp = _Func(*src1p, *src2p);
        }
    }
}

template < PCType VRad, PCType HRad, typename _Dt1, typename _St1, typename _Fn1 > inline
void _Convolute(_Dt1& dst, const _St1& src, _Fn1& _Func)
{
    const char * FunctionName = "_Convolute";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }
    
    for (PCType j = 0; j < dst.Height(); ++j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;

        decltype(src.Data()) srcpV[VRad * 2 + 1];
        decltype(src[0]) srcb2D[VRad * 2 + 1][HRad * 2 + 1];

        srcpV[VRad] = src.Data() + src.Stride() * j;
        for (PCType y = 1; y <= VRad; ++y)
        {
            srcpV[VRad - y] = j < y ? srcpV[VRad - y + 1] : srcpV[VRad - y + 1] - src.Stride();
            srcpV[VRad + y] = j >= src.Height() - y ? srcpV[VRad + y - 1] : srcpV[VRad + y - 1] + src.Stride();
        }
        
        for (PCType y = 0; y < VRad * 2 + 1; ++y)
        {
            PCType x = 0;
            for (; x < HRad + 2 && x < HRad * 2 + 1; ++x)
            {
                srcb2D[y][x] = srcpV[y][0];
            }
            for (; x < HRad * 2 + 1; ++x)
            {
                srcb2D[y][x] = srcpV[y][x - HRad - 1];
            }
        }

        PCType i = HRad;
        for (; i < dst.Width(); ++i)
        {
            for (PCType y = 0; y < VRad * 2 + 1; ++y)
            {
                PCType x = 0;
                for (; x < HRad * 2; ++x)
                {
                    srcb2D[y][x] = srcb2D[y][x + 1];
                }
                srcb2D[y][x] = srcpV[y][i];
            }

            dstp[i - HRad] = _Func(srcb2D);
        }
        for (; i < dst.Width() + HRad; ++i)
        {
            for (PCType y = 0; y < VRad * 2 + 1; ++y)
            {
                PCType x = 0;
                for (; x < HRad * 2; ++x)
                {
                    srcb2D[y][x] = srcb2D[y][x + 1];
                }
            }

            dstp[i - HRad] = _Func(srcb2D);
        }
    }
}


// Template functions of algorithm with PPL
template < typename _St1, typename _Fn1 > inline
void _For_each_PPL(_St1& data, _Fn1& _Func)
{
    concurrency::parallel_for(PCType(0), dst.Height(), [&_Func, &data](PCType j)
    {
        auto datap = data.Data() + data.Stride() * j;

        for (auto upper = datap + data.Width(); datap != upper; ++datap)
        {
            _Func(*datap);
        }
    });
}

template < typename _St1, typename _Fn1 > inline
void _Transform_PPL(_St1& data, _Fn1& _Func)
{
    concurrency::parallel_for(PCType(0), dst.Height(), [&_Func, &data](PCType j)
    {
        auto datap = data.Data() + data.Stride() * j;

        for (auto upper = datap + data.Width(); datap != upper; ++datap)
        {
            *datap = _Func(*datap);
        }
    });
}

template < typename _Dt1, typename _St1, typename _Fn1 > inline
void _Transform_PPL(_Dt1& dst, const _St1& src, _Fn1& _Func)
{
    const char * FunctionName = "_Transform_PPL";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&_Func, &dst, &src](PCType j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto srcp = src.Data() + src.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++srcp)
        {
            *dstp = _Func(*srcp);
        }
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _Fn1 > inline
void _Transform_PPL(_Dt1& dst, const _St1& src1, const _St2& src2, _Fn1& _Func)
{
    const char * FunctionName = "_Transform_PPL";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height() || dst.Width() != src2.Width() || dst.Height() != src2.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1 and src2 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&_Func, &dst, &src1, &src2](PCType j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto src1p = src1.Data() + src1.Stride() * j;
        auto src2p = src2.Data() + src2.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++src1p, ++src2p)
        {
            *dstp = _Func(*src1p, *src2p);
        }
    });
}

template < PCType VRad, PCType HRad, typename _Dt1, typename _St1, typename _Fn1 > inline
void _Convolute_PPL(_Dt1& dst, const _St1& src, _Fn1& _Func)
{
    const char * FunctionName = "_Convolute_PPL";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&_Func, &dst, &src](PCType j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;

        decltype(src.Data()) srcpV[VRad * 2 + 1];
        decltype(src[0]) srcb2D[VRad * 2 + 1][HRad * 2 + 1];

        srcpV[VRad] = src.Data() + src.Stride() * j;
        for (PCType y = 1; y <= VRad; ++y)
        {
            srcpV[VRad - y] = j < y ? srcpV[VRad - y + 1] : srcpV[VRad - y + 1] - src.Stride();
            srcpV[VRad + y] = j >= src.Height() - y ? srcpV[VRad + y - 1] : srcpV[VRad + y - 1] + src.Stride();
        }

        for (PCType y = 0; y < VRad * 2 + 1; ++y)
        {
            PCType x = 0;
            for (; x < HRad + 2 && x < HRad * 2 + 1; ++x)
            {
                srcb2D[y][x] = srcpV[y][0];
            }
            for (; x < HRad * 2 + 1; ++x)
            {
                srcb2D[y][x] = srcpV[y][x - HRad - 1];
            }
        }

        PCType i = HRad;
        for (; i < dst.Width(); ++i)
        {
            for (PCType y = 0; y < VRad * 2 + 1; ++y)
            {
                PCType x = 0;
                for (; x < HRad * 2; ++x)
                {
                    srcb2D[y][x] = srcb2D[y][x + 1];
                }
                srcb2D[y][x] = srcpV[y][i];
            }

            dstp[i - HRad] = _Func(srcb2D);
        }
        for (; i < dst.Width() + HRad; ++i)
        {
            for (PCType y = 0; y < VRad * 2 + 1; ++y)
            {
                PCType x = 0;
                for (; x < HRad * 2; ++x)
                {
                    srcb2D[y][x] = srcb2D[y][x + 1];
                }
            }

            dstp[i - HRad] = _Func(srcb2D);
        }
    });
}


// Template functions for class Plane
template < typename T > inline
DType Plane::Quantize(T input) const
{
    T input_up = input + T(0.5);
    return input <= Floor_ ? Floor_ : input_up >= Ceil_ ? Ceil_ : static_cast<DType>(input_up);
}


template < typename _Fn1 > inline
void Plane::for_each(_Fn1 _Func) const
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::transform(_Fn1 _Func)
{
    TRANSFORM(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane::transform(const _St1& src, _Fn1 _Func)
{
    TRANSFORM(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane::transform(const _St1& src1, const _St2& src2, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane::convolute(const _St1& src, _Fn1 _Func)
{
    CONVOLUTE<VRad, HRad>(*this, src, _Func);
}


template < typename _Fn1 > inline
void Plane::for_each_PPL(_Fn1 _Func) const
{
    FOR_EACH_PPL(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::transform_PPL(_Fn1 _Func)
{
    TRANSFORM_PPL(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane::transform_PPL(const _St1& src, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane::transform_PPL(const _St1& src1, const _St2& src2, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src1, src2, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane::convolute_PPL(const _St1& src, _Fn1 _Func)
{
    CONVOLUTE_PPL<VRad, HRad>(*this, src, _Func);
}


// Template functions for class Plane_FL
template < typename T > inline
FLType Plane_FL::Quantize(T input) const
{
    return input <= Floor_ ? Floor_ : input >= Ceil_ ? Ceil_ : input;
}


#endif