#ifndef IMAGE_TYPE_HPP_
#define IMAGE_TYPE_HPP_


#include <algorithm>


// Enable C++ PPL Support
// Define it in the source file before #include "Image_Type.h" for fast compiling
//#define ENABLE_PPL

// Enable C++ AMP Support
// Define it in the source file before #include "Image_Type.h" for fast compiling
//#define ENABLE_AMP


#define FOR_EACH _For_each
#define TRANSFORM _Transform
#define CONVOLUTE _Convolute

#ifdef ENABLE_PPL
#include <ppl.h>
#include <ppltasks.h>
#define FOR_EACH_PPL _For_each_PPL
#define TRANSFORM_PPL _Transform_PPL
#define CONVOLUTE_PPL _Convolute_PPL
#else
#define FOR_EACH_PPL FOR_EACH
#define TRANSFORM_PPL TRANSFORM
#define CONVOLUTE_PPL CONVOLUTE
#endif

#ifdef ENABLE_AMP
#include <amp.h>
#include <amp_math.h>
#define FOR_EACH_AMP _For_each_AMP
#define TRANSFORM_AMP _Transform_AMP
#define CONVOLUTE_AMP _Convolute_AMP
#else
#define FOR_EACH_AMP FOR_EACH
#define TRANSFORM_AMP TRANSFORM
#define CONVOLUTE_AMP CONVOLUTE
#endif


// Template functions
template < typename _Ty > inline
void Quantize_Value(_Ty &Floor, _Ty &Neutral, _Ty &Ceil, _Ty BitDepth, QuantRange _QuantRange, bool Chroma)
{
    if (Chroma)
    {
        Floor = _QuantRange == QuantRange::PC ? _Ty(0) : _Ty(16) << (BitDepth - _Ty(8));
        Ceil = _QuantRange == QuantRange::PC ? (_Ty(1) << BitDepth) - _Ty(1) : _Ty(240) << (BitDepth - _Ty(8));
        Neutral = 1 << (BitDepth - 1);
        //ValueRange = _QuantRange == QuantRange::PC ? (_Ty(1) << BitDepth) - _Ty(1) : _Ty(224) << (BitDepth - _Ty(8));
    }
    else
    {
        Floor = _QuantRange == QuantRange::PC ? _Ty(0) : _Ty(16) << (BitDepth - _Ty(8));
        Ceil = _QuantRange == QuantRange::PC ? (_Ty(1) << BitDepth) - _Ty(1) : _Ty(235) << (BitDepth - _Ty(8));
        Neutral = Floor;
        //ValueRange = _QuantRange == QuantRange::PC ? (_Ty(1) << BitDepth) - _Ty(1) : _Ty(219) << (BitDepth - _Ty(8));
    }
}


// Template functions of algorithm
template < typename _St1, typename _Fn1 > inline
void _For_each(_St1 &data, _Fn1 &_Func)
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
void _Transform(_St1 &data, _Fn1 &_Func)
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
void _Transform(_Dt1 &dst, const _St1 &src, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform";
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
void _Transform(_Dt1 &dst, const _St1 &src1, const _St2 &src2, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height())
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

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void _Transform(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2 and src3 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    for (PCType j = 0; j < dst.Height(); ++j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto src1p = src1.Data() + src1.Stride() * j;
        auto src2p = src2.Data() + src2.Stride() * j;
        auto src3p = src3.Data() + src3.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++src1p, ++src2p, ++src3p)
        {
            *dstp = _Func(*src1p, *src2p, *src3p);
        }
    }
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void _Transform(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height()
        || dst.Width() != src4.Width() || dst.Height() != src4.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2, src3 and src4 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    for (PCType j = 0; j < dst.Height(); ++j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto src1p = src1.Data() + src1.Stride() * j;
        auto src2p = src2.Data() + src2.Stride() * j;
        auto src3p = src3.Data() + src3.Stride() * j;
        auto src4p = src4.Data() + src4.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++src1p, ++src2p, ++src3p, ++src4p)
        {
            *dstp = _Func(*src1p, *src2p, *src3p, *src4p);
        }
    }
}

template < PCType VRad, PCType HRad, typename _Dt1, typename _St1, typename _Fn1 > inline
void _Convolute(_Dt1 &dst, const _St1 &src, _Fn1 &_Func)
{
    const char *FunctionName = "_Convolute";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }
    
    for (PCType j = 0; j < dst.Height(); ++j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;

        typename _St1::const_pointer srcpV[VRad * 2 + 1];
        typename _St1::value_type srcb2D[VRad * 2 + 1][HRad * 2 + 1];

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
void _For_each_PPL(_St1 &data, _Fn1 &_Func)
{
    concurrency::parallel_for(PCType(0), dst.Height(), [&](PCType j)
    {
        auto datap = data.Data() + data.Stride() * j;

        for (auto upper = datap + data.Width(); datap != upper; ++datap)
        {
            _Func(*datap);
        }
    });
}

template < typename _St1, typename _Fn1 > inline
void _Transform_PPL(_St1 &data, _Fn1 &_Func)
{
    concurrency::parallel_for(PCType(0), dst.Height(), [&](PCType j)
    {
        auto datap = data.Data() + data.Stride() * j;

        for (auto upper = datap + data.Width(); datap != upper; ++datap)
        {
            *datap = _Func(*datap);
        }
    });
}

template < typename _Dt1, typename _St1, typename _Fn1 > inline
void _Transform_PPL(_Dt1 &dst, const _St1 &src, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_PPL";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&](PCType j)
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
void _Transform_PPL(_Dt1 &dst, const _St1 &src1, const _St2 &src2, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_PPL";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height() || dst.Width() != src2.Width() || dst.Height() != src2.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1 and src2 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&](PCType j)
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

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void _Transform_PPL(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_PPL";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2 and src3 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&](PCType j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto src1p = src1.Data() + src1.Stride() * j;
        auto src2p = src2.Data() + src2.Stride() * j;
        auto src3p = src3.Data() + src3.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++src1p, ++src2p, ++src3p)
        {
            *dstp = _Func(*src1p, *src2p, *src3p);
        }
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void _Transform_PPL(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_PPL";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height()
        || dst.Width() != src4.Width() || dst.Height() != src4.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2, src3 and src4 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&](PCType j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;
        auto src1p = src1.Data() + src1.Stride() * j;
        auto src2p = src2.Data() + src2.Stride() * j;
        auto src3p = src3.Data() + src3.Stride() * j;
        auto src4p = src4.Data() + src4.Stride() * j;

        for (auto upper = dstp + dst.Width(); dstp != upper; ++dstp, ++src1p, ++src2p, ++src3p, ++src4p)
        {
            *dstp = _Func(*src1p, *src2p, *src3p, *src4p);
        }
    });
}

template < PCType VRad, PCType HRad, typename _Dt1, typename _St1, typename _Fn1 > inline
void _Convolute_PPL(_Dt1 &dst, const _St1 &src, _Fn1 &_Func)
{
    const char *FunctionName = "_Convolute_PPL";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&](PCType j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;

        typename _St1::const_pointer srcpV[VRad * 2 + 1];
        typename _St1::value_type srcb2D[VRad * 2 + 1][HRad * 2 + 1];

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


// Template functions of algorithm with AMP
template < typename _St1, typename _Fn1 > inline
void _For_each_AMP(_St1 &data, _Fn1 &_Func)
{
    concurrency::array_view<decltype(data.value(0)), 1> datav(data.PixelCount(), datap);

    concurrency::parallel_for_each(datav.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        _Func(datav[idx]);
    });
}

template < typename _St1, typename _Fn1 > inline
void _Transform_AMP(_St1 &data, _Fn1 &_Func)
{
    concurrency::array_view<decltype(data.value(0)), 1> datav(data.PixelCount(), data);

    concurrency::parallel_for_each(datav.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        datav[idx] = _Func(datav[idx]);
    });
}

template < typename _Dt1, typename _St1, typename _Fn1 > inline
void _Transform_AMP(_Dt1 &dst, const _St1 &src, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_AMP";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::array_view<decltype(dst.value(0)), 1> dstv(dst.PixelCount(), dst);
    concurrency::array_view<const decltype(src.value(0)), 1> srcv(src.PixelCount(), src);
    dstv.discard_data();

    concurrency::parallel_for_each(dstv.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        dstv[idx] = _Func(srcv[idx]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _Fn1 > inline
void _Transform_AMP(_Dt1 &dst, const _St1 &src1, const _St2 &src2, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_AMP";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1 and src2 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::array_view<decltype(dst.value(0)), 1> dstv(dst.PixelCount(), dst);
    concurrency::array_view<const decltype(src1.value(0)), 1> src1v(src1.PixelCount(), src1);
    concurrency::array_view<const decltype(src2.value(0)), 1> src2v(src2.PixelCount(), src2);
    dstv.discard_data();

    concurrency::parallel_for_each(dstv.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        dstv[idx] = _Func(src1v[idx], src2v[idx]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void _Transform_AMP(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_AMP";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2 and src3 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::array_view<decltype(dst.value(0)), 1> dstv(dst.PixelCount(), dst);
    concurrency::array_view<const decltype(src1.value(0)), 1> src1v(src1.PixelCount(), src1);
    concurrency::array_view<const decltype(src2.value(0)), 1> src2v(src2.PixelCount(), src2);
    concurrency::array_view<const decltype(src3.value(0)), 1> src3v(src3.PixelCount(), src3);
    dstv.discard_data();

    concurrency::parallel_for_each(dstv.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        dstv[idx] = _Func(src1v[idx], src2v[idx], src3v[idx]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void _Transform_AMP(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 &_Func)
{
    const char *FunctionName = "_Transform_AMP";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height()
        || dst.Width() != src4.Width() || dst.Height() != src4.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2, src3 and src4 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::array_view<decltype(dst.value(0)), 1> dstv(dst.PixelCount(), dst);
    concurrency::array_view<const decltype(src1.value(0)), 1> src1v(src1.PixelCount(), src1);
    concurrency::array_view<const decltype(src2.value(0)), 1> src2v(src2.PixelCount(), src2);
    concurrency::array_view<const decltype(src3.value(0)), 1> src3v(src3.PixelCount(), src3);
    concurrency::array_view<const decltype(src4.value(0)), 1> src4v(src4.PixelCount(), src4);
    dstv.discard_data();

    concurrency::parallel_for_each(dstv.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        dstv[idx] = _Func(src1v[idx], src2v[idx], src3v[idx], src4v[idx]);
    });
}

template < PCType VRad, PCType HRad, typename _Dt1, typename _St1, typename _Fn1 > inline
void _Convolute_AMP(_Dt1 &dst, const _St1 &src, _Fn1 &_Func)
{
    const char *FunctionName = "_Convolute_AMP";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    concurrency::parallel_for(PCType(0), dst.Height(), [&_Func, &dst, &src](PCType j)
    {
        auto dstp = dst.Data() + dst.Stride() * j;

        typename _St1::const_pointer srcpV[VRad * 2 + 1];
        typename _St1::value_type srcb2D[VRad * 2 + 1][HRad * 2 + 1];

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
void Plane::for_each(_Fn1 _Func)
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::transform(_Fn1 _Func)
{
    TRANSFORM(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane::transform(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane::transform(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane::convolute(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE<VRad, HRad>(*this, src, _Func);
}


template < typename _Fn1 > inline
void Plane::for_each_PPL(_Fn1 _Func) const
{
    FOR_EACH_PPL(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::for_each_PPL(_Fn1 _Func)
{
    FOR_EACH_PPL(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::transform_PPL(_Fn1 _Func)
{
    TRANSFORM_PPL(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane::transform_PPL(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane::transform_PPL(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane::transform_PPL(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane::transform_PPL(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane::convolute_PPL(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE_PPL<VRad, HRad>(*this, src, _Func);
}


template < typename _Fn1 > inline
void Plane::for_each_AMP(_Fn1 _Func) const
{
    FOR_EACH_AMP(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::for_each_AMP(_Fn1 _Func)
{
    FOR_EACH_AMP(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::transform_AMP(_Fn1 _Func)
{
    TRANSFORM_AMP(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane::transform_AMP(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane::transform_AMP(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane::transform_AMP(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane::transform_AMP(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane::convolute_AMP(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE_AMP<VRad, HRad>(*this, src, _Func);
}


// Template functions for class Plane_FL
template < typename T > inline
FLType Plane_FL::Quantize(T input) const
{
    return input <= Floor_ ? Floor_ : input >= Ceil_ ? Ceil_ : input;
}


template < typename _Fn1 > inline
void Plane_FL::for_each(_Fn1 _Func) const
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::for_each(_Fn1 _Func)
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::transform(_Fn1 _Func)
{
    TRANSFORM(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane_FL::convolute(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE<VRad, HRad>(*this, src, _Func);
}


template < typename _Fn1 > inline
void Plane_FL::for_each_PPL(_Fn1 _Func) const
{
    FOR_EACH_PPL(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::for_each_PPL(_Fn1 _Func)
{
    FOR_EACH_PPL(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::transform_PPL(_Fn1 _Func)
{
    TRANSFORM_PPL(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane_FL::transform_PPL(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane_FL::transform_PPL(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane_FL::transform_PPL(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane_FL::transform_PPL(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM_PPL(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane_FL::convolute_PPL(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE_PPL<VRad, HRad>(*this, src, _Func);
}


template < typename _Fn1 > inline
void Plane_FL::for_each_AMP(_Fn1 _Func) const
{
    FOR_EACH_AMP(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::for_each_AMP(_Fn1 _Func)
{
    FOR_EACH_AMP(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::transform_AMP(_Fn1 _Func)
{
    TRANSFORM_AMP(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane_FL::transform_AMP(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane_FL::transform_AMP(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane_FL::transform_AMP(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane_FL::transform_AMP(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM_AMP(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane_FL::convolute_AMP(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE_AMP<VRad, HRad>(*this, src, _Func);
}


#endif