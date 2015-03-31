#ifndef IMAGE_TYPE_HPP_
#define IMAGE_TYPE_HPP_


#include <algorithm>
#include "Histogram.h"


// Enable C++ PPL Support
// Define it in the source file before #include "Image_Type.h" for fast compiling
//#define ENABLE_PPL

// Enable C++ AMP Support
// Define it in the source file before #include "Image_Type.h" for fast compiling
//#define ENABLE_AMP


#define LOOP_V _Loop_V
#define LOOP_H _Loop_H
#define LOOP_Hinv _Loop_Hinv
#define LOOP_VH _Loop_VH
#define FOR_EACH _For_each
#define TRANSFORM _Transform
#define CONVOLUTE _Convolute

#ifdef ENABLE_PPL
#include <ppl.h>
#include <ppltasks.h>
#define LOOP_V_PPL _Loop_V_PPL
#define LOOP_H_PPL _Loop_H_PPL
#define LOOP_Hinv_PPL _Loop_Hinv_PPL
#define LOOP_VH_PPL _Loop_VH_PPL
#define FOR_EACH_PPL _For_each_PPL
#define TRANSFORM_PPL _Transform_PPL
#define CONVOLUTE_PPL _Convolute_PPL
const PCType PPL_HP = 256; // Height piece size for PPL
const PCType PPL_WP = 256; // Width piece size for PPL
#else
#define LOOP_V_PPL _Loop_V
#define LOOP_H_PPL _Loop_H
#define LOOP_Hinv_PPL _Loop_Hinv
#define LOOP_VH_PPL _Loop_VH
#define FOR_EACH_PPL FOR_EACH
#define TRANSFORM_PPL TRANSFORM
#define CONVOLUTE_PPL CONVOLUTE
#endif

#ifdef ENABLE_AMP
#include <amp.h>
#include <amp_math.h>
#define FOR_EACH_AMP _For_each_AMP
#define TRANSFORM_AMP _Transform_AMP
#else
#define FOR_EACH_AMP FOR_EACH
#define TRANSFORM_AMP TRANSFORM
#endif


// Template functions
template < typename _St1 > inline
bool isValueFloat(const _St1 &src)
{
    return isFloat(typename _St1::value_type);
}


template < typename _Ty >
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


template < typename _Ty > inline
bool isChroma(_Ty Floor, _Ty Neutral)
{
    return Floor < Neutral;
}

template < typename _Ty > inline
bool isPCChromaInt(_Ty Floor, _Ty Neutral, _Ty Ceil)
{
    return Floor < Neutral && (Floor + Ceil) % 2 == 1;
}
template < typename _Ty > inline
bool isPCChromaFloat(_Ty Floor, _Ty Neutral, _Ty Ceil)
{
    return false;
}
template < typename _Ty > inline
bool isPCChroma(_Ty Floor, _Ty Neutral, _Ty Ceil)
{
    return isPCChromaInt(Floor, Neutral, Ceil);
}
template < > inline
bool isPCChroma(float Floor, float Neutral, float Ceil)
{
    return isPCChromaFloat(Floor, Neutral, Ceil);
}
template < > inline
bool isPCChroma(double Floor, double Neutral, double Ceil)
{
    return isPCChromaFloat(Floor, Neutral, Ceil);
}
template < > inline
bool isPCChroma(long double Floor, long double Neutral, long double Ceil)
{
    return isPCChromaFloat(Floor, Neutral, Ceil);
}

template < typename _Ty >
void ReSetChromaInt(_Ty &Floor, _Ty &Neutral, _Ty &Ceil, bool Chroma = false)
{
    const bool srcChroma = isChroma(Floor, Neutral);

    if (Chroma && !srcChroma)
    {
        Neutral = (Floor + Ceil + 1) / 2;
    }
    else if (!Chroma && srcChroma)
    {
        Neutral = Floor;
    }
}
template < typename _Ty >
void ReSetChromaFloat(_Ty &Floor, _Ty &Neutral, _Ty &Ceil, bool Chroma = false)
{
    const bool srcChroma = isChroma(Floor, Neutral);

    const _Ty Range = Ceil - Floor;

    if (Chroma && !srcChroma)
    {
        Floor = -Range / 2;
        Neutral = 0;
        Ceil = Range / 2;
    }
    else if (!Chroma && srcChroma)
    {
        Floor = 0;
        Neutral = 0;
        Ceil = Range;
    }
}
template < typename _Ty > inline
void ReSetChroma(_Ty &Floor, _Ty &Neutral, _Ty &Ceil, bool Chroma = false)
{
    ReSetChromaInt(Floor, Neutral, Ceil, Chroma);
}
template < > inline
void ReSetChroma(float &Floor, float &Neutral, float &Ceil, bool Chroma)
{
    ReSetChromaFloat(Floor, Neutral, Ceil, Chroma);
}
template < > inline
void ReSetChroma(double &Floor, double &Neutral, double &Ceil, bool Chroma)
{
    ReSetChromaFloat(Floor, Neutral, Ceil, Chroma);
}
template < > inline
void ReSetChroma(long double &Floor, long double &Neutral, long double &Ceil, bool Chroma)
{
    ReSetChromaFloat(Floor, Neutral, Ceil, Chroma);
}


template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 &dst, const _St1 &src,
    typename _Dt1::value_type dFloor, typename _Dt1::value_type dNeutral, typename _Dt1::value_type dCeil,
    typename _St1::value_type sFloor, typename _St1::value_type sNeutral, typename _St1::value_type sCeil,
    bool clip = false)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool srcFloat = isFloat(srcType);
    const bool dstFloat = isFloat(dstType);
    const bool sameType = typeid(srcType) == typeid(dstType);

    bool srcChroma = isChroma(sFloor, sNeutral);
    bool dstChroma = isChroma(dFloor, dNeutral);
    bool srcPCChroma = isPCChroma(sFloor, sNeutral, sCeil);
    bool dstPCChroma = isPCChroma(dFloor, dNeutral, dCeil);

    if (dstChroma && !srcChroma)
    {
        if (srcFloat)
        {
            sNeutral = (sFloor + sCeil) / 2;
        }
        else
        {
            sNeutral = (sFloor + sCeil + 1) / 2;
        }
        srcChroma = true;
    }
    else if (!dstChroma && srcChroma)
    {
        sNeutral = sFloor;
        srcChroma = false;
    }

    dst.ReSize(src.Width(), src.Height());

    const PCType height = dst.Height();
    const PCType width = dst.Width();
    const PCType stride = dst.Stride();

    const auto sRange = sCeil - sFloor;
    const auto dRange = dCeil - dFloor;

    if (sFloor == dFloor && sNeutral == dNeutral && sCeil == dCeil) // src and dst are of the same quantization range
    {
        if (reinterpret_cast<const void *>(dst.Data()) != reinterpret_cast<const void *>(src.Data()))
        {
            return;
        }
        if (sameType && !clip) // Copy memory if they are of the same type and clipping is not performed
        {
            memcpy(dst.Data(), src.Data(), sizeof(dstType) * dst.PixelCount());
        }
        else if (srcFloat && !dstFloat) // Conversion from float to integer
        {
            srcType offset = srcType(dstPCChroma ? 0.499999 : 0.5);

            if (clip)
            {
                const srcType lowerL = static_cast<srcType>(dFloor);
                const srcType upperL = static_cast<srcType>(dCeil);
                
                LOOP_VH_PPL(height, width, stride, [&](PCType i)
                {
                    dst[i] = static_cast<dstType>(Clip(src[i] + offset, lowerL, upperL));
                });
            }
            else
            {
                LOOP_VH_PPL(height, width, stride, [&](PCType i)
                {
                    dst[i] = static_cast<dstType>(src[i] + offset);
                });
            }
        }
        else // Otherwise cast type with/without clipping
        {
            if (clip)
            {
                const srcType lowerL = static_cast<srcType>(dFloor);
                const srcType upperL = static_cast<srcType>(dCeil);

                LOOP_VH_PPL(height, width, stride, [&](PCType i)
                {
                    dst[i] = static_cast<dstType>(Clip(src[i], lowerL, upperL));
                });
            }
            else
            {
                LOOP_VH_PPL(height, width, stride, [&](PCType i)
                {
                    dst[i] = static_cast<dstType>(src[i]);
                });
            }
        }
    }
    else // src and dst are of different quantization range
    {
        // Always apply clipping if source is PC range chroma
        if (srcPCChroma) clip = true;

        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = dNeutral - sNeutral * gain;
        if (!dstFloat) offset += FLType(dstPCChroma ? 0.499999 : 0.5);

        if (clip)
        {
            const FLType lowerL = static_cast<FLType>(dFloor);
            const FLType upperL = static_cast<FLType>(dCeil);

            LOOP_VH_PPL(height, width, stride, [&](PCType i)
            {
                dst[i] = static_cast<dstType>(Clip(static_cast<FLType>(src[i]) * gain + offset, lowerL, upperL));
            });
        }
        else
        {
            LOOP_VH_PPL(height, width, stride, [&](PCType i)
            {
                dst[i] = static_cast<dstType>(static_cast<FLType>(src[i]) * gain + offset);
            });
        }
    }
}

template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 &dst, const _St1 &src, bool clip = false)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool sameType = typeid(srcType) == typeid(dstType);

    if (sameType && reinterpret_cast<const void *>(dst.Data()) == reinterpret_cast<const void *>(src.Data()))
    {
        return;
    }

    if (dst.isChroma() != src.isChroma())
    {
        dst.ReSetChroma(src.isChroma());
    }

    RangeConvert(dst, src, dst.Floor(), dst.Neutral(), dst.Ceil(), src.Floor(), src.Neutral(), src.Ceil(), clip);
}

template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB,
    typename _Dt1::value_type dFloor, typename _Dt1::value_type dNeutral, typename _Dt1::value_type dCeil,
    typename _St1::value_type sFloor, typename _St1::value_type sNeutral, typename _St1::value_type sCeil,
    bool clip = false)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool srcFloat = isFloat(srcType);
    const bool dstFloat = isFloat(dstType);

    bool srcChroma = isChroma(sFloor, sNeutral);
    bool dstChroma = isChroma(dFloor, dNeutral);
    bool srcPCChroma = isPCChroma(sFloor, sNeutral, sCeil);
    bool dstPCChroma = isPCChroma(dFloor, dNeutral, dCeil);

    if (dstChroma && !srcChroma)
    {
        if (srcFloat)
        {
            sNeutral = (sFloor + sCeil) / 2;
        }
        else
        {
            sNeutral = (sFloor + sCeil + 1) / 2;
        }
        srcChroma = true;
    }
    else if (!dstChroma && srcChroma)
    {
        sNeutral = sFloor;
        srcChroma = false;
    }

    dstR.ReSize(srcR.Width(), srcR.Height());
    dstG.ReSize(srcG.Width(), srcG.Height());
    dstB.ReSize(srcB.Width(), srcB.Height());

    const PCType height = dstR.Height();
    const PCType width = dstR.Width();
    const PCType stride = dstR.Stride();

    const auto sRange = sCeil - sFloor;
    const auto dRange = dCeil - dFloor;

    FLType gain = static_cast<FLType>(dRange) / sRange;
    FLType offset = dNeutral - sNeutral * gain;
    if (!dstFloat) offset += FLType(0.5);

    // Always apply clipping if source is PC range chroma
    if (srcPCChroma) clip = true;

    if (clip)
    {
        const FLType lowerL = static_cast<FLType>(dFloor);
        const FLType upperL = static_cast<FLType>(dCeil);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dstR[i] = static_cast<dstType>(Clip(static_cast<FLType>(srcR[i]) * gain + offset, lowerL, upperL));
            dstG[i] = static_cast<dstType>(Clip(static_cast<FLType>(srcG[i]) * gain + offset, lowerL, upperL));
            dstB[i] = static_cast<dstType>(Clip(static_cast<FLType>(srcB[i]) * gain + offset, lowerL, upperL));
        });
    }
    else
    {
        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dstR[i] = static_cast<dstType>(static_cast<FLType>(srcR[i]) * gain + offset);
            dstG[i] = static_cast<dstType>(static_cast<FLType>(srcG[i]) * gain + offset);
            dstB[i] = static_cast<dstType>(static_cast<FLType>(srcB[i]) * gain + offset);
        });
    }
}

template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, bool clip = false)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool sameType = typeid(srcType) == typeid(dstType);

    if (sameType && reinterpret_cast<const void *>(dstR.Data()) == reinterpret_cast<const void *>(srcR.Data())
        && reinterpret_cast<const void *>(dstG.Data()) == reinterpret_cast<const void *>(srcG.Data())
        && reinterpret_cast<const void *>(dstB.Data()) == reinterpret_cast<const void *>(srcB.Data()))
    {
        return;
    }

    RangeConvert(dstR, dstG, dstB, srcR, srcG, srcB, dstR.Floor(), dstR.Neutral(), dstR.Ceil(), srcR.Floor(), srcR.Neutral(), srcR.Ceil(), clip);
}


// Template functions of algorithm
template < typename _Fn1 >
void _Loop_V(const PCType height, _Fn1 &&_Func)
{
    for (PCType j = 0; j < height; ++j)
    {
        _Func(j);
    }
}

template < typename _Fn1 >
void _Loop_H(const PCType height, const PCType width, const PCType stride, _Fn1 &&_Func)
{
    const PCType offset = 0;
    const PCType range = width;

    for (PCType j = 0; j < height; ++j)
    {
        const PCType lower = j * stride + offset;
        const PCType upper = lower + range;

        _Func(j, lower, upper);
    }
}

template < typename _Fn1 >
void _Loop_Hinv(const PCType height, const PCType width, const PCType stride, _Fn1 &&_Func)
{
    const PCType offset = 0;
    const PCType range = width;

    for (PCType j = height - 1; j >= 0; --j)
    {
        const PCType lower = j * stride + offset;
        const PCType upper = lower + range;

        _Func(j, lower, upper);
    }
}

template < typename _Fn1 >
void _Loop_VH(const PCType height, const PCType width, const PCType stride, _Fn1 &&_Func)
{
    for (PCType j = 0; j < height; ++j)
    {
        PCType i = j * stride;

        for (const PCType upper = i + width; i < upper; ++i)
        {
            _Func(i);
        }
    }
}


template < typename _St1, typename _Fn1 >
void _For_each(_St1 &data, _Fn1 &&_Func)
{
    LOOP_VH(data.Height(), data.Width(), data.Stride(), [&](PCType i)
    {
        _Func(data[i]);
    });
}

template < typename _St1, typename _Fn1 >
void _Transform(_St1 &data, _Fn1 &&_Func)
{
    LOOP_VH(data.Height(), data.Width(), data.Stride(), [&](PCType i)
    {
        data[i] = _Func(data[i]);
    });
}

template < typename _Dt1, typename _St1, typename _Fn1 >
void _Transform(_Dt1 &dst, const _St1 &src, _Fn1 &&_Func)
{
    const char *FunctionName = "_Transform";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_VH(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src[i]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _Fn1 >
void _Transform(_Dt1 &dst, const _St1 &src1, const _St2 &src2, _Fn1 &&_Func)
{
    const char *FunctionName = "_Transform";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1 and src2 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_VH(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src1[i], src2[i]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _Fn1 >
void _Transform(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 &&_Func)
{
    const char *FunctionName = "_Transform";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2 and src3 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_VH(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src1[i], src2[i], src3[i]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 >
void _Transform(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 &&_Func)
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

    LOOP_VH(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src1[i], src2[i], src3[i], src4[i]);
    });
}

template < PCType VRad, PCType HRad, typename _Dt1, typename _St1, typename _Fn1 >
void _Convolute(_Dt1 &dst, const _St1 &src, _Fn1 &&_Func)
{
    const char *FunctionName = "_Convolute";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_V(dst.Height(), [&](PCType j)
    {
        auto dstp = dst.Data() + j * dst.Stride();

        typename _St1::const_pointer srcpV[VRad * 2 + 1];
        typename _St1::value_type srcb2D[VRad * 2 + 1][HRad * 2 + 1];

        srcpV[VRad] = src.Data() + j * src.Stride();
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


// Template functions of algorithm with PPL
#ifdef ENABLE_PPL
template < typename _Fn1 >
void _Loop_V_PPL(const PCType height, _Fn1 &&_Func)
{
    const PCType pNum = (height + PPL_HP - 1) / PPL_HP;

    concurrency::parallel_for(PCType(0), pNum, [&](PCType p)
    {
        const PCType lower = p * PPL_HP;
        const PCType upper = Min(height, lower + PPL_HP);

        for (PCType j = lower; j < upper; ++j)
        {
            _Func(j);
        }
    });
}

template < typename _Fn1 >
void _Loop_H_PPL(const PCType height, const PCType width, const PCType stride, _Fn1 &&_Func)
{
    const PCType pNum = (width + PPL_WP - 1) / PPL_WP;

    concurrency::parallel_for(PCType(0), pNum, [&](PCType p)
    {
        const PCType offset = p * PPL_WP;
        const PCType range = Min(width - offset, PPL_WP);

        for (PCType j = 0; j < height; ++j)
        {
            const PCType lower = j * stride + offset;
            const PCType upper = lower + range;

            _Func(j, lower, upper);
        }
    });
}

template < typename _Fn1 >
void _Loop_Hinv_PPL(const PCType height, const PCType width, const PCType stride, _Fn1 &&_Func)
{
    const PCType pNum = (width + PPL_WP - 1) / PPL_WP;

    concurrency::parallel_for(PCType(0), pNum, [&](PCType p)
    {
        const PCType offset = p * PPL_WP;
        const PCType range = Min(width - offset, PPL_WP);

        for (PCType j = height - 1; j >= 0; --j)
        {
            const PCType lower = j * stride + offset;
            const PCType upper = lower + range;

            _Func(j, lower, upper);
        }
    });
}

template < typename _Fn1 >
void _Loop_VH_PPL(const PCType height, const PCType width, const PCType stride, _Fn1 &&_Func)
{
    const PCType pNum = (height + PPL_HP - 1) / PPL_HP;

    concurrency::parallel_for(PCType(0), pNum, [&](PCType p)
    {
        const PCType lower = p * PPL_HP;
        const PCType upper = Min(height, lower + PPL_HP);

        for (PCType j = lower; j < upper; ++j)
        {
            PCType i = j * stride;

            for (const PCType upper = i + width; i < upper; ++i)
            {
                _Func(i);
            }
        }
    });
}


template < typename _St1, typename _Fn1 >
void _For_each_PPL(_St1 &data, _Fn1 &&_Func)
{
    LOOP_VH_PPL(data.Height(), data.Width(), data.Stride(), [&](PCType i)
    {
        _Func(data[i]);
    });
}

template < typename _St1, typename _Fn1 >
void _Transform_PPL(_St1 &data, _Fn1 &&_Func)
{
    LOOP_VH_PPL(data.Height(), data.Width(), data.Stride(), [&](PCType i)
    {
        data[i] = _Func(data[i]);
    });
}

template < typename _Dt1, typename _St1, typename _Fn1 >
void _Transform_PPL(_Dt1 &dst, const _St1 &src, _Fn1 &&_Func)
{
    const char *FunctionName = "_Transform_PPL";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_VH_PPL(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src[i]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _Fn1 >
void _Transform_PPL(_Dt1 &dst, const _St1 &src1, const _St2 &src2, _Fn1 &&_Func)
{
    const char *FunctionName = "_Transform_PPL";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height() || dst.Width() != src2.Width() || dst.Height() != src2.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1 and src2 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_VH_PPL(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src1[i], src2[i]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _Fn1 >
void _Transform_PPL(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 &&_Func)
{
    const char *FunctionName = "_Transform_PPL";
    if (dst.Width() != src1.Width() || dst.Height() != src1.Height()
        || dst.Width() != src2.Width() || dst.Height() != src2.Height()
        || dst.Width() != src3.Width() || dst.Height() != src3.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst, src1, src2 and src3 must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_VH_PPL(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src1[i], src2[i], src3[i]);
    });
}

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 >
void _Transform_PPL(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 &&_Func)
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

    LOOP_VH_PPL(dst.Height(), dst.Width(), dst.Stride(), [&](PCType i)
    {
        dst[i] = _Func(src1[i], src2[i], src3[i], src4[i]);
    });
}

template < PCType VRad, PCType HRad, typename _Dt1, typename _St1, typename _Fn1 >
void _Convolute_PPL(_Dt1 &dst, const _St1 &src, _Fn1 &&_Func)
{
    const char *FunctionName = "_Convolute_PPL";
    if (dst.Width() != src.Width() || dst.Height() != src.Height())
    {
        std::cerr << FunctionName << ": Width() and Height() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    LOOP_V_PPL(dst.Height(), [&](PCType j)
    {
        auto dstp = dst.Data() + j * dst.Stride();

        typename _St1::const_pointer srcpV[VRad * 2 + 1];
        typename _St1::value_type srcb2D[VRad * 2 + 1][HRad * 2 + 1];

        srcpV[VRad] = src.Data() + j * src.Stride();
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
#endif


// Template functions of algorithm with AMP
#ifdef ENABLE_AMP
template < typename _St1, typename _Fn1 >
void _For_each_AMP(_St1 &data, _Fn1 &&_Func)
{
    concurrency::array_view<decltype(data.value(0)), 1> datav(data.PixelCount(), datap);

    concurrency::parallel_for_each(datav.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        _Func(datav[idx]);
    });
}

template < typename _St1, typename _Fn1 >
void _Transform_AMP(_St1 &data, _Fn1 &&_Func)
{
    concurrency::array_view<decltype(data.value(0)), 1> datav(data.PixelCount(), data);

    concurrency::parallel_for_each(datav.extent, [=](concurrency::index<1> idx) restrict(amp)
    {
        datav[idx] = _Func(datav[idx]);
    });
}

template < typename _Dt1, typename _St1, typename _Fn1 >
void _Transform_AMP(_Dt1 &dst, const _St1 &src, _Fn1 &&_Func)
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

template < typename _Dt1, typename _St1, typename _St2, typename _Fn1 >
void _Transform_AMP(_Dt1 &dst, const _St1 &src1, const _St2 &src2, _Fn1 &&_Func)
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

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _Fn1 >
void _Transform_AMP(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 &&_Func)
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

template < typename _Dt1, typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 >
void _Transform_AMP(_Dt1 &dst, const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 &&_Func)
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
#endif


#endif
