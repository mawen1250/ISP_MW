#ifndef CONVERSION_HPP_
#define CONVERSION_HPP_


#include "Image_Type.h"
#include "LUT.h"
#include "Histogram.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _St1 >
typename _St1::value_type GetMin(const _St1 &src)
{
    typedef typename _St1::value_type srcType;

    srcType min = src[0];

    _For_each(src, [&](srcType x)
    {
        if (min > x) min = x;
    });

    return min;
}


template < typename _St1 >
typename _St1::value_type GetMax(const _St1 &src)
{
    typedef typename _St1::value_type srcType;

    srcType max = src[0];

    _For_each(src, [&](srcType x)
    {
        if (max < x) max = x;
    });

    return max;
}


template < typename _St1 >
void GetMinMax(const _St1 &src, typename _St1::reference min, typename _St1::reference max)
{
    typedef typename _St1::value_type srcType;

    max = min = src[0];

    _For_each(src, [&](srcType x)
    {
        if (min > x) min = x;
        if (max < x) max = x;
    });
}

template < typename _St1 >
void GetMinMax(const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, typename _St1::reference min, typename _St1::reference max)
{
    typedef typename _St1::value_type srcType;

    max = min = srcR[0];

    _For_each(srcR, [&](srcType x)
    {
        if (min > x) min = x;
        if (max < x) max = x;
    });
    _For_each(srcG, [&](srcType x)
    {
        if (min > x) min = x;
        if (max < x) max = x;
    });
    _For_each(srcB, [&](srcType x)
    {
        if (min > x) min = x;
        if (max < x) max = x;
    });
}


template < typename _St1 >
void ValidRange(const _St1 &src, typename _St1::reference min, typename _St1::reference max,
    double lower_thr = 0., double upper_thr = 0., int HistBins = 1024, bool protect = false)
{
    typedef typename _St1::value_type srcType;

    GetMinMax(src, min, max);

    if (protect && max <= min)
    {
        min = src.Floor();
        max = src.Ceil();
    }
    else if (lower_thr > 0 || upper_thr > 0)
    {
        Histogram<srcType> hist(src, min, max, HistBins);

        if (lower_thr > 0) min = hist.Min(lower_thr);
        if (upper_thr > 0) max = hist.Max(upper_thr);
    }
}

template < typename _St1 >
void ValidRange(const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, typename _St1::reference min, typename _St1::reference max,
    double lower_thr = 0., double upper_thr = 0., int HistBins = 1024, bool protect = false)
{
    typedef typename _St1::value_type srcType;

    GetMinMax(srcR, srcG, srcB, min, max);

    if (protect && max <= min)
    {
        min = srcR.Floor();
        max = srcR.Ceil();
    }
    else if (lower_thr > 0 || upper_thr > 0)
    {
        Histogram<srcType> hist(srcR, min, max, HistBins);
        hist.Add(srcG);
        hist.Add(srcB);

        if (lower_thr > 0) min = hist.Min(lower_thr);
        if (upper_thr > 0) max = hist.Max(upper_thr);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 *dst, const _St1 *src,
    const PCType height, const PCType width, const PCType dst_stride, const PCType src_stride,
    _Dt1 dFloor, _Dt1 dNeutral, _Dt1 dCeil,
    _St1 sFloor, _St1 sNeutral, _St1 sCeil,
    bool clip = false)
{
    typedef _St1 srcType;
    typedef _Dt1 dstType;

    const bool srcFloat = isFloat(srcType);
    const bool dstFloat = isFloat(dstType);
    const bool sameType = std::is_same<srcType, dstType>::value;

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

    const auto sRange = sCeil - sFloor;
    const auto dRange = dCeil - dFloor;

    if (sFloor == dFloor && sNeutral == dNeutral && sCeil == dCeil) // src and dst are of the same quantization range
    {
        if (sameType && !clip) // Copy memory if they are of the same type and clipping is not performed
        {
            if (reinterpret_cast<const void *>(dst) == reinterpret_cast<const void *>(src))
            {
                return;
            }
            else
            {
                MatCopy(dst, src, height, width, dst_stride, src_stride);
            }
        }
        else if (srcFloat && !dstFloat) // Conversion from float to integer
        {
            srcType offset = srcType(dstPCChroma ? 0.499999 : 0.5);

            if (clip)
            {
                const srcType lowerL = static_cast<srcType>(dFloor);
                const srcType upperL = static_cast<srcType>(dCeil);

                LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
                {
                    dst[i0] = static_cast<dstType>(Clip(src[i1] + offset, lowerL, upperL));
                });
            }
            else
            {
                LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
                {
                    dst[i0] = static_cast<dstType>(src[i1] + offset);
                });
            }
        }
        else // Otherwise cast type with/without clipping
        {
            if (clip)
            {
                const srcType lowerL = static_cast<srcType>(dFloor);
                const srcType upperL = static_cast<srcType>(dCeil);

                LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
                {
                    dst[i0] = static_cast<dstType>(Clip(src[i1], lowerL, upperL));
                });
            }
            else
            {
                LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
                {
                    dst[i0] = static_cast<dstType>(src[i1]);
                });
            }
        }
    }
    else // src and dst are of different quantization range
    {
        // Always apply clipping if source is PC range chroma
        if (srcPCChroma) clip = true;

        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = -static_cast<FLType>(sNeutral) * gain + dNeutral;
        if (!dstFloat) offset += FLType(dstPCChroma ? 0.499999 : 0.5);

        if (clip)
        {
            const FLType lowerL = static_cast<FLType>(dFloor);
            const FLType upperL = static_cast<FLType>(dCeil);

            LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
            {
                dst[i0] = static_cast<dstType>(Clip(static_cast<FLType>(src[i1]) * gain + offset, lowerL, upperL));
            });
        }
        else
        {
            LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
            {
                dst[i0] = static_cast<dstType>(static_cast<FLType>(src[i1]) * gain + offset);
            });
        }
    }
}


template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 &dst, const _St1 &src,
    typename _Dt1::value_type dFloor, typename _Dt1::value_type dNeutral, typename _Dt1::value_type dCeil,
    typename _St1::value_type sFloor, typename _St1::value_type sNeutral, typename _St1::value_type sCeil,
    bool clip = true)
{
    dst.ReSize(src.Width(), src.Height());

    const PCType height = dst.Height();
    const PCType width = dst.Width();
    const PCType stride = dst.Stride();

    RangeConvert(dst.data(), src.data(), dst.Height(), dst.Width(), dst.Stride(), src.Stride(),
        dFloor, dNeutral, dCeil, sFloor, sNeutral, sCeil, clip);
}

template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 &dst, const _St1 &src, bool clip = true)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool sameType = std::is_same<srcType, dstType>::value;

    if (sameType && reinterpret_cast<const void *>(dst.data()) == reinterpret_cast<const void *>(src.data()))
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
    bool clip = true)
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
    const PCType dst_stride = dstR.Stride();
    const PCType src_stride = srcR.Stride();

    const auto sRange = sCeil - sFloor;
    const auto dRange = dCeil - dFloor;

    FLType gain = static_cast<FLType>(dRange) / sRange;
    FLType offset = -static_cast<FLType>(sNeutral) * gain + dNeutral;
    if (!dstFloat) offset += FLType(0.5);

    // Always apply clipping if source is PC range chroma
    if (srcPCChroma) clip = true;

    if (clip)
    {
        const FLType lowerL = static_cast<FLType>(dFloor);
        const FLType upperL = static_cast<FLType>(dCeil);

        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            dstR[i0] = static_cast<dstType>(Clip(static_cast<FLType>(srcR[i1]) * gain + offset, lowerL, upperL));
            dstG[i0] = static_cast<dstType>(Clip(static_cast<FLType>(srcG[i1]) * gain + offset, lowerL, upperL));
            dstB[i0] = static_cast<dstType>(Clip(static_cast<FLType>(srcB[i1]) * gain + offset, lowerL, upperL));
        });
    }
    else
    {
        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            dstR[i0] = static_cast<dstType>(static_cast<FLType>(srcR[i1]) * gain + offset);
            dstG[i0] = static_cast<dstType>(static_cast<FLType>(srcG[i1]) * gain + offset);
            dstB[i0] = static_cast<dstType>(static_cast<FLType>(srcB[i1]) * gain + offset);
        });
    }
}

template < typename _Dt1, typename _St1 >
void RangeConvert(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, bool clip = true)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool sameType = std::is_same<srcType, dstType>::value;

    if (sameType && reinterpret_cast<const void *>(dstR.data()) == reinterpret_cast<const void *>(srcR.data())
        && reinterpret_cast<const void *>(dstG.data()) == reinterpret_cast<const void *>(srcG.data())
        && reinterpret_cast<const void *>(dstB.data()) == reinterpret_cast<const void *>(srcB.data()))
    {
        return;
    }

    RangeConvert(dstR, dstG, dstB, srcR, srcG, srcB, dstR.Floor(), dstR.Neutral(), dstR.Ceil(), srcR.Floor(), srcR.Neutral(), srcR.Ceil(), clip);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _Dt1, typename _St1 >
void ConvertToY(_Dt1 *dst, const _St1 *srcR, const _St1 *srcG, const _St1 *srcB,
    const PCType height, const PCType width, const PCType dst_stride, const PCType src_stride,
    _Dt1 dFloor, _Dt1 dCeil, _St1 sFloor, _St1 sCeil,
    ColorMatrix matrix = ColorMatrix::OPP, bool clip = false)
{
    typedef _St1 srcType;
    typedef _Dt1 dstType;

    const bool dstFloat = isFloat(dstType);

    const auto sRange = sCeil - sFloor;
    const auto dRange = dCeil - dFloor;

    const FLType lowerL = static_cast<FLType>(dFloor);
    const FLType upperL = static_cast<FLType>(dCeil);

    if (matrix == ColorMatrix::GBR)
    {
        RangeConvert(dst, srcG, height, width, dst_stride, src_stride, dFloor, dFloor, dCeil, sFloor, sFloor, sCeil, false);
    }
    else if (matrix == ColorMatrix::OPP)
    {
        FLType gain = static_cast<FLType>(dRange) / (sRange * FLType(3));
        FLType offset = -static_cast<FLType>(sFloor)* FLType(3) * gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            FLType temp = (static_cast<FLType>(srcR[i1])
                + static_cast<FLType>(srcG[i1])
                + static_cast<FLType>(srcB[i1]))
                * gain + offset;
            dst[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);
        });
    }
    else if (matrix == ColorMatrix::Minimum)
    {
        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = -static_cast<FLType>(sFloor)* gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            FLType temp = static_cast<FLType>(
                ::Min(srcR[i1], ::Min(srcG[i1], srcB[i1]))
                ) * gain + offset;
            dst[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);
        });
    }
    else if (matrix == ColorMatrix::Maximum)
    {
        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = -static_cast<FLType>(sFloor)* gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            FLType temp = static_cast<FLType>(
                ::Max(srcR[i1], ::Max(srcG[i1], srcB[i1]))
                ) * gain + offset;
            dst[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);
        });
    }
    else
    {
        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = -static_cast<FLType>(sFloor)* gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        FLType Kr, Kg, Kb;
        ColorMatrix_Parameter(matrix, Kr, Kg, Kb);

        Kr *= gain;
        Kg *= gain;
        Kb *= gain;

        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            FLType temp = Kr * static_cast<FLType>(srcR[i1])
                + Kg * static_cast<FLType>(srcG[i1])
                + Kb * static_cast<FLType>(srcB[i1])
                + offset;
            dst[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);
        });
    }
}


template < typename _Dt1, typename _St1 >
void ConvertToY(_Dt1 &dst, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, ColorMatrix matrix = ColorMatrix::OPP, bool clip = true)
{
    if (dst.isChroma())
    {
        dst.ReSetChroma(false);
    }

    dst.ReSize(srcR.Width(), srcR.Height());

    ConvertToY(dst.data(), srcR.data(), srcG.data(), srcB.data(),
        dst.Height(), dst.Width(), dst.Stride(), srcR.Stride(),
        dst.Floor(), dst.Ceil(), srcR.Floor(), srcR.Ceil(), matrix, clip);
}

template < typename _Dt1, typename _St1 > inline
void ConvertToY(_Dt1 &dst, const _St1 &src, ColorMatrix matrix = ColorMatrix::OPP, bool clip = true)
{
    if (src.isRGB())
    {
        ConvertToY(dst, src.R(), src.G(), src.B(), matrix, clip);
    }
    else if (src.isYUV())
    {
        RangeConvert(dst, src.Y(), clip);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _Dt1, typename _St1 >
void MatrixConvert_RGB2YUV(_Dt1 *dstY, _Dt1 *dstU, _Dt1 *dstV,
    const _St1 *srcR, const _St1 *srcG, const _St1 *srcB,
    const PCType height, const PCType width, const PCType dst_stride, const PCType src_stride,
    _Dt1 dFloorY, _Dt1 dCeilY, _Dt1 dFloorC, _Dt1 dNeutralC, _Dt1 dCeilC, _St1 sFloor, _St1 sCeil,
    ColorMatrix matrix = ColorMatrix::OPP, bool clip = false)
{
    typedef _St1 srcType;
    typedef _Dt1 dstType;

    const bool dstFloat = isFloat(dstType);

    const bool dstPCChroma = isPCChroma(dFloorC, dNeutralC, dCeilC);

    const auto sRange = sCeil - sFloor;
    const auto dRangeY = dCeilY - dFloorY;
    const auto dRangeC = dCeilC - dFloorC;

    const FLType lowerLY = static_cast<FLType>(dFloorY);
    const FLType upperLY = static_cast<FLType>(dCeilY);
    const FLType lowerLC = static_cast<FLType>(dFloorC);
    const FLType upperLC = static_cast<FLType>(dCeilC);

    if (matrix == ColorMatrix::GBR)
    {
        RangeConvert(dstY, srcG, height, width, dst_stride, src_stride, dFloorY, dFloorY, dCeilY, sFloor, sFloor, sCeil, clip);
        RangeConvert(dstU, srcB, height, width, dst_stride, src_stride, dFloorY, dFloorY, dCeilY, sFloor, sFloor, sCeil, clip);
        RangeConvert(dstV, srcR, height, width, dst_stride, src_stride, dFloorY, dFloorY, dCeilY, sFloor, sFloor, sCeil, clip);
    }
    else if (matrix == ColorMatrix::OPP)
    {
        FLType gainY = static_cast<FLType>(dRangeY) / (sRange * FLType(3));
        FLType offsetY = -static_cast<FLType>(sFloor)* FLType(3) * gainY + dFloorY;
        if (!dstFloat) offsetY += FLType(0.5);
        FLType gainU = static_cast<FLType>(dRangeC) / (sRange * FLType(2));
        FLType gainV = static_cast<FLType>(dRangeC) / (sRange * FLType(4));
        FLType offsetC = static_cast<FLType>(dNeutralC);
        if (!dstFloat) offsetC += FLType(dstPCChroma ? 0.499999 : 0.5);

        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            FLType temp;

            temp = (static_cast<FLType>(srcR[i1])
                + static_cast<FLType>(srcG[i1])
                + static_cast<FLType>(srcB[i1]))
                * gainY + offsetY;
            dstY[i0] = static_cast<dstType>(clip ? Clip(temp, lowerLY, upperLY) : temp);

            temp = (static_cast<FLType>(srcR[i1])
                - static_cast<FLType>(srcB[i1]))
                * gainU + offsetC;
            dstU[i0] = static_cast<dstType>(clip ? Clip(temp, lowerLC, upperLC) : temp);

            temp = (static_cast<FLType>(srcR[i1])
                - static_cast<FLType>(srcG[i1]) * FLType(2)
                + static_cast<FLType>(srcB[i1]))
                * gainV + offsetC;
            dstV[i0] = static_cast<dstType>(clip ? Clip(temp, lowerLC, upperLC) : temp);
        });
    }
    else if (matrix == ColorMatrix::Minimum || matrix == ColorMatrix::Maximum)
    {
        DEBUG_FAIL("MatrixConvert_RGB2YUV: ColorMatrix::Minimum or ColorMatrix::Maximum is invalid!");
    }
    else
    {
        FLType gainY = static_cast<FLType>(dRangeY) / sRange;
        FLType offsetY = -static_cast<FLType>(sFloor)* gainY + dFloorY;
        if (!dstFloat) offsetY += FLType(0.5);
        FLType gainC = static_cast<FLType>(dRangeC) / sRange;
        FLType offsetC = static_cast<FLType>(dNeutralC);
        if (!dstFloat) offsetC += FLType(dstPCChroma ? 0.499999 : 0.5);

        FLType Yr, Yg, Yb, Ur, Ug, Ub, Vr, Vg, Vb;
        ColorMatrix_RGB2YUV_Parameter(matrix, Yr, Yg, Yb, Ur, Ug, Ub, Vr, Vg, Vb);

        Yr *= gainY;
        Yg *= gainY;
        Yb *= gainY;
        Ur *= gainC;
        Ug *= gainC;
        Ub *= gainC;
        Vr *= gainC;
        Vg *= gainC;
        Vb *= gainC;

        LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
        {
            FLType temp;

            temp = Yr * static_cast<FLType>(srcR[i1])
                + Yg * static_cast<FLType>(srcG[i1])
                + Yb * static_cast<FLType>(srcB[i1])
                + offsetY;
            dstY[i0] = static_cast<dstType>(clip ? Clip(temp, lowerLY, upperLY) : temp);

            temp = Ur * static_cast<FLType>(srcR[i1])
                + Ug * static_cast<FLType>(srcG[i1])
                + Ub * static_cast<FLType>(srcB[i1])
                + offsetC;
            dstU[i0] = static_cast<dstType>(clip ? Clip(temp, lowerLC, upperLC) : temp);

            temp = Vr * static_cast<FLType>(srcR[i1])
                + Vg * static_cast<FLType>(srcG[i1])
                + Vb * static_cast<FLType>(srcB[i1])
                + offsetC;
            dstV[i0] = static_cast<dstType>(clip ? Clip(temp, lowerLC, upperLC) : temp);
        });
    }
}

template < typename _Dt1, typename _St1 >
void MatrixConvert_RGB2YUV(_Dt1 &dstY, _Dt1 &dstU, _Dt1 &dstV,
    const _St1 &srcR, const _St1 &srcG, const _St1 &srcB,
    ColorMatrix matrix = ColorMatrix::OPP, bool clip = true)
{
    dstY.ReSize(srcR.Width(), srcR.Height());
    dstU.ReSize(srcR.Width(), srcR.Height());
    dstV.ReSize(srcR.Width(), srcR.Height());

    if (matrix == ColorMatrix::GBR)
    {
        dstY.ReSetChroma(false);
        dstU.ReSetChroma(false);
        dstV.ReSetChroma(false);

        RangeConvert(dstY, dstU, dstV, srcG, srcB, srcR, true);
    }
    else
    {
        dstY.ReSetChroma(false);
        dstU.ReSetChroma(true);
        dstV.ReSetChroma(true);

        MatrixConvert_RGB2YUV(dstY.data(), dstU.data(), dstV.data(), srcR.data(), srcG.data(), srcB.data(),
            dstY.Height(), dstY.Width(), dstY.Stride(), srcR.Stride(),
            dstY.Floor(), dstY.Ceil(), dstU.Floor(), dstU.Neutral(), dstU.Ceil(), srcR.Floor(), srcR.Ceil(),
            matrix, clip);
    }
}


template < typename _Dt1, typename _St1 >
void MatrixConvert_YUV2RGB(_Dt1 *dstR, _Dt1 *dstG, _Dt1 *dstB,
    const _St1 *srcY, const _St1 *srcU, const _St1 *srcV,
    const PCType height, const PCType width, const PCType dst_stride, const PCType src_stride,
    _Dt1 dFloor, _Dt1 dCeil, _St1 sFloorY, _St1 sCeilY, _St1 sFloorC, _St1 sNeutralC, _St1 sCeilC,
    ColorMatrix matrix = ColorMatrix::OPP, bool clip = false)
{
    typedef _St1 srcType;
    typedef _Dt1 dstType;

    const bool dstFloat = isFloat(dstType);

    const auto sRangeY = sCeilY - sFloorY;
    const auto sRangeC = sCeilC - sFloorC;
    const auto dRange = dCeil - dFloor;

    const FLType lowerL = static_cast<FLType>(dFloor);
    const FLType upperL = static_cast<FLType>(dCeil);

    if (matrix == ColorMatrix::GBR)
    {
        RangeConvert(dstG, srcY, height, width, dst_stride, src_stride, dFloor, dFloor, dCeil, sFloorY, sFloorY, sCeilY, clip);
        RangeConvert(dstB, srcU, height, width, dst_stride, src_stride, dFloor, dFloor, dCeil, sFloorY, sFloorY, sCeilY, clip);
        RangeConvert(dstR, srcV, height, width, dst_stride, src_stride, dFloor, dFloor, dCeil, sFloorY, sFloorY, sCeilY, clip);
    }
    else if (matrix == ColorMatrix::Minimum || matrix == ColorMatrix::Maximum)
    {
        DEBUG_FAIL("MatrixConvert_YUV2RGB: ColorMatrix::Minimum or ColorMatrix::Maximum is invalid!");
    }
    else
    {
        FLType gainY = static_cast<FLType>(dRange) / sRangeY;
        FLType gainC = static_cast<FLType>(dRange) / sRangeC;

        FLType Ry, Ru, Rv, Gy, Gu, Gv, By, Bu, Bv;
        ColorMatrix_YUV2RGB_Parameter(matrix, Ry, Ru, Rv, Gy, Gu, Gv, By, Bu, Bv);

        Ry *= gainY;
        Ru *= gainC;
        Rv *= gainC;
        Gy *= gainY;
        Gu *= gainC;
        Gv *= gainC;
        By *= gainY;
        Bu *= gainC;
        Bv *= gainC;

        FLType offsetR = -static_cast<FLType>(sFloorY)* Ry - sNeutralC * (Ru + Rv) + dFloor;
        if (!dstFloat) offsetR += FLType(0.5);
        FLType offsetG = -static_cast<FLType>(sFloorY)* Gy - sNeutralC * (Gu + Gv) + dFloor;
        if (!dstFloat) offsetG += FLType(0.5);
        FLType offsetB = -static_cast<FLType>(sFloorY)* By - sNeutralC * (Bu + Bv) + dFloor;
        if (!dstFloat) offsetB += FLType(0.5);

        if (matrix == ColorMatrix::YCgCo)
        {
            LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
            {
                FLType temp;

                temp = Ry * static_cast<FLType>(srcY[i1])
                    + Ru * static_cast<FLType>(srcU[i1])
                    + Rv * static_cast<FLType>(srcV[i1])
                    + offsetR;
                dstR[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);

                temp = Gy * static_cast<FLType>(srcY[i1])
                    + Gu * static_cast<FLType>(srcU[i1])
                    + offsetG;
                dstG[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);

                temp = By * static_cast<FLType>(srcY[i1])
                    + Bu * static_cast<FLType>(srcU[i1])
                    + Bv * static_cast<FLType>(srcV[i1])
                    + offsetB;
                dstB[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);
            });
        }
        else if (matrix == ColorMatrix::OPP)
        {
            LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
            {
                FLType temp;

                temp = Ry * static_cast<FLType>(srcY[i1])
                    + Ru * static_cast<FLType>(srcU[i1])
                    + Rv * static_cast<FLType>(srcV[i1])
                    + offsetR;
                dstR[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);

                temp = Gy * static_cast<FLType>(srcY[i1])
                    + Gv * static_cast<FLType>(srcV[i1])
                    + offsetG;
                dstG[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);

                temp = By * static_cast<FLType>(srcY[i1])
                    + Bu * static_cast<FLType>(srcU[i1])
                    + Bv * static_cast<FLType>(srcV[i1])
                    + offsetB;
                dstB[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);
            });
        }
        else
        {
            LOOP_VH_PPL(height, width, dst_stride, src_stride, [&](PCType i0, PCType i1)
            {
                FLType temp;

                temp = Ry * static_cast<FLType>(srcY[i1])
                    + Rv * static_cast<FLType>(srcV[i1])
                    + offsetR;
                dstR[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);

                temp = Gy * static_cast<FLType>(srcY[i1])
                    + Gu * static_cast<FLType>(srcU[i1])
                    + Gv * static_cast<FLType>(srcV[i1])
                    + offsetG;
                dstG[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);

                temp = By * static_cast<FLType>(srcY[i1])
                    + Bu * static_cast<FLType>(srcU[i1])
                    + offsetB;
                dstB[i0] = static_cast<dstType>(clip ? Clip(temp, lowerL, upperL) : temp);
            });
        }
    }
}

template < typename _Dt1, typename _St1 >
void MatrixConvert_YUV2RGB(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB,
    const _St1 &srcY, const _St1 &srcU, const _St1 &srcV,
    ColorMatrix matrix = ColorMatrix::OPP, bool clip = true)
{
    dstR.ReSize(srcY.Width(), srcY.Height());
    dstG.ReSize(srcY.Width(), srcY.Height());
    dstB.ReSize(srcY.Width(), srcY.Height());

    dstR.ReSetChroma(false);
    dstG.ReSetChroma(false);
    dstB.ReSetChroma(false);

    if (matrix == ColorMatrix::GBR)
    {
        RangeConvert(dstG, dstB, dstR, srcY, srcU, srcV, clip);
    }
    else
    {
        MatrixConvert_YUV2RGB(dstR.data(), dstG.data(), dstB.data(), srcY.data(), srcU.data(), srcV.data(),
            dstR.Height(), dstR.Width(), dstR.Stride(), srcY.Stride(),
            dstR.Floor(), dstR.Ceil(), srcY.Floor(), srcY.Ceil(), srcU.Floor(), srcU.Neutral(), srcU.Ceil(),
            matrix, clip);
    }
}


template < typename _Dt1, typename _St1 > inline
void MatrixConvert(_Dt1 &dst, const _St1 &src, ColorMatrix _ColorMatrix = ColorMatrix::Unspecified, bool clip = true)
{
    if (_ColorMatrix == ColorMatrix::Unspecified)
    {
        _ColorMatrix = dst.GetColorMatrix();
    }
    else
    {
        dst.SetColorMatrix(_ColorMatrix);
    }

    if (src.isRGB() && dst.isYUV())
    {
        MatrixConvert_RGB2YUV(dst.Y(), dst.U(), dst.V(), src.R(), src.G(), src.B(), _ColorMatrix, clip);
    }
    else if (src.isYUV() && dst.isRGB())
    {
        MatrixConvert_YUV2RGB(dst.Y(), dst.U(), dst.V(), src.R(), src.G(), src.B(), _ColorMatrix, clip);
    }
    else
    {
        dst = src;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _Dt1 >
void TransferConvert(_Dt1 &dst, const Plane &src, TransferChar dstTransferChar, TransferChar srcTransferChar)
{
    typedef Plane _St1;
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool dstFloat = isFloat(dstType);

    // Transfer Characteristics process
    dst.SetTransferChar(dstTransferChar);
    TransferChar_Conv<FLType> ConvFilter(dstTransferChar, srcTransferChar);

    // Skip Transfer conversion if not needed
    if (ConvFilter.Type() == TransferChar_Conv<FLType>::ConvType::none)
    {
        RangeConvert(dst, src, false);
        return;
    }

    // Lambda for conversion to/from float point
    FLType gain, offset;

    srcType sFloor = src.Floor();
    gain = FLType(1) / src.ValueRange();

    auto ToFloat = [=](srcType i)
    {
        return static_cast<FLType>(i - sFloor) * gain;
    };

    gain = static_cast<FLType>(dst.ValueRange());
    offset = static_cast<FLType>(dst.Floor());
    if (!dstFloat) offset += FLType(0.5);

    auto FloatTo = [=](FLType i)
    {
        return static_cast<dstType>(i * gain + offset);
    };

    // Generate conversion LUT
    LUT<dstType> _LUT(src);

    if (dstFloat && dst.Floor() == 0 && dst.ValueRange() == 1)
    {
        _LUT.Set(src, [&](srcType i)
        {
            return static_cast<dstType>(ConvFilter(ToFloat(i)));
        });
    }
    else
    {
        _LUT.Set(src, [&](srcType i)
        {
            return FloatTo(ConvFilter(ToFloat(i)));
        });
    }

    // Conversion
    dst.ReSize(src.Width(), src.Height());
    _LUT.Lookup(dst, src);
}

template < typename _Dt1 >
void TransferConvert(_Dt1 &dst, const Plane_FL &src, TransferChar dstTransferChar, TransferChar srcTransferChar)
{
    typedef Plane_FL _St1;
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool dstFloat = isFloat(dstType);

    // Transfer Characteristics process
    dst.SetTransferChar(dstTransferChar);
    TransferChar_Conv<FLType> ConvFilter(dstTransferChar, srcTransferChar);

    // Skip Transfer conversion if not needed
    if (ConvFilter.Type() == TransferChar_Conv<FLType>::ConvType::none)
    {
        RangeConvert(dst, src, false);
        return;
    }

    // Lambda for conversion to/from float point
    FLType gain, offset;

    gain = FLType(1) / src.ValueRange();
    offset = -src.Floor() * gain;

    auto ToFloat = [=](srcType i)
    {
        return static_cast<FLType>(i * gain + offset);
    };

    gain = dst.ValueRange();
    offset = dst.Floor();
    if (!dstFloat) offset += FLType(0.5);

    auto FloatTo = [=](FLType i)
    {
        return static_cast<dstType>(i * gain + offset);
    };

    // Conversion
    dst.ReSize(src.Width(), src.Height());

    if (src.Floor() == 0 && src.ValueRange() == 1)
    {
        if (dstFloat && dst.Floor() == 0 && dst.ValueRange() == 1)
        {
            TRANSFORM_PPL(dst, src, [&](srcType i)
            {
                return static_cast<dstType>(ConvFilter(static_cast<FLType>(i)));
            });
        }
        else
        {
            TRANSFORM_PPL(dst, src, [&](srcType i)
            {
                return FloatTo(ConvFilter(static_cast<FLType>(i)));
            });
        }
    }
    else
    {
        if (dstFloat && dst.Floor() == 0 && dst.ValueRange() == 1)
        {
            TRANSFORM_PPL(dst, src, [&](srcType i)
            {
                return static_cast<dstType>(ConvFilter(ToFloat(i)));
            });
        }
        else
        {
            TRANSFORM_PPL(dst, src, [&](srcType i)
            {
                return FloatTo(ConvFilter(ToFloat(i)));
            });
        }
    }
}

template < typename _Dt1, typename _St1 > inline
void TransferConvert(_Dt1 &dst, const _St1 &src, TransferChar dstTransferChar)
{
    TransferConvert(dst, src, dstTransferChar, src.GetTransferChar());
}

template < typename _Dt1, typename _St1 > inline
void TransferConvert(_Dt1 &dst, const _St1 &src)
{
    TransferConvert(dst, src, dst.GetTransferChar(), src.GetTransferChar());
}

template < typename _Dt1 >
void TransferConvert(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const Plane &srcR, const Plane &srcG, const Plane &srcB, TransferChar dstTransferChar, TransferChar srcTransferChar)
{
    typedef Plane _St1;
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool dstFloat = isFloat(dstType);

    // Transfer Characteristics process
    dstR.SetTransferChar(dstTransferChar);
    dstG.SetTransferChar(dstTransferChar);
    dstB.SetTransferChar(dstTransferChar);
    TransferChar_Conv<FLType> ConvFilter(dstTransferChar, srcTransferChar);

    // Skip Transfer conversion if not needed
    if (ConvFilter.Type() == TransferChar_Conv<FLType>::ConvType::none)
    {
        RangeConvert(dstR, dstG, dstB, srcR, srcG, srcB, false);
        return;
    }

    // Lambda for conversion to/from float point
    FLType gain, offset;

    srcType sFloor = srcR.Floor();
    gain = FLType(1) / srcR.ValueRange();

    auto ToFloat = [=](srcType i)
    {
        return static_cast<FLType>(i - sFloor) * gain;
    };

    gain = static_cast<FLType>(dstR.ValueRange());
    offset = static_cast<FLType>(dstR.Floor());
    if (!dstFloat) offset += FLType(0.5);

    auto FloatTo = [=](FLType i)
    {
        return static_cast<dstType>(i * gain + offset);
    };

    // Generate conversion LUT
    LUT<dstType> _LUT(srcR);

    if (dstFloat && dstR.Floor() == 0 && dstR.ValueRange() == 1)
    {
        _LUT.Set(srcR, [&](srcType i)
        {
            return static_cast<dstType>(ConvFilter(ToFloat(i)));
        });
    }
    else
    {
        _LUT.Set(srcR, [&](srcType i)
        {
            return FloatTo(ConvFilter(ToFloat(i)));
        });
    }

    // Conversion
    dstR.ReSize(srcR.Width(), srcR.Height());
    dstG.ReSize(srcG.Width(), srcG.Height());
    dstB.ReSize(srcB.Width(), srcB.Height());
    _LUT.Lookup(dstR, srcR);
    _LUT.Lookup(dstG, srcG);
    _LUT.Lookup(dstB, srcB);
}

template < typename _Dt1 >
void TransferConvert(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const Plane_FL &srcR, const Plane_FL &srcG, const Plane_FL &srcB, TransferChar dstTransferChar, TransferChar srcTransferChar)
{
    typedef Plane_FL _St1;
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool dstFloat = isFloat(dstType);

    // Transfer Characteristics process
    dstR.SetTransferChar(dstTransferChar);
    dstG.SetTransferChar(dstTransferChar);
    dstB.SetTransferChar(dstTransferChar);
    TransferChar_Conv<FLType> ConvFilter(dstTransferChar, srcTransferChar);

    // Skip Transfer conversion if not needed
    if (ConvFilter.Type() == TransferChar_Conv<FLType>::ConvType::none)
    {
        RangeConvert(dstR, dstG, dstB, srcR, srcG, srcB, false);
        return;
    }

    // Lambda for conversion to/from float point
    FLType gain, offset;

    gain = FLType(1) / srcR.ValueRange();
    offset = -srcR.Floor() * gain;

    auto ToFloat = [=](srcType i)
    {
        return static_cast<FLType>(i * gain + offset);
    };

    gain = dstR.ValueRange();
    offset = dstR.Floor();
    if (!dstFloat) offset += FLType(0.5);

    auto FloatTo = [=](FLType i)
    {
        return static_cast<dstType>(i * gain + offset);
    };

    // Conversion
    dstR.ReSize(srcR.Width(), srcR.Height());
    dstG.ReSize(srcG.Width(), srcG.Height());
    dstB.ReSize(srcB.Width(), srcB.Height());

    const PCType height = dstR.Height();
    const PCType width = dstR.Width();
    const PCType stride = dstR.Stride();

    if (srcR.Floor() == 0 && srcR.ValueRange() == 1)
    {
        if (dstFloat && dstR.Floor() == 0 && dstR.ValueRange() == 1)
        {
            LOOP_VH_PPL(height, width, stride, [&](PCType i)
            {
                dstR[i] = static_cast<dstType>(ConvFilter(static_cast<FLType>(srcR[i])));
                dstG[i] = static_cast<dstType>(ConvFilter(static_cast<FLType>(srcG[i])));
                dstB[i] = static_cast<dstType>(ConvFilter(static_cast<FLType>(srcB[i])));
            });
        }
        else
        {
            LOOP_VH_PPL(height, width, stride, [&](PCType i)
            {
                dstR[i] = FloatTo(ConvFilter(static_cast<FLType>(srcR[i])));
                dstG[i] = FloatTo(ConvFilter(static_cast<FLType>(srcG[i])));
                dstB[i] = FloatTo(ConvFilter(static_cast<FLType>(srcB[i])));
            });
        }
    }
    else
    {
        if (dstFloat && dstR.Floor() == 0 && dstR.ValueRange() == 1)
        {
            LOOP_VH_PPL(height, width, stride, [&](PCType i)
            {
                dstR[i] = static_cast<dstType>(ConvFilter(ToFloat(srcR[i])));
                dstG[i] = static_cast<dstType>(ConvFilter(ToFloat(srcG[i])));
                dstB[i] = static_cast<dstType>(ConvFilter(ToFloat(srcB[i])));
            });
        }
        else
        {
            LOOP_VH_PPL(height, width, stride, [&](PCType i)
            {
                dstR[i] = FloatTo(ConvFilter(ToFloat(srcR[i])));
                dstG[i] = FloatTo(ConvFilter(ToFloat(srcG[i])));
                dstB[i] = FloatTo(ConvFilter(ToFloat(srcB[i])));
            });
        }
    }
}

template < typename _Dt1, typename _St1 > inline
void TransferConvert(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, TransferChar dstTransferChar)
{
    TransferConvert(dstR, dstG, dstB, srcR, srcG, srcB, dstTransferChar, srcR.GetTransferChar());
}

template < typename _Dt1, typename _St1 > inline
void TransferConvert(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB)
{
    TransferConvert(dstR, dstG, dstB, srcR, srcG, srcB, dstR.GetTransferChar(), srcR.GetTransferChar());
}

inline void TransferConvert(Frame &dst, const Frame &src, TransferChar dstTransferChar, TransferChar srcTransferChar)
{
    if (dst.GetPixelType() != src.GetPixelType())
    {
        DEBUG_FAIL("TransferConvert: \"PixelType\" of dst and src should be the same!");
    }

    dst.SetTransferChar(dstTransferChar);

    if (dst.isRGB())
    {
        if (dst.GetPixelType() == PixelType::RGB)
        {
            TransferConvert(dst.R(), dst.G(), dst.B(), src.R(), src.G(), src.B(), dstTransferChar, srcTransferChar);
        }
        else
        {
            if (dst.GetPixelType() == PixelType::R)
            {
                TransferConvert(dst.R(), src.R(), dstTransferChar, srcTransferChar);
            }

            if (dst.GetPixelType() == PixelType::G)
            {
                TransferConvert(dst.G(), src.G(), dstTransferChar, srcTransferChar);
            }

            if (dst.GetPixelType() == PixelType::B)
            {
                TransferConvert(dst.B(), src.B(), dstTransferChar, srcTransferChar);
            }
        }
    }
    else if (dst.isYUV())
    {
        if (dst.GetPixelType() != PixelType::U && dst.GetPixelType() != PixelType::V)
        {
            TransferConvert(dst.Y(), src.Y(), dstTransferChar, srcTransferChar);
        }

        if (dst.GetPixelType() != PixelType::Y)
        {
            if (dst.GetPixelType() != PixelType::V)
            {
                dst.U() = src.U();
            }

            if (dst.GetPixelType() != PixelType::U)
            {
                dst.V() = src.V();
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _Dt1, typename _St1 >
void SimplestColorBalance(_Dt1 &dst, const _St1 &src, double lower_thr = 0., double upper_thr = 0., int HistBins = 1024)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    srcType min, max;
    ValidRange(src, min, max, lower_thr, upper_thr, HistBins, true);
    RangeConvert(dst, src, dst.Floor(), dst.Neutral(), dst.Ceil(), min, min, max, lower_thr > 0 || upper_thr > 0);
}

template < typename _Dt1, typename _St1 >
void SimplestColorBalance(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB,
    double lower_thr = 0., double upper_thr = 0., int HistBins = 1024)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    srcType min, max;
    ValidRange(srcR, srcG, srcB, min, max, lower_thr, upper_thr, HistBins, true);
    RangeConvert(dstR, dstG, dstB, srcR, srcG, srcB,
        dstR.Floor(), dstR.Neutral(), dstR.Ceil(), min, min, max,
        lower_thr > 0 || upper_thr > 0);
}

template < typename _St1 > inline
void SimplestColorBalance(Frame &dst, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB,
double lower_thr = 0., double upper_thr = 0., int HistBins = 1024)
{
    SimplestColorBalance(dst.R(), dst.G(), dst.B(), srcR, srcG, srcB, lower_thr, upper_thr, HistBins);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif
