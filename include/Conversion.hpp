#ifndef CONVERSION_HPP_
#define CONVERSION_HPP_


#include "Image_Type.h"
#include "LUT.h"
#include "Histogram.h"


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


template < typename _Dt1, typename _St1 >
void ConvertToY(_Dt1 &dst, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, ColorMatrix dstColorMatrix = ColorMatrix::OPP)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool dstFloat = isFloat(dstType);

    if (dst.isChroma())
    {
        dst.ReSetChroma(false);
    }

    dst.ReSize(srcR.Width(), srcR.Height());

    const PCType height = dst.Height();
    const PCType width = dst.Width();
    const PCType stride = dst.Stride();

    const auto sFloor = srcR.Floor();
    const auto sRange = srcR.ValueRange();
    const auto dFloor = dst.Floor();
    const auto dRange = dst.ValueRange();

    if (dstColorMatrix == ColorMatrix::OPP)
    {
        FLType gain = static_cast<FLType>(dRange) / (sRange * FLType(3));
        FLType offset = -sFloor * 3 * gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dst[i] = static_cast<dstType>(static_cast<FLType>(srcR[i] + srcG[i] + srcB[i]) * gain + offset);
        });
    }
    else if (dstColorMatrix == ColorMatrix::Minimum)
    {
        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = -sFloor * gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dst[i] = static_cast<dstType>(static_cast<FLType>(::Min(srcR[i], ::Min(srcG[i], srcB[i]))) * gain + offset);
        });
    }
    else if (dstColorMatrix == ColorMatrix::Maximum)
    {
        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = -sFloor * gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dst[i] = static_cast<dstType>(static_cast<FLType>(::Max(srcR[i], ::Max(srcG[i], srcB[i]))) * gain + offset);
        });
    }
    else if (dstColorMatrix == ColorMatrix::GBR)
    {
        RangeConvert(dst, srcG);
    }
    else
    {
        FLType gain = static_cast<FLType>(dRange) / sRange;
        FLType offset = -sFloor * gain + dFloor;
        if (!dstFloat) offset += FLType(0.5);

        FLType Kr, Kg, Kb;
        ColorMatrix_Parameter(dstColorMatrix, Kr, Kg, Kb);

        Kr *= gain;
        Kg *= gain;
        Kb *= gain;

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dst[i] = static_cast<dstType>(Kr * static_cast<FLType>(srcR[i]) + Kg * static_cast<FLType>(srcG[i])
                + Kb * static_cast<FLType>(srcB[i]) + offset);
        });
    }
}

template < typename _Dt1, typename _St1 > inline
void ConvertToY(_Dt1 &dst, const _St1 &src, ColorMatrix dstColorMatrix = ColorMatrix::OPP)
{
    if (src.isRGB())
    {
        ConvertToY(dst, src.R(), src.G(), src.B(), dstColorMatrix);
    }
    else if (src.isYUV())
    {
        RangeConvert(dst, src.Y());
    }
}


template < typename _Dt1, typename _St1 >
void MatrixConvert_RGB2YUV(_Dt1 &dstY, _Dt1 &dstU, _Dt1 &dstV,
    const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, ColorMatrix dstColorMatrix = ColorMatrix::OPP)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool dstFloat = isFloat(dstType);

    dstY.ReSize(srcR.Width(), srcR.Height());
    dstU.ReSize(srcR.Width(), srcR.Height());
    dstV.ReSize(srcR.Width(), srcR.Height());
    dstY.ReSetChroma(false);
    dstU.ReSetChroma(true);
    dstV.ReSetChroma(true);

    const PCType height = dstY.Height();
    const PCType width = dstY.Width();
    const PCType stride = dstY.Stride();

    const FLType sFloor = static_cast<FLType>(srcR.Floor());
    const FLType sRange = static_cast<FLType>(srcR.ValueRange());
    const FLType dFloorY = static_cast<FLType>(dstY.Floor());
    const FLType dRangeY = static_cast<FLType>(dstY.ValueRange());
    const FLType dNeutralC = static_cast<FLType>(dstU.Neutral());
    const FLType dRangeC = static_cast<FLType>(dstU.ValueRange());

    if (dstColorMatrix == ColorMatrix::OPP)
    {
        FLType gainY = dRangeY / (sRange * FLType(3));
        FLType offsetY = -sFloor * 3 * gainY + dFloorY;
        if (!dstFloat) offsetY += FLType(0.5);
        FLType gainU = dRangeC / (sRange * FLType(2));
        FLType gainV = dRangeC / (sRange * FLType(4));
        FLType offsetC = dNeutralC;
        if (!dstFloat) offsetC += FLType(dstU.isPCChroma() ? 0.499999 : 0.5);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dstY[i] = static_cast<dstType>(
                static_cast<FLType>(srcR[i] + srcG[i] + srcB[i]) * gainY + offsetY
                );
            dstU[i] = static_cast<dstType>(
                static_cast<FLType>(srcR[i] - srcB[i]) * gainU + offsetC
                );
            dstV[i] = static_cast<dstType>(
                static_cast<FLType>(srcR[i] - srcG[i] * 2 + srcB[i]) * gainV + offsetC
                );
        });
    }
    else if (dstColorMatrix == ColorMatrix::Minimum || dstColorMatrix == ColorMatrix::Maximum)
    {
        std::cerr << "MatrixConvert_RGB2YUV: ColorMatrix::Minimum or ColorMatrix::Maximum is invalid!\n";
        return;
    }
    else if (dstColorMatrix == ColorMatrix::GBR)
    {
        dstY.ReSetChroma(false);
        dstU.ReSetChroma(false);
        dstV.ReSetChroma(false);
        RangeConvert(dstY, dstU, dstV, srcG, srcB, srcR);
    }
    else
    {
        FLType gainY = dRangeY / sRange;
        FLType offsetY = -sFloor * gainY + dFloorY;
        if (!dstFloat) offsetY += FLType(0.5);
        FLType gainC = dRangeC / sRange;
        FLType offsetC = dNeutralC;
        if (!dstFloat) offsetC += FLType(dstU.isPCChroma() ? 0.499999 : 0.5);

        FLType Yr, Yg, Yb, Ur, Ug, Ub, Vr, Vg, Vb;
        ColorMatrix_RGB2YUV_Parameter(dstColorMatrix, Yr, Yg, Yb, Ur, Ug, Ub, Vr, Vg, Vb);

        Yr *= gainY;
        Yg *= gainY;
        Yb *= gainY;
        Ur *= gainC;
        Ug *= gainC;
        Ub *= gainC;
        Vr *= gainC;
        Vg *= gainC;
        Vb *= gainC;

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dstY[i] = static_cast<dstType>(Yr * static_cast<FLType>(srcR[i])
                + Yg * static_cast<FLType>(srcG[i])
                + Yb * static_cast<FLType>(srcB[i])
                + offsetY);
            dstU[i] = static_cast<dstType>(Ur * static_cast<FLType>(srcR[i])
                + Ug * static_cast<FLType>(srcG[i])
                + Ub * static_cast<FLType>(srcB[i])
                + offsetC);
            dstV[i] = static_cast<dstType>(Vr * static_cast<FLType>(srcR[i])
                + Vg * static_cast<FLType>(srcG[i])
                + Vb * static_cast<FLType>(srcB[i])
                + offsetC);
        });
    }
}

template < typename _Dt1, typename _St1 >
void MatrixConvert_YUV2RGB(_Dt1 &dstR, _Dt1 &dstG, _Dt1 &dstB,
    const _St1 &srcY, const _St1 &srcU, const _St1 &srcV, ColorMatrix dstColorMatrix = ColorMatrix::OPP)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool dstFloat = isFloat(dstType);

    dstR.ReSize(srcY.Width(), srcY.Height());
    dstG.ReSize(srcY.Width(), srcY.Height());
    dstB.ReSize(srcY.Width(), srcY.Height());
    dstR.ReSetChroma(false);
    dstG.ReSetChroma(false);
    dstB.ReSetChroma(false);

    const PCType height = dstR.Height();
    const PCType width = dstR.Width();
    const PCType stride = dstR.Stride();

    const FLType sFloorY = static_cast<FLType>(srcY.Floor());
    const FLType sRangeY = static_cast<FLType>(srcY.ValueRange());
    const FLType sNeutralC = static_cast<FLType>(srcU.Neutral());
    const FLType sRangeC = static_cast<FLType>(srcU.ValueRange());
    const FLType dFloor = static_cast<FLType>(dstR.Floor());
    const FLType dCeil = static_cast<FLType>(dstR.Ceil());
    const FLType dRange = static_cast<FLType>(dstR.ValueRange());

    if (dstColorMatrix == ColorMatrix::OPP)
    {
        FLType gainY = dRange / sRangeY;
        FLType gainC = dRange / sRangeC;

        FLType Ry = gainY;
        FLType Ru = gainC;
        FLType Rv = gainC * FLType(2. / 3.);
        FLType Gy = gainY;
        FLType Gv = gainC * FLType(-4. / 3.);
        FLType By = gainY;
        FLType Bu = gainC * FLType(-1);
        FLType Bv = gainC * FLType(2. / 3.);

        FLType offsetR = -sFloorY * gainY - sNeutralC * gainC * FLType(5. / 3.) + dFloor;
        if (!dstFloat) offsetR += FLType(0.5);
        FLType offsetG = -sFloorY * gainY - sNeutralC * gainC * FLType(-4. / 3.) + dFloor;
        if (!dstFloat) offsetG += FLType(0.5);
        FLType offsetB = -sFloorY * gainY - sNeutralC * gainC * FLType(-1. / 3.) + dFloor;
        if (!dstFloat) offsetB += FLType(0.5);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dstR[i] = static_cast<dstType>(Clip(
                Ry * static_cast<FLType>(srcY[i])
                + Ru * static_cast<FLType>(srcU[i]) 
                + Rv * static_cast<FLType>(srcV[i])
                + offsetR, dFloor, dCeil));
            dstG[i] = static_cast<dstType>(Clip(
                Gy * static_cast<FLType>(srcY[i])
                + Gv * static_cast<FLType>(srcV[i])
                + offsetG, dFloor, dCeil));
            dstB[i] = static_cast<dstType>(Clip(
                By * static_cast<FLType>(srcY[i])
                + Bu * static_cast<FLType>(srcU[i])
                + Bv * static_cast<FLType>(srcV[i])
                + offsetB, dFloor, dCeil));
        });
    }
    else if (dstColorMatrix == ColorMatrix::Minimum || dstColorMatrix == ColorMatrix::Maximum)
    {
        std::cerr << "MatrixConvert_YUV2RGB: ColorMatrix::Minimum or ColorMatrix::Maximum is invalid!\n";
        return;
    }
    else if (dstColorMatrix == ColorMatrix::GBR)
    {
        RangeConvert(dstG, dstB, dstR, srcY, srcU, srcV);
    }
    else
    {
        FLType gainY = dRange / sRangeY;
        FLType gainC = dRange / sRangeC;

        FLType Ry, Ru, Rv, Gy, Gu, Gv, By, Bu, Bv;
        ColorMatrix_YUV2RGB_Parameter(dstColorMatrix, Ry, Ru, Rv, Gy, Gu, Gv, By, Bu, Bv);

        Ry *= gainY;
        Rv *= gainC;
        Gy *= gainY;
        Gu *= gainC;
        Gv *= gainC;
        By *= gainY;
        Bu *= gainC;

        FLType offsetR = -sFloorY * Ry - sNeutralC * Rv + dFloor;
        if (!dstFloat) offsetR += FLType(0.5);
        FLType offsetG = -sFloorY * Gy - sNeutralC * (Gu + Gv) + dFloor;
        if (!dstFloat) offsetG += FLType(0.5);
        FLType offsetB = -sFloorY * By - sNeutralC * Bu + dFloor;
        if (!dstFloat) offsetB += FLType(0.5);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dstR[i] = static_cast<dstType>(Clip(
                Ry * static_cast<FLType>(srcY[i])
                + Rv * static_cast<FLType>(srcV[i])
                + offsetR, dFloor, dCeil));
            dstG[i] = static_cast<dstType>(Clip(
                Gy * static_cast<FLType>(srcY[i])
                + Gu * static_cast<FLType>(srcU[i])
                + Gv * static_cast<FLType>(srcV[i])
                + offsetG, dFloor, dCeil));
            dstB[i] = static_cast<dstType>(Clip(
                By * static_cast<FLType>(srcY[i])
                + Bu * static_cast<FLType>(srcU[i])
                + offsetB, dFloor, dCeil));
        });
    }
}

template < typename _Dt1, typename _St1 > inline
void MatrixConvert(_Dt1 &dst, const _St1 &src, ColorMatrix _ColorMatrix = ColorMatrix::Unspecified)
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
        MatrixConvert_RGB2YUV(dst.Y(), dst.U(), dst.V(), src.R(), src.G(), src.B(), _ColorMatrix);
    }
    else if (src.isYUV() && dst.isRGB())
    {
        MatrixConvert_YUV2RGB(dst.Y(), dst.U(), dst.V(), src.R(), src.G(), src.B(), _ColorMatrix);
    }
    else
    {
        dst = src;
    }
}


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
    const char *FunctionName = "TransferConvert";
    if (dst.GetPixelType() != src.GetPixelType())
    {
        std::cerr << FunctionName << ": \"PixelType\" of dst and src should be the same!\n";
        exit(EXIT_FAILURE);
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


#endif
