#ifndef CONVERSION_HPP_
#define CONVERSION_HPP_


#include "Image_Type.h"
#include "LUT.h"
#include "Histogram.h"


template < typename _St1 >
void ValidRange(const _St1 &src, typename _St1::reference min, typename _St1::reference max, double lower_thr = 0., double upper_thr = 0., int HistBins = 1024, bool protect = false)
{
    typedef typename _St1::value_type dataType;

    src.MinMax(min, max);

    if (protect && max <= min)
    {
        min = src.Floor();
        max = src.Ceil();
    }
    else if (lower_thr > 0 || upper_thr > 0)
    {
        Histogram<dataType> hist(src, min, max, HistBins);
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
void ConvertToY(_Dt1 &dst, const _St1 &srcR, const _St1 &srcG, const _St1 &srcB, ColorMatrix dstColorMatrix = ColorMatrix::Average)
{
    typedef typename _St1::value_type srcType;
    typedef typename _Dt1::value_type dstType;

    const bool srcFloat = isFloat(srcType);
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

    FLType gain = static_cast<FLType>(dRange) / sRange;
    FLType offset = dFloor - sFloor * gain;
    if (!dstFloat) offset += FLType(0.5);

    if (dstColorMatrix == ColorMatrix::Average)
    {
        gain = static_cast<FLType>(dRange) / (sRange * 3);
        offset = dFloor - sFloor * 3 * gain;

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dst[i] = static_cast<dstType>(static_cast<FLType>(srcR[i] + srcG[i] + srcB[i]) * gain + offset);
        });
    }
    else if (dstColorMatrix == ColorMatrix::Minimum)
    {
        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dst[i] = static_cast<dstType>(static_cast<FLType>(::Min(srcR[i], ::Min(srcG[i], srcB[i]))) * gain + offset);
        });
    }
    else if (dstColorMatrix == ColorMatrix::Maximum)
    {
        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dst[i] = static_cast<dstType>(static_cast<FLType>(::Max(srcR[i], ::Max(srcG[i], srcB[i]))) * gain + offset);
        });
    }
    else
    {
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
void ConvertToY(_Dt1 &dst, const _St1 &src, ColorMatrix dstColorMatrix = ColorMatrix::Average)
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
    srcType minR, maxR, minG, maxG, minB, maxB;

    ValidRange(srcR, minR, maxR, lower_thr, upper_thr, HistBins, false);
    ValidRange(srcG, minG, maxG, lower_thr, upper_thr, HistBins, false);
    ValidRange(srcB, minB, maxB, lower_thr, upper_thr, HistBins, false);

    min = Min(minR, Min(minG, minB));
    max = Max(maxR, Max(maxG, maxB));
    if (min >= max)
    {
        min = srcR.Floor();
        max = srcR.Ceil();
    }

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
