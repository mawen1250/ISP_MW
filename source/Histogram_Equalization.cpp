#include "Histogram_Equalization.h"


LUT<DType> Equalization_LUT(const Histogram<DType> &hist, const Plane &dst, const Plane &src, FLType strength)
{
    typedef Histogram<DType> HistType;
    typedef LUT<DType> LUTType;

    DType sRange = src.ValueRange();
    DType dFloor = dst.Floor();
    DType dRange = dst.ValueRange();
    LUTType::LevelType LUT_Levels = sRange + 1;

    LUTType _LUT(LUT_Levels);

    FLType weight1 = strength;
    FLType weight2 = 1 - strength;

    HistType::CountType Sum;
    FLType scale1, scale2, offset;

    if (hist.Bins() == LUT_Levels)
    {
        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / hist.Count() * weight1;
        scale2 = static_cast<FLType>(dRange) / sRange * weight2;
        offset = static_cast<FLType>(dFloor)+FLType(0.5);

        for (HistType::BinType i = 0; i < hist.Bins(); ++i)
        {
            Sum += hist[i];
            _LUT[i] = static_cast<DType>(Sum * scale1 + i * scale2 + offset);
        }
    }
    else
    {
        FLType *temp = new FLType[hist.Bins()];

        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / hist.Count() * weight1;
        scale2 = static_cast<FLType>(dRange) / (hist.Bins() - 1) * weight2;
        offset = static_cast<FLType>(dFloor);

        for (HistType::BinType i = 0; i < hist.Bins(); ++i)
        {
            Sum += hist[i];
            temp[i] = Sum * scale1 + i * scale2 + offset;
        }

        FLType val;
        DType val_lower, val_upper;
        DType val_lower_upper = hist.Bins() - 2;
        scale1 = static_cast<FLType>(hist.Bins() - 1) / sRange;

        for (LUT<DType>::LevelType i = 0; i < LUT_Levels; ++i)
        {
            val = i * scale1;
            val_lower = ::Min(static_cast<DType>(val), val_lower_upper);
            val_upper = val_lower + 1;
            _LUT[i] = static_cast<DType>(temp[val_lower] * (val_upper - val) + temp[val_upper] * (val - val_lower) + FLType(0.5));
        }
    }

    return _LUT;
}

LUT<FLType> Equalization_LUT_Gain(const Histogram<DType> &hist, const Plane &dst, const Plane &src, FLType strength)
{
    typedef Histogram<DType> HistType;
    typedef LUT<FLType> LUTType;

    DType sRange = src.ValueRange();
    DType dFloor = dst.Floor();
    DType dRange = dst.ValueRange();
    LUTType::LevelType LUT_Levels = sRange + 1;

    LUTType _LUT(LUT_Levels);

    FLType weight1 = strength;
    FLType weight2 = 1 - strength;

    HistType::CountType Sum;
    FLType scale1, scale2;

    if (hist.Bins() == LUT_Levels)
    {
        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / hist.Count() * weight1;
        scale2 = static_cast<FLType>(dRange) / sRange * weight2;

        for (HistType::BinType i = 0; i < hist.Bins(); ++i)
        {
            Sum += hist[i];
            _LUT[i] = Sum * scale1 / i + scale2;
        }
    }
    else
    {
        FLType *temp = new FLType[hist.Bins()];

        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / hist.Count() * weight1;
        scale2 = static_cast<FLType>(dRange) / (hist.Bins() - 1) * weight2;

        for (HistType::BinType i = 0; i < hist.Bins(); ++i)
        {
            Sum += hist[i];
            temp[i] = Sum * scale1 + i * scale2;
        }

        FLType val;
        DType val_lower, val_upper;
        DType val_lower_upper = hist.Bins() - 2;
        scale1 = static_cast<FLType>(hist.Bins() - 1) / sRange;

        for (LUTType::LevelType i = 0; i < LUT_Levels; ++i)
        {
            val = i * scale1;
            val_lower = ::Min(static_cast<DType>(val), val_lower_upper);
            val_upper = val_lower + 1;
            _LUT[i] = (temp[val_lower] * (val_upper - val) + temp[val_upper] * (val - val_lower)) / i;
        }
    }

    return _LUT;
}


Plane & Histogram_Equalization(Plane &dst, const Plane &src, FLType strength)
{
    Histogram<DType> hist(src);
    auto _LUT = Equalization_LUT(hist, dst, strength);

    _LUT.Lookup(dst, src);

    return dst;
}


Frame & Histogram_Equalization(Frame &dst, const Frame &src, FLType strength, bool separate)
{
    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    if (src.isYUV())
    {
        const Plane &srcY = src.Y();
        Plane &dstY = dst.Y();

        Histogram<DType> hist(srcY);
        auto _LUT = Equalization_LUT_Gain(hist, dstY, strength);

        _LUT.Lookup_Gain(dst, src, srcY);
    }
    else if (src.isRGB())
    {
        if (separate)
        {
            const Plane &srcR = src.R();
            const Plane &srcG = src.G();
            const Plane &srcB = src.B();
            Plane &dstR = dst.R();
            Plane &dstG = dst.G();
            Plane &dstB = dst.B();

            Histogram_Equalization(dstR, srcR, strength);
            Histogram_Equalization(dstG, srcG, strength);
            Histogram_Equalization(dstB, srcB, strength);
        }
        else
        {
            const Plane &srcR = src.R();
            const Plane &srcG = src.G();
            const Plane &srcB = src.B();
            Plane &dstR = dst.R();
            Plane &dstG = dst.G();
            Plane &dstB = dst.B();

            Plane srcY(srcR, false);

            for (j = 0; j < height; j++)
            {
                i = stride * j;
                for (upper = i + width; i < upper; i++)
                {
                    srcY[i] = Round_Div((srcR[i] + srcG[i] + srcB[i]), DType(3));
                }
            }

            Histogram<DType> hist(srcY);
            auto _LUT = Equalization_LUT_Gain(hist, srcY, strength);

            _LUT.Lookup_Gain(dst, src, srcY);
        }
    }

    return dst;
}
