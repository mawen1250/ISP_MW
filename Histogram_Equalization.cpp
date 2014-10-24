#include "Histogram_Equalization.h"


Plane & Histogram_Equalization(Plane &dst, const Plane &src, FLType strength)
{
    Histogram<DType> Hist(src);
    LUT<DType> _LUT = Hist.Equalization_LUT(dst, strength);

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

        Histogram<DType> Hist(srcY);
        LUT<FLType> _LUT = Hist.Equalization_LUT_Gain(dstY, strength);

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

            Histogram<DType> Hist(srcY);
            LUT<FLType> _LUT = Hist.Equalization_LUT_Gain(srcY, strength);

            _LUT.Lookup_Gain(dst, src, srcY);
        }
    }

    return dst;
}
