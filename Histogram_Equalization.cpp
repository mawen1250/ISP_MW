#include "Histogram_Equalization.h"


Plane & Histogram_Equalization(Plane &dst, const Plane &src)
{
    Histogram<DType> Hist(src);
    LUT<DType> _LUT = Hist.Equalization_LUT(dst);

    _LUT.Lookup(dst, src);

    return dst;
}


Frame & Histogram_Equalization(Frame &dst, const Frame &src, bool separate)
{
    if (src.isYUV())
    {
        const Plane &srcY = src.Y();
        Plane &dstY = dst.Y();

        Histogram<DType> Hist(srcY);
        LUT<FLType> _LUT = Hist.Equalization_LUT_Gain(dstY);

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

            Histogram_Equalization(dstR, srcR);
            Histogram_Equalization(dstG, srcG);
            Histogram_Equalization(dstB, srcB);
        }
        else
        {
            Plane srcY(src.R(), false);
            srcY.YFrom(src);

            Histogram<DType> Hist(srcY);
            LUT<FLType> _LUT = Hist.Equalization_LUT_Gain(srcY);

            _LUT.Lookup_Gain(dst, src, srcY);
        }
    }

    return dst;
}
