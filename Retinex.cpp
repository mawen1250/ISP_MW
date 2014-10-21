#include <cmath>
#include "Retinex.h"
#include "Specification.h"
#include "Gaussian.h"


Plane & Retinex_SSR(Plane & dst, const Plane & src, double sigma, double lower_thr, double upper_thr)
{
    if (sigma <= 0)
    {
        dst = src;
        return dst;
    }

    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    FLType B, B1, B2, B3;
    Recursive_Gaussian_Parameters(sigma, B, B1, B2, B3);

    Plane_FL data(src);
    Plane_FL gauss(data, false);

    Recursive_Gaussian2D_Horizontal(gauss, data, B, B1, B2, B3);
    Recursive_Gaussian2D_Vertical(gauss, B, B1, B2, B3);

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            data[i] = gauss[i] <= 0 ? 0 : log(data[i] / gauss[i] + 1);
        }
    }
    
    FLType min, max;
    data.MinMax(min, max);

    if (max <= min)
    {
        dst = src;
        return dst;
    }

    if (lower_thr> 0 || upper_thr > 0)
    {
        data.ReQuantize(min, min, max, false);
        Histogram<FLType> Histogram(data, Retinex_Default.HistBins);
        min = Histogram.Min(lower_thr);
        max = Histogram.Max(upper_thr);
    }

    data.ReQuantize(min, min, max, false);

    FLType gain = dst.ValueRange() / (max - min);
    FLType offset = dst.Floor() - min*gain + FLType(0.5);

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            dst[i] = static_cast<DType>(data.Quantize(data[i]) * gain + offset);
        }
    }
    
    return dst;
}


Plane_FL Retinex_MSR(const Plane_FL & idata, const std::vector<double> & sigmaVector, double lower_thr, double upper_thr)
{
    size_t s, scount = sigmaVector.size();

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        Plane_FL odata(idata);
        return odata;
    }

    PCType i, j, upper;
    PCType height = idata.Height();
    PCType width = idata.Width();
    PCType stride = idata.Width();

    Plane_FL odata(idata, true, 1);
    Plane_FL gauss(idata, false);

    FLType B, B1, B2, B3;

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0)
        {
            Recursive_Gaussian_Parameters(sigmaVector[s], B, B1, B2, B3);
            Recursive_Gaussian2D_Horizontal(gauss, idata, B, B1, B2, B3);
            Recursive_Gaussian2D_Vertical(gauss, B, B1, B2, B3);

            for (j = 0; j < height; j++)
            {
                i = stride * j;
                for (upper = i + width; i < upper; i++)
                {
                    odata[i] *= gauss[i] <= 0 ? 1 : idata[i] / gauss[i] + 1;
                }
            }
        }
        else
        {
            for (j = 0; j < height; j++)
            {
                i = stride * j;
                for (upper = i + width; i < upper; i++)
                {
                    odata[i] *= FLType(2);
                }
            }
        }
    }

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            odata[i] = log(odata[i]) / static_cast<FLType>(scount);
        }
    }

    FLType min, max;
    odata.MinMax(min, max);

    if (max <= min)
    {
        odata = idata;
        return odata;
    }

    if (lower_thr> 0 || upper_thr > 0)
    {
        odata.ReQuantize(min, min, max, false);
        Histogram<FLType> Histogram(odata, Retinex_Default.HistBins);
        min = Histogram.Min(lower_thr);
        max = Histogram.Max(upper_thr);
    }

    odata.ReQuantize(min, min, max, false);
    odata.ReQuantize(idata.Floor(), idata.Neutral(), idata.Ceil(), true, lower_thr> 0 || upper_thr > 0);

    return odata;
}

Plane & Retinex_MSR(Plane & dst, const Plane & src, const std::vector<double> & sigmaVector, double lower_thr, double upper_thr)
{
    size_t s, scount = sigmaVector.size();

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        dst = src;
        return dst;
    }

    Plane_FL idata(src);
    Plane_FL odata = Retinex_MSR(idata, sigmaVector, lower_thr, upper_thr);

    odata.To(dst);

    return dst;
}

Frame & Retinex_MSRCP(Frame & dst, const Frame & src, const std::vector<double> & sigmaVector, double lower_thr, double upper_thr, double chroma_protect)
{
    size_t s, scount = sigmaVector.size();

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        dst = src;
        return dst;
    }

    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    FLType gain, offset;

    if (src.isYUV())
    {
        const Plane & srcY = src.Y();
        const Plane & srcU = src.U();
        const Plane & srcV = src.V();
        Plane & dstY = dst.Y();
        Plane & dstU = dst.U();
        Plane & dstV = dst.V();

        sint32 sNeutral = srcU.Neutral();
        FLType sRangeC2FL = static_cast<FLType>(srcU.ValueRange()) / 2.;

        Plane_FL idata(srcY);
        Plane_FL odata = Retinex_MSR(idata, sigmaVector, lower_thr, upper_thr);

        odata.To(dstY);

        FLType chroma_protect_mul1 = static_cast<FLType>(chroma_protect - 1);
        FLType chroma_protect_mul2 = static_cast<FLType>(1 / log(chroma_protect));

        sint32 Uval, Vval;
        if (dstU.isPCChroma())
            offset = dstU.Neutral() + FLType(0.499999);
        else
            offset = dstU.Neutral() + FLType(0.5);

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                if (chroma_protect > 1)
                    gain = idata[i] <= 0 ? 1 : log(odata[i] / idata[i] * chroma_protect_mul1 + 1) * chroma_protect_mul2;
                else
                    gain = idata[i] <= 0 ? 1 : odata[i] / idata[i];
                Uval = srcU[i] - sNeutral;
                Vval = srcV[i] - sNeutral;
                gain = Min(sRangeC2FL / Max(Abs(Uval), Abs(Vval)), gain);
                dstU[i] = static_cast<DType>(Uval * gain + offset);
                dstV[i] = static_cast<DType>(Vval * gain + offset);
            }
        }
    }
    else if (src.isRGB())
    {
        const Plane & srcR = src.R();
        const Plane & srcG = src.G();
        const Plane & srcB = src.B();
        Plane & dstR = dst.R();
        Plane & dstG = dst.G();
        Plane & dstB = dst.B();

        DType sRange = srcR.ValueRange();
        FLType sRangeFL = static_cast<FLType>(sRange);

        Plane_FL idata(srcR, false);

        offset = static_cast<FLType>(srcR.Floor() * -3);
        gain = FLType(1) / (srcR.ValueRange() * 3);

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                idata[i] = (srcR[i] + srcG[i] + srcB[i] + offset) * gain;
            }
        }

        Plane_FL odata = Retinex_MSR(idata, sigmaVector, lower_thr, upper_thr);

        DType sFloor = srcR.Floor();
        offset = dstR.Floor() + FLType(0.5);

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                gain = idata[i] <= 0 ? 1 : odata[i] / idata[i];
                gain = Min(sRangeFL / Max(srcR[i], Max(srcG[i], srcB[i])), gain);
                dstR[i] = static_cast<DType>((srcR[i] - sFloor) * gain + offset);
                dstG[i] = static_cast<DType>((srcG[i] - sFloor) * gain + offset);
                dstB[i] = static_cast<DType>((srcB[i] - sFloor) * gain + offset);
            }
        }
    }

    return dst;
}
