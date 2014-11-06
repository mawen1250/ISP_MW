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
    
    dst.SimplestColorBalance(data, src, lower_thr, upper_thr, Retinex_Default.HistBins);

    return dst;
}


Plane_FL Retinex_MSR(const Plane_FL & idata, const std::vector<double> & sigmaVector)
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

    return odata;
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

    Plane_FL odata = Retinex_MSR(idata, sigmaVector);

    odata.SimplestColorBalance(odata, idata, lower_thr, upper_thr, Retinex_Default.HistBins);

    return odata;
}

Plane_FL Retinex_MSRCR_GIMP(const Plane_FL & idata, const std::vector<double> & sigmaVector, double dynamic)
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

    Plane_FL odata = Retinex_MSR(idata, sigmaVector);

    FLType min, max;
    FLType mean, var;

    mean = odata.Mean();
    var = odata.Variance(mean);
    min = mean - dynamic * var;
    max = mean + dynamic * var;

    odata.ReQuantize(min, min, max, false);
    odata.ReQuantize(idata.Floor(), idata.Neutral(), idata.Ceil(), true, true);

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
    Plane_FL odata = Retinex_MSR(idata, sigmaVector);

    dst.SimplestColorBalance(odata, src, lower_thr, upper_thr, Retinex_Default.HistBins);

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
        FLType dRangeFL = static_cast<FLType>(dstY.ValueRange());

        Plane_FL idata(srcY);
        Plane_FL odata = Retinex_MSR(idata, sigmaVector, lower_thr, upper_thr);

        FLType chroma_protect_mul1 = static_cast<FLType>(chroma_protect - 1);
        FLType chroma_protect_mul2 = static_cast<FLType>(1 / log(chroma_protect));

        sint32 Uval, Vval;
        if (dstU.isPCChroma())
            offset = dstU.Neutral() + FLType(0.499999);
        else
            offset = dstU.Neutral() + FLType(0.5);
        FLType offsetY = dstY.Floor() + FLType(0.5);

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Uval = srcU[i] - sNeutral;
                Vval = srcV[i] - sNeutral;
                if (chroma_protect > 1)
                    gain = idata[i] <= 0 ? 1 : log(odata[i] / idata[i] * chroma_protect_mul1 + 1) * chroma_protect_mul2;
                else
                    gain = idata[i] <= 0 ? 1 : odata[i] / idata[i];
                gain = Min(sRangeC2FL / Max(Abs(Uval), Abs(Vval)), gain);
                dstY[i] = static_cast<DType>(odata[i] * dRangeFL + offsetY);
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

        DType sFloor = srcR.Floor();
        FLType sRangeFL = static_cast<FLType>(srcR.ValueRange());

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

        DType Rval, Gval, Bval;
        offset = dstR.Floor() + FLType(0.5);

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Rval = srcR[i] - sFloor;
                Gval = srcG[i] - sFloor;
                Bval = srcB[i] - sFloor;
                gain = idata[i] <= 0 ? 1 : odata[i] / idata[i];
                gain = Min(sRangeFL / Max(Rval, Max(Gval, Bval)), gain);
                dstR[i] = static_cast<DType>(Rval * gain + offset);
                dstG[i] = static_cast<DType>(Gval * gain + offset);
                dstB[i] = static_cast<DType>(Bval * gain + offset);
            }
        }
    }

    return dst;
}

Frame & Retinex_MSRCR(Frame & dst, const Frame & src, const std::vector<double> & sigmaVector, double lower_thr, double upper_thr, double restore)
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

    if (src.isRGB())
    {
        const Plane & srcR = src.R();
        const Plane & srcG = src.G();
        const Plane & srcB = src.B();
        Plane & dstR = dst.R();
        Plane & dstG = dst.G();
        Plane & dstB = dst.B();

        DType sFloor = srcR.Floor();
        DType dFloor = dstR.Floor();
        FLType dRangeFL = static_cast<FLType>(dstR.ValueRange());

        Plane_FL idata(srcR, false);

        idata.From(srcR);
        Plane_FL odataR = Retinex_MSR(idata, sigmaVector);
        idata.From(srcG);
        Plane_FL odataG = Retinex_MSR(idata, sigmaVector);
        idata.From(srcB);
        Plane_FL odataB = Retinex_MSR(idata, sigmaVector);

        DType Rval, Gval, Bval;
        FLType temp;

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Rval = srcR[i] - sFloor;
                Gval = srcG[i] - sFloor;
                Bval = srcB[i] - sFloor;
                temp = Rval + Gval + Bval;
                temp = temp <= 0 ? 0 : restore / temp;
                odataR[i] *= log(Rval * temp + 1);
                odataG[i] *= log(Gval * temp + 1);
                odataB[i] *= log(Bval * temp + 1);
            }
        }

        dstR.SimplestColorBalance(odataR, srcR, lower_thr, upper_thr, Retinex_Default.HistBins);
        dstG.SimplestColorBalance(odataG, srcG, lower_thr, upper_thr, Retinex_Default.HistBins);
        dstB.SimplestColorBalance(odataB, srcB, lower_thr, upper_thr, Retinex_Default.HistBins);
    }
    else
    {
        const char * FunctionName = "Retinex_MSRCR";
        std::cerr << FunctionName << ": invalid PixelType of Frame \"src\", should be RGB.\n";
        exit(EXIT_FAILURE);
    }

    return dst;
}

Frame & Retinex_MSRCR_GIMP(Frame & dst, const Frame & src, const std::vector<double> & sigmaVector, double dynamic)
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

    if (src.isRGB())
    {
        const Plane & srcR = src.R();
        const Plane & srcG = src.G();
        const Plane & srcB = src.B();
        Plane & dstR = dst.R();
        Plane & dstG = dst.G();
        Plane & dstB = dst.B();

        Plane_FL idata(srcR, false);
        Plane_FL odata(dstR, false);

        idata.From(srcR);
        odata = Retinex_MSRCR_GIMP(idata, sigmaVector, dynamic);
        odata.To(dstR);
        idata.From(srcG);
        odata = Retinex_MSRCR_GIMP(idata, sigmaVector, dynamic);
        odata.To(dstG);
        idata.From(srcB);
        odata = Retinex_MSRCR_GIMP(idata, sigmaVector, dynamic);
        odata.To(dstB);
    }
    else
    {
        const char * FunctionName = "Retinex_MSRCR_GIMP";
        std::cerr << FunctionName << ": invalid PixelType of Frame \"src=\", should be RGB.\n";
        exit(EXIT_FAILURE);
    }

    return dst;
}
