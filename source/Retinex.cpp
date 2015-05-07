#include "Retinex.h"
#include "Conversion.hpp"
#include "Gaussian.h"


// Functions of class Retinex
Plane_FL &Retinex::Kernel(Plane_FL &dst, const Plane_FL &src)
{
    const size_t scount = para.sigmaVector.size();
    size_t s;

    for (s = 0; s < scount; ++s)
    {
        if (para.sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        dst = src;
        return dst;
    }

    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Stride();

    Plane_FL gauss(src, false);

    if (scount == 1 && para.sigmaVector[0] > 0) // single-scale Gaussian filter
    {
        RecursiveGaussian GFilter(para.sigmaVector[0], true);
        Plane_FL gauss = GFilter(src);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                dst[i] = gauss[i] <= 0 ? 0 : log(src[i] / gauss[i] + 1);
            }
        }
    }
    else // multi-scale Gaussian filter
    {
        if (dst.Data() == src.Data())
        {
            const char *FunctionName = "Retinex::Kernel";
            std::cerr << FunctionName << ": Data of Plane_FL \"dst\" and Plane_FL \"src\" can not be of the same address for multi-scale.\n";
            exit(EXIT_FAILURE);
        }

        dst.InitValue(1);

        for (s = 0; s < scount; ++s)
        {
            if (para.sigmaVector[s] > 0)
            {
                RecursiveGaussian GFilter(para.sigmaVector[s], true);
                GFilter.Filter(gauss, src);

                for (j = 0; j < height; ++j)
                {
                    i = j * stride;
                    for (upper = i + width; i < upper; ++i)
                    {
                        if (gauss[i] > 0)
                        {
                            dst[i] *= src[i] / gauss[i] + 1;
                        }
                    }
                }
            }
            else
            {
                for (j = 0; j < height; ++j)
                {
                    i = j * stride;
                    for (upper = i + width; i < upper; ++i)
                    {
                        dst[i] *= FLType(2);
                    }
                }
            }
        }

        FLType scountRec = 1 / static_cast<FLType>(scount);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                dst[i] = log(dst[i]) * scountRec;
            }
        }
    }

    return dst;
}


// Functions of class Retinex_SSR
Plane_FL &Retinex_SSR::process_Plane_FL(Plane_FL &dst, const Plane_FL &src)
{
    Kernel(dst, src);

    SimplestColorBalance(dst, dst, para.lower_thr, para.upper_thr, para.HistBins);

    return dst;
}

Plane &Retinex_SSR::process_Plane(Plane &dst, const Plane &src)
{
    Plane_FL data(src);
    
    Kernel(data, data);
    
    SimplestColorBalance(dst, data, para.lower_thr, para.upper_thr, para.HistBins);

    return dst;
}


// Functions of class Retinex_MSR
Plane_FL &Retinex_MSR::process_Plane_FL(Plane_FL &dst, const Plane_FL &src)
{
    Kernel(dst, src);
    
    SimplestColorBalance(dst, dst, para.lower_thr, para.upper_thr, para.HistBins);

    return dst;
}

Plane &Retinex_MSR::process_Plane(Plane &dst, const Plane &src)
{
    Plane_FL idata(src);
    Plane_FL odata = Kernel(idata);

    SimplestColorBalance(dst, odata, para.lower_thr, para.upper_thr, para.HistBins);

    return dst;
}


// Functions of class Retinex_MSRCR_GIMP
Plane_FL &Retinex_MSRCR_GIMP::process_Plane_FL(Plane_FL &dst, const Plane_FL &src)
{
    Kernel(dst, src);

    FLType mean = dst.Mean();
    FLType var = dst.Variance(mean);
    FLType min = mean - para.dynamic * var;
    FLType max = mean + para.dynamic * var;

    RangeConvert(dst, dst, dst.Floor(), dst.Neutral(), dst.Ceil(), min, min, max, true);

    return dst;
}

Plane &Retinex_MSRCR_GIMP::process_Plane(Plane &dst, const Plane &src)
{
    Plane_FL idata(src);
    Plane_FL odata = Kernel(idata);

    FLType mean = odata.Mean();
    FLType var = odata.Variance(mean);
    FLType min = mean - para.dynamic * var;
    FLType max = mean + para.dynamic * var;

    RangeConvert(dst, odata, dst.Floor(), dst.Neutral(), dst.Ceil(), min, min, max, true);

    return dst;
}


// Functions of class Retinex_MSRCP
Frame &Retinex_MSRCP::process_Frame(Frame &dst, const Frame &src)
{
    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Stride();

    FLType gain, offset;

    if (src.isYUV())
    {
        const Plane &srcY = src.Y();
        const Plane &srcU = src.U();
        const Plane &srcV = src.V();
        Plane &dstY = dst.Y();
        Plane &dstU = dst.U();
        Plane &dstV = dst.V();

        sint32 sNeutral = srcU.Neutral();
        FLType sRangeC2FL = static_cast<FLType>(srcU.ValueRange()) / FLType(2);
        FLType dRangeFL = static_cast<FLType>(dstY.ValueRange());

        Plane_FL idata(srcY);
        Plane_FL odata = operator()(idata);

        FLType chroma_protect_mul1 = static_cast<FLType>(para.chroma_protect - 1);
        FLType chroma_protect_mul2 = static_cast<FLType>(1 / log(para.chroma_protect));

        sint32 Uval, Vval;
        FLType offsetY = dstY.Floor() + FLType(0.5);
        offset = dstU.Neutral() + FLType(dstU.isPCChroma() ? 0.499999 : 0.5);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                Uval = srcU[i] - sNeutral;
                Vval = srcV[i] - sNeutral;
                if (para.chroma_protect > 1)
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
        const Plane &srcR = src.R();
        const Plane &srcG = src.G();
        const Plane &srcB = src.B();
        Plane &dstR = dst.R();
        Plane &dstG = dst.G();
        Plane &dstB = dst.B();
        
        DType sFloor = srcR.Floor();
        FLType sRangeFL = static_cast<FLType>(srcR.ValueRange());

        Plane_FL idata(srcR, false);
        ConvertToY(idata, src, ColorMatrix::OPP);
        
        Plane_FL odata = operator()(idata);
        
        DType Rval, Gval, Bval;
        offset = dstR.Floor() + FLType(0.5);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
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


// Functions of class Retinex_MSRCR
Frame &Retinex_MSRCR::process_Frame(Frame &dst, const Frame &src)
{
    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Stride();

    if (src.isRGB())
    {
        const Plane &srcR = src.R();
        const Plane &srcG = src.G();
        const Plane &srcB = src.B();
        Plane &dstR = dst.R();
        Plane &dstG = dst.G();
        Plane &dstB = dst.B();

        DType sFloor = srcR.Floor();
        DType dFloor = dstR.Floor();
        FLType dRangeFL = static_cast<FLType>(dstR.ValueRange());

        Plane_FL idata(srcR, false);

        RangeConvert(idata, srcR);
        Plane_FL odataR = Kernel(idata);
        RangeConvert(idata, srcG);
        Plane_FL odataG = Kernel(idata);
        RangeConvert(idata, srcB);
        Plane_FL odataB = Kernel(idata);

        DType Rval, Gval, Bval;
        FLType temp;

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                Rval = srcR[i] - sFloor;
                Gval = srcG[i] - sFloor;
                Bval = srcB[i] - sFloor;
                temp = static_cast<FLType>(Rval + Gval + Bval);
                temp = temp <= 0 ? 0 : para.restore / temp;
                odataR[i] *= log(Rval * temp + 1);
                odataG[i] *= log(Gval * temp + 1);
                odataB[i] *= log(Bval * temp + 1);
            }
        }

        SimplestColorBalance(dst, odataR, odataG, odataB, para.lower_thr, para.upper_thr, para.HistBins);
    }
    else
    {
        const char *FunctionName = "Retinex_MSRCR::process";
        std::cerr << FunctionName << ": invalid PixelType of Frame \"src\", should be RGB.\n";
        exit(EXIT_FAILURE);
    }

    return dst;
}
