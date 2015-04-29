#define ENABLE_PPL


#include "Haze_Removal.h"
#include "Conversion.hpp"
#include "Gaussian.h"


const Haze_Removal_Para Haze_Removal_Default;


// Functions for class Haze_Removal
Frame &Haze_Removal::process_Frame(Frame &dst, const Frame &src)
{
    height = dst.Height();
    width = dst.Width();
    stride = dst.Stride();

    if (src.isYUV() || dst.isYUV())
    {
        const char *FunctionName = "Haze_Removal::process";
        std::cerr << FunctionName << ": YUV input/output is not supported.\n";
        exit(EXIT_FAILURE);
    }
    else
    {
        TransferConvert(dataR, dataG, dataB, src.R(), src.G(), src.B(), TransferChar::linear, para.TransferChar_);

        GetTMapInv();
        GetAtmosLight();
        RemoveHaze();
        StoreResult(dst);

        if (para.TransferChar_ != TransferChar::linear)
            TransferConvert(dst, dst, para.TransferChar_, TransferChar::linear);
    }

    return dst;
}

void Haze_Removal::GetAtmosLight()
{
    const Histogram<FLType> Histogram(tMapInv, para.HistBins);
    const FLType tMapLowerThr = Histogram.Max(para.tMap_thr);

    int count = 0;
    FLType AL_Rsum = 0;
    FLType AL_Gsum = 0;
    FLType AL_Bsum = 0;

    LOOP_VH(height, width, stride, [&](PCType i)
    {
        if (tMapInv[i] >= tMapLowerThr)
        {
            ++count;
            AL_Rsum += dataR[i];
            AL_Gsum += dataG[i];
            AL_Bsum += dataB[i];
        }
    });

    AL_R = Min(para.ALmax, AL_Rsum / count);
    AL_G = Min(para.ALmax, AL_Gsum / count);
    AL_B = Min(para.ALmax, AL_Bsum / count);

    if (para.debug > 0)
    {
        std::cout << "Global Atmospheric Light (R, G, B) = (" << AL_R << ", " << AL_G << ", " << AL_B << ")\n";
    }
}

void Haze_Removal::RemoveHaze()
{
    const FLType mulR = para.strength / AL_R;
    const FLType mulG = para.strength / AL_G;
    const FLType mulB = para.strength / AL_B;

    if (para.debug == 2)
    {
        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dataR[i] = tMapInv[i];
            dataG[i] = tMapInv[i];
            dataB[i] = tMapInv[i];
        });
    }
    else if (para.debug == 3)
    {
        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            dataR[i] = Max(para.tMapMin, para.tMapMax - tMapInv[i] * mulR);
            dataG[i] = Max(para.tMapMin, para.tMapMax - tMapInv[i] * mulG);
            dataB[i] = Max(para.tMapMin, para.tMapMax - tMapInv[i] * mulB);
        });
    }
    else
    {
        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            const FLType divR = Max(para.tMapMin, para.tMapMax - tMapInv[i] * mulR);
            const FLType divG = Max(para.tMapMin, para.tMapMax - tMapInv[i] * mulG);
            const FLType divB = Max(para.tMapMin, para.tMapMax - tMapInv[i] * mulB);
            dataR[i] = (dataR[i] - AL_R) / divR + AL_R;
            dataG[i] = (dataG[i] - AL_G) / divG + AL_G;
            dataB[i] = (dataB[i] - AL_B) / divB + AL_B;
        });
    }
}

void Haze_Removal::StoreResult(Frame &dst)
{
    Plane &dstR = dst.R();
    Plane &dstG = dst.G();
    Plane &dstB = dst.B();

    FLType min, max;

    if (para.debug >= 2 || para.ppmode <= 0) // No range scaling
    {
        min = dataR.Floor();
        max = dataR.Ceil();
    }
    else if (para.ppmode == 1) // Canonical range scaling
    {
        min = FLType(0.13); // Experimental lower limit
        max = (AL_R + AL_G + AL_B) / FLType(3); // Take average Global Atmospheric Light as upper limit
    }
    else if (para.ppmode == 2) // Gaussian filtered average channel for range detection
    {
        Plane_FL temp(dataR, false);
        RecursiveGaussian GFilter(para.pp_sigma, true);

        ConvertToY(temp, dataR, dataG, dataB, ColorMatrix::Average);

        GFilter.Filter(temp, temp);
        GetMinMax(temp, min, max);

        if (max <= min)
        {
            min = dataR.Floor();
            max = dataR.Ceil();
        }
    }
    else // Simpliest Color Balance with all channels combined for range detection
    {
        ValidRange(dataR, dataG, dataB, min, max, para.lower_thr, para.upper_thr, para.HistBins, true);
    }

    RangeConvert(dstR, dstG, dstB, dataR, dataG, dataB,
        dstR.Floor(), dstR.Neutral(), dstR.Ceil(), min, min, max, true);
}


// Functions for class Haze_Removal_Retinex
void Haze_Removal_Retinex::GetTMapInv()
{
    Plane_FL refY(dataR, false);
    ConvertToY(refY, dataR, dataG, dataB, para.Ymode == 1 ? ColorMatrix::Minimum
        : para.Ymode == 2 ? ColorMatrix::Maximum : ColorMatrix::Average);

    const size_t scount = para.sigmaVector.size();
    size_t s;

    // Use refY as tMapInv if no Gaussian need to be applied
    for (s = 0; s < scount; ++s)
    {
        if (para.sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        tMapInv = std::move(refY);
        return;
    }

    if (scount == 1 && para.sigmaVector[0] > 0) // single-scale Gaussian filter
    {
        Plane_FL gauss(refY, false);
        RecursiveGaussian GFilter(para.sigmaVector[0], true);
        GFilter.Filter(gauss, refY);

        tMapInv = Plane_FL(refY, false);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            if (gauss[i] > 0)
            {
                tMapInv[i] = gauss[i];
            }
            else
            {
                tMapInv[i] = 0;
            }
        });
    }
    else if (scount == 2 && para.sigmaVector[0] > 0 && para.sigmaVector[1] > 0) // double-scale Gaussian filter
    {
        Plane_FL gauss0(refY, false);
        RecursiveGaussian GFilter0(para.sigmaVector[0], true);
        GFilter0.Filter(gauss0, refY);

        Plane_FL gauss1(refY, false);
        RecursiveGaussian GFilter1(para.sigmaVector[1], true);
        GFilter1.Filter(gauss1, refY);

        tMapInv = Plane_FL(refY, false);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            if (gauss0[i] > 0 && gauss1[i] > 0)
            {
                tMapInv[i] = sqrt(gauss0[i] * gauss1[i]);
            }
            else
            {
                tMapInv[i] = 0;
            }
        });
    }
    else // multi-scale Gaussian filter
    {
        tMapInv = Plane_FL(refY, true, 1);
        Plane_FL gauss(refY, false);

        for (s = 0; s < scount; ++s)
        {
            if (para.sigmaVector[s] > 0)
            {
                RecursiveGaussian GFilter(para.sigmaVector[s], true);
                GFilter.Filter(gauss, refY);

                LOOP_VH_PPL(height, width, stride, [&](PCType i)
                {
                    if (gauss[i] > 0)
                    {
                        tMapInv[i] *= gauss[i];
                    }
                    else
                    {
                        tMapInv[i] = 0;
                    }
                });
            }
            else
            {
                LOOP_VH_PPL(height, width, stride, [&](PCType i)
                {
                    tMapInv[i] *= refY[i];
                });
            }
        }

        // Calculate geometric mean of multiple scales
        const FLType scountRec = FLType(1) / static_cast<FLType>(scount);

        LOOP_VH_PPL(height, width, stride, [&](PCType i)
        {
            tMapInv[i] = pow(tMapInv[i], scountRec);
        });
    }
}
