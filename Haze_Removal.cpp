#include "Haze_Removal.h"
#include "Gaussian.h"


// Functions for class Haze_Removal
Frame &Haze_Removal::process(Frame &dst, const Frame &src)
{
    height = src.Height();
    width = src.Width();
    stride = src.Stride();

    if (src.isYUV())
    {
        exit(EXIT_FAILURE);
    }
    else
    {
        dataR = Plane_FL(src.R());
        dataG = Plane_FL(src.G());
        dataB = Plane_FL(src.B());

        GetTMapInv(src);
        GetAirLight();
        RemoveHaze();
        StoreResult(dst);
    }

    return dst;
}

void Haze_Removal::GetAirLight()
{
    PCType i, j, upper;

    Histogram<FLType> Histogram(tMapInv, para.HistBins);
    FLType tMapLowerThr = Histogram.Max(para.tMap_thr);

    int count = 0;
    FLType AL_Rsum = 0;
    FLType AL_Gsum = 0;
    FLType AL_Bsum = 0;

    for (j = 0; j < height; ++j)
    {
        i = j * stride;
        for (upper = i + width; i < upper; ++i)
        {
            if (tMapInv[i] >= tMapLowerThr)
            {
                ++count;
                AL_Rsum += dataR[i];
                AL_Gsum += dataG[i];
                AL_Bsum += dataB[i];
            }
        }
    }

    AL_R = Min(para.ALmax, AL_Rsum / count);
    AL_G = Min(para.ALmax, AL_Gsum / count);
    AL_B = Min(para.ALmax, AL_Bsum / count);

    std::cout << "Global Atmospheric Light (R, G, B) = (" << AL_R << ", " << AL_G << ", " << AL_B << ")\n";
}

void Haze_Removal::RemoveHaze()
{
    PCType i, j, upper;

    FLType divR, divG, divB;
    FLType mulR = para.strength / AL_R;
    FLType mulG = para.strength / AL_G;
    FLType mulB = para.strength / AL_B;

    for (j = 0; j < height; ++j)
    {
        i = j * stride;
        for (upper = i + width; i < upper; ++i)
        {
            divR = Max(para.tMapMin, 1 - tMapInv[i] * mulR);
            divG = Max(para.tMapMin, 1 - tMapInv[i] * mulG);
            divB = Max(para.tMapMin, 1 - tMapInv[i] * mulB);
            dataR[i] = (dataR[i] - AL_R) / divR + AL_R;
            dataG[i] = (dataG[i] - AL_G) / divG + AL_G;
            dataB[i] = (dataB[i] - AL_B) / divB + AL_B;
        }
    }
}

void Haze_Removal::StoreResult(Frame &dst)
{
    Plane &dstR = dst.R();
    Plane &dstG = dst.G();
    Plane &dstB = dst.B();

    if (para.ppmode <= 0) // No range scaling
    {
        RangeConvert(dstR, dstG, dstB, dataR, dataG, dataB,
            dstR.Floor(), dstR.Neutral(), dstR.Ceil(), dataR.Floor(), dataR.Neutral(), dataR.Ceil(), true);
    }
    else if (para.ppmode == 1) // Canonical range scaling
    {
        FLType min = -0.10; // Experimental lower limit
        FLType max = (AL_R + AL_G + AL_B) / FLType(3); // Take average Global Atmospheric Light as upper limit
        RangeConvert(dstR, dstG, dstB, dataR, dataG, dataB,
            dstR.Floor(), dstR.Neutral(), dstR.Ceil(), min, min, max, true);
    }
    else if (para.ppmode == 2) // Apply separate Simpliest Color Balance to each channel
    {
        dstR.SimplestColorBalance(dataR, para.lower_thr, para.upper_thr, para.HistBins);
        dstG.SimplestColorBalance(dataG, para.lower_thr, para.upper_thr, para.HistBins);
        dstB.SimplestColorBalance(dataB, para.lower_thr, para.upper_thr, para.HistBins);
    }
    else // Apply Simpliest Color Balance to all channels with the same range conversion
    {
        dst.SimplestColorBalance(dataR, dataG, dataB, para.lower_thr, para.upper_thr, para.HistBins);
    }
}


// Functions for class Haze_Removal_Retinex
void Haze_Removal_Retinex::GetTMapInv(const Frame &src)
{
    PCType i, j, upper;

    Plane_FL refY(src.P(0), false);
    refY.YFrom(src, ColorMatrix::Average);

    size_t s, scount = para.sigmaVector.size();

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
        RecursiveGaussian GFilter(para.sigmaVector[0]);
        GFilter.Filter(gauss, refY);

        tMapInv = Plane_FL(refY, false);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                if (gauss[i] > 0)
                {
                    tMapInv[i] = gauss[i];
                }
                else
                {
                    tMapInv[i] = 0;
                }
            }
        }
    }
    else if (scount == 2 && para.sigmaVector[0] > 0 && para.sigmaVector[1] > 0) // double-scale Gaussian filter
    {
        Plane_FL gauss0(refY, false);
        RecursiveGaussian GFilter0(para.sigmaVector[0]);
        GFilter0.Filter(gauss0, refY);

        Plane_FL gauss1(refY, false);
        RecursiveGaussian GFilter1(para.sigmaVector[1]);
        GFilter1.Filter(gauss1, refY);

        tMapInv = Plane_FL(refY, false);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                if (gauss0[i] > 0 && gauss1[i] > 0)
                {
                    tMapInv[i] = sqrt(gauss0[i] * gauss1[i]);
                }
                else
                {
                    tMapInv[i] = 0;
                }
            }
        }
    }
    else // multi-scale Gaussian filter
    {
        tMapInv = Plane_FL(refY, true, 1);
        Plane_FL gauss(refY, false);

        for (s = 0; s < scount; ++s)
        {
            if (para.sigmaVector[s] > 0)
            {
                RecursiveGaussian GFilter(para.sigmaVector[s]);
                GFilter.Filter(gauss, refY);

                for (j = 0; j < height; ++j)
                {
                    i = j * stride;
                    for (upper = i + width; i < upper; ++i)
                    {
                        if (gauss[i] > 0)
                        {
                            tMapInv[i] *= gauss[i];
                        }
                        else
                        {
                            tMapInv[i] = 0;
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
                        tMapInv[i] *= refY[i];
                    }
                }
            }
        }

        // Calculate geometric mean of multiple scales
        FLType scountRec = 1 / static_cast<FLType>(scount);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                tMapInv[i] = pow(tMapInv[i], scountRec);
            }
        }
    }
}
