#include "Haze_Removal.h"
#include "Gaussian.h"


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
        GetHazeFree(dst);
    }

    return dst;
}


void Haze_Removal::GetAirLight()
{
    PCType i, j, upper;

    Histogram<FLType> Histogram(tMapInv, HistBins);
    FLType tMapLowerThr = Histogram.Max(tMap_thr);

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

    AL_R = Min(ALmax, AL_Rsum / count);
    AL_G = Min(ALmax, AL_Gsum / count);
    AL_B = Min(ALmax, AL_Bsum / count);

    std::cout << "Air Light (R, G, B) = (" << AL_R << ", " << AL_G << ", " << AL_B << ")\n";
}


void Haze_Removal::GetHazeFree(Frame &dst)
{
    PCType i, j, upper;

    Plane &dstR = dst.R();
    Plane &dstG = dst.G();
    Plane &dstB = dst.B();

    FLType dFloorFL = dstR.Floor();
    FLType dCeilFL = dstR.Ceil();

    FLType divR, divG, divB;
    FLType mulR = strength / AL_R;
    FLType mulG = strength / AL_G;
    FLType mulB = strength / AL_B;

    for (j = 0; j < height; ++j)
    {
        i = j * stride;
        for (upper = i + width; i < upper; ++i)
        {
            divR = Max(tMapMin, 1 - tMapInv[i] * mulR);
            divG = Max(tMapMin, 1 - tMapInv[i] * mulG);
            divB = Max(tMapMin, 1 - tMapInv[i] * mulB);
            dataR[i] = (dataR[i] - AL_R) / divR + AL_R;
            dataG[i] = (dataG[i] - AL_G) / divG + AL_G;
            dataB[i] = (dataB[i] - AL_B) / divB + AL_B;
        }
    }

    if (ppmode <= 0)
    {
        dstR.From(dataR, true);
        dstG.From(dataG, true);
        dstB.From(dataB, true);
    }
    else if (ppmode == 1)
    {
        FLType min = -0.10;
        FLType max = (AL_R + AL_G + AL_B) / FLType(3);
        RangeConvert(dstR, dstG, dstB, dataR, dataG, dataB,
            dstR.Floor(), dstR.Neutral(), dstR.Ceil(), min, min, max, true);
    }
    else if (ppmode == 2)
    {
        dstR.SimplestColorBalance(dataR, lower_thr, upper_thr, HistBins);
        dstG.SimplestColorBalance(dataG, lower_thr, upper_thr, HistBins);
        dstB.SimplestColorBalance(dataB, lower_thr, upper_thr, HistBins);
    }
    else
    {
        dst.SimplestColorBalance(dataR, dataG, dataB, lower_thr, upper_thr, HistBins);
    }
}


void Haze_Removal_Retinex::GetTMapInv(const Frame &src)
{
    PCType i, j, upper;

    Plane_FL refY(src.P(0), false);
    refY.YFrom(src, ColorMatrix::Average);

    size_t s, scount = sigmaVector.size();

    // Use refY as tMapInv if no Gaussian need to be applied
    for (s = 0; s < scount; ++s)
    {
        if (sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        tMapInv = std::move(refY);
        return;
    }

    if (scount == 1 && sigmaVector[0] > 0) // single-scale Gaussian filter
    {
        Plane_FL gauss(refY, false);
        RecursiveGaussian GFilter(sigmaVector[0]);
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
    else if (scount == 2 && sigmaVector[0] > 0 && sigmaVector[1] > 0) // dual-scale Gaussian filter
    {
        Plane_FL gauss1(refY, false);
        RecursiveGaussian GFilter1(sigmaVector[0]);
        GFilter1.Filter(gauss1, refY);

        Plane_FL gauss2(refY, false);
        RecursiveGaussian GFilter2(sigmaVector[1]);
        GFilter2.Filter(gauss2, refY);

        tMapInv = Plane_FL(refY, false);

        for (j = 0; j < height; ++j)
        {
            i = j * stride;
            for (upper = i + width; i < upper; ++i)
            {
                if (gauss1[i] > 0 && gauss2[i] > 0)
                {
                    tMapInv[i] = sqrt(gauss1[i] * gauss2[i]);
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
            if (sigmaVector[s] > 0)
            {
                RecursiveGaussian GFilter(sigmaVector[s]);
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
