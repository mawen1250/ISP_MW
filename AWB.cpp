#include "AWB.h"


DType AWB::Hist_value(const Histogram<DType> &Hist, FLType ratio) const
{
    if (ratio < 0.5)
        return Hist.Min(ratio);
    else
        return Hist.Max(1 - ratio);
}


Plane AWB::GetIntensity(const Frame &ref) const
{
    PCType i, j, upper;

    const Plane &refR = ref.R();
    const Plane &refG = ref.G();
    const Plane &refB = ref.B();

    Plane dst(refR, false);

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            dst[i] = Round_Div(refR[i] + refG[i] + refB[i], DType(3));
        }
    }

    return dst;
}


void AWB::sum(const Frame &ref, FLType lower_ratio, FLType upper_ratio)
{
    const Plane &refR = ref.R();
    const Plane &refG = ref.G();
    const Plane &refB = ref.B();

    sumR = 0;
    sumG = 0;
    sumB = 0;

    if (lower_ratio > 0 || upper_ratio < 1)
    {
        DType k;

        DType range_upper, range_lower;
        Histogram<DType> HistR(refR);
        Histogram<DType> HistG(refG);
        Histogram<DType> HistB(refB);

        range_lower = Hist_value(HistR, lower_ratio);
        range_upper = Hist_value(HistR, upper_ratio);

        for (k = range_lower; k <= range_upper; k++)
        {
            sumR += HistR[k] * k;
        }

        range_lower = Hist_value(HistG, lower_ratio);
        range_upper = Hist_value(HistG, upper_ratio);

        for (k = range_lower; k <= range_upper; k++)
        {
            sumG += HistG[k] * k;
        }

        range_lower = Hist_value(HistB, lower_ratio);
        range_upper = Hist_value(HistB, upper_ratio);

        for (k = range_lower; k <= range_upper; k++)
        {
            sumB += HistB[k] * k;
        }
    }
    else
    {
        PCType i, j, upper;

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                sumR += refR[i];
                sumG += refG[i];
                sumB += refB[i];
            }
        }
    }

    sumMax = Max(sumR, Max(sumG, sumB));
}


void AWB::sum_masked(const Frame &ref, const Plane &mask, DType thr)
{
    PCType i, j, upper;

    const Plane &refR = ref.R();
    const Plane &refG = ref.G();
    const Plane &refB = ref.B();

    sumR = 0;
    sumG = 0;
    sumB = 0;

    if (thr <= 0)
    {
        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                sumR += refR[i];
                sumG += refG[i];
                sumB += refB[i];
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
                if (mask[i] >= thr)
                {
                    sumR += refR[i];
                    sumG += refG[i];
                    sumB += refB[i];
                }
            }
        }
    }

    sumMax = Max(sumR, Max(sumG, sumB));
}


void AWB::apply_gain() const
{
    PCType i, j, upper;

    FLType _offsetR = dFloor + dst_offsetR - (sFloor - src_offsetR) * gainR + FLType(0.5);
    FLType _offsetG = dFloor + dst_offsetG - (sFloor - src_offsetG) * gainG + FLType(0.5);
    FLType _offsetB = dFloor + dst_offsetB - (sFloor - src_offsetB) * gainB + FLType(0.5);

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            dstR[i] = static_cast<DType>(Clip(srcR[i] * gainR + _offsetR, dFloorFL, dCeilFL));
            dstG[i] = static_cast<DType>(Clip(srcG[i] * gainG + _offsetG, dFloorFL, dCeilFL));
            dstB[i] = static_cast<DType>(Clip(srcB[i] * gainB + _offsetB, dFloorFL, dCeilFL));
        }
    }

    std::cout << "gainR: " << gainR << ", gainG: " << gainG << ", gainB: " << gainB << std::endl;
}


Frame AWB::process()
{
    kernel();

    apply_gain();

    return dst;
}


void AWB1::kernel()
{
    Plane srcY = GetIntensity(src);
    Histogram<DType> Hist_srcY(srcY);

    srcY.Binarize(Hist_srcY.Max(0.050), Hist_srcY.Max(0.005));

    sum_masked(src, srcY);

    gainR = static_cast<FLType>(sumMax) / sumR;
    gainG = static_cast<FLType>(sumMax) / sumG;
    gainB = static_cast<FLType>(sumMax) / sumB;
    src_offsetR = 0;
    src_offsetG = 0;
    src_offsetB = 0;
    dst_offsetR = 0;
    dst_offsetG = 0;
    dst_offsetB = 0;
}


void AWB2::kernel()
{
    EdgeDetect(dstR, srcR, EdgeKernel::Sobel);
    EdgeDetect(dstG, srcG, EdgeKernel::Sobel);
    EdgeDetect(dstB, srcB, EdgeKernel::Sobel);

    sum(dst, 0.50, 0.99);
    
    Plane srcY = GetIntensity(src);
    Plane edgeY = EdgeDetect(srcY, EdgeKernel::Sobel);
    Histogram<DType> Hist_edgeY(edgeY);

    edgeY.Binarize(Hist_edgeY.Max(0.010), Hist_edgeY.Max(0.001));

    sum_masked(src, edgeY);

    gainR = static_cast<FLType>(sumMax) / sumR;
    gainG = static_cast<FLType>(sumMax) / sumG;
    gainB = static_cast<FLType>(sumMax) / sumB;
    src_offsetR = 0;
    src_offsetG = 0;
    src_offsetB = 0;
    dst_offsetR = 0;
    dst_offsetG = 0;
    dst_offsetB = 0;
}
