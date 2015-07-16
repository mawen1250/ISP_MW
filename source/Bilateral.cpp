#include <cmath>
#include "Bilateral.h"


Plane &Bilateral2D(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane)
{
    // Skip processing if either sigma is not positive
    if (d.process[plane] == 0)
    {
        dst = src;
        return dst;
    }

    bool joint = src.data() != ref.data();

    switch (d.algorithm[plane])
    {
    case 1:
        Bilateral2D_1(dst, src, ref, d, plane);
        break;
    case 2:
        if(joint)
            Bilateral2D_2(dst, src, ref, d, plane);
        else
            Bilateral2D_2(dst, src, d, plane);
        break;
    default:
        Bilateral2D_0(dst, src, ref, d, plane);
        break;
    }
    
    // output
    return dst;
}


// Implementation of cross/joint Bilateral filter with truncated spatial window
Plane &Bilateral2D_0(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane)
{
    int radiusx = d.radius0[plane];
    int radiusy = d.radius0[plane];
    int xUpper = radiusx + 1, yUpper = radiusy + 1;

    int index0, i, j, index1, x, y;
    FLType Weight, WeightSum;
    FLType Sum;

    const LUT<FLType> &GS_LUT = d.GS_LUT[plane];
    const LUT<FLType> &GR_LUT = d.GR_LUT[plane];

    index0 = radiusy*src.Width();
    memcpy(dst.data(), src.data(), index0 * sizeof(DType));
    for (j = radiusy; j < src.Height() - radiusy; ++j)
    {
        for (i = 0; i < radiusx; ++i, ++index0)
        {
            dst[index0] = src[index0];
        }
        for (; i < src.Width() - radiusx; ++i, ++index0)
        {
            WeightSum = 0;
            Sum = 0;
            for (y = -radiusy; y < yUpper; ++y)
            {
                index1 = index0 + y*src.Width() - radiusx;
                for (x = -radiusx; x < xUpper; ++x, ++index1)
                {
                    Weight = Gaussian_Distribution2D_Spatial_LUT_Lookup(GS_LUT, xUpper, Abs(x), Abs(y)) * Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, ref[index0], ref[index1]);
                    WeightSum += Weight;
                    Sum += src[index1] * Weight;
                }
            }
            dst[index0] = dst.Quantize(Sum / WeightSum);
        }
        for (; i < src.Width(); ++i, ++index0)
        {
            dst[index0] = src[index0];
        }
    }
    memcpy(dst.data() + index0, src.data() + index0, radiusy*src.Width()*sizeof(DType));

    return dst;
}


// Implementation of O(1) cross/joint Bilateral filter algorithm from "Qingxiong Yang, Kar-Han Tan, Narendra Ahuja - Real-Time O(1) Bilateral Filtering"
Plane &Bilateral2D_1(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane)
{
    int i, j, upper;
    int k;

    int height = ref.Height();
    int width = ref.Width();
    int stride = ref.Width();
    int pcount = ref.PixelCount();

    double sigmaS = d.sigmaS[plane];
    double sigmaR = d.sigmaR[plane];
    int PBFICnum = d.PBFICnum[plane];

    const LUT<FLType> &GR_LUT = d.GR_LUT[plane];

    // Value range of Plane "ref"
    DType rLower, rUpper, rRange;

    rLower = ref.Floor();
    rUpper = ref.Ceil();
    rRange = rUpper - rLower;

    // Generate quantized PBFICs' parameters
    DType * PBFICk = new DType[PBFICnum];

    for (k = 0; k < PBFICnum; ++k)
    {
        PBFICk[k] = static_cast<DType>(static_cast<double>(rRange)*k / (PBFICnum - 1) + rLower + 0.5);
    }

    // Generate recursive Gaussian filter object
    RecursiveGaussian GFilter(sigmaS, true);

    // Generate quantized PBFICs
    Plane_FL * PBFIC = new Plane_FL[PBFICnum];
    Plane_FL Wk(ref, false);
    Plane_FL Jk(ref, false);
    
    for (k = 0; k < PBFICnum; ++k)
    {
        PBFIC[k] = Plane_FL(ref, false);

        for (j = 0; j < height; ++j)
        {
            i = stride * j;
            for (upper = i + width; i < upper; ++i)
            {
                Wk[i] = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, PBFICk[k], ref[i]);
                Jk[i] = Wk[i] * src[i];
            }
        }

        GFilter(Wk, Wk);
        GFilter(Jk, Jk);

        for (j = 0; j < height; ++j)
        {
            i = stride * j;
            for (upper = i + width; i < upper; ++i)
            {
                PBFIC[k][i] = Wk[i] == 0 ? 0 : Jk[i] / Wk[i];
            }
        }
    }

    // Generate filtered result from PBFICs using linear interpolation
    for (j = 0; j < height; ++j)
    {
        i = stride * j;
        for (upper = i + width; i < upper; ++i)
        {
            for (k = 0; k < PBFICnum - 2; ++k)
            {
                if (ref[i] < PBFICk[k + 1] && ref[i] >= PBFICk[k]) break;
            }

            dst[i] = dst.Quantize(((PBFICk[k + 1] - ref[i])*PBFIC[k][i] + (ref[i] - PBFICk[k])*PBFIC[k + 1][i]) / (PBFICk[k + 1] - PBFICk[k]));
        }
    }

    // Clear and output
    delete[] PBFIC;
    delete[] PBFICk;

    return dst;
}


// copy data to buff
template < typename T >
void data2buff(T * dst, const T * src, int xoffset, int yoffset,
    int bufheight, int bufwidth, int bufstride, int height, int width, int stride)
{
    int x, y;
    T *dstp;
    const T *srcp;

    for (y = 0; y < height; ++y)
    {
        dstp = dst + (yoffset + y) * bufstride;
        srcp = src + y * stride;
        for (x = 0; x < xoffset; ++x)
            dstp[x] = srcp[0];
        memcpy(dstp + xoffset, srcp, sizeof(T)*width);
        for (x = xoffset + width; x < bufwidth; ++x)
            dstp[x] = srcp[width - 1];
    }

    srcp = dst + yoffset * bufstride;
    for (y = 0; y < yoffset; ++y)
    {
        dstp = dst + y * bufstride;
        memcpy(dstp, srcp, sizeof(T)*bufwidth);
    }

    srcp = dst + (yoffset + height - 1) * bufstride;
    for (y = yoffset + height; y < bufheight; ++y)
    {
        dstp = dst + y * bufstride;
        memcpy(dstp, srcp, sizeof(T)*bufwidth);
    }
}

// Implementation of cross/joint Bilateral filter with truncated spatial window and sub-sampling
Plane &Bilateral2D_2(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane)
{
    int i, j, x, y;

    int radiusx = d.radius[plane];
    int radiusy = d.radius[plane];
    int buffnum = radiusy * 2 + 1;
    int samplenum = d.samples[plane];
    int samplestep = d.step[plane];
    int samplecenter = buffnum / 2;

    int height = src.Height();
    int width = src.Width();
    int stride = src.Width();
    int bufheight = src.Height() + radiusy * 2;
    int bufwidth = src.Width() + radiusx * 2;
    int bufstride = src.Width() + radiusx * 2;

    const DType * srcp = src.data();
    const DType * refp = ref.data();
    DType * dstp = dst.data();

    const LUT<FLType> &GS_LUT = d.GS_LUT[plane];
    const LUT<FLType> &GR_LUT = d.GR_LUT[plane];

    // Allocate buffs
    DType *srcbuff = new DType[bufstride * bufheight];
    DType *refbuff = new DType[bufstride * bufheight];
    DType *srcbuffp1, *refbuffp1, *srcbuffp2, *refbuffp2;

    data2buff(srcbuff, srcp, radiusx, radiusy, bufheight, bufwidth, bufstride, height, width, stride);
    data2buff(refbuff, refp, radiusx, radiusy, bufheight, bufwidth, bufstride, height, width, stride);

    // Process
    FLType SWei, RWei1, RWei2, RWei3, RWei4, WeightSum, Sum;
    int xoffset, yoffset;
    const int xUpper = radiusx + 1, yUpper = radiusy + 1;

    yoffset = samplecenter;
    for (j = 0; j < height; ++j, srcp += stride, refp += stride, dstp += stride)
    {
        srcbuffp1 = srcbuff + (yoffset + j) * bufstride;
        refbuffp1 = refbuff + (yoffset + j) * bufstride;

        for (i = 0; i < width; ++i)
        {
            xoffset = samplecenter + i;
            srcbuffp2 = srcbuffp1 + xoffset;
            refbuffp2 = refbuffp1 + xoffset;

            WeightSum = GS_LUT[0] * GR_LUT[0];
            Sum = srcp[i] * WeightSum;

            for (y = 1; y < yUpper; y += samplestep)
            {
                for (x = 1; x < xUpper; x += samplestep)
                {
                    SWei = Gaussian_Distribution2D_Spatial_LUT_Lookup(GS_LUT, xUpper, x, y);
                    RWei1 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, refp[i], refbuffp2[+y*bufstride + x]);
                    RWei2 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, refp[i], refbuffp2[+y*bufstride - x]);
                    RWei3 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, refp[i], refbuffp2[-y*bufstride - x]);
                    RWei4 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, refp[i], refbuffp2[-y*bufstride + x]);

                    WeightSum += SWei * (RWei1 + RWei2 + RWei3 + RWei4);
                    Sum += SWei * (
                        srcbuffp2[+y*bufstride + x] * RWei1 +
                        srcbuffp2[+y*bufstride - x] * RWei2 +
                        srcbuffp2[-y*bufstride - x] * RWei3 +
                        srcbuffp2[-y*bufstride + x] * RWei4);
                }
            }

            dstp[i] = dst.Quantize(Sum / WeightSum);
        }
    }

    // Clear and output
    delete[] srcbuff;
    delete[] refbuff;

    return dst;
}

// Implementation of Bilateral filter with truncated spatial window and sub-sampling
Plane &Bilateral2D_2(Plane &dst, const Plane &src, const Bilateral2D_Data &d, int plane)
{
    int i, j, x, y;

    int radiusx = d.radius[plane];
    int radiusy = d.radius[plane];
    int buffnum = radiusy * 2 + 1;
    int samplenum = d.samples[plane];
    int samplestep = d.step[plane];
    int samplecenter = buffnum / 2;

    int height = src.Height();
    int width = src.Width();
    int stride = src.Width();
    int bufheight = src.Height() + radiusy * 2;
    int bufwidth = src.Width() + radiusx * 2;
    int bufstride = src.Width() + radiusx * 2;

    const DType * srcp = src.data();
    DType * dstp = dst.data();

    const LUT<FLType> &GS_LUT = d.GS_LUT[plane];
    const LUT<FLType> &GR_LUT = d.GR_LUT[plane];

    // Allocate buffs
    DType *srcbuff = new DType[bufstride * bufheight];
    DType *srcbuffp1, *srcbuffp2;

    data2buff(srcbuff, srcp, radiusx, radiusy, bufheight, bufwidth, bufstride, height, width, stride);

    // Process
    FLType SWei, RWei1, RWei2, RWei3, RWei4, WeightSum, Sum;
    int xoffset, yoffset;
    const int xUpper = radiusx + 1, yUpper = radiusy + 1;

    yoffset = samplecenter;
    for (j = 0; j < height; ++j, srcp += stride, dstp += stride)
    {
        srcbuffp1 = srcbuff + (yoffset + j) * bufstride;

        for (i = 0; i < width; ++i)
        {
            xoffset = samplecenter + i;
            srcbuffp2 = srcbuffp1 + xoffset;

            WeightSum = GS_LUT[0] * GR_LUT[0];
            Sum = srcp[i] * WeightSum;

            for (y = 1; y < yUpper; y += samplestep)
            {
                for (x = 1; x < xUpper; x += samplestep)
                {
                    SWei = Gaussian_Distribution2D_Spatial_LUT_Lookup(GS_LUT, xUpper, x, y);
                    RWei1 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, srcp[i], srcbuffp2[+y*bufstride + x]);
                    RWei2 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, srcp[i], srcbuffp2[+y*bufstride - x]);
                    RWei3 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, srcp[i], srcbuffp2[-y*bufstride - x]);
                    RWei4 = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, srcp[i], srcbuffp2[-y*bufstride + x]);

                    WeightSum += SWei * (RWei1 + RWei2 + RWei3 + RWei4);
                    Sum += SWei * (
                        srcbuffp2[+y*bufstride + x] * RWei1 +
                        srcbuffp2[+y*bufstride - x] * RWei2 +
                        srcbuffp2[-y*bufstride - x] * RWei3 +
                        srcbuffp2[-y*bufstride + x] * RWei4);
                }
            }

            dstp[i] = dst.Quantize(Sum / WeightSum);
        }
    }

    // Clear and output
    delete[] srcbuff;

    return dst;
}
