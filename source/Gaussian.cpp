#define ENABLE_PPL


#include "Gaussian.h"
#include "Conversion.hpp"


const Gaussian2D_Para Gaussian2D_Default;


// Public functions of class Gaussian2D
Plane &Gaussian2D::process_Plane(Plane &dst, const Plane &src)
{
    if (para.sigma <= 0)
    {
        dst = src;
        return dst;
    }

    Plane_FL data(src);
    RecursiveGaussian GFilter(para.sigma);
    
    GFilter.Filter(data);
    RangeConvert(dst, data, true);
    
    return dst;
}


// Public functions of class RecursiveGaussian
void RecursiveGaussian::GetPara(long double sigma)
{
    const long double q = sigma < 2.5 ? 3.97156 - 4.14554 * sqrt(1 - 0.26891 * sigma) : 0.98711 * sigma - 0.96330;

    const long double b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
    const long double b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
    const long double b2 = -(1.4281*q*q + 1.26661*q*q*q);
    const long double b3 = 0.422205*q*q*q;
    
    B = static_cast<FLType>(1 - (b1 + b2 + b3) / b0);
    B1 = static_cast<FLType>(b1 / b0);
    B2 = static_cast<FLType>(b2 / b0);
    B3 = static_cast<FLType>(b3 / b0);
}


void RecursiveGaussian::FilterV(Plane_FL &dst, const Plane_FL &src)
{
    d.Init(dst, src);

    FilterV_Kernel(d.dst_data, d.src_data);

    d.End();
}

void RecursiveGaussian::FilterH(Plane_FL &dst, const Plane_FL &src)
{
    d.Init(dst, src);

    FilterH_Kernel(d.dst_data, d.src_data);

    d.End();
}

void RecursiveGaussian::Filter(Plane_FL &dst, const Plane_FL &src)
{
    d.Init(dst, src);

    FilterH_Kernel(d.dst_data, d.src_data);
    FilterV_Kernel(d.dst_data, d.dst_data);

    d.End();
}

void RecursiveGaussian::Filter(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    d.Init(dst, src, height, width, stride);

    FilterH_Kernel(d.dst_data, d.src_data);
    FilterV_Kernel(d.dst_data, d.dst_data);

    d.End();
}


// Protected functions of class RecursiveGaussian
void RecursiveGaussian::FilterV_Kernel(FLType *dst, const FLType *src) const
{
    if (dst != src)
    {
        memcpy(dst, src, sizeof(FLType) * d.width);
    }

    LOOP_H_PPL(d.height, d.width, d.stride, [&](const PCType j, const PCType lower, const PCType upper)
    {
        PCType i0 = lower;
        PCType i1 = j < 1 ? i0 : i0 - d.stride;
        PCType i2 = j < 2 ? i1 : i1 - d.stride;
        PCType i3 = j < 3 ? i2 : i2 - d.stride;

        FLType P0, P1, P2, P3;

        for (; i0 < upper; ++i0, ++i1, ++i2, ++i3)
        {
            P3 = dst[i3];
            P2 = dst[i2];
            P1 = dst[i1];
            P0 = src[i0];
            dst[i0] = B * P0 + B1 * P1 + B2 * P2 + B3 * P3;
        }
    });

    LOOP_Hinv_PPL(d.height, d.width, d.stride, [&](const PCType j, const PCType lower, const PCType upper)
    {
        PCType i0 = lower;
        PCType i1 = j >= d.height - 1 ? i0 : i0 + d.stride;
        PCType i2 = j >= d.height - 2 ? i1 : i1 + d.stride;
        PCType i3 = j >= d.height - 3 ? i2 : i2 + d.stride;

        FLType P0, P1, P2, P3;

        for (; i0 < upper; ++i0, ++i1, ++i2, ++i3)
        {
            P3 = dst[i3];
            P2 = dst[i2];
            P1 = dst[i1];
            P0 = dst[i0];
            dst[i0] = B * P0 + B1 * P1 + B2 * P2 + B3 * P3;
        }
    });
}

void RecursiveGaussian::FilterH_Kernel(FLType *dst, const FLType *src) const
{
    LOOP_V_PPL(d.height, [&](const PCType j)
    {
        const PCType lower = j * d.stride;
        const PCType upper = lower + d.width - 1;

        PCType i = lower;
        FLType P0, P1, P2, P3;
        P3 = P2 = P1 = src[i];
        dst[i] = src[i];

        for (; i < upper;)
        {
            ++i;
            P0 = B * src[i] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            dst[i] = P0;
        }

        P3 = P2 = P1 = dst[i];

        for (; i > lower;)
        {
            --i;
            P0 = B * dst[i] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            dst[i] = P0;
        }
    });
}
