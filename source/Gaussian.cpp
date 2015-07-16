#define ENABLE_PPL


#include "Gaussian.h"
#include "Conversion.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


const Gaussian2D_Para Gaussian2D_Default;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Public functions of class Gaussian2D


Plane &Gaussian2D::process_Plane(Plane &dst, const Plane &src)
{
    if (para.sigma <= 0)
    {
        dst = src;
        return dst;
    }

    Plane_FL data(src);
    RecursiveGaussian GFilter(para.sigma, true);
    
    GFilter(data, data);
    RangeConvert(dst, data, true);
    
    return dst;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Public functions of class RecursiveGaussian


void RecursiveGaussian::setPara(ldbl sigma, bool _allow_negative)
{
    allow_negative = _allow_negative;

    // Constants
    const ldbl max_sigma_iter = sizeof(FLType) <= 4 ? 80.0L : 100000000.0L;
    const int max_iter = 25;
    const ldbl const1 = 1.36L;

    // Iteration parameters
    ldbl sigma_iter = sigma;

    if (sigma_iter > max_sigma_iter)
    {
        for (iter = 1; iter < max_iter;)
        {
            ++iter;
            sigma_iter = const1 * sigma / iter;
            // It should be (sigma / sqrt(iter)) mathematically, but this approximates better in my tests

            if (sigma_iter <= max_sigma_iter)
            {
                break;
            }
        }

        if (sigma_iter > max_sigma_iter)
        {
            sigma_iter = max_sigma_iter;
            ldbl actual_sigma = iter * sigma_iter / const1;

            std::cerr << "RecursiveGaussian: warning!\n"
                << "    The implemented sigma is " << actual_sigma << " instead of given sigma (" << sigma
                << ") due to the limitation of float point precision and iterative times!\n"
                << "    sigma per iteration is " << sigma_iter << ", iterative times is " << iter << ".\n";
        }
    }

    // Recursive Gaussian parameters
    const ldbl q = sigma_iter < 2.5L ? 3.97156L - 4.14554L * sqrt(1L - 0.26891L * sigma_iter) : 0.98711L * sigma_iter - 0.96330L;

    const ldbl b0 = 1.57825L + 2.44413L*q + 1.4281L*q*q + 0.422205L*q*q*q;
    const ldbl b1 = 2.44413L*q + 2.85619L*q*q + 1.26661L*q*q*q;
    const ldbl b2 = -(1.4281L*q*q + 1.26661L*q*q*q);
    const ldbl b3 = 0.422205L*q*q*q;
    
    B = static_cast<FLType>(1 - (b1 + b2 + b3) / b0);
    B1 = static_cast<FLType>(b1 / b0);
    B2 = static_cast<FLType>(b2 / b0);
    B3 = static_cast<FLType>(b3 / b0);
}


void RecursiveGaussian::filterV(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    _Dt d;
    d.Init(dst, src, height, width, stride);

    filterV_Kernel(d.dst_data, d.src_data, d.height, d.width, d.stride);

    for (int i = 1; i < iter; ++i)
    {
        filterV_Kernel(d.dst_data, d.dst_data, d.height, d.width, d.stride);
    }

    d.End();
}

void RecursiveGaussian::filterH(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    _Dt d;
    d.Init(dst, src, height, width, stride);

    filterH_Kernel(d.dst_data, d.src_data, d.height, d.width, d.stride);

    for (int i = 1; i < iter; ++i)
    {
        filterH_Kernel(d.dst_data, d.dst_data, d.height, d.width, d.stride);
    }

    d.End();
}

void RecursiveGaussian::filter(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride)
{
    _Dt d;
    d.Init(dst, src, height, width, stride);

    filterH_Kernel(d.dst_data, d.src_data, d.height, d.width, d.stride);
    filterV_Kernel(d.dst_data, d.dst_data, d.height, d.width, d.stride);

    for (int i = 1; i < iter; ++i)
    {
        filterH_Kernel(d.dst_data, d.dst_data, d.height, d.width, d.stride);
        filterV_Kernel(d.dst_data, d.dst_data, d.height, d.width, d.stride);
    }

    d.End();
}


void RecursiveGaussian::filterV(Plane_FL &dst, const Plane_FL &src)
{
    filterV(dst.data(), src.data(), dst.Height(), dst.Width(), dst.Stride());
}

void RecursiveGaussian::filterH(Plane_FL &dst, const Plane_FL &src)
{
    filterH(dst.data(), src.data(), dst.Height(), dst.Width(), dst.Stride());
}

void RecursiveGaussian::filter(Plane_FL &dst, const Plane_FL &src)
{
    filter(dst.data(), src.data(), dst.Height(), dst.Width(), dst.Stride());
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Protected functions of class RecursiveGaussian


void RecursiveGaussian::filterV_Kernel(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride) const
{
    if (dst != src)
    {
        memcpy(dst, src, sizeof(FLType) * width);
    }

    LOOP_H_PPL(height, width, stride, [&](const PCType j, const PCType lower, const PCType upper)
    {
        PCType i0 = lower;
        PCType i1 = j < 1 ? i0 : i0 - stride;
        PCType i2 = j < 2 ? i1 : i1 - stride;
        PCType i3 = j < 3 ? i2 : i2 - stride;

        for (; i0 < upper; ++i0, ++i1, ++i2, ++i3)
        {
            dst[i0] = B * src[i0] + B1 * dst[i1] + B2 * dst[i2] + B3 * dst[i3];
        }
    });

    LOOP_Hinv_PPL(height, width, stride, [&](const PCType j, const PCType lower, const PCType upper)
    {
        PCType i0 = lower;
        PCType i1 = j >= height - 1 ? i0 : i0 + stride;
        PCType i2 = j >= height - 2 ? i1 : i1 + stride;
        PCType i3 = j >= height - 3 ? i2 : i2 + stride;

        for (; i0 < upper; ++i0, ++i1, ++i2, ++i3)
        {
            FLType res = B * dst[i0] + B1 * dst[i1] + B2 * dst[i2] + B3 * dst[i3];
            if (allow_negative || res >= 0) dst[i0] = res;
            else dst[i0] = 0;
        }
    });
}

void RecursiveGaussian::filterH_Kernel(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride) const
{
    LOOP_V_PPL(height, [&](const PCType j)
    {
        const PCType lower = j * stride;
        const PCType upper = lower + width - 1;

        PCType i = lower;
        FLType P0, P1, P2, P3;
        P3 = P2 = P1 = src[i];
        dst[i] = src[i];

        while (i < upper)
        {
            ++i;
            P0 = B * src[i] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            dst[i] = P0;
        }

        P3 = P2 = P1 = dst[i];

        while (i > lower)
        {
            --i;
            P0 = B * dst[i] + B1 * P1 + B2 * P2 + B3 * P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            if (allow_negative || P0 >= 0) dst[i] = P0;
            else dst[i] = 0;
        }
    });
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
