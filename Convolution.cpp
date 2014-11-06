#include <cstring>
#include "Convolution.h"

 
Plane & Convolution3V(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, bool norm)
{
    PCType i0, i1, i2, j, upper;
    FLType P0, P1, P2;
    FLType R;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    FLType FloorFL = static_cast<FLType>(dst.Floor());
    FLType CeilFL = static_cast<FLType>(dst.Ceil());

    FLType sum = K0 + K1 + K2;
    bool absVal = false;
    bool clip = false;

    if (sum == 0)
    {
        sum = (Abs(K0) + Abs(K1) + Abs(K2)) / 2;

        if (K0 < 0 || K1 < 0 || K2 < 0)
        {
            absVal = true;
        }
    }
    else if (K0 < 0 || K1 < 0 || K2 < 0)
    {
        clip = true;
    }

    if (norm)
    {
        if (sum != 0)
        {
            K0 /= sum;
            K1 /= sum;
            K2 /= sum;
        }
        else
        {
            norm = false;
            clip = true;
        }
    }
    else if (sum > 1)
    {
        clip = true;
    }

    for (j = 0; j < height; j++)
    {
        i1 = stride * j;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = static_cast<FLType>(src[i0]);
            P1 = static_cast<FLType>(src[i1]);
            P2 = static_cast<FLType>(src[i2]);

            R = K0 * P0 + K1 * P1 + K2 * P2;
            if (absVal) R = Abs(R);
            if (clip) R = Clip(R, FloorFL, CeilFL);
            dst[i1] = static_cast<DType>(R + FLType(0.5));
        }
    }

    return dst;
}


Plane & Convolution3H(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, bool norm)
{
    PCType i, j, upper;
    FLType P0, P1, P2;
    FLType R;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    FLType FloorFL = static_cast<FLType>(dst.Floor());
    FLType CeilFL = static_cast<FLType>(dst.Ceil());

    FLType sum = K0 + K1 + K2;
    bool absVal = false;
    bool clip = false;

    if (sum == 0)
    {
        sum = (Abs(K0) + Abs(K1) + Abs(K2)) / 2;

        if (K0 < 0 || K1 < 0 || K2 < 0)
        {
            absVal = true;
        }
    }
    else if (K0 < 0 || K1 < 0 || K2 < 0)
    {
        clip = true;
    }

    if (norm)
    {
        if (sum != 0)
        {
            K0 /= sum;
            K1 /= sum;
            K2 /= sum;
        }
        else
        {
            norm = false;
            clip = true;
        }
    }
    else if (sum > 1)
    {
        clip = true;
    }

    for (j = 0; j < height; j++)
    {
        i = stride * j + radius;

        P0 = P1 = P2 = static_cast<FLType>(src[i - radius]);

        for (upper = stride * j + width; i < upper; i++)
        {
            P0 = P1;
            P1 = P2;
            P2 = static_cast<FLType>(src[i]);

            R = K0 * P0 + K1 * P1 + K2 * P2;
            if (absVal) R = Abs(R);
            if (clip) R = Clip(R, FloorFL, CeilFL);
            dst[i - radius] = static_cast<DType>(R + FLType(0.5));
        }

        for (upper = stride * j + width + radius; i < upper; i++)
        {
            P0 = P1;
            P1 = P2;

            R = K0 * P0 + K1 * P1 + K2 * P2;
            if (absVal) R = Abs(R);
            if (clip) R = Clip(R, FloorFL, CeilFL);
            dst[i - radius] = static_cast<DType>(R + FLType(0.5));
        }
    }

    return dst;
}


Plane & Convolution3(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, FLType K3, FLType K4, FLType K5, FLType K6, FLType K7, FLType K8, bool norm)
{
    PCType i0, i1, i2, j, upper;
    FLType P0, P1, P2, P3, P4, P5, P6, P7, P8;
    FLType R;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    FLType FloorFL = static_cast<FLType>(dst.Floor());
    FLType CeilFL = static_cast<FLType>(dst.Ceil());

    FLType sum = K0 + K1 + K2 + K3 + K4 + K5 + K6 + K7 + K8;
    bool absVal = false;
    bool clip = false;

    if (sum == 0)
    {
        sum = (Abs(K0) + Abs(K1) + Abs(K2) + Abs(K3) + Abs(K4) + Abs(K5) + Abs(K6) + Abs(K7) + Abs(K8)) / 2;

        if (K0 < 0 || K1 < 0 || K2 < 0 || K3 < 0 || K4 < 0 || K5 < 0 || K6 < 0 || K7 < 0 || K8 < 0)
        {
            absVal = true;
        }
    }
    else if (K0 < 0 || K1 < 0 || K2 < 0 || K3 < 0 || K4 < 0 || K5 < 0 || K6 < 0 || K7 < 0 || K8 < 0)
    {
        clip = true;
    }

    if (norm)
    {
        if (sum != 0)
        {
            K0 /= sum;
            K1 /= sum;
            K2 /= sum;
            K3 /= sum;
            K4 /= sum;
            K5 /= sum;
            K6 /= sum;
            K7 /= sum;
            K8 /= sum;
        }
        else
        {
            norm = false;
            clip = true;
        }
    }
    else if (sum > 1)
    {
        clip = true;
    }

    for (j = 0; j < height; j++)
    {
        i1 = stride * j + radius;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        P0 = P1 = P2 = static_cast<FLType>(src[i0 - radius]);
        P3 = P4 = P5 = static_cast<FLType>(src[i1 - radius]);
        P6 = P7 = P8 = static_cast<FLType>(src[i2 - radius]);

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2; P2 = static_cast<FLType>(src[i0]);
            P3 = P4; P4 = P5; P5 = static_cast<FLType>(src[i1]);
            P6 = P7; P7 = P8; P8 = static_cast<FLType>(src[i2]);

            R = K0 * P0 + K1 * P1 + K2 * P2 + K3 * P3 + K4 * P4 + K5 * P5 + K6 * P6 + K7 * P7 + K8 * P8;
            if (absVal) R = Abs(R);
            if (clip) R = Clip(R, FloorFL, CeilFL);
            dst[i1 - radius] = static_cast<DType>(R + FLType(0.5));
        }

        for (upper = stride * j + width + radius; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2;
            P3 = P4; P4 = P5;
            P6 = P7; P7 = P8;

            R = K0 * P0 + K1 * P1 + K2 * P2 + K3 * P3 + K4 * P4 + K5 * P5 + K6 * P6 + K7 * P7 + K8 * P8;
            if (absVal) R = Abs(R);
            if (clip) R = Clip(R, FloorFL, CeilFL);
            dst[i1 - radius] = static_cast<DType>(R + FLType(0.5));
        }
    }

    return dst;
}


Plane & FirstOrderDerivative3(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, FLType K6, FLType K7, FLType K8, bool norm)
{
    PCType i0, i1, i2, j, upper;
    FLType P0, P1, P2, P3, P4, P5, P6, P7, P8;
    FLType R, R0, R1;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    FLType FloorFL = static_cast<FLType>(dst.Floor());
    FLType CeilFL = static_cast<FLType>(dst.Ceil());

    FLType sum = (K0 + K1 + K2 + K6 + K7 + K8) * 2;
    bool absVal = false;
    bool clip = false;

    if (sum == 0)
    {
        sum = Abs(K0) + Abs(K1) + Abs(K2) + Abs(K6) + Abs(K7) + Abs(K8);

        if (K0 < 0 || K1 < 0 || K2 < 0 || K6 < 0 || K7 < 0 || K8 < 0)
        {
            absVal = true;
        }
    }
    else if (K0 < 0 || K1 < 0 || K2 < 0 || K6 < 0 || K7 < 0 || K8 < 0)
    {
        clip = true;
    }

    if (norm)
    {
        if (sum != 0)
        {
            K0 /= sum;
            K1 /= sum;
            K2 /= sum;
            K6 /= sum;
            K7 /= sum;
            K8 /= sum;
        }
        else
        {
            norm = false;
            clip = true;
        }
    }
    else if (sum > 1)
    {
        clip = true;
    }

    for (j = 0; j < height; j++)
    {
        i1 = stride * j + radius;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        P0 = P1 = P2 = static_cast<FLType>(src[i0 - radius]);
        P3 = P4 = P5 = static_cast<FLType>(src[i1 - radius]);
        P6 = P7 = P8 = static_cast<FLType>(src[i2 - radius]);

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2; P2 = static_cast<FLType>(src[i0]);
            P3 = P4; P4 = P5; P5 = static_cast<FLType>(src[i1]);
            P6 = P7; P7 = P8; P8 = static_cast<FLType>(src[i2]);

            R = K0 * P0 + K8 * P8;
            R0 = R + K1 * P1 + K2 * P2 + K6 * P6 + K7 * P7;
            R1 = R + K6 * P2 + K1 * P3 + K7 * P5 + K2 * P6;
            if (absVal) R = Abs(R0) + Abs(R1);
            else R = R0 + R1;
            if (clip) R = Clip(R, FloorFL, CeilFL);
            dst[i1 - radius] = static_cast<DType>(R + FLType(0.5));
        }

        for (upper = stride * j + width + radius; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2;
            P3 = P4; P4 = P5;
            P6 = P7; P7 = P8;

            R = K0 * P0 + K8 * P8;
            R0 = R + K1 * P1 + K2 * P2 + K6 * P6 + K7 * P7;
            R1 = R + K6 * P2 + K1 * P3 + K7 * P5 + K2 * P6;
            if (absVal) R = Abs(R0) + Abs(R1);
            else R = R0 + R1;
            if (clip) R = Clip(R, FloorFL, CeilFL);
            dst[i1 - radius] = static_cast<DType>(R + FLType(0.5));
        }
    }

    return dst;
}


Plane & EdgeDetect_Sobel(Plane &dst, const Plane &src)
{
    PCType i0, i1, i2, j, upper;
    sint32 P0, P1, P2, P3, P4, P5, P6, P7, P8;
    sint32 R, R0, R1;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    dst.ReQuantize(dst.BitDepth(), QuantRange::PC, false);
    sint32 Ceil = static_cast<sint32>(dst.Ceil());

    for (j = 0; j < height; j++)
    {
        i1 = stride * j + radius;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        P0 = P1 = P2 = static_cast<sint32>(src[i0 - radius]);
        P3 = P4 = P5 = static_cast<sint32>(src[i1 - radius]);
        P6 = P7 = P8 = static_cast<sint32>(src[i2 - radius]);

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2; P2 = static_cast<sint32>(src[i0]);
            P3 = P4; P4 = P5; P5 = static_cast<sint32>(src[i1]);
            P6 = P7; P7 = P8; P8 = static_cast<sint32>(src[i2]);

            R = P8 - P0;
            R0 = R + P6 + 2 * (P7 - P1) - P2;
            R1 = R + P2 + 2 * (P5 - P3) - P6;
            R = Round_Div(Abs(R0) + Abs(R1), sint32(8));
            dst[i1 - radius] = static_cast<DType>(R);
        }

        for (upper = stride * j + width + radius; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2;
            P3 = P4; P4 = P5;
            P6 = P7; P7 = P8;

            R = P8 - P0;
            R0 = R + P6 + 2 * (P7 - P1) - P2;
            R1 = R + P2 + 2 * (P5 - P3) - P6;
            R = Round_Div(Abs(R0) + Abs(R1), sint32(8));
            dst[i1 - radius] = static_cast<DType>(R);
        }
    }

    return dst;
}


Plane & EdgeDetect_Prewitt(Plane &dst, const Plane &src)
{
    PCType i0, i1, i2, j, upper;
    sint32 P0, P1, P2, P3, P4, P5, P6, P7, P8;
    sint32 R, R0, R1;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    dst.ReQuantize(dst.BitDepth(), QuantRange::PC, false);
    sint32 Ceil = static_cast<sint32>(dst.Ceil());

    for (j = 0; j < height; j++)
    {
        i1 = stride * j + radius;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        P0 = P1 = P2 = static_cast<sint32>(src[i0 - radius]);
        P3 = P4 = P5 = static_cast<sint32>(src[i1 - radius]);
        P6 = P7 = P8 = static_cast<sint32>(src[i2 - radius]);

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2; P2 = static_cast<sint32>(src[i0]);
            P3 = P4; P4 = P5; P5 = static_cast<sint32>(src[i1]);
            P6 = P7; P7 = P8; P8 = static_cast<sint32>(src[i2]);

            R = P8 - P0;
            R0 = R + P6 + P7 - P1 - P2;
            R1 = R + P2 + P5 - P3 - P6;
            R = Round_Div(Abs(R0) + Abs(R1), sint32(6));
            dst[i1 - radius] = static_cast<DType>(R);
        }

        for (upper = stride * j + width + radius; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2;
            P3 = P4; P4 = P5;
            P6 = P7; P7 = P8;

            R = P8 - P0;
            R0 = R + P6 + P7 - P1 - P2;
            R1 = R + P2 + P5 - P3 - P6;
            R = Round_Div(Abs(R0) + Abs(R1), sint32(6));
            dst[i1 - radius] = static_cast<DType>(R);
        }
    }

    return dst;
}


Plane & EdgeDetect_Laplace1(Plane &dst, const Plane &src)
{
    PCType i0, i1, i2, j, upper;
    sint32 P0, P1, P2, P3, P4, P5, P6, P7, P8;
    sint32 R;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    sint32 Floor = static_cast<sint32>(dst.Floor());
    sint32 Ceil = static_cast<sint32>(dst.Ceil());

    for (j = 0; j < height; j++)
    {
        i1 = stride * j + radius;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        P0 = P1 = P2 = static_cast<sint32>(src[i0 - radius]);
        P3 = P4 = P5 = static_cast<sint32>(src[i1 - radius]);
        P6 = P7 = P8 = static_cast<sint32>(src[i2 - radius]);

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2; P2 = static_cast<sint32>(src[i0]);
            P3 = P4; P4 = P5; P5 = static_cast<sint32>(src[i1]);
            P6 = P7; P7 = P8; P8 = static_cast<sint32>(src[i2]);

            R = 4 * P4 - (P1 + P3 + P5 + P7);
            R = Round_Div(Abs(R), sint32(4));
            dst[i1 - radius] = static_cast<DType>(R);
        }

        for (upper = stride * j + width + radius; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2;
            P3 = P4; P4 = P5;
            P6 = P7; P7 = P8;

            R = 4 * P4 - (P1 + P3 + P5 + P7);
            R = Round_Div(Abs(R), sint32(4));
            dst[i1 - radius] = static_cast<DType>(R);
        }
    }

    return dst;
}


Plane & EdgeDetect_Laplace2(Plane &dst, const Plane &src)
{
    PCType i0, i1, i2, j, upper;
    sint32 P0, P1, P2, P3, P4, P5, P6, P7, P8;
    sint32 R;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    sint32 Floor = static_cast<sint32>(dst.Floor());
    sint32 Ceil = static_cast<sint32>(dst.Ceil());

    for (j = 0; j < height; j++)
    {
        i1 = stride * j + radius;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        P0 = P1 = P2 = static_cast<sint32>(src[i0 - radius]);
        P3 = P4 = P5 = static_cast<sint32>(src[i1 - radius]);
        P6 = P7 = P8 = static_cast<sint32>(src[i2 - radius]);

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2; P2 = static_cast<sint32>(src[i0]);
            P3 = P4; P4 = P5; P5 = static_cast<sint32>(src[i1]);
            P6 = P7; P7 = P8; P8 = static_cast<sint32>(src[i2]);

            R = 8 * P4 - (P0 + P1 + P2 + P3 + P5 + P6 + P7 + P8);
            R = Round_Div(Abs(R), sint32(8));
            dst[i1 - radius] = static_cast<DType>(R);
        }

        for (upper = stride * j + width + radius; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2;
            P3 = P4; P4 = P5;
            P6 = P7; P7 = P8;

            R = 8 * P4 - (P0 + P1 + P2 + P3 + P5 + P6 + P7 + P8);
            R = Round_Div(Abs(R), sint32(8));
            dst[i1 - radius] = static_cast<DType>(R);
        }
    }

    return dst;
}


Plane & EdgeDetect_Laplace3(Plane &dst, const Plane &src)
{
    PCType i0, i1, i2, j, upper;
    sint32 P0, P1, P2, P3, P4, P5, P6, P7, P8;
    sint32 R;

    const PCType radius = 1;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();

    sint32 Floor = static_cast<sint32>(dst.Floor());
    sint32 Ceil = static_cast<sint32>(dst.Ceil());

    for (j = 0; j < height; j++)
    {
        i1 = stride * j + radius;
        i0 = j < 1 ? i1 : i1 - stride;
        i2 = j >= height - 1 ? i1 : i1 + stride;

        P0 = P1 = P2 = static_cast<sint32>(src[i0 - radius]);
        P3 = P4 = P5 = static_cast<sint32>(src[i1 - radius]);
        P6 = P7 = P8 = static_cast<sint32>(src[i2 - radius]);

        for (upper = stride * j + width; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2; P2 = static_cast<sint32>(src[i0]);
            P3 = P4; P4 = P5; P5 = static_cast<sint32>(src[i1]);
            P6 = P7; P7 = P8; P8 = static_cast<sint32>(src[i2]);

            R = 12 * P4 - 2 * (P1 + P3 + P5 + P7) - (P0 + P2 + P6 + P8);
            R = Round_Div(Abs(R), sint32(12));
            dst[i1 - radius] = static_cast<DType>(R);
        }

        for (upper = stride * j + width + radius; i1 < upper; i0++, i1++, i2++)
        {
            P0 = P1; P1 = P2;
            P3 = P4; P4 = P5;
            P6 = P7; P7 = P8;

            R = 12 * P4 - 2 * (P1 + P3 + P5 + P7) - (P0 + P2 + P6 + P8);
            R = Round_Div(Abs(R), sint32(12));
            dst[i1 - radius] = static_cast<DType>(R);
        }
    }

    return dst;
}


Plane & EdgeDetect(Plane &dst, const Plane &src, EdgeKernel Kernel)
{
    switch (Kernel)
    {
    case EdgeKernel::Sobel:
        EdgeDetect_Sobel(dst, src);
        break;
    case EdgeKernel::Prewitt:
        EdgeDetect_Prewitt(dst, src);
        break;
    case EdgeKernel::Laplace1:
        EdgeDetect_Laplace1(dst, src);
        break;
    case EdgeKernel::Laplace2:
        EdgeDetect_Laplace2(dst, src);
        break;
    case EdgeKernel::Laplace3:
        EdgeDetect_Laplace3(dst, src);
        break;
    }

    return dst;
}
