#include "Gaussian.h"
#include "Transform.h"


// Implementation of recursive Gaussian algorithm from "Ian T. Young, Lucas J. van Vliet - Recursive implementation of the Gaussian filter"
Plane & Gaussian2D(Plane & output, const Plane & input, const double sigma)
{
    if (sigma <= 0)
    {
        output = input;
        return output;
    }

    FLType B, B1, B2, B3;
    Recursive_Gaussian_Parameters(sigma, B, B1, B2, B3);

    Plane_FL data(input);

    Recursive_Gaussian2D_Horizontal(data, B, B1, B2, B3);
    Recursive_Gaussian2D_Vertical(data, B, B1, B2, B3);
    
    output.From(data);
    
    return output;
}


void Recursive_Gaussian_Parameters(const double sigma, FLType & B, FLType & B1, FLType & B2, FLType & B3)
{
    const double q = sigma < 2.5 ? 3.97156 - 4.14554*sqrt(1 - 0.26891*sigma) : 0.98711*sigma - 0.96330;

    const double b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
    const double b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
    const double b2 = -(1.4281*q*q + 1.26661*q*q*q);
    const double b3 = 0.422205*q*q*q;

    B = static_cast<FLType>(1 - (b1 + b2 + b3) / b0);
    B1 = static_cast<FLType>(b1 / b0);
    B2 = static_cast<FLType>(b2 / b0);
    B3 = static_cast<FLType>(b3 / b0);
}

/*void Recursive_Gaussian2D_Vertical(Plane_FL & output, const Plane_FL & input, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    PCType i, j, lower, upper;
    PCType height = input.Height();
    PCType width = input.Width();
    PCType stride = input.Width();
    PCType pcount = stride * height;
    FLType P0, P1, P2, P3;

    for (j = 0; j < width; j++)
    {
        lower = j;
        upper = pcount;

        i = lower;
        output[i] = P3 = P2 = P1 = input[i];

        for (i += stride; i < upper; i += stride)
        {
            P0 = B*input[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        i -= stride;
        P3 = P2 = P1 = output[i];
        
        for (i -= stride; i >= lower; i -= stride)
        {
            P0 = B*output[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
    }
}*/
void Recursive_Gaussian2D_Vertical(Plane_FL & output, const Plane_FL & input, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    PCType i0, i1, i2, i3, j, lower, upper;
    PCType height = input.Height();
    PCType width = input.Width();
    PCType stride = input.Width();
    FLType P0, P1, P2, P3;

    if (output.Data() != input.Data())
    {
        memcpy(output.Data(), input.Data(), sizeof(FLType) * width);
    }

    for (j = 0; j < height; j++)
    {
        lower = stride * j;
        upper = lower + width;

        i0 = lower;
        i1 = j < 1 ? i0 : i0 - stride;
        i2 = j < 2 ? i1 : i1 - stride;
        i3 = j < 3 ? i2 : i2 - stride;

        for (; i0 < upper; i0++, i1++, i2++, i3++)
        {
            P3 = output[i3];
            P2 = output[i2];
            P1 = output[i1];
            P0 = input[i0];
            output[i0] = B*P0 + B1*P1 + B2*P2 + B3*P3;
        }
    }

    for (j = height - 1; j >= 0; j--)
    {
        lower = stride * j;
        upper = lower + width;

        i0 = lower;
        i1 = j >= height - 1 ? i0 : i0 + stride;
        i2 = j >= height - 2 ? i1 : i1 + stride;
        i3 = j >= height - 3 ? i2 : i2 + stride;

        for (; i0 < upper; i0++, i1++, i2++, i3++)
        {
            P3 = output[i3];
            P2 = output[i2];
            P1 = output[i1];
            P0 = output[i0];
            output[i0] = B*P0 + B1*P1 + B2*P2 + B3*P3;
        }
    }
}

void Recursive_Gaussian2D_Horizontal(Plane_FL & output, const Plane_FL & input, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    PCType i, j, lower, upper;
    PCType height = input.Height();
    PCType width = input.Width();
    PCType stride = input.Width();
    FLType P0, P1, P2, P3;
    
    for (j = 0; j < height; j++)
    {
        lower = stride * j;
        upper = lower + width;

        i = lower;
        output[i] = P3 = P2 = P1 = input[i];

        for (i++; i < upper; i++)
        {
            P0 = B*input[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
        
        i--;
        P3 = P2 = P1 = output[i];

        for (i--; i >= lower; i--)
        {
            P0 = B*output[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
    }
}
