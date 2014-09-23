#include <iostream>

#include "include\Transform.h"
#include "include\Image_Type.h"


Plane & Transpose(Plane & output, const Plane & input)
{
    PCType i, j, k;
    PCType sw = input.Width();
    PCType sh = input.Height();
    PCType pcount = input.PixelCount();
    k = pcount - 1;

    // Change Plane info
    output.ReSize(sw, sh);

    // Apply transpose
    for (i = 0, j = 0; i < pcount; i++, j+=sw)
    {
        if (j > k) j -= k;
        output[i] = input[j];
    }

    // Output
    return output;
}


Plane_FL & Transpose(Plane_FL & output, const Plane_FL & input)
{
    PCType i, j, k;
    PCType sw = input.Width();
    PCType sh = input.Height();
    PCType pcount = input.PixelCount();
    k = pcount - 1;

    // Change Plane info
    output.ReSize(sw, sh);

    // Apply transpose
    for (i = 0, j = 0; i < pcount; i++, j += sw)
    {
        if (j > k) j -= k;
        output[i] = input[j];
    }

    // Output
    return output;
}


void Transpose(FLType * output, const FLType * const input, const PCType sw, const PCType sh)
{
    PCType i, j, k;
    PCType pcount = sw*sh;
    k = pcount - 1;

    // Apply transpose
    for (i = 0, j = 0; i < pcount; i++, j += sw)
    {
        if (j > k) j -= k;
        output[i] = input[j];
    }
}