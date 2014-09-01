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
    
    // Apply transpose
    for (i = 0, j = 0; i < pcount; i++, j+=sw)
    {
        if (j > k) j -= k;
        output[i] = input[j];
    }

    // Change Plane info
    output.Width(sh);
    output.Height(sw);
    output.PixelCount(pcount);

    // Output
    return output;
}


void Transpose(float * output, const float * const input, const PCType sw, const PCType sh)
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