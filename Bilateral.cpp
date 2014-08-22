#include <iostream>
#include <cmath>
#include "include/Bilateral.h"
#include "include/Image_Type.h"


Plane Bilateral2D(const Plane & input, const Plane & ref, const double sigmaS, const double sigmaR)
{
    if (sigmaS <= 0 || sigmaR <= 0)
    {
        return input;
    }

    PCType index0, i, j, index1, x, y;
    float Weight, WeightSum;
    float Sum;

    const PCType radius = Max((PCType)(sigmaS*RangeSigmaMul), (PCType)1);
    const PCType radiusx = radius, radiusy = radius;
    const PCType xUpper = radiusx + 1, yUpper = radiusy + 1;

    Plane output(input, false);
    float * GS_LUT = new float[xUpper*yUpper];
    float * GR_LUT = new float[ref.ValueRange()];

    Gaussian_Distribution2D_Spatial_LUT_Generation(GS_LUT, xUpper, yUpper, sigmaS);
    Gaussian_Distribution2D_Range_LUT_Generation(GR_LUT, ref.ValueRange(), sigmaR*ref.ValueRange()/RangeSigmaMul);

    index0 = radiusy*input.Width();
    memcpy(output.Data(), input.Data(), index0*sizeof(DType));
    for (j = radiusy; j < input.Height() - radiusy; j++)
    {
        for (i = 0; i < radiusx; i++, index0++)
        {
            output[index0] = input[index0];
        }
        for (; i < input.Width() - radiusx; i++, index0++)
        {
            WeightSum = 0;
            Sum = 0;
            for (y = -radiusy; y < yUpper; y++)
            {
                index1 = index0 + y*input.Width() - radiusx;
                for (x = -radiusx; x < xUpper; x++, index1++)
                {
                    Weight = Gaussian_Distribution2D_Spatial_LUT_Lookup(GS_LUT, xUpper, x, y) * Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, ref[index0], ref[index1]);
                    WeightSum += Weight;
                    Sum += input[index1] * Weight;
                }
            }
            output[index0] = Quantize(Sum / WeightSum, output);
        }
        for (; i < input.Width(); i++, index0++)
        {
            output[index0] = input[index0];
        }
    }
    memcpy(output.Data() + index0, input.Data() + index0, radiusy*input.Width()*sizeof(DType));

    delete[] GS_LUT;
    delete[] GR_LUT;

    return output;
}


void Gaussian_Distribution2D_Spatial_LUT_Generation(float * GS_LUT, const PCType xUpper, const PCType yUpper, const double sigmaS)
{
    PCType x, y;

    for (y = 0; y < yUpper; y++)
    {
        for (x = 0; x < xUpper; x++)
        {
            GS_LUT[y*xUpper + x] = (float)Gaussian_Distribution2D_x2((double)(x*x + y*y), sigmaS);
        }
    }
}

void Gaussian_Distribution2D_Range_LUT_Generation(float * GR_LUT, const DType ValueRange, const double sigmaR)
{
    DType i;

    for (i = 0; i < ValueRange; i++)
    {
        GR_LUT[i] = (float)Gaussian_Distribution2D((double)i, sigmaR);
    }
}

