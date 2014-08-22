#ifndef BILATERAL_H_
#define BILATERAL_H_


#include <cmath>
#include "Image_Type.h"
#include "Type_Conv.h"


const double Pi = 3.1415926535897932384626433832795;
const double RangeSigmaMul = 3.;


Plane Bilateral2D(const Plane & input, const Plane & ref, const double sigmaS = 1.0, const double sigmaR = 0.1);
inline Frame_YUV Bilateral2D(const Frame_YUV & input, const Frame_YUV & ref, const double sigmaS = 1.0, const double sigmaR = 0.1)
{
    Frame_YUV output(input, false);
    output.Y() = Bilateral2D(input.Y(), ref.Y(), sigmaS, sigmaR);
    output.U() = Bilateral2D(input.U(), ref.U(), sigmaS, sigmaR);
    output.V() = Bilateral2D(input.V(), ref.V(), sigmaS, sigmaR);
    return output;
}
inline Frame_YUV Bilateral2D(const Frame_YUV & input, const double sigmaS = 1.0, const double sigmaR = 0.1)
{
    Frame_YUV output(input, false);
    output.Y() = Bilateral2D(input.Y(), input.Y(), sigmaS, sigmaR);
    output.U() = Bilateral2D(input.U(), input.U(), sigmaS, sigmaR);
    output.V() = Bilateral2D(input.V(), input.V(), sigmaS, sigmaR);
    return output;
}
inline Frame_RGB Bilateral2D(const Frame_RGB & input, const Frame_RGB & ref, const double sigmaS = 1.0, const double sigmaR = 0.1)
{
    Frame_RGB output(input, false);
    output.R() = Bilateral2D(input.R(), ref.R(), sigmaS, sigmaR);
    output.G() = Bilateral2D(input.G(), ref.G(), sigmaS, sigmaR);
    output.B() = Bilateral2D(input.B(), ref.B(), sigmaS, sigmaR);
    return output;
}
inline Frame_RGB Bilateral2D(const Frame_RGB & input, const double sigmaS = 1.0, const double sigmaR = 0.1)
{
    Frame_RGB output(input, false);
    output.R() = Bilateral2D(input.R(), input.R(), sigmaS, sigmaR);
    output.G() = Bilateral2D(input.G(), input.G(), sigmaS, sigmaR);
    output.B() = Bilateral2D(input.B(), input.B(), sigmaS, sigmaR);
    return output;
}

void Gaussian_Distribution2D_Spatial_LUT_Generation(float * GS_LUT, const PCType xUpper, const PCType yUpper, const double sigmaS);
void Gaussian_Distribution2D_Range_LUT_Generation(float * GR_LUT, const DType ValueRange, const double sigmaR);


inline double Gaussian_Distribution2D(double x, double sigma)
{
    sigma *= sigma * 2;

    return exp(-x*x / sigma) / (Pi*sigma);
}

inline double Gaussian_Distribution2D_x2(double x2, double sigma)
{
    sigma *= sigma * 2;

    return exp(-x2 / sigma) / (Pi*sigma);
}

inline float Gaussian_Distribution2D_Spatial_LUT_Lookup(const float * const GS_LUT, const PCType xUpper, const PCType x, const PCType y)
{
    return GS_LUT[Abs(y)*xUpper + Abs(x)];
}

inline float Gaussian_Distribution2D_Range_LUT_Lookup(const float * const GR_LUT, const DType Value1, const DType Value2)
{
    return GR_LUT[Value1 > Value2 ? Value1 - Value2 : Value2 - Value1];
}


#endif