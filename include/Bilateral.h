#ifndef BILATERAL_H_
#define BILATERAL_H_


#include <cmath>
#include "Image_Type.h"
#include "Type_Conv.h"


const double Pi = 3.1415926535897932384626433832795;
const double sigmaSMul = 2.;
const double sigmaRMul = 30.;

const struct Bilateral2D_Default {
    const double sigmaS = 2.0;
    const double sigmaR = 0.1;
    const DType PBFICnum = 8;
} Bilateral2D_Default;


int Bilateral2D_IO(int argc, char ** argv);


Plane & Bilateral2D(Plane & output, const Plane & input, const Plane & ref, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum);
inline Plane Bilateral2D(const Plane & input, const Plane & ref, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Plane output(input, false);
    Bilateral2D(output, input, ref, sigmaS, sigmaR, PBFICnum);
    return output;
}
inline Plane Bilateral2D(const Plane & input, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Plane output(input, false);
    Bilateral2D(output, input, input, sigmaS, sigmaR, PBFICnum);
    return output;
}
inline Frame_YUV Bilateral2D(const Frame_YUV & input, const Frame_YUV & ref, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Frame_YUV output(input, false);
    Bilateral2D(output.Y(), input.Y(), ref.Y(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.U(), input.U(), ref.U(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.V(), input.V(), ref.V(), sigmaS, sigmaR, PBFICnum);
    return output;
}
inline Frame_YUV Bilateral2D(const Frame_YUV & input, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Frame_YUV output(input, false);
    Bilateral2D(output.Y(), input.Y(), input.Y(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.U(), input.U(), input.U(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.V(), input.V(), input.V(), sigmaS, sigmaR, PBFICnum);
    return output;
}
inline Frame_RGB Bilateral2D(const Frame_RGB & input, const Frame_RGB & ref, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Frame_RGB output(input, false);
    Bilateral2D(output.R(), input.R(), ref.R(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.G(), input.G(), ref.G(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.B(), input.B(), ref.B(), sigmaS, sigmaR, PBFICnum);
    return output;
}
inline Frame_RGB Bilateral2D(const Frame_RGB & input, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Frame_RGB output(input, false);
    Bilateral2D(output.R(), input.R(), input.R(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.G(), input.G(), input.G(), sigmaS, sigmaR, PBFICnum);
    Bilateral2D(output.B(), input.B(), input.B(), sigmaS, sigmaR, PBFICnum);
    return output;
}


void Gaussian_Distribution2D_Spatial_LUT_Generation(FLType * GS_LUT, const PCType xUpper, const PCType yUpper, const double sigmaS);
void Gaussian_Distribution2D_Range_LUT_Generation(FLType * GR_LUT, const DType ValueRange, const double sigmaR = Bilateral2D_Default.sigmaR);

Plane & Bilateral2D_0(Plane & output, const Plane & input, const Plane & ref, FLType * GS_LUT, FLType * GR_LUT, const PCType radiusx, const PCType radiusy);
Plane & Bilateral2D_1(Plane & output, const Plane & input, const Plane & ref, FLType * GR_LUT,
    const double sigmaS = Bilateral2D_Default.sigmaS, const DType PBFICnum = Bilateral2D_Default.PBFICnum);


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

inline FLType Gaussian_Distribution2D_Spatial_LUT_Lookup(const FLType * const GS_LUT, const PCType xUpper, const PCType x, const PCType y)
{
    return GS_LUT[Abs(y)*xUpper + Abs(x)];
}

inline FLType Gaussian_Distribution2D_Range_LUT_Lookup(const FLType * const GR_LUT, const DType Value1, const DType Value2)
{
    return GR_LUT[Value1 > Value2 ? Value1 - Value2 : Value2 - Value1];
}


#endif