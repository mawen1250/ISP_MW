#ifndef BILATERAL_H_
#define BILATERAL_H_


#include <cmath>
#include "LUT.h"
#include "Image_Type.h"
#include "Type_Conv.h"


const double sigmaSMul = 2.;
const double sigmaRMul = 30.;

const struct Bilateral2D_Default {
    const double sigmaS = 2.0;
    const double sigmaR = 0.1;
    const DType PBFICnum = 8;
} Bilateral2D_Default;


int Bilateral2D_IO(const int argc, const std::string * args);


Plane & Bilateral2D(Plane & output, const Plane & input, const Plane & ref, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum);
inline Plane Bilateral2D(const Plane & input, const Plane & ref, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Plane output(input, false);
    return Bilateral2D(output, input, ref, sigmaS, sigmaR, PBFICnum);
}
inline Plane Bilateral2D(const Plane & input, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Plane output(input, false);
    return Bilateral2D(output, input, input, sigmaS, sigmaR, PBFICnum);
}
inline Frame Bilateral2D(const Frame & input, const Frame & ref, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Frame output(input, false);
    for (Frame::PlaneCountType i = 0; i < input.PlaneCount(); i++)
        Bilateral2D(output.P(i), input.P(i), ref.P(i), sigmaS, sigmaR, PBFICnum);
    return output;
}
inline Frame Bilateral2D(const Frame & input, const double sigmaS = Bilateral2D_Default.sigmaS,
    const double sigmaR = Bilateral2D_Default.sigmaR, const DType PBFICnum = Bilateral2D_Default.PBFICnum)
{
    Frame output(input, false);
    for (Frame::PlaneCountType i = 0; i < input.PlaneCount(); i++)
        Bilateral2D(output.P(i), input.P(i), input.P(i), sigmaS, sigmaR, PBFICnum);
    return output;
}


Plane & Bilateral2D_0(Plane & output, const Plane & input, const Plane & ref, const LUT<FLType> & GS_LUT, const LUT<FLType> & GR_LUT, const PCType radiusx, const PCType radiusy);
Plane & Bilateral2D_1(Plane & output, const Plane & input, const Plane & ref, const LUT<FLType> & GR_LUT,
    const double sigmaS = Bilateral2D_Default.sigmaS, const DType PBFICnum = Bilateral2D_Default.PBFICnum);


LUT<FLType> Gaussian_Function_Spatial_LUT_Generation(const PCType xUpper, const PCType yUpper, const double sigmaS = Bilateral2D_Default.sigmaS);
LUT<FLType> Gaussian_Function_Range_LUT_Generation(const DType ValueRange, const double sigmaR = Bilateral2D_Default.sigmaR);

inline FLType Gaussian_Distribution2D_Spatial_LUT_Lookup(const LUT<FLType> & GS_LUT, const PCType xUpper, const PCType x, const PCType y)
{
    return GS_LUT[Abs(y)*xUpper + Abs(x)];
}

inline FLType Gaussian_Distribution2D_Range_LUT_Lookup(const LUT<FLType> & GR_LUT, const DType Value1, const DType Value2)
{
    return GR_LUT[Value1 > Value2 ? Value1 - Value2 : Value2 - Value1];
}


#endif