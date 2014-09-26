#ifndef TRANSFORM_H_
#define TRANSFORM_H_


#include "Image_Type.h"


Plane & Transpose(Plane & output, const Plane & input);
inline Plane & Transpose(Plane & input)
{
    Plane temp(input);
    return Transpose(input, temp);
}
inline Frame & Transpose(Frame & output, const Frame & input)
{
    for (Frame::PlaneCountType i = 0; i < input.PlaneCount(); i++)
        Transpose(output.P(i), input.P(i));
    return output;
}
inline Frame & Transpose(Frame & input)
{
    Frame temp(input);
    return Transpose(input, temp);
}


Plane_FL & Transpose(Plane_FL & output, const Plane_FL & input);
inline Plane_FL & Transpose(Plane_FL & input)
{
    Plane_FL temp(input);
    return Transpose(input, temp);
}


void Transpose(FLType * output, const FLType * const input, const PCType sw, const PCType sh);


#endif