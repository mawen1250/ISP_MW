#ifndef TRANSFORM_H_
#define TRANSFORM_H_


#include "Image_Type.h"


Plane & Transpose(Plane & output, const Plane & input);
inline Plane & Transpose(Plane & input)
{
    Plane temp(input);
    Transpose(input, temp);
    return input;
}
inline Frame_YUV & Transpose(Frame_YUV & output, const Frame_YUV & input)
{
    Transpose(output.Y(), input.Y());
    Transpose(output.U(), input.U());
    Transpose(output.V(), input.V());
    return output;
}
inline Frame_YUV & Transpose(Frame_YUV & input)
{
    Frame_YUV temp(input);
    Transpose(input, temp);
    return input;
}
inline Frame_RGB & Transpose(Frame_RGB & output, const Frame_RGB & input)
{
    Transpose(output.R(), input.R());
    Transpose(output.G(), input.G());
    Transpose(output.B(), input.B());
    return output;
}
inline Frame_RGB & Transpose(Frame_RGB & input)
{
    Frame_RGB temp(input);
    Transpose(input, temp);
    return input;
}


void Transpose(float * output, const float * const input, const PCType sw, const PCType sh);


#endif