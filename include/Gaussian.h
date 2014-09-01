#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_


#include "Image_Type.h"


int Gaussian2D_IO(int argc, char ** argv);


Plane & Gaussian2D(Plane & output, const Plane & input, const double sigma = 1.0);
inline Plane Gaussian2D(const Plane & input, const double sigma = 1.0)
{
    Plane output(input, false);
    Gaussian2D(output, input, sigma);
    return output;
}
inline Frame_YUV Gaussian2D(const Frame_YUV & input, const double sigma = 1.0)
{
    Frame_YUV output(input, false);
    Gaussian2D(output.Y(), input.Y(), sigma);
    Gaussian2D(output.U(), input.U(), sigma);
    Gaussian2D(output.V(), input.V(), sigma);
    return output;
}
inline Frame_RGB Gaussian2D(const Frame_RGB & input, const double sigma = 1.0)
{
    Frame_RGB output(input, false);
    Gaussian2D(output.R(), input.R(), sigma);
    Gaussian2D(output.G(), input.G(), sigma);
    Gaussian2D(output.B(), input.B(), sigma);
    return output;
}


void Recursive_Gaussian_Parameters(const double sigma, double & B, double & B1, double & B2, double & B3);
void Recursive_Gaussian2D_Vertical(FLType * data, const PCType sw, const PCType sh, const FLType B, const FLType B1, const FLType B2, const FLType B3);
void Recursive_Gaussian2D_Horizontal(FLType * data, const PCType sw, const PCType sh, const FLType B, const FLType B1, const FLType B2, const FLType B3);


#endif