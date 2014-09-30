#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_


#include <cmath>
#include "Image_Type.h"


const double Pi = 3.1415926535897932384626433832795;
const double sqrt_2Pi = sqrt(2 * Pi);

const struct Gaussian2D_Default {
    const double sigma = 2.0;
} Gaussian2D_Default;


int Gaussian2D_IO(const int argc, const std::string * args);


Plane & Gaussian2D(Plane & output, const Plane & input, const double sigma = Gaussian2D_Default.sigma);
inline Plane Gaussian2D(const Plane & input, const double sigma = Gaussian2D_Default.sigma)
{
    Plane output(input, false);
    return Gaussian2D(output, input, sigma);
}
inline Frame Gaussian2D(const Frame & input, const double sigma = Gaussian2D_Default.sigma)
{
    Frame output(input, false);
    for (Frame::PlaneCountType i = 0; i < input.PlaneCount(); i++)
        Gaussian2D(output.P(i), input.P(i), sigma);
    return output;
}


void Recursive_Gaussian_Parameters(const double sigma, double & B, double & B1, double & B2, double & B3);
void Recursive_Gaussian2D_Vertical(Plane_FL & output, const Plane_FL & input, const FLType B, const FLType B1, const FLType B2, const FLType B3);
void Recursive_Gaussian2D_Horizontal(Plane_FL & output, const Plane_FL & input, const FLType B, const FLType B1, const FLType B2, const FLType B3);
inline void Recursive_Gaussian2D_Vertical(Plane_FL & data, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    Recursive_Gaussian2D_Vertical(data, data, B, B1, B2, B3);
}
inline void Recursive_Gaussian2D_Horizontal(Plane_FL & data, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    Recursive_Gaussian2D_Horizontal(data, data, B, B1, B2, B3);
}


inline FLType Gaussian_Function(FLType x, FLType sigma)
{
    x /= sigma;
    return std::exp(x*x / -2);
}

inline FLType Gaussian_Function_sqr_x(FLType sqr_x, FLType sigma)
{
    return std::exp(sqr_x / (sigma*sigma*-2));
}

inline FLType Normalized_Gaussian_Function(FLType x, FLType sigma)
{
    x /= sigma;
    return std::exp(x*x / -2) / (sqrt_2Pi*sigma);
}

inline FLType Normalized_Gaussian_Function_sqr_x(FLType sqr_x, FLType sigma)
{
    return std::exp(sqr_x / (sigma*sigma*-2)) / (sqrt_2Pi*sigma);
}


#endif