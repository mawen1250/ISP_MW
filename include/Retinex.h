#ifndef RETINEX_H_
#define RETINEX_H_


#include <vector>
#include "Image_Type.h"
#include "Histogram.h"


const struct Retinex_Default {
    const double sigma = 100.0;
    const std::vector<double> sigmaVector;
    const double lower_thr = 0.0;
    const double upper_thr = 0.01;
    const Histogram<FLType>::BinType HistBins = 4096;

    Retinex_Default() : sigmaVector({ 15.0, 80.0, 250.0 }) {};
} Retinex_Default;


int Retinex_MSR_IO(const int argc, const std::string * args);


Plane & Retinex_SSR(Plane & output, const Plane & input, const double sigma = Retinex_Default.sigma,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr);

inline Plane Retinex_SSR(const Plane & input, const double sigma = Retinex_Default.sigma,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr)
{
    Plane output(input, false);
    return Retinex_SSR(output, input, sigma, lower_thr, upper_thr);
}
inline Frame Retinex_SSR(const Frame & input, const double sigma = Retinex_Default.sigma,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr)
{
    Frame output(input, false);
    for (Frame::PlaneCountType i = 0; i < input.PlaneCount(); i++)
        Retinex_SSR(output.P(i), input.P(i), sigma, lower_thr, upper_thr);
    return output;
}


Plane_FL Retinex_MSR(const Plane_FL & idata, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr);
Plane & Retinex_MSR(Plane & output, const Plane & input, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr);
Frame & Retinex_MSR(Frame & output, const Frame & input, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr);

inline Plane Retinex_MSR(const Plane & input, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr)
{
    Plane output(input, false);
    return Retinex_MSR(output, input, sigmaVector, lower_thr, upper_thr);
}
inline Frame Retinex_MSR(const Frame & input, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr)
{
    Frame output(input, false);
    return Retinex_MSR(output, input, sigmaVector, lower_thr, upper_thr);
}


#endif