#ifndef RETINEX_H_
#define RETINEX_H_


#include <vector>
#include "IO.h"
#include "Image_Type.h"
#include "Histogram.h"


const struct Retinex_Para {
    double sigma = 100.0;
    std::vector<double> sigmaVector;
    double lower_thr = 0.0;
    double upper_thr = 0.0;
    Histogram<FLType>::BinType HistBins = 4096;

    Retinex_Para() : sigmaVector({ 25.0, 80.0, 250.0 }) {};
} Retinex_Default;


int Retinex_MSRCP_IO(const int argc, const std::vector<std::string> &args);


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
Frame & Retinex_MSRCP(Frame & output, const Frame & input, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr);

inline Plane Retinex_MSR(const Plane & input, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr)
{
    Plane output(input, false);
    return Retinex_MSR(output, input, sigmaVector, lower_thr, upper_thr);
}
inline Frame Retinex_MSRCP(const Frame & input, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    const double lower_thr = Retinex_Default.lower_thr, const double upper_thr = Retinex_Default.upper_thr)
{
    Frame output(input, false);
    return Retinex_MSRCP(output, input, sigmaVector, lower_thr, upper_thr);
}


#endif