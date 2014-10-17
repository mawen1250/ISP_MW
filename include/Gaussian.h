#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_


#include <cmath>
#include "IO.h"
#include "Image_Type.h"
#include "LUT.h"


const double Pi = 3.1415926535897932384626433832795;
const double sqrt_2Pi = sqrt(2 * Pi);

const double sigmaSMul = 2.;
const double sigmaRMul = sizeof(FLType) < 8 ? 8. : 32.; // 8. when FLType is float, 32. when FLType is double


const struct Gaussian2D_Para {
    double sigma = 3.0;
} Gaussian2D_Default;


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


class Gaussian2D_IO
    : public FilterIO
{
protected:
    double sigma = Gaussian2D_Default.sigma;

    virtual void arguments_process()
    {
        FilterIO::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, sigma);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        ArgsObj.Check();
    }

    virtual Frame processFrame(const Frame &src)
    {
        return Gaussian2D(src, sigma);
    }

public:
    Gaussian2D_IO(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".Gaussian")
        : FilterIO(_argc, _args, _Tag) {}

    ~Gaussian2D_IO() {}
};


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


inline double Gaussian_Function(double x, double sigma)
{
    x /= sigma;
    return exp(x*x / -2);
}

inline double Gaussian_Function_sqr_x(double sqr_x, double sigma)
{
    return exp(sqr_x / (sigma*sigma*-2));
}

inline double Normalized_Gaussian_Function(double x, double sigma)
{
    x /= sigma;
    return exp(x*x / -2) / (sqrt_2Pi*sigma);
}

inline double Normalized_Gaussian_Function_sqr_x(double sqr_x, double sigma)
{
    return exp(sqr_x / (sigma*sigma*-2)) / (sqrt_2Pi*sigma);
}


inline LUT<FLType> Gaussian_Function_Spatial_LUT_Generation(const PCType xUpper, const PCType yUpper, const double sigmaS)
{
    PCType x, y;
    LUT<FLType> GS_LUT(xUpper*yUpper);

    for (y = 0; y < yUpper; y++)
    {
        for (x = 0; x < xUpper; x++)
        {
            GS_LUT[y*xUpper + x] = static_cast<FLType>(Gaussian_Function_sqr_x(static_cast<double>(x*x + y*y), sigmaS));
        }
    }

    return GS_LUT;
}

inline LUT<FLType> Gaussian_Function_Range_LUT_Generation(const DType ValueRange, const double sigmaR)
{
    DType i;
    DType Levels = ValueRange + 1;
    const DType upper = Min(ValueRange, static_cast<DType>(sigmaR*sigmaRMul*ValueRange + 0.5));
    LUT<FLType> GR_LUT(Levels);

    for (i = 0; i <= upper; i++)
    {
        GR_LUT[i] = static_cast<FLType>(Normalized_Gaussian_Function(static_cast<double>(i) / ValueRange, sigmaR));
    }
    // For unknown reason, when more range weights are too small or equal 0, the runtime speed gets lower - mainly in function Recursive_Gaussian2D_Horizontal.
    // To avoid this issue, we set range weights whose range values are larger than sigmaR*sigmaRMul to the Gaussian function value at sigmaR*sigmaRMul.
    if (i < Levels)
    {
        const FLType upperLUTvalue = GR_LUT[upper];
        for (; i < Levels; i++)
        {
            GR_LUT[i] = upperLUTvalue;
        }
    }

    return GR_LUT;
}

inline FLType Gaussian_Distribution2D_Spatial_LUT_Lookup(const LUT<FLType> & GS_LUT, const PCType xUpper, const PCType x, const PCType y)
{
    return GS_LUT[y*xUpper + x];
}

inline FLType Gaussian_Distribution2D_Range_LUT_Lookup(const LUT<FLType> & GR_LUT, const DType Value1, const DType Value2)
{
    return GR_LUT[Value1 > Value2 ? Value1 - Value2 : Value2 - Value1];
}


#endif