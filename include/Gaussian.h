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


const struct Gaussian2D_Para
{
    double sigma = 3.0;
} Gaussian2D_Default;


Plane &Gaussian2D(Plane &dst, const Plane &src, const double sigma = Gaussian2D_Default.sigma);
inline Plane Gaussian2D(const Plane &src, const double sigma = Gaussian2D_Default.sigma)
{
    Plane dst(src, false);
    return Gaussian2D(dst, src, sigma);
}
inline Frame Gaussian2D(const Frame &src, const double sigma = Gaussian2D_Default.sigma)
{
    Frame dst(src, false);
    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
        Gaussian2D(dst.P(i), src.P(i), sigma);
    return dst;
}


class Gaussian2D_IO
    : public FilterIO
{
public:
    typedef Gaussian2D_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    Gaussian2D_Para para = Gaussian2D_Default;

    virtual void arguments_process()
    {
        FilterIO::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, para.sigma);
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

    virtual Frame process(const Frame &src)
    {
        return Gaussian2D(src, para.sigma);
    }

public:
    _Myt(std::string _Tag = ".Gaussian")
        : _Mybase(std::move(_Tag)) {}
};


// Implementation of recursive Gaussian algorithm from "Ian T. Young, Lucas J. van Vliet - Recursive implementation of the Gaussian filter"
// For 32bit float type, the maximum sigma with valid result is about 120.
class RecursiveGaussian
{
public:
    typedef RecursiveGaussian _Myt;

protected:
    FLType B;
    FLType B1;
    FLType B2;
    FLType B3;

public:
    _Myt(long double sigma)
    {
        GetPara(sigma);
    }

    Plane_FL operator()(const Plane_FL &src)
    {
        Plane_FL dst(src, false);
        Filter(dst, src);
        return dst;
    }

    void GetPara(long double sigma);

    void FilterV(Plane_FL &dst, const Plane_FL &src);
    void FilterV(Plane_FL &data);

    void FilterH(Plane_FL &dst, const Plane_FL &src);
    void FilterH(Plane_FL &data);

    void Filter(Plane_FL &dst, const Plane_FL &src) { FilterH(dst, src); FilterV(dst); }
    void Filter(Plane_FL &data) { FilterH(data); FilterV(data); }
};


template < typename _Ty = FLType >
class GaussianFunction
{
public:
    typedef GaussianFunction<_Ty> _Myt;

protected:
    const _Ty const1_;

protected:
    _Ty BaseFunc(_Ty sqr_x)
    {
        return exp(sqr_x * const1_);
    }

public:
    _Myt(_Ty sigma)
        : const1_(_Ty(1) / (sigma * sigma * -2))
    {}

    _Ty operator()(_Ty sqr_x)
    {
        return BaseFunc(sqr_x);
    }
};

template < typename _Ty = FLType >
class NormalizedGaussianFunction
    : protected GaussianFunction<_Ty>
{
public:
    typedef NormalizedGaussianFunction<_Ty> _Myt;
    typedef GaussianFunction<_Ty> _Mybase;

protected:
    const _Ty const2_;

protected:
    _Ty BaseFunc(_Ty sqr_x)
    {
        return exp(sqr_x * const1_) * const2_;
    }

public:
    _Myt(_Ty sigma)
        : _Mybase(sigma), const2_(_Ty(1) / (_Ty(sqrt_2Pi) * sigma))
    {}

    _Ty operator()(_Ty sqr_x)
    {
        return BaseFunc(sqr_x);
    }
};

template < typename _Ty = FLType >
class GaussianFunctionX
    : protected GaussianFunction<_Ty>
{
public:
    typedef GaussianFunctionX<_Ty> _Myt;
    typedef GaussianFunction<_Ty> _Mybase;

public:
    _Myt(_Ty sigma)
        : _Mybase(sigma)
    {}

    _Ty operator()(_Ty x)
    {
        return BaseFunc(x * x);
    }
};

template < typename _Ty = FLType >
class NormalizedGaussianFunctionX
    : protected NormalizedGaussianFunction<_Ty>
{
public:
    typedef NormalizedGaussianFunctionX<_Ty> _Myt;
    typedef NormalizedGaussianFunction<_Ty> _Mybase;

public:
    _Myt(_Ty sigma)
        : _Mybase(sigma)
    {}

    _Ty operator()(_Ty x)
    {
        return BaseFunc(x * x);
    }
};


inline LUT<FLType> Gaussian_Function_Spatial_LUT_Generation(const PCType xUpper, const PCType yUpper, const double sigmaS)
{
    GaussianFunction<double> GFunc(sigmaS);

    LUT<FLType> GS_LUT(xUpper*yUpper);

    for (PCType y = 0; y < yUpper; y++)
    {
        for (PCType x = 0; x < xUpper; x++)
        {
            GS_LUT[y*xUpper + x] = static_cast<FLType>(GFunc(static_cast<double>(x*x + y*y)));
        }
    }

    return GS_LUT;
}

inline LUT<FLType> Gaussian_Function_Range_LUT_Generation(const DType ValueRange, const double sigmaR)
{
    NormalizedGaussianFunctionX<double> NGFuncX(sigmaR);

    DType Levels = ValueRange + 1;
    const DType upper = Min(ValueRange, static_cast<DType>(sigmaR*sigmaRMul*ValueRange + 0.5));
    LUT<FLType> GR_LUT(Levels);

    DType i = 0;
    for (; i <= upper; i++)
    {
        GR_LUT[i] = static_cast<FLType>(NGFuncX(static_cast<double>(i) / ValueRange));
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

inline FLType Gaussian_Distribution2D_Spatial_LUT_Lookup(const LUT<FLType> &GS_LUT, const PCType xUpper, const PCType x, const PCType y)
{
    return GS_LUT[y*xUpper + x];
}

inline FLType Gaussian_Distribution2D_Range_LUT_Lookup(const LUT<FLType> &GR_LUT, const DType Value1, const DType Value2)
{
    return GR_LUT[Value1 > Value2 ? Value1 - Value2 : Value2 - Value1];
}


#endif
