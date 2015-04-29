#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_


#include <cmath>
#include "Filter.h"
#include "Image_Type.h"
#include "LUT.h"


const ldbl sqrt_2Pi = sqrt(2 * Pi);

const ldbl sigmaSMul = 2.0L;
const ldbl sigmaRMul = sizeof(FLType) < 8 ? 8.0L : 32.0L; // 8. when FLType is float, 32. when FLType is double


struct Gaussian2D_Para
{
    ldbl sigma = 3.0L;
};

extern const Gaussian2D_Para Gaussian2D_Default;


class Gaussian2D
    : public FilterIF
{
public:
    typedef Gaussian2D _Myt;
    typedef FilterIF _Mybase;

protected:
    Gaussian2D_Para para;

public:
    _Myt(const Gaussian2D_Para &_para = Gaussian2D_Default)
        : para(_para)
    {}

protected:
    virtual Plane &process_Plane(Plane &dst, const Plane &src);
};


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
        _Mybase::arguments_process();

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
        Gaussian2D filter(para);
        return filter(src);
    }

public:
    _Myt(std::string _Tag = ".Gaussian")
        : _Mybase(std::move(_Tag))
    {}
};


// Implementation of recursive Gaussian algorithm from "Ian T. Young, Lucas J. van Vliet - Recursive implementation of the Gaussian filter"
// For 32bit float type, the maximum sigma with valid result is about 80.
class RecursiveGaussian
{
public:
    typedef RecursiveGaussian _Myt;
    typedef FilterData<FLType> _Dt;

public:
    FLType B;
    FLType B1;
    FLType B2;
    FLType B3;
    bool allow_negative = true;
    int iter = 1;

public:
    _Myt(ldbl sigma, bool _allow_negative = true)
        : allow_negative(_allow_negative)
    {
        GetPara(sigma);
    }

    Plane_FL operator()(const Plane_FL &src)
    {
        Plane_FL dst(src, false);
        Filter(dst, src);
        return dst;
    }

    void GetPara(ldbl sigma);

    virtual void FilterV(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride);
    virtual void FilterH(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride);
    virtual void Filter(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride);

    virtual void FilterV(Plane_FL &dst, const Plane_FL &src);
    virtual void FilterH(Plane_FL &dst, const Plane_FL &src);
    virtual void Filter(Plane_FL &dst, const Plane_FL &src);

    virtual void FilterV(Plane_FL &data) { FilterV(data, data); }
    virtual void FilterH(Plane_FL &data) { FilterH(data, data); }
    virtual void Filter(Plane_FL &data) { Filter(data, data); }

protected:
    virtual void FilterV_Kernel(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride) const;
    virtual void FilterH_Kernel(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride) const;
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


inline LUT<FLType> Gaussian_Function_Spatial_LUT_Generation(const PCType xUpper, const PCType yUpper, const ldbl sigmaS)
{
    GaussianFunction<ldbl> GFunc(sigmaS);

    LUT<FLType> GS_LUT(xUpper * yUpper);

    for (PCType y = 0; y < yUpper; y++)
    {
        for (PCType x = 0; x < xUpper; x++)
        {
            GS_LUT[y * xUpper + x] = static_cast<FLType>(GFunc(static_cast<ldbl>(x * x + y * y)));
        }
    }

    return GS_LUT;
}

inline LUT<FLType> Gaussian_Function_Range_LUT_Generation(const DType ValueRange, const ldbl sigmaR)
{
    NormalizedGaussianFunctionX<ldbl> NGFuncX(sigmaR);

    DType Levels = ValueRange + 1;
    const DType upper = Min(ValueRange, static_cast<DType>(sigmaR * sigmaRMul * ValueRange + 0.5));
    LUT<FLType> GR_LUT(Levels);

    DType i = 0;
    for (; i <= upper; i++)
    {
        GR_LUT[i] = static_cast<FLType>(NGFuncX(static_cast<ldbl>(i) / ValueRange));
    }
    // For unknown reason, when more range weights are too small or equal 0, the runtime speed gets lower - mainly in function Recursive_Gaussian2D_Horizontal.
    // To avoid this issue, we set range weights whose range values are larger than sigmaR * sigmaRMul to the Gaussian function value at sigmaR * sigmaRMul.
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
    return GS_LUT[y * xUpper + x];
}

inline FLType Gaussian_Distribution2D_Range_LUT_Lookup(const LUT<FLType> &GR_LUT, const DType Value1, const DType Value2)
{
    return GR_LUT[Value1 > Value2 ? Value1 - Value2 : Value2 - Value1];
}


#endif
