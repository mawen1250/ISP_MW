#ifdef __CUDACC__

#ifndef SPECIFICATION_CUH_
#define SPECIFICATION_CUH_


#include <math_functions.h>
#include "Specification.h"
#include "Helper.cuh"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Conversion classes
template < typename T = FLType >
struct CUDA_TransferChar_Conv_Sub
{
    typedef CUDA_TransferChar_Conv_Sub<T> _Myt;

    // General Parameters
    T k0 = 1;
    T k0Mphi;

    // Parameters for gamma function
    T phi = 1;
    T alpha = 0;
    T beta;
    T power = 1;
    T gamma;

    // Parameters for log function
    T div = 1;

    CUDA_TransferChar_Conv_Sub(TransferChar _TransferChar)
    {
        ldbl _k0 = 1.0L, _phi = 1.0L, _alpha = 0.0L, _power = 1.0L, _div = 1.0L;
        TransferChar_Parameter(_TransferChar, _k0, _phi, _alpha, _power, _div);
        ldbl _beta = _alpha + 1.0L;

        k0 = static_cast<T>(_k0);
        phi = static_cast<T>(_phi);
        alpha = static_cast<T>(_alpha);
        power = static_cast<T>(_power);
        div = static_cast<T>(_div);
        k0Mphi = static_cast<T>(_k0 * _phi);
        beta = static_cast<T>(_beta);
        gamma = static_cast<T>(1 / _power);
    }

    bool operator==(const _Myt &right) const
    {
        auto &left = *this;

        if (left.k0 == right.k0 && left.phi == right.phi && left.alpha == right.alpha
            && left.power == right.power && left.div == right.div)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ T gamma2linear(T x) const
    {
        return x < k0Mphi ? x / phi : pow((x + alpha) / beta, gamma);
    }

    __device__ T linear2gamma(T x) const
    {
        return x < k0 ? x * phi : pow(x, power) * beta - alpha;
    }

    __device__ T log2linear(T x) const
    {
        return x == 0 ? 0 : pow(T(10), (x - 1) * div);
    }

    __device__ T linear2log(T x) const
    {
        return x < k0 ? 0 : T(1) + log10(x) / div;
    }
};


template < typename T = FLType >
class CUDA_TransferChar_Conv
    : public TransferChar_Conv_Base
{
public:
    typedef CUDA_TransferChar_Conv<T> _Myt;
    typedef TransferChar_Conv_Base _Mybase;

protected:
    CUDA_TransferChar_Conv_Sub<T> ToLinear;
    CUDA_TransferChar_Conv_Sub<T> LinearTo;

public:
    CUDA_TransferChar_Conv(TransferChar dst, TransferChar src)
        : _Mybase(dst, src), ToLinear(src), LinearTo(dst)
    {
        ConvTypeDecision(ToLinear == LinearTo);
    }

    __device__ T operator()(T x) const
    {
        switch (type_)
        {
        case 0:
            return none(x);
        case 1:
            return linear2gamma(x);
        case 2:
            return linear2log(x);
        case 3:
            return gamma2linear(x);
        case 4:
            return gamma2gamma(x);
        case 5:
            return gamma2log(x);
        case 6:
            return log2linear(x);
        case 7:
            return log2gamma(x);
        case 8:
            return log2log(x);
        default:
            return none(x);
        }
    }

    __device__ T none(T x) const
    {
        return x;
    }

    __device__ T linear2gamma(T x) const
    {
        return LinearTo.linear2gamma(x);
    }

    __device__ T linear2log(T x) const
    {
        return LinearTo.linear2log(x);
    }

    __device__ T gamma2linear(T x) const
    {
        return ToLinear.gamma2linear(x);
    }

    __device__ T gamma2gamma(T x) const
    {
        return LinearTo.linear2gamma(ToLinear.gamma2linear(x));
    }

    __device__ T gamma2log(T x) const
    {
        return LinearTo.linear2log(ToLinear.gamma2linear(x));
    }

    __device__ T log2linear(T x) const
    {
        return ToLinear.log2linear(x);
    }

    __device__ T log2gamma(T x) const
    {
        return LinearTo.linear2gamma(ToLinear.log2linear(x));
    }

    __device__ T log2log(T x) const
    {
        return LinearTo.linear2log(ToLinear.log2linear(x));
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif

#endif // __CUDACC__
