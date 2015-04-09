#ifndef HELPER_H_
#define HELPER_H_


#include "Type.h"


// Memory allocation
#ifdef _CUDA_
const size_t MEMORY_ALIGNMENT = 4096;
#else
const size_t MEMORY_ALIGNMENT = 64;
#endif


template < typename _Ty = void >
void AlignedMalloc(_Ty *&Memory, size_t Count, size_t Alignment = MEMORY_ALIGNMENT)
{
    Memory = reinterpret_cast<_Ty *>(_aligned_malloc(sizeof(_Ty) * Count, Alignment));

    if (Memory == nullptr)
    {
        std::cerr << "AlignedMalloc: memory allocation failed!\n";
        exit(EXIT_FAILURE);
    }
}


template < typename _Ty = void >
void AlignedFree(_Ty *&Memory)
{
    _aligned_free(Memory);
    Memory = nullptr;
}


// Round_Div
template < typename T >
inline T Round_Div(T dividend, T divisor)
{
    return (dividend + divisor / 2) / divisor;
}

template < >
inline float Round_Div(float dividend, float divisor)
{
    return dividend / divisor;
}

template < >
inline double Round_Div(double dividend, double divisor)
{
    return dividend / divisor;
}

template < >
inline long double Round_Div(long double dividend, long double divisor)
{
    return dividend / divisor;
}


// Round_BitRsh
template < typename T >
inline T Round_BitRsh(T input, int shift)
{
    return (input + (1 << (shift - 1))) >> shift;
}


// Round_XXX
template < typename T >
inline uint8 Round_U8(T input)
{
    return input <= 0 ? 0 : input >= T(_UI8_MAX) ? _UI8_MAX : static_cast<uint8>(input + T(0.5));
}

template < typename T >
inline uint16 Round_U16(T input)
{
    return input <= 0 ? 0 : input >= T(_UI16_MAX) ? _UI16_MAX : static_cast<uint16>(input + T(0.5));
}

template < typename T >
inline uint32 Round_U32(T input)
{
    return input <= 0 ? 0 : input >= T(_UI32_MAX) ? _UI32_MAX : static_cast<uint32>(input + T(0.5));
}

template < typename T >
inline uint64 Round_U64(T input)
{
    return input <= 0 ? 0 : input >= T(_UI64_MAX) ? _UI64_MAX : static_cast<uint64>(input + T(0.5));
}

template < typename T >
inline sint8 Round_S8(T input)
{
    return input <= T(_I8_MIN) ? _I8_MIN : input >= T(_I8_MAX) ? _I8_MAX : static_cast<sint8>(input + T(0.5));
}

template < typename T >
inline sint16 Round_S16(T input)
{
    return input <= T(_I16_MIN) ? _I16_MIN : input >= T(_I16_MAX) ? _I16_MAX : static_cast<sint16>(input + T(0.5));
}

template < typename T >
inline sint32 Round_S32(T input)
{
    return input <= T(_I32_MIN) ? _I32_MIN : input >= T(_I32_MAX) ? _I32_MAX : static_cast<sint32>(input + T(0.5));
}

template < typename T >
inline sint64 Round_S64(T input)
{
    return input <= T(_I64_MIN) ? _I64_MIN : input >= T(_I64_MAX) ? _UI64_MAX : static_cast<sint64>(input + T(0.5));
}


// Clip_XXX
template < typename T >
inline uint8 Clip_U8(T input)
{
    return input >= _UI8_MAX ? _UI8_MAX : input <= 0 ? 0 : static_cast<uint8>(input);
}

template < typename T >
inline uint16 Clip_U16(T input)
{
    return input >= _UI16_MAX ? _UI16_MAX : input <= 0 ? 0 : static_cast<uint16>(input);
}

template < typename T >
inline uint32 Clip_U32(T input)
{
    return input >= _UI32_MAX ? _UI32_MAX : input <= 0 ? 0 : static_cast<uint32>(input);
}

template < typename T >
inline uint64 Clip_U64(T input)
{
    return input >= _UI64_MAX ? _UI64_MAX : input <= 0 ? 0 : static_cast<uint64>(input);
}

template < typename T >
inline sint8 Clip_S8(T input)
{
    return input >= _I8_MAX ? _I8_MAX : input <= _I8_MIN ? _I8_MIN : static_cast<sint8>(input);
}

template < typename T >
inline sint16 Clip_S16(T input)
{
    return input >= _I16_MAX ? _I16_MAX : input <= _I16_MIN ? _I16_MIN : static_cast<sint16>(input);
}

template < typename T >
inline sint32 Clip_S32(T input)
{
    return input >= _I32_MAX ? _I32_MAX : input <= _I32_MIN ? _I32_MIN : static_cast<sint32>(input);
}

template < typename T >
inline sint64 Clip_S64(T input)
{
    return input >= _I64_MAX ? _I64_MAX : input <= _I64_MIN ? _I64_MIN : static_cast<sint64>(input);
}


// Max Min
template < typename T >
inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template < typename T >
inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template < typename T >
inline T Clip(T input, T lower, T upper)
{
    return input >= upper ? upper : input <= lower ? lower : input;
}


// Abs
template < typename T >
inline T Abs(T input)
{
    return input < 0 ? -input : input;
}

template < typename T >
inline T AbsSub(T a, T b)
{
    return a >= b ? a - b : b - a;
}


// Limit
template < typename T >
T limit_dif_abs(T value, T ref, T dthr, T uthr, T elast, bool smooth)
{
    T diff, abdiff, thr, alpha, beta, output;

    dthr = Max(dthr, T(0.0));
    uthr = Max(uthr, T(0.0));
    elast = Max(elast, T(1.0));
    smooth = elast == 1 ? false : smooth;

    diff = value - ref;
    abdiff = Abs(diff);
    thr = diff > 0 ? uthr : dthr;
    alpha = 1. / (thr * (elast - 1));
    beta = elast * thr;

    if (smooth)
    {
        output = abdiff <= thr ? value :
            abdiff >= beta ? ref :
            ref + alpha * diff * (beta - abdiff);
    }
    else
    {
        output = abdiff <= thr ? value :
            abdiff >= beta ? ref :
            ref + thr * (diff / abdiff);
    }

    return output;
}

template < typename T >
inline T limit_dif_abs(T value, T ref, T thr, T elast = 2.0, bool smooth = true)
{
    return limit_dif_abs(value, ref, thr, thr, elast, smooth);
}

template < typename T >
T limit_dif_ratio(T value, T ref, T dthr, T uthr, T elast, bool smooth)
{
    T ratio, abratio, negative, inverse, nratio, thr, lratio, output;

    dthr = Max(dthr, T(0.0));
    uthr = Max(uthr, T(0.0));
    elast = Max(elast, T(1.0));
    smooth = elast == 1 ? false : smooth;

    ratio = value / ref;
    abratio = Abs(ratio);
    negative = ratio*abratio < 0;
    inverse = abratio < 1;
    nratio = inverse ? 1. / abratio : abratio;
    thr = inverse ? dthr : uthr;
    thr = thr < 1 ? 1. / thr - 1 : thr - 1;

    lratio = limit_dif_abs(nratio, 1., smooth, thr, thr, elast);
    lratio = inverse ? 1. / lratio : lratio;
    output = negative ? -lratio*ref : lratio*ref;

    return output;
}

template < typename T >
inline T limit_dif_ratio(T value, T ref, T thr, T elast = 2.0, bool smooth = true)
{
    return limit_dif_ratio(value, ref, thr, thr, elast, smooth);
}

template < typename T >
T damp_ratio(T ratio, T damp)
{
    T abratio, negative, inverse, nratio, dratio, output;

    abratio = Abs(ratio);
    negative = ratio*abratio < 0;
    inverse = abratio < 1;
    nratio = inverse ? 1. / abratio : abratio;

    dratio = (nratio - 1)*damp + 1;
    dratio = inverse ? 1. / dratio : dratio;
    output = negative ? -dratio : dratio;

    return output;
}


// Interpolation
template < typename T >
inline T Linear2(T x, T x1, T y1, T x2, T y2)
{
    return (y2 - y1) / (x2 - x1) * (x - x1) + y1;
}


#endif
