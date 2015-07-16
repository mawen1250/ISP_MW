#ifndef HELPER_H_
#define HELPER_H_


#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include "Type.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Exception handle


#ifdef _DEBUG
#define DEBUG_BREAK __debugbreak();
#define DEBUG_FAIL(mesg) __debugbreak(); _STD _DEBUG_ERROR(mesg);
#else
#define DEBUG_BREAK exit(EXIT_FAILURE);
#define DEBUG_FAIL(mesg) std::cerr << mesg << std::endl; exit(EXIT_FAILURE);
#endif


enum class STAT
{
    Null = 0,
    OK,
    Error,
    Alloc_Fail,
    PixelType_Invalid
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constants


const ldbl Pi = 3.14159265358979323846264338327950288L;
const ldbl Exp1 = 2.71828182845904523536028747135266250L;
const ldbl Two32 = 4294967296.0L;
const ldbl Two31 = 2147483648.0L;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert to std::string


template < typename _Ty >
std::string GetStr(const _Ty &src)
{
    std::stringstream ss;
    ss << src;
    return ss.str();
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Memory allocation


#ifdef _CUDA_
const size_t MEMORY_ALIGNMENT = 4096;
#else
const size_t MEMORY_ALIGNMENT = 64;
#endif


inline void *AlignedMalloc(size_t Size, size_t Alignment = MEMORY_ALIGNMENT)
{
    void *Memory = nullptr;
#ifdef _WIN32
    Memory = _aligned_malloc(Size, Alignment);
#else
    if (posix_memalign(&Memory, Alignment, Size))
    {
        Memory = nullptr;
    }
#endif
    if (Memory == nullptr)
    {
        DEBUG_FAIL("AlignedMalloc: memory allocation failed!");
    }
    return Memory;
}

template < typename _Ty >
void AlignedMalloc(_Ty *&Memory, size_t Count, size_t Alignment = MEMORY_ALIGNMENT)
{
    Memory = reinterpret_cast<_Ty *>(AlignedMalloc(Count * sizeof(_Ty), Alignment));
}


inline void AlignedFree(void **Memory)
{
#ifdef _WIN32
    _aligned_free(*Memory);
#else
    free(*Memory);
#endif
    *Memory = nullptr;
}

template < typename _Ty >
void AlignedFree(_Ty *&Memory)
{
    void *temp = reinterpret_cast<void *>(Memory);
    AlignedFree(&temp);
    Memory = reinterpret_cast<_Ty *>(temp);
}


inline void *AlignedRealloc(void *Memory, size_t NewSize, size_t Alignment = MEMORY_ALIGNMENT)
{
#ifdef _WIN32
    Memory = _aligned_realloc(Memory, NewSize, Alignment);
    if (Memory == nullptr)
    {
        DEBUG_FAIL("AlignedRealloc: memory allocation failed!");
    }
#else
    AlignedFree(&Memory);
    Memory = AlignedMalloc(NewSize, Alignment);
#endif
    return Memory;
}

template < typename _Ty >
void AlignedRealloc(_Ty *&Memory, size_t NewCount, size_t Alignment = MEMORY_ALIGNMENT)
{
    Memory = reinterpret_cast<_Ty *>(AlignedRealloc(reinterpret_cast<void *>(Memory), NewCount * sizeof(_Ty), Alignment));
}


template < typename _Ty >
size_t CalStride(PCType width, size_t Alignment = MEMORY_ALIGNMENT)
{
    size_t line_size = static_cast<size_t>(width) * sizeof(_Ty);
    return line_size % Alignment == 0 ? line_size : (line_size / Alignment + 1) * Alignment;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2D array copy


template < typename _Dt1, typename _St1 >
void MatCopy(void *dstp, const void *srcp, PCType height, PCType width, size_t dst_stride, size_t src_stride)
{
    for (PCType j = 0; j < height; ++j)
    {
        auto dst = reinterpret_cast<_Dt1 *>(reinterpret_cast<uint8_t *>(dstp) + dst_stride);
        auto src = reinterpret_cast<const _St1 *>(reinterpret_cast<const uint8_t *>(srcp) + src_stride);

        for (PCType i = 0; i < width; ++i)
        {
            dst[i] = static_cast<_Dt1>(src[i]);
        }
    }
}

template < typename _Dt1, typename _St1 >
void MatCopy(_Dt1 *dstp, const _St1 *srcp, PCType height, PCType width, PCType dst_stride, PCType src_stride)
{
    static_assert(!std::is_same<_Dt1, void>::value && !std::is_same<_St1, void>::value,
        "MatCopy: instantiating with void pointer is not allowed here in this template function.");
    MatCopy<_Dt1, _St1>(reinterpret_cast<void *>(dstp), reinterpret_cast<const void *>(srcp),
        height, width, dst_stride * sizeof(_Dt1), src_stride * sizeof(_St1));
}


inline void MatCopy(void *dstp, const void *srcp, PCType height, size_t row_size, size_t dst_stride, size_t src_stride)
{
    if (dstp == srcp)
    {
        return;
    }

    if (height > 0)
    {
        if (src_stride == dst_stride && src_stride == row_size)
        {
            memcpy(dstp, srcp, height * row_size);
        }
        else
        {
            for (PCType j = 0; j < height; ++j)
            {
                memcpy(dstp, srcp, row_size);
                dstp = reinterpret_cast<uint8_t *>(dstp) + dst_stride;
                srcp = reinterpret_cast<const uint8_t *>(srcp) + src_stride;
            }
        }
    }
}

template < typename _Ty >
void MatCopy(void *dstp, const void *srcp, PCType height, PCType width, size_t dst_stride, size_t src_stride)
{
    MatCopy(dstp, srcp, height, width * sizeof(_Ty), dst_stride * sizeof(_Ty), src_stride * sizeof(_Ty));
}

template < typename _Ty >
void MatCopy(_Ty *dstp, const _Ty *srcp, PCType height, PCType width, PCType dst_stride, PCType src_stride)
{
    MatCopy(reinterpret_cast<void *>(dstp), reinterpret_cast<const void *>(srcp),
        height, width * sizeof(_Ty), dst_stride * sizeof(_Ty), src_stride * sizeof(_Ty));
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Min Max Clip


template < typename _Ty >
_Ty Min(const _Ty &a, const _Ty &b)
{
    return b < a ? b : a;
}

template < typename _Ty >
_Ty Max(const _Ty &a, const _Ty &b)
{
    return b > a ? b : a;
}

template < typename _Ty >
_Ty Clip(const _Ty &input, const _Ty &lower, const _Ty &upper)
{
    return input <= lower ? lower : input >= upper ? upper : input;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Abs AbsSub


template < typename _Ty >
_Ty Abs(const _Ty &input)
{
    return input < 0 ? -input : input;
}

template < typename _Ty >
_Ty AbsSub(const _Ty &a, const _Ty &b)
{
    return b < a ? a - b : b - a;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Division with rounding to integer


template < typename _Ty >
_Ty _RoundDiv(_Ty dividend, _Ty divisor, const std::false_type &)
{
    return (dividend + divisor / 2) / divisor;
}

template < typename _Ty >
_Ty _RoundDiv(_Ty dividend, _Ty divisor, const std::true_type &)
{
    return dividend / divisor;
}

template < typename _Ty >
_Ty RoundDiv(_Ty dividend, _Ty divisor)
{
    return _RoundDiv(dividend, divisor, _IsFloat<_Ty>());
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bit shift with rounding to integer


template < typename _Ty >
_Ty RoundBitRsh(_Ty input, int shift)
{
    static_assert(_IsInt<_Ty>::value, "Invalid arguments for template instantiation! Must be integer.");
    return (input + (1 << (shift - 1))) >> shift;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Round - numeric type conversion with up-rounding (float to int) and saturation


// UInt to UInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(_St1 input, const std::false_type &, const std::false_type &)
{
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input >= max ? max : input);
}

// UInt to SInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(_St1 input, const std::false_type &, const std::true_type &)
{
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input >= max ? max : input);
}

// SInt to UInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(_St1 input, const std::true_type &, const std::false_type &)
{
    _St1 min = _St1(0);
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input);
}

// SInt to SInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(_St1 input, const std::true_type &, const std::true_type &)
{
    _St1 min = _St1(TypeMin<_Dt1>() > TypeMin<_St1>() ? TypeMin<_Dt1>() : TypeMin<_St1>());
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input);
}

// Int to Int
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2(_St1 input, const std::false_type &)
{
    return _Round_Int2Int<_Dt1, _St1>(input, _IsSInt<_St1>(), _IsSInt<_Dt1>());
}

// Int to Float
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2(_St1 input, const std::true_type &)
{
    return static_cast<_Dt1>(input);
}

// Int to Any
template < typename _Dt1, typename _St1 >
_Dt1 _Round(_St1 input, const std::false_type &)
{
    return _Round_Int2<_Dt1, _St1>(input, _IsFloat<_Dt1>());
}

// Float to Int
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Float2(_St1 input, const std::false_type &)
{
    _St1 min = _St1(TypeMin<_Dt1>());
    _St1 max = _St1(TypeMax<_Dt1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input + _St1(0.5));
}

// Float to Float
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Float2(_St1 input, const std::true_type &)
{
    _St1 min = _St1(TypeMin<_Dt1>() > TypeMin<_St1>() ? TypeMin<_Dt1>() : TypeMin<_St1>());
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input);
}

// Float to Any
template < typename _Dt1, typename _St1 >
_Dt1 _Round(_St1 input, const std::true_type &)
{
    return _Round_Float2<_Dt1, _St1>(input, _IsFloat<_Dt1>());
}

// Any to Any
template < typename _Dt1, typename _St1 >
_Dt1 Round(_St1 input)
{
    return _Round<_Dt1, _St1>(input, _IsFloat<_St1>());
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Limit difference


template < typename _Ty >
_Ty limit_dif_abs(_Ty value, _Ty ref, _Ty dthr, _Ty uthr, _Ty elast, bool smooth)
{
    _Ty diff, abdiff, thr, alpha, beta, output;

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

template < typename _Ty >
_Ty limit_dif_abs(_Ty value, _Ty ref, _Ty thr, _Ty elast = 2.0, bool smooth = true)
{
    return limit_dif_abs(value, ref, thr, thr, elast, smooth);
}


template < typename _Ty >
_Ty limit_dif_ratio(_Ty value, _Ty ref, _Ty dthr, _Ty uthr, _Ty elast, bool smooth)
{
    _Ty ratio, abratio, negative, inverse, nratio, thr, lratio, output;

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

template < typename _Ty >
_Ty limit_dif_ratio(_Ty value, _Ty ref, _Ty thr, _Ty elast = 2.0, bool smooth = true)
{
    return limit_dif_ratio(value, ref, thr, thr, elast, smooth);
}


template < typename _Ty >
_Ty damp_ratio(_Ty ratio, _Ty damp)
{
    _Ty abratio, negative, inverse, nratio, dratio, output;

    abratio = Abs(ratio);
    negative = ratio*abratio < 0;
    inverse = abratio < 1;
    nratio = inverse ? 1. / abratio : abratio;

    dratio = (nratio - 1)*damp + 1;
    dratio = inverse ? 1. / dratio : dratio;
    output = negative ? -dratio : dratio;

    return output;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interpolation


template < typename _Ty >
_Ty Linear2(_Ty x, _Ty x1, _Ty y1, _Ty x2, _Ty y2)
{
    return (y2 - y1) / (x2 - x1) * (x - x1) + y1;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif
