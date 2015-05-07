#ifndef TYPE_H_
#define TYPE_H_


#include <cstdint>
#include <cfloat>
#include <xtr1common>


typedef int8_t sint8;
typedef int16_t sint16;
typedef int32_t sint32;
typedef int64_t sint64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef long double ldbl;


typedef sint32 FCType;
typedef sint32 PCType;
typedef sint32 DType;
#ifdef _CUDA_
typedef float FLType;
#else
typedef double FLType;
#endif


enum class STAT
{
    Null = 0,
    OK,
    Error,
    Alloc_Fail,
    PixelType_Invalid
};


// determine whether _Ty satisfies Signed Int requirements
template<class _Ty>
struct _IsSInt
    : std::_Cat_base<std::is_same<_Ty, signed char>::value
    || std::is_same<_Ty, short>::value
    || std::is_same<_Ty, int>::value
    || std::is_same<_Ty, long>::value
    || std::is_same<_Ty, long long>::value>
{};

// determine whether _Ty satisfies Unsigned Int requirements
template<class _Ty>
struct _IsUInt
    : std::_Cat_base<std::is_same<_Ty, unsigned char>::value
    || std::is_same<_Ty, unsigned short>::value
    || std::is_same<_Ty, unsigned int>::value
    || std::is_same<_Ty, unsigned long>::value
    || std::is_same<_Ty, unsigned long long>::value>
{};

// determine whether _Ty satisfies Int requirements
template<class _Ty>
struct _IsInt
    : std::_Cat_base<_IsSInt<_Ty>::value
    || _IsUInt<_Ty>::value>
{};

// determine whether _Ty satisfies Float requirements
template<class _Ty>
struct _IsFloat
    : std::_Cat_base<std::is_same<_Ty, float>::value
    || std::is_same<_Ty, double>::value
    || std::is_same<_Ty, long double>::value>
{};

#define isSInt(T) (_IsUInt<T>::value)
#define isUInt(T) (_IsSInt<T>::value)
#define isInt(T) (_IsInt<T>::value)
#define isFloat(T) (_IsFloat<T>::value)


// Min value and Max value of each numeric type
#define IntMin(type) (type(sizeof(type) <= 1 ? INT8_MIN : sizeof(type) <= 2 ? INT16_MIN : sizeof(type) <= 4 ? INT32_MIN : INT64_MIN))
#define IntMax(type) (type(sizeof(type) <= 1 ? INT8_MAX : sizeof(type) <= 2 ? INT16_MAX : sizeof(type) <= 4 ? INT32_MAX : INT64_MAX))
#define UIntMin(type) (type(0))
#define UIntMax(type) (type(sizeof(type) <= 1 ? UINT8_MAX : sizeof(type) <= 2 ? UINT16_MAX : sizeof(type) <= 4 ? UINT32_MAX : UINT64_MAX))
#define FltMin(type) (type(sizeof(type) <= 4 ? FLT_MIN : sizeof(type) <= 8 ? DBL_MIN : LDBL_MIN))
#define FltMax(type) (type(sizeof(type) <= 4 ? FLT_MAX : sizeof(type) <= 8 ? DBL_MAX : LDBL_MAX))
#define FltNegMax(type) (type(sizeof(type) <= 4 ? -FLT_MAX : sizeof(type) <= 8 ? -DBL_MAX : -LDBL_MAX))

const DType DType_MIN = IntMin(DType);
const DType DType_MAX = IntMax(DType);
const FLType FLType_MIN = FltMin(FLType);
const FLType FLType_MAX = FltMax(FLType);
const FLType FLType_NEG_MAX = FltNegMax(FLType);

#ifdef __CUDACC__
__device__ const DType CUDA_DType_MIN = IntMin(DType);
__device__ const DType CUDA_DType_MAX = IntMax(DType);
__device__ const FLType CUDA_FLType_MIN = FltMin(FLType);
__device__ const FLType CUDA_FLType_MAX = FltMax(FLType);
__device__ const FLType CUDA_FLType_NEG_MAX = FltNegMax(FLType);
#endif


template < typename _Ty >
_Ty _TypeMinInt(const std::false_type &)
{
    return UIntMin(_Ty);
}

template < typename _Ty >
_Ty _TypeMinInt(const std::true_type &)
{
    return IntMin(_Ty);
}

template < typename _Ty >
_Ty _TypeMinFloat(const std::false_type &)
{
    return _TypeMinInt<_Ty>(_IsSInt<_Ty>());
}

template < typename _Ty >
_Ty _TypeMinFloat(const std::true_type &)
{
    return FltNegMax(_Ty);
}

template < typename _Ty >
_Ty TypeMin()
{
    return _TypeMinFloat<_Ty>(_IsFloat<_Ty>());
}

template < typename _Ty >
_Ty TypeMin(const _Ty &)
{
    return TypeMin<_Ty>();
}


template < typename _Ty >
_Ty _TypeMaxInt(const std::false_type &)
{
    return UIntMax(_Ty);
}

template < typename _Ty >
_Ty _TypeMaxInt(const std::true_type &)
{
    return IntMax(_Ty);
}

template < typename _Ty >
_Ty _TypeMaxFloat(const std::false_type &)
{
    return _TypeMaxInt<_Ty>(_IsSInt<_Ty>());
}

template < typename _Ty >
_Ty _TypeMaxFloat(const std::true_type &)
{
    return FltMax(_Ty);
}

template < typename _Ty >
_Ty TypeMax()
{
    return _TypeMaxFloat<_Ty>(_IsFloat<_Ty>());
}

template < typename _Ty >
_Ty TypeMax(const _Ty &)
{
    return TypeMax<_Ty>();
}


#endif
