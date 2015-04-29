#ifndef TYPE_H_
#define TYPE_H_


#include <cstdint>


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
typedef uint32 DType;
#ifdef _CUDA_
typedef float FLType;
#else
typedef double FLType;
#endif


#define IntMin(type) type(sizeof(type) <= 1 ? INT8_MIN : sizeof(type) <= 2 ? INT16_MIN : sizeof(type) <= 4 ? INT32_MIN : INT64_MIN)
#define IntMax(type) type(sizeof(type) <= 1 ? INT8_MAX : sizeof(type) <= 2 ? INT16_MAX : sizeof(type) <= 4 ? INT32_MAX : INT64_MAX)
#define UIntMin(type) type(0)
#define UIntMax(type) type(sizeof(type) <= 1 ? UINT8_MAX : sizeof(type) <= 2 ? UINT16_MAX : sizeof(type) <= 4 ? UINT32_MAX : UINT64_MAX)
#define FltMin(type) type(sizeof(type) <= 4 ? FLT_MIN : sizeof(type) <= 8 ? DBL_MIN : LDBL_MIN)
#define FltMax(type) type(sizeof(type) <= 4 ? FLT_MAX : sizeof(type) <= 8 ? DBL_MAX : LDBL_MAX)
#define FltNegMax(type) type(sizeof(type) <= 4 ? -FLT_MAX : sizeof(type) <= 8 ? -DBL_MAX : -LDBL_MAX)

const DType DType_MIN = UIntMin(DType);
const DType DType_MAX = UIntMax(DType);
const FLType FLType_MIN = FltMin(FLType);
const FLType FLType_MAX = FltMax(FLType);
const FLType FLType_NEG_MAX = FltNegMax(FLType);

#ifdef __CUDACC__
__device__ const DType CUDA_DType_MIN = UIntMin(DType);
__device__ const DType CUDA_DType_MAX = UIntMax(DType);
__device__ const FLType CUDA_FLType_MIN = FltMin(FLType);
__device__ const FLType CUDA_FLType_MAX = FltMax(FLType);
__device__ const FLType CUDA_FLType_NEG_MAX = FltNegMax(FLType);
#endif


enum class STAT {
    Null = 0,
    OK,
    Error,
    Alloc_Fail,
    PixelType_Invalid
};


template < typename T >
class ClassChecker
{
public:
    typedef struct { char c[2]; } Yes;
    typedef struct { char c[1]; } No;
    static Yes _CheckVal(T t);
    static No _CheckVal(...);
};

#define TypeCheckerVal(T, i) ((sizeof (ClassChecker<T>::_CheckVal(i))) == 2 ? true : false)
#define TypeChecker(T1, T2) ((sizeof (ClassChecker<T1>::_CheckVal(T2(0)))) == 2 ? true : false)

//#define isFloat(T) (TypeChecker(T, float) || TypeChecker(T, double) || TypeChecker(T, ldbl))
#define isFloat(T) (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(ldbl))


#endif
