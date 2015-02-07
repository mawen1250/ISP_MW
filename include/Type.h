#ifndef TYPE_H_
#define TYPE_H_


#include <climits>
#include <cfloat>


typedef unsigned char  uint8;
typedef unsigned short uint16;
typedef unsigned long  uint32;
typedef unsigned long long uint64;
typedef   signed char  sint8;
typedef   signed short sint16;
typedef   signed long  sint32;
typedef   signed long long sint64;


typedef sint32 FCType;
typedef sint32 PCType;
typedef uint32 DType;
typedef double FLType;


const DType DType_MAX = ULONG_MAX;
const DType DType_MIN = 0;
const FLType FLType_MAX = sizeof(FLType) < 8 ? FLT_MAX : DBL_MAX;
const FLType FLType_MIN = sizeof(FLType) < 8 ? FLT_MIN : DBL_MIN;


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

//#define isFloat(T) (TypeChecker(T, float) || TypeChecker(T, double) || TypeChecker(T, long double))
#define isFloat(T) (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(long double))


#endif
