#ifndef TYPE_H_
#define TYPE_H_


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


const FLType FLType_MAX = sizeof(FLType) < 8 ? FLT_MAX : DBL_MAX;
const FLType FLType_MIN = sizeof(FLType) < 8 ? FLT_MIN : DBL_MIN;


enum class STAT {
    Null = 0,
    OK,
    Error,
    Alloc_Fail,
    PixelType_Invalid
};


#endif