#ifndef TYPE_H_
#define TYPE_H_


typedef unsigned char  uint8;
typedef unsigned short uint16;
typedef unsigned long  uint32;
typedef unsigned long long uint64;
typedef   signed char  sint8;
typedef   signed short sint16;
typedef   signed long  sint32;
typedef   signed long long sint64;

typedef bool  Bool;
#define True  true;
#define False false;


enum class STAT {
    Null = 0,
    OK,
    Error,
    Alloc_Fail,
    PixelType_Invalid
};


#endif