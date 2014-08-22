#ifndef TYPE_CONV_H_
#define TYPE_CONV_H_


#include "Type.h"


// Round_Div
template <typename T>
inline T Round_Div(T dividend, T divisor)
{
    return (dividend + divisor / 2) / divisor;
}

template <>
inline float Round_Div(float dividend, float divisor)
{
    return dividend / divisor;
}

template <>
inline double Round_Div(double dividend, double divisor)
{
    return dividend / divisor;
}

template <>
inline long double Round_Div(long double dividend, long double divisor)
{
    return dividend / divisor;
}


// Round_BitRsh
template <typename T>
inline T Round_BitRsh(T input, int shift)
{
    return (input + (1 << (shift - 1))) >> shift;
}


// Round_XXX
inline uint8 Round_U8(float input)
{
    return input <= 0 ? 0 : input >= 255.f ? 0xFF : (uint8)(input + 0.5);
}

inline uint8 Round_U8(double input)
{
    return input <= 0 ? 0 : input >= 255. ? 0xFF : (uint8)(input + 0.5);
}

inline uint16 Round_U16(float input)
{
    return input <= 0 ? 0 : input >= 65535.f ? 0xFFFF : (uint16)(input + 0.5);
}

inline uint16 Round_U16(double input)
{
    return input <= 0 ? 0 : input >= 65535. ? 0xFFFF : (uint16)(input + 0.5);
}

inline uint32 Round_U32(float input)
{
    return input <= 0 ? 0 : input >= 4294967295.f ? 0xFFFFFFFF : (uint32)(input + 0.5);
}

inline uint32 Round_U32(double input)
{
    return input <= 0 ? 0 : input >= 4294967295. ? 0xFFFFFFFF : (uint32)(input + 0.5);
}

inline uint64 Round_U64(float input)
{
    return input <= 0 ? 0 : input >= 18446744073709551615.f ? 0xFFFFFFFFFFFFFFFF : (uint64)(input + 0.5);
}

inline uint64 Round_U64(double input)
{
    return input <= 0 ? 0 : input >= 18446744073709551615. ? 0xFFFFFFFFFFFFFFFF : (uint64)(input + 0.5);
}

inline sint8 Round_S8(float input)
{
    return input <= -128.f ? 0x80 : input >= 127.f ? 0x7F : (sint8)(input + 0.5);
}

inline sint8 Round_S8(double input)
{
    return input <= -128. ? 0x80 : input >= 127. ? 0x7F : (sint8)(input + 0.5);
}

inline sint16 Round_S16(float input)
{
    return input <= -32768.f ? 0x8000 : input >= 32767.f ? 0x7FFF : (sint16)(input + 0.5);
}

inline sint16 Round_S16(double input)
{
    return input <= -32768. ? 0x8000 : input >= 32767. ? 0x7FFF : (sint16)(input + 0.5);
}

inline sint32 Round_S32(float input)
{
    return input <= -2147483648.f ? 0x80000000 : input >= 2147483647.f ? 0x7FFFFFFF : (sint32)(input + 0.5);
}

inline sint32 Round_S32(double input)
{
    return input <= -2147483648. ? 0x80000000 : input >= 2147483647. ? 0x7FFFFFFF : (sint32)(input + 0.5);
}

inline sint64 Round_S64(float input)
{
    return input <= -9223372036854775808.f ? 0x8000000000000000 : input >= 9223372036854775807.f ? 0x7FFFFFFFFFFFFFFF : (sint64)(input + 0.5);
}

inline sint64 Round_S64(double input)
{
    return input <= -9223372036854775808. ? 0x8000000000000000 : input >= 9223372036854775807. ? 0x7FFFFFFFFFFFFFFF : (sint64)(input + 0.5);
}


// Clip_XXX
template <typename T>
inline uint8 Clip_U8(T input)
{
    return input >= 255 ? 0xFF : input <= 0 ? 0 : (uint8)input;
}

template <typename T>
inline uint16 Clip_U16(T input)
{
    return input >= 65535 ? 0xFFFF : input <= 0 ? 0 : (uint16)input;
}

template <typename T>
inline uint32 Clip_U32(T input)
{
    return input >= 4294967295 ? 0xFFFFFFFF : input <= 0 ? 0 : (uint32)input;
}

template <typename T>
inline uint64 Clip_U64(T input)
{
    return input >= 18446744073709551615 ? 0xFFFFFFFFFFFFFFFF : input <= 0 ? 0 : (uint64)input;
}

template <typename T>
inline sint8 Clip_S8(T input)
{
    return input >= 127 ? 0x7F : input <= -128 ? 0x80 : (sint8)input;
}

template <typename T>
inline sint16 Clip_S16(T input)
{
    return input >= 32767 ? 0x7FFF : input <= -32768 ? 0x8000 : (sint16)input;
}

template <typename T>
inline sint32 Clip_S32(T input)
{
    return input >= 2147483647 ? 0x7FFFFFFF : input <= -2147483648 ? 0x80000000 : (sint32)input;
}

template <typename T>
inline sint64 Clip_S64(T input)
{
    return input >= 9223372036854775807 ? 0x7FFFFFFFFFFFFFFF : input <= -9223372036854775808 ? 0x8000000000000000 : (sint64)input;
}


// Max Min
template <typename T>
inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <typename T>
inline T Min(T a, T b)
{
    return a > b ? b : a;
}


// Abs
template <typename T>
inline T Abs(T input)
{
    return input < 0 ? -input : input;
}


#endif