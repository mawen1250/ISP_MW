#ifndef TONE_MAPPING_H_
#define TONE_MAPPING_H_


#include "Image_Type.h"
#include "LUT.h"


int Adaptive_Global_Tone_Mapping_IO(int argc, char ** argv);


Frame_YUV & Adaptive_Global_Tone_Mapping(Frame_YUV & output, const Frame_YUV & input);
inline Frame_YUV Adaptive_Global_Tone_Mapping(const Frame_YUV & input)
{
    Frame_YUV output(input, false);
    return Adaptive_Global_Tone_Mapping(output, input);
}

Frame_RGB & Adaptive_Global_Tone_Mapping(Frame_RGB & output, const Frame_RGB & input);
inline Frame_RGB Adaptive_Global_Tone_Mapping(const Frame_RGB & input)
{
    Frame_RGB output(input, false);
    return Adaptive_Global_Tone_Mapping(output, input);
}


LUT<FLType> Adaptive_Global_Tone_Mapping_Gain_LUT_Generation(const Plane & input);


#endif