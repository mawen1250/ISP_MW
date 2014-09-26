#ifndef TONE_MAPPING_H_
#define TONE_MAPPING_H_


#include "Image_Type.h"
#include "LUT.h"
#include "Histogram.h"


const struct AGTM_Default {
    const double sigma = 100.0;
    const Histogram<DType>::BinType HistBins = 8;
} AGTM_Default;


int Adaptive_Global_Tone_Mapping_IO(const int argc, const std::string * args);


Frame & Adaptive_Global_Tone_Mapping(Frame & output, const Frame & input);
inline Frame Adaptive_Global_Tone_Mapping(const Frame & input)
{
    Frame output(input, false);
    return Adaptive_Global_Tone_Mapping(output, input);
}


LUT<FLType> Adaptive_Global_Tone_Mapping_Gain_LUT_Generation(const Plane & input);


#endif