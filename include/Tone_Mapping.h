#ifndef TONE_MAPPING_H_
#define TONE_MAPPING_H_


#include "Filter.h"
#include "Image_Type.h"
#include "LUT.h"
#include "Histogram.h"


const struct AGTM_Para {
    const Histogram<DType>::BinType HistBins = 8;
} AGTM_Default;


Frame &Adaptive_Global_Tone_Mapping(Frame &dst, const Frame &src);
inline Frame Adaptive_Global_Tone_Mapping(const Frame &src)
{
    Frame dst(src, false);
    return Adaptive_Global_Tone_Mapping(dst, src);
}


class Adaptive_Global_Tone_Mapping_IO
    : public FilterIO
{
public:
    typedef Adaptive_Global_Tone_Mapping_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    virtual Frame process(const Frame &src)
    {
        return Adaptive_Global_Tone_Mapping(src);
    }

public:
    _Myt(std::string _Tag = ".AGTM")
        : _Mybase(std::move(_Tag)) {}
};


LUT<FLType> Adaptive_Global_Tone_Mapping_Gain_LUT_Generation(const Plane &src);


#endif
