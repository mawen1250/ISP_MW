#ifndef TONE_MAPPING_H_
#define TONE_MAPPING_H_


#include "IO.h"
#include "Image_Type.h"
#include "LUT.h"
#include "Histogram.h"


const struct AGTM_Para {
    const Histogram<DType>::BinType HistBins = 8;
} AGTM_Default;


Frame & Adaptive_Global_Tone_Mapping(Frame & output, const Frame & input);
inline Frame Adaptive_Global_Tone_Mapping(const Frame & input)
{
    Frame output(input, false);
    return Adaptive_Global_Tone_Mapping(output, input);
}


class Adaptive_Global_Tone_Mapping_IO
    : public FilterIO
{
protected:
    virtual void arguments_process()
    {
        FilterIO::arguments_process();
    }

    virtual Frame processFrame(const Frame &src)
    {
        return Adaptive_Global_Tone_Mapping(src);
    }

public:
    Adaptive_Global_Tone_Mapping_IO(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".AGTM")
        : FilterIO(_argc, _args, _Tag) {}

    ~Adaptive_Global_Tone_Mapping_IO() {}
};


LUT<FLType> Adaptive_Global_Tone_Mapping_Gain_LUT_Generation(const Plane & input);


#endif