#ifndef HISTOGRAM_EQUALIZATION_H_
#define HISTOGRAM_EQUALIZATION_H_


#include "IO.h"
#include "Image_Type.h"
#include "Histogram.h"
#include "LUT.h"


const struct HE_Para
{
    FLType strength = 1.0;
    bool separate = false;
} HE_Default;


LUT<DType> Equalization_LUT(const Histogram<DType> &hist, const Plane &dst, const Plane &src, FLType strength = 1.0);
inline LUT<DType> Equalization_LUT(const Histogram<DType> &hist, const Plane &src, FLType strength = 1.0)
{
    return Equalization_LUT(hist, src, src, strength);
}

LUT<FLType> Equalization_LUT_Gain(const Histogram<DType> &hist, const Plane &dst, const Plane &src, FLType strength = 1.0);
inline LUT<FLType> Equalization_LUT_Gain(const Histogram<DType> &hist, const Plane &src, FLType strength = 1.0)
{
    return Equalization_LUT_Gain(hist, src, src, strength);
}

Plane &Histogram_Equalization(Plane &dst, const Plane &src, FLType strength = HE_Default.strength);
Frame &Histogram_Equalization(Frame &dst, const Frame &src, FLType strength = HE_Default.strength, bool separate = HE_Default.separate);

inline Plane Histogram_Equalization(const Plane &src, FLType strength = HE_Default.strength)
{
    Plane dst(src, false);
    return Histogram_Equalization(dst, src, strength);
}
inline Frame Histogram_Equalization(const Frame &src, FLType strength = HE_Default.strength, bool separate = HE_Default.separate)
{
    Frame dst(src, false);
    return Histogram_Equalization(dst, src, strength, separate);
}


class Histogram_Equalization_IO
    : public FilterIO
{
public:
    typedef Histogram_Equalization_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    HE_Para para;

    virtual void arguments_process()
    {
        FilterIO::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-STR" || args[i] == "--strength")
            {
                ArgsObj.GetPara(i, para.strength);
                continue;
            }
            if (args[i] == "-SEP" || args[i] == "--separate")
            {
                ArgsObj.GetPara(i, para.separate);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        ArgsObj.Check();
    }

    virtual Frame process(const Frame &src)
    {
        return Histogram_Equalization(src, para.strength, para.separate);
    }

public:
    _Myt(std::string _Tag = ".HE")
        : _Mybase(std::move(_Tag)) {}
};


#endif
