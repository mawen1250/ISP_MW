#ifndef HISTOGRAM_EQUALIZATION_H_
#define HISTOGRAM_EQUALIZATION_H_


#include "IO.h"
#include "Image_Type.h"
#include "Histogram.h"


const struct HE_Para {
    bool separate = false;
} HE_Default;


Plane & Histogram_Equalization(Plane &dst, const Plane &src);
Frame & Histogram_Equalization(Frame &dst, const Frame &src, bool separate = HE_Default.separate);

inline Plane Histogram_Equalization(const Plane &src)
{
    Plane dst(src, false);
    return Histogram_Equalization(dst, src);
}
inline Frame Histogram_Equalization(const Frame &src, bool separate = HE_Default.separate)
{
    Frame dst(src, false);
    return Histogram_Equalization(dst, src, separate);
}


class Histogram_Equalization_IO
    : public FilterIO
{
protected:
    bool separate = HE_Default.separate;

    virtual void arguments_process()
    {
        FilterIO::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-S" || args[i] == "--separate")
            {
                ArgsObj.GetPara(i, separate);
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

    virtual Frame processFrame(const Frame &src)
    {
        return Histogram_Equalization(src, separate);
    }

public:
    Histogram_Equalization_IO(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".HE")
        : FilterIO(_argc, _args, _Tag) {}

    ~Histogram_Equalization_IO() {}
};


#endif