#ifndef NLMEANS_H_
#define NLMEANS_H_


#include "IO.h"
#include "Image_Type.h"
#include "Helper.h"
#include "Block.h"


const struct NLMeans_Para {
    double sigma = 8.0;
    double strength = sigma * 5;
    PCType GroupSizeMax = 16;
    PCType BlockSize = 8;
    PCType Overlap = 4;
    PCType BMrange = 24;
    PCType BMstep = 3;
    double thMSE = sigma * 50;
} NLMeans_Default;


Plane &NLMeans(Plane &dst, const Plane &src, const Plane &ref, double sigma = NLMeans_Default.sigma, double strength = NLMeans_Default.strength,
    PCType GroupSizeMax = NLMeans_Default.GroupSizeMax, PCType BlockSize = NLMeans_Default.BlockSize, PCType Overlap = NLMeans_Default.Overlap,
    PCType BMrange = NLMeans_Default.BMrange, PCType BMstep = NLMeans_Default.BMstep, double thMSE = NLMeans_Default.thMSE);


inline Plane NLMeans(const Plane &src, const Plane &ref, double sigma = NLMeans_Default.sigma, double strength = NLMeans_Default.strength,
    PCType GroupSizeMax = NLMeans_Default.GroupSizeMax, PCType BlockSize = NLMeans_Default.BlockSize, PCType Overlap = NLMeans_Default.Overlap,
    PCType BMrange = NLMeans_Default.BMrange, PCType BMstep = NLMeans_Default.BMstep, double thMSE = NLMeans_Default.thMSE)
{
    Plane dst(src, false);

    return NLMeans(dst, src, ref, sigma, strength, GroupSizeMax, BlockSize, Overlap, BMrange, BMstep, thMSE);
}

inline Plane NLMeans(const Plane &src, double sigma = NLMeans_Default.sigma, double strength = NLMeans_Default.strength,
    PCType GroupSizeMax = NLMeans_Default.GroupSizeMax, PCType BlockSize = NLMeans_Default.BlockSize, PCType Overlap = NLMeans_Default.Overlap,
    PCType BMrange = NLMeans_Default.BMrange, PCType BMstep = NLMeans_Default.BMstep, double thMSE = NLMeans_Default.thMSE)
{
    Plane dst(src, false);

    return NLMeans(dst, src, src, sigma, strength, GroupSizeMax, BlockSize, Overlap, BMrange, BMstep, thMSE);
}

inline Frame NLMeans(const Frame &src, const Frame &ref, double sigma = NLMeans_Default.sigma, double strength = NLMeans_Default.strength,
    PCType GroupSizeMax = NLMeans_Default.GroupSizeMax, PCType BlockSize = NLMeans_Default.BlockSize, PCType Overlap = NLMeans_Default.Overlap,
    PCType BMrange = NLMeans_Default.BMrange, PCType BMstep = NLMeans_Default.BMstep, double thMSE = NLMeans_Default.thMSE)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < dst.PlaneCount(); i++)
    {
        NLMeans(dst.P(i), src.P(i), ref.P(i), sigma, strength, GroupSizeMax, BlockSize, Overlap, BMrange, BMstep, thMSE);
    }

    return dst;
}

inline Frame NLMeans(const Frame &src, double sigma = NLMeans_Default.sigma, double strength = NLMeans_Default.strength,
    PCType GroupSizeMax = NLMeans_Default.GroupSizeMax, PCType BlockSize = NLMeans_Default.BlockSize, PCType Overlap = NLMeans_Default.Overlap,
    PCType BMrange = NLMeans_Default.BMrange, PCType BMstep = NLMeans_Default.BMstep, double thMSE = NLMeans_Default.thMSE)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < dst.PlaneCount(); i++)
    {
        NLMeans(dst.P(i), src.P(i), src.P(i), sigma, strength, GroupSizeMax, BlockSize, Overlap, BMrange, BMstep, thMSE);
    }

    return dst;
}


class NLMeans_IO
    : public FilterIO
{
protected:
    std::string RPath;
    double sigma = NLMeans_Default.sigma;
    double strength = sigma * 5;
    PCType GroupSizeMax = NLMeans_Default.GroupSizeMax;
    PCType BlockSize = NLMeans_Default.BlockSize;
    PCType Overlap = NLMeans_Default.Overlap;
    PCType BMrange = NLMeans_Default.BMrange;
    PCType BMstep = NLMeans_Default.BMstep;
    double thMSE = sigma * 50;

    virtual void arguments_process()
    {
        FilterIO::arguments_process();

        Args ArgsObj(argc, args);

        bool strength_def = false;
        bool thMSE_def = false;

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "--ref")
            {
                ArgsObj.GetPara(i, RPath);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, sigma);
                if (!strength_def) strength = sigma * 5;
                if (!thMSE_def) thMSE = sigma * 50;
                continue;
            }
            if (args[i] == "-H" || args[i] == "--strength")
            {
                ArgsObj.GetPara(i, strength);
                strength_def = true;
                continue;
            }
            if (args[i] == "-GSM" || args[i] == "--GroupSizeMax")
            {
                ArgsObj.GetPara(i, GroupSizeMax);
                continue;
            }
            if (args[i] == "-B" || args[i] == "--BlockSize")
            {
                ArgsObj.GetPara(i, BlockSize);
                continue;
            }
            if (args[i] == "-O" || args[i] == "--Overlap")
            {
                ArgsObj.GetPara(i, Overlap);
                continue;
            }
            if (args[i] == "-MR" || args[i] == "--BMrange")
            {
                ArgsObj.GetPara(i, BMrange);
                continue;
            }
            if (args[i] == "-MS" || args[i] == "--BMstep")
            {
                ArgsObj.GetPara(i, BMstep);
                continue;
            }
            if (args[i] == "-TH" || args[i] == "--thMSE")
            {
                ArgsObj.GetPara(i, thMSE);
                thMSE_def = true;
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
        if (RPath.size() == 0)
        {
            return NLMeans(src, sigma, strength, GroupSizeMax, BlockSize, Overlap, BMrange, BMstep, thMSE);
        }
        else
        {
            const Frame ref = ImageReader(RPath);
            return NLMeans(src, ref, sigma, strength, GroupSizeMax, BlockSize, Overlap, BMrange, BMstep, thMSE);
        }
    }

public:
    NLMeans_IO(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".NLMeans")
        : FilterIO(_argc, _args, _Tag) {}

    ~NLMeans_IO() {}
};


#endif
