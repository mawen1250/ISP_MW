#ifndef NLMEANS_H_
#define NLMEANS_H_


#include "IO.h"
#include "Image_Type.h"
#include "Helper.h"
#include "Block.h"


const struct NLMeans_Para
{
    bool correction = true;
    double sigma = 8.0;
    double strength = correction ? sigma * 5 : sigma * 1.5;
    PCType GroupSizeMax = 16;
    PCType BlockSize = 8;
    PCType Overlap = 4;
    PCType BMrange = 24;
    PCType BMstep = 3;
    double thMSE = correction ? sigma * 50 : sigma * 25;
} NLMeans_Default;


// Non-local Means denoising algorithm based on block matching and weighted average of grouped blocks
class NLMeans
{
private:
    bool correction;
    double sigma;
    double strength;
    PCType GroupSizeMax;
    PCType BlockSize;
    PCType Overlap;
    PCType BMrange;
    PCType BMstep;
    double thMSE;

public:
    NLMeans(bool _correction = NLMeans_Default.correction, double _sigma = NLMeans_Default.sigma, double _strength = NLMeans_Default.strength,
        PCType _GroupSizeMax = NLMeans_Default.GroupSizeMax, PCType _BlockSize = NLMeans_Default.BlockSize, PCType _Overlap = NLMeans_Default.Overlap,
        PCType _BMrange = NLMeans_Default.BMrange, PCType _BMstep = NLMeans_Default.BMstep, double _thMSE = NLMeans_Default.thMSE)
        : correction(_correction), sigma(_sigma), strength(_strength),
        GroupSizeMax(_GroupSizeMax), BlockSize(_BlockSize), Overlap(_Overlap),
        BMrange(_BMrange), BMstep(_BMstep), thMSE(_thMSE)
    {}

    ~NLMeans()
    {}

    Plane &process(Plane &dst, const Plane &src, const Plane &ref);

    Plane process(const Plane &src, const Plane &ref)
    {
        Plane dst(src, false);

        return process(dst, src, ref);
    }

    Plane process(const Plane &src)
    {
        Plane dst(src, false);

        return process(dst, src, src);
    }

    Frame &process(Frame &dst, const Frame &src, const Frame &ref);

    Frame process(const Frame &src, const Frame &ref)
    {
        Frame dst(src, false);

        return process(dst, src, ref);
    }

    Frame process(const Frame &src)
    {
        Frame dst(src, false);

        return process(dst, src, src);
    }

protected:
    template < typename _St1, typename _Ty, typename _FTy >
    void WeightedAverage(Block<_Ty, _FTy> &dstBlock, const Block<_Ty, _FTy> &refBlock, const _St1 &src,
        const typename Block<_Ty, _FTy>::PosPairCode &posPairCode);

    template < typename _St1, typename _Ty, typename _FTy >
    void WeightedAverage_Correction(Block<_Ty, _FTy> &dstBlock, const Block<_Ty, _FTy> &refBlock, const _St1 &src,
        const typename Block<_Ty, _FTy>::PosPairCode &posPairCode);
};


class NLMeans_IO
    : public FilterIO
{
protected:
    std::string RPath;
    bool correction = NLMeans_Default.correction;
    double sigma = NLMeans_Default.sigma;
    double strength = NLMeans_Default.strength;
    PCType GroupSizeMax = NLMeans_Default.GroupSizeMax;
    PCType BlockSize = NLMeans_Default.BlockSize;
    PCType Overlap = NLMeans_Default.Overlap;
    PCType BMrange = NLMeans_Default.BMrange;
    PCType BMstep = NLMeans_Default.BMstep;
    double thMSE = NLMeans_Default.thMSE;

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
            if (args[i] == "-C" || args[i] == "--correction")
            {
                ArgsObj.GetPara(i, correction);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, sigma);
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

        if (!strength_def) strength = correction ? sigma * 5 : sigma * 1.5;
        if (!thMSE_def) thMSE = correction ? sigma * 50 : sigma * 25;
    }

    virtual Frame processFrame(const Frame &src)
    {
        NLMeans filter(correction, sigma, strength, GroupSizeMax, BlockSize, Overlap, BMrange, BMstep, thMSE);

        if (RPath.size() == 0)
        {
            return filter.process(src);
        }
        else
        {
            const Frame ref = ImageReader(RPath);
            return filter.process(src, ref);
        }
    }

public:
    NLMeans_IO(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".NLMeans")
        : FilterIO(_argc, _args, _Tag) {}

    ~NLMeans_IO() {}
};


#endif
