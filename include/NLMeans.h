#ifndef NLMEANS_H_
#define NLMEANS_H_


#include "Filter.h"
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
    : public FilterIF2
{
public:
    typedef NLMeans _Myt;
    typedef FilterIF2 _Mybase;

protected:
    NLMeans_Para para;

public:
    _Myt(const NLMeans_Para &_para = NLMeans_Default)
        : para(_para)
    {}

protected:
    virtual Plane &process_Plane(Plane &dst, const Plane &src, const Plane &ref);

    virtual Frame &process_Frame(Frame &dst, const Frame &src, const Frame &ref);

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
public:
    typedef NLMeans_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    NLMeans_Para para;
    std::string RPath;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

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
                ArgsObj.GetPara(i, para.correction);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, para.sigma);
                continue;
            }
            if (args[i] == "-H" || args[i] == "--strength")
            {
                ArgsObj.GetPara(i, para.strength);
                strength_def = true;
                continue;
            }
            if (args[i] == "-GSM" || args[i] == "--GroupSizeMax")
            {
                ArgsObj.GetPara(i, para.GroupSizeMax);
                continue;
            }
            if (args[i] == "-B" || args[i] == "--BlockSize")
            {
                ArgsObj.GetPara(i, para.BlockSize);
                continue;
            }
            if (args[i] == "-O" || args[i] == "--Overlap")
            {
                ArgsObj.GetPara(i, para.Overlap);
                continue;
            }
            if (args[i] == "-MR" || args[i] == "--BMrange")
            {
                ArgsObj.GetPara(i, para.BMrange);
                continue;
            }
            if (args[i] == "-MS" || args[i] == "--BMstep")
            {
                ArgsObj.GetPara(i, para.BMstep);
                continue;
            }
            if (args[i] == "-TH" || args[i] == "--thMSE")
            {
                ArgsObj.GetPara(i, para.thMSE);
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

        if (!strength_def) para.strength = para.correction ? para.sigma * 5 : para.sigma * 1.5;
        if (!thMSE_def) para.thMSE = para.correction ? para.sigma * 50 : para.sigma * 25;
    }

    virtual Frame process(const Frame &src)
    {
        NLMeans filter(para);

        if (RPath.size() == 0)
        {
            return filter(src);
        }
        else
        {
            const Frame ref = ImageReader(RPath);
            return filter(src, ref);
        }
    }

public:
    _Myt(std::string _Tag = ".NLMeans")
        : _Mybase(std::move(_Tag)) {}
};


#endif
