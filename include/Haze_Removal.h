#ifndef HAZE_REMOVAL_H_
#define HAZE_REMOVAL_H_


#include "IO.h"
#include "Image_Type.h"
#include "Histogram.h"


const struct Haze_Removal_Para
{
    double tMap_thr = 0.001;
    FLType ALmax = 1;
    FLType tMapMin = 0.1;
    FLType strength = 0.65;
    int ppmode = 3;
    double lower_thr = 0.02;
    double upper_thr = 0.01;
    Histogram<FLType>::BinType HistBins = 1024;

    int Ymode = 1;
    std::vector<double> sigmaVector;

    Haze_Removal_Para() : sigmaVector({ 25.0, 250.0 }) {}
} Haze_Removal_Default;


class Haze_Removal
{
public:
    typedef Haze_Removal _Myt;

protected:
    Haze_Removal_Para para;

protected:
    Plane_FL dataR;
    Plane_FL dataG;
    Plane_FL dataB;
    PCType height;
    PCType width;
    PCType stride;

    Plane_FL tMapInv;
    FLType AL_R;
    FLType AL_G;
    FLType AL_B;

public:
    _Myt(const Haze_Removal_Para &_para = Haze_Removal_Default)
        : para(_para)
    {}

    Frame &process(Frame &dst, const Frame &src);

    Frame operator()(const Frame &src)
    {
        Frame dst(src, false);
        return process(dst, src);
    }

protected:
    // Generate the Inverted Transmission Map from intensity image
    virtual void GetTMapInv(const Frame &src) = 0;

    // Get the Global Atmospheric Light from Inverted Transmission Map and src
    void GetAtmosLight();

    // Generate the haze-free image
    void RemoveHaze();

    // Store the filtered result to a Frame with range scaling
    void StoreResult(Frame &dst);
};


class Haze_Removal_Retinex
    : public Haze_Removal
{
public:
    typedef Haze_Removal_Retinex _Myt;
    typedef Haze_Removal _Mybase;

public:
    _Myt(const Haze_Removal_Para &_para = Haze_Removal_Default)
        : _Mybase(_para)
    {}

protected:
    // Generate the Inverted Transmission Map from intensity image
    virtual void GetTMapInv(const Frame &src);
};


class Haze_Removal_IO
    : public FilterIO
{
public:
    typedef Haze_Removal_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    Haze_Removal_Para para;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "--tMap_thr")
            {
                ArgsObj.GetPara(i, para.tMap_thr);
                continue;
            }
            if (args[i] == "--ALmax")
            {
                ArgsObj.GetPara(i, para.ALmax);
                continue;
            }
            if (args[i] == "--tMapMin")
            {
                ArgsObj.GetPara(i, para.tMapMin);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--strength")
            {
                ArgsObj.GetPara(i, para.strength);
                continue;
            }
            if (args[i] == "-PP" || args[i] == "--ppmode")
            {
                ArgsObj.GetPara(i, para.ppmode);
                continue;
            }
            if (args[i] == "-L" || args[i] == "--lower_thr")
            {
                ArgsObj.GetPara(i, para.lower_thr);
                continue;
            }
            if (args[i] == "-U" || args[i] == "--upper_thr")
            {
                ArgsObj.GetPara(i, para.upper_thr);
                continue;
            }
            if (args[i] == "-HB" || args[i] == "--HistBins")
            {
                ArgsObj.GetPara(i, para.HistBins);
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

    virtual Frame process(const Frame &src) = 0;

public:
    _Myt(std::string _Tag = ".Haze_Removal")
        : _Mybase(std::move(_Tag)) {}
};


class Haze_Removal_Retinex_IO
    : public Haze_Removal_IO
{
public:
    typedef Haze_Removal_Retinex_IO _Myt;
    typedef Haze_Removal_IO _Mybase;

protected:
    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);
        double sigma;
        para.sigmaVector.clear();

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-Y" || args[i] == "--Ymode")
            {
                ArgsObj.GetPara(i, para.Ymode);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, sigma);
                para.sigmaVector.push_back(sigma);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        ArgsObj.Check();

        if (para.sigmaVector.size() == 0)
        {
            para.sigmaVector = Haze_Removal_Default.sigmaVector;
        }
    }

    virtual Frame process(const Frame &src)
    {
        Haze_Removal_Retinex filter(para);
        return filter(src);
    }

public:
    _Myt(std::string _Tag = ".Haze_Removal_Retinex")
        : _Mybase(std::move(_Tag)) {}
};


#endif
