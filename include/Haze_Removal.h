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

    std::vector<double> sigmaVector;

    int ppmode = 3;
    double lower_thr = 0.02;
    double upper_thr = 0.01;
    Histogram<FLType>::BinType HistBins = 1024;

    Haze_Removal_Para() : sigmaVector({ 25.0, 250.0 }) {}
} Haze_Removal_Default;


class Haze_Removal
{
public:
    typedef Haze_Removal _Myt;

private:
    double tMap_thr;
    FLType ALmax;
    FLType tMapMin;
    FLType strength;
    int ppmode;
    double lower_thr;
    double upper_thr;
    Histogram<FLType>::BinType HistBins;

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
    _Myt(const Haze_Removal_Para &Para = Haze_Removal_Default)
        : tMap_thr(Para.tMap_thr), ALmax(Para.ALmax), tMapMin(Para.tMapMin), strength(Para.strength),
        ppmode(Para.ppmode), lower_thr(Para.lower_thr), upper_thr(Para.upper_thr), HistBins(Para.HistBins)
    {}

    Frame &process(Frame &dst, const Frame &src);

    Frame process(const Frame &src)
    {
        Frame dst(src, false);
        return process(dst, src);
    }

protected:
    // Generate the Inverted Transmission Map from intensity image
    virtual void GetTMapInv(const Frame &src) = 0;

    // Get the Global Atmospheric Light from Inverted Transmission Map and src
    void GetAirLight();

    // Generate the haze-free image
    void GetHazeFree(Frame &dst);
};


class Haze_Removal_Retinex
    : public Haze_Removal
{
public:
    typedef Haze_Removal_Retinex _Myt;
    typedef Haze_Removal _Mybase;

private:
    std::vector<double> sigmaVector;

public:
    _Myt(const Haze_Removal_Para &Para = Haze_Removal_Default)
        : _Mybase(Para), sigmaVector(Para.sigmaVector)
    {}

protected:
    // Generate the Inverted Transmission Map from intensity image
    void GetTMapInv(const Frame &src);
};


class Haze_Removal_IO
    : public FilterIO
{
public:
    typedef Haze_Removal_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    Haze_Removal_Para Para;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "--tMap_thr")
            {
                ArgsObj.GetPara(i, Para.tMap_thr);
                continue;
            }
            if (args[i] == "--ALmax")
            {
                ArgsObj.GetPara(i, Para.ALmax);
                continue;
            }
            if (args[i] == "--tMapMin")
            {
                ArgsObj.GetPara(i, Para.tMapMin);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--strength")
            {
                ArgsObj.GetPara(i, Para.strength);
                continue;
            }
            if (args[i] == "-PP" || args[i] == "--ppmode")
            {
                ArgsObj.GetPara(i, Para.ppmode);
                continue;
            }
            if (args[i] == "-L" || args[i] == "--lower_thr")
            {
                ArgsObj.GetPara(i, Para.lower_thr);
                continue;
            }
            if (args[i] == "-U" || args[i] == "--upper_thr")
            {
                ArgsObj.GetPara(i, Para.upper_thr);
                continue;
            }
            if (args[i] == "-HB" || args[i] == "--HistBins")
            {
                ArgsObj.GetPara(i, Para.HistBins);
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

    virtual Frame processFrame(const Frame &src) = 0;

public:
    _Myt(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".Haze_Removal")
        : _Mybase(_argc, _args, _Tag) {}
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
        Para.sigmaVector.clear();

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, sigma);
                Para.sigmaVector.push_back(sigma);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        ArgsObj.Check();

        if (Para.sigmaVector.size() == 0)
        {
            Para.sigmaVector = Haze_Removal_Default.sigmaVector;
        }
    }

    virtual Frame processFrame(const Frame &src)
    {
        Haze_Removal_Retinex filter(Para);
        return filter.process(src);
    }

public:
    _Myt(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".Haze_Removal_Retinex")
        : _Mybase(_argc, _args, _Tag) {}
};


#endif
