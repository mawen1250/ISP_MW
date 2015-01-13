#ifndef RETINEX_H_
#define RETINEX_H_


#include "IO.h"
#include "Image_Type.h"
#include "Histogram.h"


const struct Retinex_Para
{
    double sigma = 100.0;
    std::vector<double> sigmaVector;

    double lower_thr = 0.001;
    double upper_thr = 0.001;
    Histogram<FLType>::BinType HistBins = 4096;

    double chroma_protect = 1.2;

    double restore = 125.0;

    double dynamic = 10.0;

    Retinex_Para() : sigmaVector({ 25.0, 80.0, 250.0 }) {}
} Retinex_Default;


Plane_FL Retinex_MSR(const Plane_FL & idata, const std::vector<double> & sigmaVector);
Plane_FL Retinex_MSR(const Plane_FL & idata, const std::vector<double> & sigmaVector, double lower_thr, double upper_thr);
Plane_FL Retinex_MSRCR_GIMP(const Plane_FL & idata, const std::vector<double> & sigmaVector, double dynamic);

Plane & Retinex_SSR(Plane & dst, const Plane & src, double sigma = Retinex_Default.sigma,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr);
Plane & Retinex_MSR(Plane & dst, const Plane & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr);
Frame & Retinex_MSRCP(Frame & dst, const Frame & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr,
    double chroma_protect = Retinex_Default.chroma_protect);
Frame & Retinex_MSRCR(Frame & dst, const Frame & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr,
    double restore = Retinex_Default.restore);
Frame & Retinex_MSRCR_GIMP(Frame & dst, const Frame & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double dynamic = Retinex_Default.dynamic);

inline Plane Retinex_SSR(const Plane & src, double sigma = Retinex_Default.sigma,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr)
{
    Plane dst(src, false);
    return Retinex_SSR(dst, src, sigma, lower_thr, upper_thr);
}

inline Frame Retinex_SSR(const Frame & src, double sigma = Retinex_Default.sigma,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr)
{
    Frame dst(src, false);
    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
        Retinex_SSR(dst.P(i), src.P(i), sigma, lower_thr, upper_thr);
    return dst;
}

inline Plane Retinex_MSR(const Plane & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr)
{
    Plane dst(src, false);
    return Retinex_MSR(dst, src, sigmaVector, lower_thr, upper_thr);
}
inline Frame Retinex_MSRCP(const Frame & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr,
    double chroma_protect = Retinex_Default.chroma_protect)
{
    Frame dst(src, false);
    return Retinex_MSRCP(dst, src, sigmaVector, lower_thr, upper_thr, chroma_protect);
}

inline Frame Retinex_MSRCR(const Frame & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double lower_thr = Retinex_Default.lower_thr, double upper_thr = Retinex_Default.upper_thr,
    double restore = Retinex_Default.restore)
{
    Frame dst(src, false);
    return Retinex_MSRCR(dst, src, sigmaVector, lower_thr, upper_thr, restore);
}

inline Frame Retinex_MSRCR_GIMP(const Frame & src, const std::vector<double> & sigmaVector = Retinex_Default.sigmaVector,
    double dynamic = Retinex_Default.dynamic)
{
    Frame dst(src, false);
    return Retinex_MSRCR_GIMP(dst, src, sigmaVector, dynamic);
}


class Retinex_MSR_IO
    : public FilterIO
{
public:
    typedef Retinex_MSR_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    std::vector<double> sigmaVector = Retinex_Default.sigmaVector;
    double lower_thr = Retinex_Default.lower_thr;
    double upper_thr = Retinex_Default.upper_thr;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);
        double sigma;
        sigmaVector.clear();

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                ArgsObj.GetPara(i, sigma);
                sigmaVector.push_back(sigma);
                continue;
            }
            if (args[i] == "-L" || args[i] == "--lower_thr")
            {
                ArgsObj.GetPara(i, lower_thr);
                continue;
            }
            if (args[i] == "-U" || args[i] == "--upper_thr")
            {
                ArgsObj.GetPara(i, upper_thr);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        ArgsObj.Check();

        if (sigmaVector.size() == 0)
        {
            sigmaVector = Retinex_Default.sigmaVector;
        }
    }

    virtual Frame processFrame(const Frame &src) = 0;

public:
    _Myt(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".MSR")
        : _Mybase(_argc, _args, _Tag) {}
};


class Retinex_MSRCP_IO
    : public Retinex_MSR_IO
{
public:
    typedef Retinex_MSRCP_IO _Myt;
    typedef Retinex_MSR_IO _Mybase;

protected:
    double chroma_protect = Retinex_Default.chroma_protect;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-CP" || args[i] == "--chroma_protect")
            {
                ArgsObj.GetPara(i, chroma_protect);
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
        return Retinex_MSRCP(src, sigmaVector, lower_thr, upper_thr, chroma_protect);
    }

public:
    _Myt(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".MSRCP")
        : _Mybase(_argc, _args, _Tag) {}
};


class Retinex_MSRCR_IO
    : public Retinex_MSR_IO
{
public:
    typedef Retinex_MSRCR_IO _Myt;
    typedef Retinex_MSR_IO _Mybase;

protected:
    double restore = Retinex_Default.restore;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-R" || args[i] == "--restore")
            {
                ArgsObj.GetPara(i, restore);
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
        return Retinex_MSRCR(src, sigmaVector, lower_thr, upper_thr, restore);
    }

public:
    _Myt(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".MSRCR")
        : _Mybase(_argc, _args, _Tag) {}
};


class Retinex_MSRCR_GIMP_IO
    : public Retinex_MSR_IO
{
public:
    typedef Retinex_MSRCR_GIMP_IO _Myt;
    typedef Retinex_MSR_IO _Mybase;

protected:
    double dynamic = Retinex_Default.dynamic;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-D" || args[i] == "--dynamic")
            {
                ArgsObj.GetPara(i, dynamic);
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
        return Retinex_MSRCR_GIMP(src, sigmaVector, dynamic);
    }

public:
    _Myt(int _argc, const std::vector<std::string> &_args, std::string _Tag = ".MSRCR_GIMP")
        : _Mybase(_argc, _args, _Tag) {}
};


#endif