#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_


#include "Filter.h"
#include "Image_Type.h"


enum class EdgeKernel
{
    Sobel = 0,
    Prewitt,
    Laplace1 = 10,
    Laplace2,
    Laplace3
};


const struct EdgeDetect_Para
{
    EdgeKernel Kernel = EdgeKernel::Sobel;
} ED_Default;


Plane & Convolution3V(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, bool norm = true);
Plane & Convolution3H(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, bool norm = true);
Plane & Convolution3(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, FLType K3, FLType K4, FLType K5, FLType K6, FLType K7, FLType K8, bool norm = true);
Plane & FirstOrderDerivative3(Plane &dst, const Plane &src, FLType K0, FLType K1, FLType K2, FLType K6, FLType K7, FLType K8, bool norm = true);
Plane & EdgeDetect(Plane &dst, const Plane &src, EdgeKernel Kernel = ED_Default.Kernel);


inline Plane Convolution3V(const Plane &src, FLType K0, FLType K1, FLType K2, bool norm = true)
{
    Plane dst(src, false);
    return Convolution3V(dst, src, K0, K1, K2, norm);
}

inline Plane Convolution3H(const Plane &src, FLType K0, FLType K1, FLType K2, bool norm = true)
{
    Plane dst(src, false);
    return Convolution3H(dst, src, K0, K1, K2, norm);
}

inline Plane Convolution3(const Plane &src, FLType K0, FLType K1, FLType K2, FLType K3, FLType K4, FLType K5, FLType K6, FLType K7, FLType K8, bool norm = true)
{
    Plane dst(src, false);
    return Convolution3(dst, src, K0, K1, K2, K3, K4, K5, K6, K7, K8, norm);
}

inline Plane EdgeDetect(const Plane &src, EdgeKernel Kernel = ED_Default.Kernel)
{
    Plane dst(src, false);
    return EdgeDetect(dst, src, Kernel);
}


inline Frame Convolution3V(const Frame &src, FLType K0, FLType K1, FLType K2, bool norm = true)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
    {
        Convolution3V(dst.P(i), src.P(i), K0, K1, K2, norm);
    }

    return dst;
}

inline Frame Convolution3H(const Frame &src, FLType K0, FLType K1, FLType K2, bool norm = true)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
    {
        Convolution3H(dst.P(i), src.P(i), K0, K1, K2, norm);
    }

    return dst;
}

inline Frame Convolution3(const Frame &src, FLType K0, FLType K1, FLType K2, FLType K3, FLType K4, FLType K5, FLType K6, FLType K7, FLType K8, bool norm = true)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
    {
        Convolution3(dst.P(i), src.P(i), K0, K1, K2, K3, K4, K5, K6, K7, K8, norm);
    }

    return dst;
}

inline Frame EdgeDetect(const Frame &src, EdgeKernel Kernel = ED_Default.Kernel)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
    {
        EdgeDetect(dst.P(i), src.P(i), Kernel);
    }

    return dst;
}


class EdgeDetect_IO
    : public FilterIO
{
public:
    typedef EdgeDetect_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    EdgeKernel Kernel = ED_Default.Kernel;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);
        std::string KernelStr;

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-K" || args[i] == "--kernel")
            {
                ArgsObj.GetPara(i, KernelStr, 1);

                if (KernelStr == "sobel")
                    Kernel = EdgeKernel::Sobel;
                else if (KernelStr == "prewitt")
                    Kernel = EdgeKernel::Prewitt;
                else if (KernelStr == "laplace1")
                    Kernel = EdgeKernel::Laplace1;
                else if (KernelStr == "laplace2")
                    Kernel = EdgeKernel::Laplace2;
                else if (KernelStr == "laplace3")
                    Kernel = EdgeKernel::Laplace3;

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
        return EdgeDetect(src, Kernel);
    }

public:
    _Myt(std::string _Tag = ".EdgeDetect")
        : _Mybase(std::move(_Tag)) {}
};


#endif
