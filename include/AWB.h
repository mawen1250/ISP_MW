#ifndef AWB_H_
#define AWB_H_


#include "IO.h"
#include "Image_Type.h"
#include "Helper.h"
#include "LUT.h"
#include "Histogram.h"
#include "Convolution.h"
#include "Gaussian.h"


const struct AWB1_Para
{} AWB1_Default;

const struct AWB2_Para
{} AWB2_Default;


class AWB
{
protected:
    const Frame &src;
    const Plane &srcR = src.R();
    const Plane &srcG = src.G();
    const Plane &srcB = src.B();
    Frame dst;
    Plane &dstR = dst.R();
    Plane &dstG = dst.G();
    Plane &dstB = dst.B();

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Stride();

    DType sFloor;
    DType sCeil;
    DType dFloor;
    DType dCeil;
    FLType dFloorFL;
    FLType dCeilFL;

    uint64 sumR = 0, sumG = 0, sumB = 0, sumMax = 0;

    FLType gainR = 1;
    FLType gainG = 1;
    FLType gainB = 1;
    FLType src_offsetR = 0;
    FLType src_offsetG = 0;
    FLType src_offsetB = 0;
    FLType dst_offsetR = 0;
    FLType dst_offsetG = 0;
    FLType dst_offsetB = 0;

protected:
    virtual void kernel() = 0;

    DType Hist_value(const Histogram<DType> &Hist, FLType ratio = 1.) const;
    Plane GetIntensity(const Frame &src) const;
    void sum(const Frame &ref, FLType lower_ratio = 0., FLType upper_ratio = 1.);
    void sum_masked(const Frame &ref, const Plane &mask, DType thr);
    void sum_masked(const Frame &ref, const Plane &mask) { sum_masked(ref, mask, mask.Floor() + 1); }
    void apply_gain() const;

public:
    AWB(const Frame &_src) 
        : src(_src), srcR(src.R()), srcG(src.G()), srcB(src.B()), dst(_src, false), dstR(dst.R()), dstG(dst.G()), dstB(dst.B()),
        height(src.Height()), width(src.Width()), stride(src.Stride()),
        sFloor(srcR.Floor()), sCeil(srcR.Ceil()), dFloor(dstR.Floor()), dCeil(dstR.Ceil()), dFloorFL(dFloor), dCeilFL(dCeil)
    {
        if (!src.isRGB())
        {
            const char * FunctionName = "class AWB";
            std::cerr << FunctionName << ": invalid PixelType of Frame \"src\", should be RGB.\n";
            exit(EXIT_FAILURE);
        }
    }

    virtual ~AWB() {}

    Frame &process();
};


class AWB1
    : public AWB
{
protected:
    void kernel();

public:
    AWB1(const Frame &_src)
        : AWB(_src)
    {}
};


class AWB2
    : public AWB
{
protected:
    void kernel();

public:
    AWB2(const Frame &_src)
        : AWB(_src)
    {}
};


class AWB_IO
    : public FilterIO
{
public:
    typedef AWB_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    virtual void arguments_process()
    {
        FilterIO::arguments_process();
    }

    virtual Frame process(const Frame &src) = 0;

public:
    _Myt(std::string _Tag = ".AWB")
        : _Mybase(std::move(_Tag)) {}
};


class AWB1_IO
    : public AWB_IO
{
public:
    typedef AWB1_IO _Myt;
    typedef AWB_IO _Mybase;

protected:
    virtual void arguments_process()
    {
        AWB_IO::arguments_process();
    }

    virtual Frame process(const Frame &src)
    {
        AWB1 Filter(src);
        return Filter.process();
    }

public:
    _Myt(std::string _Tag = ".AWB1")
        : _Mybase(std::move(_Tag)) {}
};


class AWB2_IO
    : public AWB_IO
{
public:
    typedef AWB2_IO _Myt;
    typedef AWB_IO _Mybase;

protected:
    virtual void arguments_process()
    {
        AWB_IO::arguments_process();
    }

    virtual Frame process(const Frame &src)
    {
        AWB2 Filter(src);
        return Filter.process();
    }

public:
    _Myt(std::string _Tag = ".AWB2")
        : _Mybase(std::move(_Tag)) {}
};


#endif
