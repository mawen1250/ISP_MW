#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include <iostream>
#include "Image_Type.h"


template < typename _Ty = DType >
class Histogram
{
public:
    typedef Histogram<_Ty> _Myt;

    typedef sint32 BinType;
    typedef sint32 CountType;

private:
    _Ty Lower_ = 0;
    _Ty Upper_ = -1;
    BinType Bins_ = 0;
    FLType Scale_ = 1;
    CountType Count_ = 0;
    CountType *Data_ = nullptr;

protected:
    template < typename T0 > BinType ToBinType(const T0 input) const { return static_cast<BinType>(input); }
    template < > BinType ToBinType<float>(const float input) const { return static_cast<BinType>(input + 0.5F); }
    template < > BinType ToBinType<double>(const double input) const { return static_cast<BinType>(input + 0.5); }
    template < > BinType ToBinType<ldbl>(const ldbl input) const { return static_cast<BinType>(input + 0.5L); }

public:
    Histogram() {} // Default constructor
    Histogram(const Histogram &src); // Copy constructor
    Histogram(Histogram &&src); // Move constructor
    explicit Histogram(const Plane &src); // Convertor/Constructor from Plane
    Histogram(const Plane &src, BinType Bins);
    Histogram(const Plane &src, _Ty Lower, _Ty Upper, BinType Bins);
    explicit Histogram(const Plane_FL &src, BinType Bins = 256);
    Histogram(const Plane_FL &src, _Ty Lower, _Ty Upper, BinType Bins = 256);
    Histogram(const _Ty *src, CountType _pcount, _Ty Lower, _Ty Upper); // Template constructor from array
    Histogram(const _Ty *src, CountType _pcount, _Ty Lower, _Ty Upper, BinType Bins); // Template constructor from array
    ~Histogram(); // Destructor

    Histogram &operator=(const Histogram &src); // Copy assignment operator
    Histogram &operator=(Histogram &&src); // Move assignment operator
    CountType &operator[](BinType i) { return Data_[i]; }
    const CountType &operator[](BinType i) const { return Data_[i]; }

    _Ty Lower() const { return Lower_; }
    _Ty Upper() const { return Upper_; }
    BinType Bins() const { return Bins_; }
    FLType Scale() const { return Scale_; }
    CountType Count() const { return Count_; }
    CountType *Data() { return Data_; }
    const CountType *Data() const { return Data_; }

    template < typename T0 > T0 BinToValue(BinType i) const { return static_cast<T0>(i / Scale_ + FLType(0.5)) + Lower_; }
    template < > float BinToValue<float>(BinType i) const { return static_cast<float>(i / Scale_) + Lower_; }
    template < > double BinToValue<double>(BinType i) const { return static_cast<double>(i / Scale_) + Lower_; }
    template < > ldbl BinToValue<ldbl>(BinType i) const { return static_cast<ldbl>(i / Scale_) + Lower_; }

    void Generate(const _Ty *src, CountType _pcount);
    void Add(const _Ty *src, CountType _pcount);
    template < typename _St1 > void Generate(const _St1 &src);
    template < typename _St1 > void Add(const _St1 &src);

    _Ty Min(double ratio = 0) const;
    _Ty Max(double ratio = 0) const;
    CountType Lookup(const _Ty input) const { return Data_[Clip(static_cast<BinType>((input - Lower_) * Scale_), BinType(0), Bins_ - BinType(1))]; }
};


// Template functions of class Histogram
template < typename _Ty >
Histogram<_Ty>::Histogram(const Histogram<_Ty> &src)
    : Lower_(src.Lower_), Upper_(src.Upper_), Bins_(src.Bins_), Scale_(src.Scale_), Count_(src.Count_)
{
    Data_ = new CountType[Bins_];

    memcpy(Data_, src.Data_, sizeof(CountType) * Bins_);
}

template < typename _Ty >
Histogram<_Ty>::Histogram(Histogram<_Ty> &&src)
    : Lower_(src.Lower_), Upper_(src.Upper_), Bins_(src.Bins_), Scale_(src.Scale_), Count_(src.Count_)
{
    Data_ = src.Data_;

    src.Lower_ = 0;
    src.Upper_ = -1;
    src.Bins_ = 0;
    src.Scale_ = 1;
    src.Count_ = 0;
    src.Data_ = nullptr;
}

template < typename _Ty >
Histogram<_Ty>::Histogram(const Plane &src)
    : Histogram(src, src.Floor(), src.Ceil(), Upper_ - Lower_ + 1)
{}

template < typename _Ty >
Histogram<_Ty>::Histogram(const Plane &src, BinType Bins)
    : Histogram(src, src.Floor(), src.Ceil(), Bins)
{}

template < typename _Ty >
Histogram<_Ty>::Histogram(const Plane &src, _Ty Lower, _Ty Upper, BinType Bins)
    : Lower_(Lower), Upper_(Upper), Bins_(Bins), Scale_((Bins_ - 1) / static_cast<FLType>(Upper_ - Lower_))
{
    if (Scale_ > 1)
    {
        Bins_ = ToBinType(Upper_ - Lower_ + 1);
        Scale_ = 1;
    }

    Data_ = new CountType[Bins_];

    Generate(src);
}

template < typename _Ty >
Histogram<_Ty>::Histogram(const Plane_FL &src, BinType Bins)
    : Histogram(src, src.Floor(), src.Ceil(), Bins)
{}

template < typename _Ty >
Histogram<_Ty>::Histogram(const Plane_FL &src, _Ty Lower, _Ty Upper, BinType Bins)
    : Lower_(Lower), Upper_(Upper), Bins_(Bins), Scale_((Bins_ - 1) / static_cast<FLType>(Upper_ - Lower_))
{
    Data_ = new CountType[Bins_];

    Generate(src);
}

template < typename _Ty >
Histogram<_Ty>::Histogram(const _Ty *src, CountType _pcount, _Ty Lower, _Ty Upper)
    : Histogram(src, _pcount, Lower, Upper, Upper_ - Lower_ + 1)
{}

template < typename _Ty >
Histogram<_Ty>::Histogram(const _Ty *src, CountType _pcount, _Ty Lower, _Ty Upper, BinType Bins)
    : Lower_(Lower), Upper_(Upper), Bins_(Bins), Scale_((Bins_ - 1) / static_cast<FLType>(Upper_ - Lower_))
{
    Data_ = new CountType[Bins_];

    Generate(src, _pcount);
}

template < typename _Ty >
Histogram<_Ty>::~Histogram()
{
    delete[] Data_;
}


template < typename _Ty >
Histogram<_Ty> &Histogram<_Ty>::operator=(const Histogram<_Ty> &src)
{
    if (this == &src)
    {
        return *this;
    }

    Lower_ = src.Lower_;
    Upper_ = src.Upper_;
    Bins_ = src.Bins_;
    Scale_ = src.Scale_;
    Count_ = src.Count_;

    delete[] Data_;
    Data_ = new CountType[Bins_];

    memcpy(Data_, src.Data_, sizeof(CountType) * Bins_);

    return *this;
}

template < typename _Ty >
Histogram<_Ty> &Histogram<_Ty>::operator=(Histogram<_Ty> &&src)
{
    if (this == &src)
    {
        return *this;
    }

    Lower_ = src.Lower_;
    Upper_ = src.Upper_;
    Bins_ = src.Bins_;
    Scale_ = src.Scale_;
    Count_ = src.Count_;

    delete[] Data_;
    Data_ = src.Data_;

    src.Lower_ = 0;
    src.Upper_ = -1;
    src.Bins_ = 0;
    src.Scale_ = 1;
    src.Count_ = 0;
    src.Data_ = nullptr;

    return *this;
}


template < typename _Ty >
void Histogram<_Ty>::Generate(const _Ty *src, CountType _pcount)
{
    Count_ = 0;
    memset(Data_, 0, sizeof(CountType) * Bins_);

    Add(src, _pcount);
}

template < typename _Ty >
void Histogram<_Ty>::Add(const _Ty *src, CountType _pcount)
{
    Count_ += _pcount;

    BinType BinUpper = Bins_ - 1;
    if (Scale_ == 1)
    {
        for (CountType i = 0; i < _pcount; ++i)
        {
            ++Data_[Clip(ToBinType(src[i] - Lower_), BinType(0), BinUpper)];
        }
    }
    else
    {
        for (CountType i = 0; i < _pcount; ++i)
        {
            ++Data_[Clip(ToBinType((src[i] - Lower_) * Scale_), BinType(0), BinUpper)];
        }
    }
}

template < typename _Ty >
template < typename _St1 >
void Histogram<_Ty>::Generate(const _St1 &src)
{
    Generate(src.data(), src.PixelCount());
}

template < typename _Ty >
template < typename _St1 >
void Histogram<_Ty>::Add(const _St1 &src)
{
    Add(src.data(), src.PixelCount());
}


template < typename _Ty >
_Ty Histogram<_Ty>::Min(double ratio) const
{
    const char *FunctionName = "Histogram::Min";
    if (ratio < 0 || ratio >= 1)
    {
        std::cerr << FunctionName << ": invalid value of \"ratio=" << ratio << "\" is set, should be within [0, 1).\n";
        DEBUG_BREAK;
    }

    BinType i;
    CountType Count = 0;
    CountType MaxCount = static_cast<CountType>(Count_ * ratio + 0.5);

    for (i = 0; i < Bins_; ++i)
    {
        Count += Data_[i];
        if (Count > MaxCount) break;
    }

    return BinToValue<_Ty>(i);
}

template < typename _Ty >
_Ty Histogram<_Ty>::Max(double ratio) const
{
    const char *FunctionName = "Histogram::Max";
    if (ratio < 0 || ratio >= 1)
    {
        std::cerr << FunctionName << ": invalid value of \"ratio=" << ratio << "\" is set, should be within [0, 1).\n";
        DEBUG_BREAK;
    }

    BinType i;
    CountType Count = 0;
    CountType MaxCount = static_cast<CountType>(Count_ * ratio + 0.5);

    for (i = Bins_ - 1; i >= 0; i--)
    {
        Count += Data_[i];
        if (Count > MaxCount) break;
    }

    return BinToValue<_Ty>(i);
}


#endif
