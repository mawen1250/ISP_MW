#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include "Image_Type.h"
#include "LUT.h"


template < typename T = DType >
class Histogram
{
public:
    typedef sint32 BinType;
    typedef sint32 CountType;

private:
    T Lower_ = 0;
    T Upper_ = -1;
    BinType Bins_ = 0;
    FLType Scale_ = 1;
    CountType Count_ = 0;
    CountType * Data_ = nullptr;

protected:
    template < typename T0 > BinType ToBinType(const T0 & input) const { return static_cast<BinType>(input); }
    template < > BinType ToBinType<float>(const float & input) const { return static_cast<BinType>(input + 0.5F); }
    template < > BinType ToBinType<double>(const double & input) const { return static_cast<BinType>(input + 0.5); }
    template < > BinType ToBinType<long double>(const long double & input) const { return static_cast<BinType>(input + 0.5L); }

public:
    Histogram() {} // Default constructor
    Histogram(const Histogram & src); // Copy constructor
    Histogram(Histogram && src); // Move constructor
    explicit Histogram(const Plane & src); // Convertor/Constructor from Plane
    Histogram(const Plane & src, BinType Bins);
    explicit Histogram(const Plane_FL & src, BinType Bins = 256);
    Histogram(const T * src, CountType size, T Lower, T Upper); // Template constructor from array
    Histogram(const T * src, CountType size, T Lower, T Upper, BinType Bins); // Template constructor from array
    ~Histogram(); // Destructor

    Histogram & operator=(const Histogram & src); // Copy assignment operator
    Histogram & operator=(Histogram && src); // Move assignment operator
    CountType & operator[](BinType i) { return Data_[i]; }
    CountType operator[](BinType i) const { return Data_[i]; }

    T Lower() const { return Lower_; }
    T Upper() const { return Upper_; }
    BinType Bins() const { return Bins_; }
    FLType Scale() const { return Scale_; }
    CountType * Data() { return Data_; }
    const CountType * Data() const { return Data_; }

    template < typename T0 > T0 BinToValue(BinType i) const { return static_cast<T0>(i / Scale_ + FLType(0.5)) + Lower_; }
    template < > float BinToValue<float>(BinType i) const { return static_cast<float>(i / Scale_) + Lower_; }
    template < > double BinToValue<double>(BinType i) const { return static_cast<double>(i / Scale_) + Lower_; }
    template < > long double BinToValue<long double>(BinType i) const { return static_cast<long double>(i / Scale_) + Lower_; }
    T Min(FLType ratio = 0) const;
    T Max(FLType ratio = 0) const;
    CountType Lookup(const T input) const { return Data_[Clip(static_cast<BinType>((input - Lower_)*Scale_), 0, Bins_ - 1)]; }

    LUT<DType> Equalization_LUT(const Plane & dst, const Plane & src, FLType strength = 1.0) const;
    LUT<DType> Equalization_LUT(const Plane & src, FLType strength = 1.0) const { return Equalization_LUT(src, src, strength); }
    LUT<FLType> Equalization_LUT_Gain(const Plane & dst, const Plane & src, FLType strength = 1.0) const;
    LUT<FLType> Equalization_LUT_Gain(const Plane & src, FLType strength = 1.0) const { return Equalization_LUT_Gain(src, src, strength); }
};


// Template functions of class Histogram
template < typename T >
Histogram<T>::Histogram(const Histogram<T> & src)
    : Lower_(src.Lower_), Upper_(src.Upper_), Bins_(src.Bins_), Scale_(src.Scale_), Count_(src.Count_)
{
    Data_ = new CountType[Bins_];

    memcpy(Data_, src.Data_, sizeof(CountType)*Bins_);
}

template < typename T >
Histogram<T>::Histogram(Histogram<T> && src)
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

template < typename T >
Histogram<T>::Histogram(const Plane & src)
    : Lower_(src.Floor()), Upper_(src.Ceil()), Bins_(Upper_ - Lower_ + 1), Scale_(1), Count_(src.PixelCount())
{
    CountType i;

    Data_ = new CountType[Bins_];
    memset(Data_, 0, sizeof(CountType)*Bins_);

    for (i = 0; i < Count_; i++)
    {
        Data_[ToBinType(src[i] - Lower_)]++;
    }
}

template < typename T >
Histogram<T>::Histogram(const Plane & src, BinType Bins)
    : Lower_(src.Floor()), Upper_(src.Ceil()), Bins_(Bins), Scale_((Bins_ - 1) / static_cast<FLType>(Upper_ - Lower_)), Count_(src.PixelCount())
{
    CountType i;

    if (Scale_ > 1)
    {
        Bins_ = ToBinType(Upper_ - Lower_ + 1);
        Scale_ = 1;
    }

    Data_ = new CountType[Bins_];
    memset(Data_, 0, sizeof(CountType)*Bins_);

    if (Scale_ == 1)
    {
        for (i = 0; i < Count_; i++)
        {
            Data_[ToBinType(src[i] - Lower_)]++;
        }
    }
    else
    {
        for (i = 0; i < Count_; i++)
        {
            Data_[ToBinType((src[i] - Lower_)*Scale_)]++;
        }
    }
}

template < typename T >
Histogram<T>::Histogram(const Plane_FL & src, BinType Bins)
    : Lower_(src.Floor()), Upper_(src.Ceil()), Bins_(Bins), Scale_((Bins_ - 1) / static_cast<FLType>(Upper_ - Lower_)), Count_(src.PixelCount())
{
    CountType i;

    Data_ = new CountType[Bins_];
    memset(Data_, 0, sizeof(CountType)*Bins_);

    if (Scale_ == 1)
    {
        for (i = 0; i < Count_; i++)
        {
            Data_[ToBinType(src[i] - Lower_)]++;
        }
    }
    else
    {
        for (i = 0; i < Count_; i++)
        {
            Data_[ToBinType((src[i] - Lower_)*Scale_)]++;
        }
    }
}

template < typename T >
Histogram<T>::Histogram(const T * src, CountType size, T Lower, T Upper)
    : Lower_(Lower), Upper_(Upper), Bins_(Upper_ - Lower_ + 1), Scale_(1), Count_(size)
{
    CountType i;

    Data_ = new CountType[Bins_];
    memset(Data_, 0, sizeof(CountType)*Bins_);

    BinType BinUpper = Bins_ - 1;
    for (i = 0; i < Count_; i++)
    {
        Data_[Clip(ToBinType(src[i] - Lower_), 0, BinUpper)]++;
    }
}

template < typename T >
Histogram<T>::Histogram(const T * src, CountType size, T Lower, T Upper, BinType Bins)
    : Lower_(Lower), Upper_(Upper), Bins_(Bins), Scale_((Bins_ - 1) / static_cast<FLType>(Upper_ - Lower_)), Count_(size)
{
    CountType i;

    Data_ = new CountType[Bins_];
    memset(Data_, 0, sizeof(CountType)*Bins_);

    BinType BinUpper = Bins_ - 1;
    if (Scale_ == 1)
    {
        for (i = 0; i < Count_; i++)
        {
            Data_[Clip(ToBinType(src[i] - Lower_), 0, BinUpper)]++;
        }
    }
    else
    {
        for (i = 0; i < Count_; i++)
        {
            Data_[Clip(ToBinType((src[i] - Lower_)*Scale_), 0, BinUpper)]++;
        }
    }
}

template < typename T >
Histogram<T>::~Histogram()
{
    delete[] Data_;
}


template < typename T >
Histogram<T> & Histogram<T>::operator=(const Histogram<T> & src)
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

    memcpy(Data_, src.Data_, sizeof(CountType)*Bins_);

    return *this;
}

template < typename T >
Histogram<T> & Histogram<T>::operator=(Histogram<T> && src)
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


template < typename T >
T Histogram<T>::Min(FLType ratio) const
{
    const char * FunctionName = "Histogram::Min";
    if (ratio < 0 || ratio >= 1)
    {
        std::cerr << FunctionName << ": invalid value of \"ratio=" << ratio << "\" is set, should be within [0, 1).\n";
        exit(EXIT_FAILURE);
    }

    BinType i;
    CountType Count = 0;
    CountType MaxCount = static_cast<CountType>(Count_*ratio + FLType(0.5));

    for (i = 0; i < Bins_; i++)
    {
        Count += Data_[i];
        if (Count > MaxCount) break;
    }

    return BinToValue<T>(i);
}

template < typename T >
T Histogram<T>::Max(FLType ratio) const
{
    const char * FunctionName = "Histogram::Max";
    if (ratio < 0 || ratio >= 1)
    {
        std::cerr << FunctionName << ": invalid value of \"ratio=" << ratio << "\" is set, should be within [0, 1).\n";
        exit(EXIT_FAILURE);
    }

    BinType i;
    CountType Count = 0;
    CountType MaxCount = static_cast<CountType>(Count_*ratio + FLType(0.5));

    for (i = Bins_ - 1; i >= 0; i--)
    {
        Count += Data_[i];
        if (Count > MaxCount) break;
    }

    return BinToValue<T>(i);
}

inline LUT<DType> Histogram<DType>::Equalization_LUT(const Plane & dst, const Plane & src, FLType strength) const
{
    DType sRange = src.ValueRange();
    DType dFloor = dst.Floor();
    DType dRange = dst.ValueRange();
    LUT<DType>::LevelType LUT_Levels = sRange + 1;

    LUT<DType> _LUT(LUT_Levels);

    FLType weight1 = strength;
    FLType weight2 = 1 - strength;

    CountType Sum;
    FLType scale1, scale2, offset;

    if (Bins_ == LUT_Levels)
    {
        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / Count_ * weight1;
        scale2 = static_cast<FLType>(dRange) / sRange * weight2;
        offset = static_cast<FLType>(dFloor)+FLType(0.5);

        for (BinType i = 0; i < Bins_; i++)
        {
            Sum += Data_[i];
            _LUT[i] = static_cast<DType>(Sum * scale1 + i * scale2 + offset);
        }
    }
    else
    {
        FLType *temp = new FLType[Bins_];

        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / Count_ * weight1;
        scale2 = static_cast<FLType>(dRange) / (Bins_ - 1) * weight2;
        offset = static_cast<FLType>(dFloor);

        for (BinType i = 0; i < Bins_; i++)
        {
            Sum += Data_[i];
            temp[i] = Sum * scale1 + i * scale2 + offset;
        }

        FLType val;
        DType val_lower, val_upper;
        DType val_lower_upper = Bins_ - 2;
        scale1 = static_cast<FLType>(Bins_ - 1) / sRange;

        for (LUT<DType>::LevelType i = 0; i < LUT_Levels; i++)
        {
            val = i * scale1;
            val_lower = ::Min(static_cast<DType>(val), val_lower_upper);
            val_upper = val_lower + 1;
            _LUT[i] = static_cast<DType>(temp[val_lower] * (val_upper - val) + temp[val_upper] * (val - val_lower) + FLType(0.5));
        }
    }

    return _LUT;
}

inline LUT<FLType> Histogram<DType>::Equalization_LUT_Gain(const Plane & dst, const Plane & src, FLType strength) const
{
    DType sRange = src.ValueRange();
    DType dFloor = dst.Floor();
    DType dRange = dst.ValueRange();
    LUT<FLType>::LevelType LUT_Levels = sRange + 1;

    LUT<FLType> _LUT(LUT_Levels);

    FLType weight1 = strength;
    FLType weight2 = 1 - strength;

    CountType Sum;
    FLType scale1, scale2;

    if (Bins_ == LUT_Levels)
    {
        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / Count_ * weight1;
        scale2 = static_cast<FLType>(dRange) / sRange * weight2;

        for (BinType i = 0; i < Bins_; i++)
        {
            Sum += Data_[i];
            _LUT[i] = Sum * scale1 / i + scale2;
        }
    }
    else
    {
        FLType *temp = new FLType[Bins_];

        Sum = 0;
        scale1 = static_cast<FLType>(dRange) / Count_ * weight1;
        scale2 = static_cast<FLType>(dRange) / (Bins_ - 1) * weight2;

        for (BinType i = 0; i < Bins_; i++)
        {
            Sum += Data_[i];
            temp[i] = Sum * scale1 + i * scale2;
        }

        FLType val;
        DType val_lower, val_upper;
        DType val_lower_upper = Bins_ - 2;
        scale1 = static_cast<FLType>(Bins_ - 1) / sRange;

        for (LUT<DType>::LevelType i = 0; i < LUT_Levels; i++)
        {
            val = i * scale1;
            val_lower = ::Min(static_cast<DType>(val), val_lower_upper);
            val_upper = val_lower + 1;
            _LUT[i] = (temp[val_lower] * (val_upper - val) + temp[val_upper] * (val - val_lower)) / i;
        }
    }

    return _LUT;
}


#endif