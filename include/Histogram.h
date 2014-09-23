#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include "Image_Type.h"


class Histogram {
public:
    typedef sint32 LevelType;
    typedef sint32 CountType;
private:
    LevelType Lower_ = 0;
    LevelType Upper_ = -1;
    LevelType Levels_ = 0;
    FLType Quantize_ = 1;
    CountType * Data_ = nullptr;
public:
    Histogram() {} // Default constructor
    Histogram(const Histogram & src); // Copy constructor
    Histogram(Histogram && src); // Move constructor
    explicit Histogram(const Plane & input); // Convertor/Constructor from Plane
    Histogram(const Plane & input, LevelType Levels);
    template < typename T >
    Histogram(const T * input, LevelType size, T Lower, T Upper); // Template constructor from array
    template < typename T >
    Histogram(const T * input, LevelType size, T Lower, T Upper, LevelType Levels); // Template constructor from array
    ~Histogram(); // Destructor

    Histogram & operator=(const Histogram & src); // Copy assignment operator
    Histogram & operator=(Histogram && src); // Move assignment operator
    CountType & operator[](LevelType i) { return Data_[i]; }
    CountType operator[](LevelType i) const { return Data_[i]; }

    LevelType Lower() const { return Lower_; }
    LevelType Upper() const { return Upper_; }
    LevelType Levels() const { return Levels_; }
    FLType Quantize() const { return Quantize_; }
    CountType * Data() { return Data_; }
    const CountType * Data() const { return Data_; }

    template < typename T >
    CountType Lookup(const T input) const;
};


// Template functions of class Histogram
template < typename T >
Histogram::Histogram(const T * input, LevelType size, T Lower, T Upper)
    : Lower_(Lower), Upper_(Upper), Levels_(Upper_ - Lower_ + 1), Quantize_(1)
{
    LevelType i;

    Data_ = new CountType[Levels_];
    memset(Data_, 0, sizeof(CountType)*Levels_);

    LevelType LevelUpper = Levels_ - 1;
    for (i = 0; i < size; i++)
    {
        Data_[Clip(input[i] - Lower_, 0, LevelUpper)]++;
    }
}

template < typename T >
Histogram::Histogram(const T * input, LevelType size, T Lower, T Upper, LevelType Levels)
    : Lower_(Lower), Upper_(Upper), Levels_(Levels), Quantize_(static_cast<FLType>(Upper_ - Lower_ + 1) / Levels_)
{
    LevelType i;

    if (Quantize_ < 1)
    {
        Levels_ = Upper_ - Lower_ + 1;
        Quantize_ = 1;
    }

    Data_ = new CountType[Levels_];
    memset(Data_, 0, sizeof(CountType)*Levels_);

    LevelType LevelUpper = Levels_ - 1;
    if (Quantize_ == 1)
    {
        for (i = 0; i < size; i++)
        {
            Data_[Clip(input[i] - Lower_, 0, LevelUpper)]++;
        }
    }
    else
    {
        for (i = 0; i < size; i++)
        {
            Data_[static_cast<LevelType>(Clip(input[i] - Lower_, 0, LevelUpper) / Quantize_)]++;
        }
    }
}


template < typename T >
Histogram::CountType Histogram::Lookup(const T input) const
{
    LevelType LevelUpper = Levels_ - 1;
    if (Quantize_ == 1)
    {
        return Data_[Clip(input - Lower_, 0, LevelUpper)];
    }
    else
    {
        return Data_[Clip(input, 0, LevelUpper)];
    }
}


#endif