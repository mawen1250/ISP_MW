#include <iostream>
#include "include\Histogram.h"
#include "include\Image_Type.h"
#include "include\Type_Conv.h"


// Functions of class Histogram
Histogram::Histogram(const Histogram & src)
    : Lower_(src.Lower_), Upper_(src.Upper_), Levels_(src.Levels_), Quantize_(src.Quantize_)
{
    Data_ = new CountType[Levels_];

    memcpy(Data_, src.Data_, sizeof(CountType)*Levels_);
}

Histogram::Histogram(Histogram && src)
    : Lower_(src.Lower_), Upper_(src.Upper_), Levels_(src.Levels_), Quantize_(src.Quantize_)
{
    Data_ = src.Data_;

    src.Lower_ = 0;
    src.Upper_ = -1;
    src.Levels_ = 0;
    src.Quantize_ = 1;
    src.Data_ = nullptr;
}

Histogram::Histogram(const Plane & input)
    : Lower_(input.Floor()), Upper_(input.Ceil()), Levels_(Upper_ - Lower_ + 1), Quantize_(1)
{
    PCType i;
    PCType pcount = input.PixelCount();

    Data_ = new CountType[Levels_];
    memset(Data_, 0, sizeof(CountType)*Levels_);

    for (i = 0; i < pcount; i++)
    {
        Data_[input[i] - Lower_]++;
    }
}

Histogram::Histogram(const Plane & input, LevelType Levels)
    : Lower_(input.Floor()), Upper_(input.Ceil()), Levels_(Levels), Quantize_(static_cast<FLType>(Upper_ - Lower_ + 1) / Levels_)
{
    PCType i;
    PCType pcount = input.PixelCount();

    if (Quantize_ < 1)
    {
        Levels_ = Upper_ - Lower_ + 1;
        Quantize_ = 1;
    }

    Data_ = new CountType[Levels_];
    memset(Data_, 0, sizeof(CountType)*Levels_);

    if (Quantize_ == 1)
    {
        for (i = 0; i < pcount; i++)
        {
            Data_[input[i] - Lower_]++;
        }
    }
    else
    {
        for (i = 0; i < pcount; i++)
        {
            Data_[static_cast<LevelType>((input[i] - Lower_) / Quantize_)]++;
        }
    }
}

Histogram::~Histogram()
{
    delete[] Data_;
}

Histogram & Histogram::operator=(const Histogram & src)
{
    if (this == &src)
    {
        return *this;
    }

    Lower_ = src.Lower_;
    Upper_ = src.Upper_;
    Levels_ = src.Levels_;
    Quantize_ = src.Quantize_;

    delete[] Data_;
    Data_ = new CountType[Levels_];

    memcpy(Data_, src.Data_, sizeof(CountType)*Levels_);

    return *this;
}

Histogram & Histogram::operator=(Histogram && src)
{
    if (this == &src)
    {
        return *this;
    }

    Lower_ = src.Lower_;
    Upper_ = src.Upper_;
    Levels_ = src.Levels_;
    Quantize_ = src.Quantize_;

    delete[] Data_;
    Data_ = src.Data_;

    src.Lower_ = 0;
    src.Upper_ = -1;
    src.Levels_ = 0;
    src.Quantize_ = 1;
    src.Data_ = nullptr;

    return *this;
}
