#include <iostream>
#include "Image_Type.h"
#include "LUT.h"
#include "Histogram.h"


// Functions of class Plane
void Plane::DefaultPara(bool Chroma, value_type BitDepth, QuantRange _QuantRange)
{
    Quantize_Value(Floor_, Neutral_, Ceil_, ValueRange_, BitDepth, _QuantRange, Chroma);
}

void Plane::CopyParaFrom(const _Myt& src)
{
    Width_ = src.Width_;
    Height_ = src.Height_;
    PixelCount_ = Width_ * Height_;
    BitDepth_ = src.BitDepth_;
    Floor_ = src.Floor_;
    Neutral_ = src.Neutral_;
    Ceil_ = src.Ceil_;
    ValueRange_ = src.ValueRange_;
    TransferChar_ = src.TransferChar_;
}

void Plane::InitValue(value_type Value, bool Init)
{
    if (Init)
    {
        Value = Quantize(Value);

        if (Value)
        {
            for (PCType i = 0; i < PixelCount_; i++)
            {
                Data_[i] = Value;
            }
        }
        else
        {
            memset(Data_, Value, sizeof(value_type) * PixelCount_);
        }
    }
}


Plane::Plane(value_type Value, PCType Width, PCType Height, value_type BitDepth, bool Init)
    : _Myt(Value, Width, Height, BitDepth, 0, 0, (value_type(1) << BitDepth_) - 1,
    TransferChar_Default(Width, Height, true), Init)
{}

Plane::Plane(value_type Value, PCType Width, PCType Height, value_type BitDepth, value_type Floor, value_type Neutral, value_type Ceil, TransferChar _TransferChar, bool Init)
    : Width_(Width), Height_(Height), PixelCount_(Width_ * Height_), BitDepth_(BitDepth),
    Floor_(Floor), Neutral_(Neutral), Ceil_(Ceil), ValueRange_(Ceil_ - Floor_), TransferChar_(_TransferChar)
{
    const char * FunctionName = "class Plane constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }
    if (Ceil <= Floor)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }
    if (ValueRange_ >= value_type(1) << BitDepth)
    {
        std::cerr << FunctionName << ": \"Ceil-Floor=" << ValueRange_ << "\" exceeds \"BitDepth=" << BitDepth << "\" limit.\n";
        exit(EXIT_FAILURE);
    }
    if (Neutral > Floor && Neutral != (Floor + Ceil + 1) / 2)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\", \"Neutral=" << Neutral << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }

    Data_ = new value_type[PixelCount_];

    InitValue(Value, Init);
}

Plane::Plane(const _Myt& src)
    : _Myt(src, false)
{
    memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);
}

Plane::Plane(const _Myt& src, bool Init, value_type Value)
    : _Myt(Value, src.Width_, src.Height_, src.BitDepth_, src.Floor_, src.Neutral_, src.Ceil_, src.TransferChar_, Init)
{}

Plane::Plane(_Myt&& src)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_), BitDepth_(src.BitDepth_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), ValueRange_(src.ValueRange_), TransferChar_(src.TransferChar_)
{
    Data_ = src.Data_;

    src.Width_ = 0;
    src.Height_ = 0;
    src.PixelCount_ = 0;
    src.Data_ = nullptr;
}

Plane::Plane(const Plane_FL& src, value_type BitDepth)
    : _Myt(src, BitDepth, 0, 0, (value_type(1) << BitDepth_) - 1)
{}

Plane::Plane(const Plane_FL& src, value_type BitDepth, value_type Floor, value_type Neutral, value_type Ceil)
    : _Myt(src, false, 0, BitDepth, Floor, Neutral, Ceil)
{
    From(src);
}

Plane::Plane(const Plane_FL& src, bool Init, value_type Value, value_type BitDepth)
    : _Myt(src, Init, Value, BitDepth, 0, 0, (value_type(1) << BitDepth_) - 1)
{}

Plane::Plane(const Plane_FL& src, bool Init, value_type Value, value_type BitDepth, value_type Floor, value_type Neutral, value_type Ceil)
    : _Myt(Value, src.Width(), src.Height(), BitDepth, Floor, Neutral, Ceil, src.GetTransferChar(), Init)
{}


Plane::~Plane()
{
    delete[] Data_;
}


Plane& Plane::operator=(const _Myt& src)
{
    if (this == &src)
    {
        return *this;
    }

    CopyParaFrom(src);

    delete[] Data_;
    Data_ = new value_type[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);

    return *this;
}

Plane& Plane::operator=(_Myt&& src)
{
    if (this == &src)
    {
        return *this;
    }

    CopyParaFrom(src);

    delete[] Data_;
    Data_ = src.Data_;

    src.Width_ = 0;
    src.Height_ = 0;
    src.PixelCount_ = 0;
    src.Data_ = nullptr;

    return *this;
}

bool Plane::operator==(const _Myt& b) const
{
    if (this == &b)
    {
        return true;
    }

    if (Width_ != b.Width_ || Height_ != b.Height_ || PixelCount_ != b.PixelCount_ || BitDepth_ != b.BitDepth_ ||
        Floor_ != b.Floor_ || Neutral_ != b.Neutral_ || Ceil_ != b.Ceil_ || ValueRange_ != b.ValueRange_ || TransferChar_ != b.TransferChar_)
    {
        return false;
    }

    PCType i;
    for (i = 0; Data_[i] == b.Data_[i] && i < PixelCount_; i++);
    if (i < PixelCount_)
    {
        return false;
    }
    else
    {
        return true;
    }
}


Plane::value_type Plane::Min() const
{
    value_type min = value_type_MAX;

    for_each([&](value_type x)
    {
        if (min > x) min = x;
    });

    return min;
}

Plane::value_type Plane::Max() const
{
    value_type max = value_type_MIN;

    for_each([&](value_type x)
    {
        if (max < x) max = x;
    });

    return max;
}

void Plane::MinMax(reference min, reference max) const
{
    min = value_type_MAX;
    max = value_type_MIN;

    for_each([&](value_type x)
    {
        if (min > x) min = x;
        if (max < x) max = x;
    });
}

FLType Plane::Mean() const
{
    uint64 Sum = 0;

    for_each([&](value_type x)
    {
        Sum += x;
    });

    return static_cast<FLType>(Sum) / PixelCount_;
}

FLType Plane::Variance(FLType Mean) const
{
    FLType diff;
    FLType Sum = 0;

    for_each([&](value_type x)
    {
        diff = x - Mean;
        Sum += diff * diff;
    });

    return Sum / PixelCount_;
}


Plane& Plane::ReSize(PCType Width, PCType Height)
{
    if (Width_ != Width || Height_ != Height)
    {
        if (PixelCount_ != Width * Height)
        {
            delete[] Data_;
            PixelCount_ = Width * Height;
            Data_ = new value_type[PixelCount_];
        }

        Width_ = Width;
        Height_ = Height;
    }

    return *this;
}

Plane& Plane::ReQuantize(value_type BitDepth, QuantRange _QuantRange, bool scale, bool clip)
{
    const char * FunctionName = "Plane::ReQuantize";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    value_type Floor, Neutral, Ceil, ValueRange;
    Quantize_Value(Floor, Neutral, Ceil, ValueRange, BitDepth, _QuantRange, isChroma());
    return ReQuantize(BitDepth, Floor, Neutral, Ceil, scale, clip);
}

Plane& Plane::ReQuantize(value_type BitDepth, value_type Floor, value_type Neutral, value_type Ceil, bool scale, bool clip)
{
    value_type ValueRange = Ceil - Floor;

    const char * FunctionName = "Plane::ReQuantize";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }
    if (Ceil <= Floor)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }
    if (ValueRange >= value_type(1) << BitDepth)
    {
        std::cerr << FunctionName << ": \"Ceil-Floor=" << ValueRange << "\" exceeds \"BitDepth=" << BitDepth << "\" limit.\n";
        exit(EXIT_FAILURE);
    }
    if (Neutral > Floor && Neutral != (Floor + Ceil + 1) / 2)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\", \"Neutral=" << Neutral << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }

    if (scale && Data_ && (Floor_ != Floor || Neutral_ != Neutral || Ceil_ != Ceil))
    {
        FLType gain = static_cast<FLType>(ValueRange) / ValueRange_;
        FLType offset = Neutral - Neutral_ * gain + FLType(Floor < Neutral && (Floor + Ceil) % 2 == 1 ? 0.499999 : 0.5);

        if (clip)
        {
            FLType FloorFL = static_cast<FLType>(Floor);
            FLType CeilFL = static_cast<FLType>(Ceil);

            transform([&](value_type x)
            {
                return static_cast<value_type>(Clip(x * gain + offset, FloorFL, CeilFL));
            });
        }
        else
        {
            transform([&](value_type x)
            {
                return static_cast<value_type>(x * gain + offset);
            });
        }
    }

    BitDepth_ = BitDepth;
    Floor_ = Floor;
    Neutral_ = Neutral;
    Ceil_ = Ceil;
    ValueRange_ = ValueRange;

    return *this;
}


Plane& Plane::From(const Plane& src)
{
    if (isChroma() != src.isChroma()) // Plane "src" and "dst" are not both luma planes or chroma planes.
    {
        Neutral_ = (Floor_ + Ceil_ + 1) / 2;
    }

    ReSize(src.Width(), src.Height());

    if (Floor_ == src.Floor() && Neutral_ == src.Neutral() && Ceil_ == src.Ceil())
    {
        memcpy(Data_, src.Data(), sizeof(value_type) * PixelCount_);
    }
    else
    {
        Plane& dst = *this;

        FLType sNeutralFL = src.Neutral();
        FLType sRangeFL = src.ValueRange();
        FLType dNeutralFL = dst.Neutral();
        FLType dRangeFL = dst.ValueRange();
        FLType dFloorFL = dst.Floor();
        FLType dCeilFL = dst.Ceil();

        FLType gain = dRangeFL / sRangeFL;
        FLType offset = dNeutralFL - sNeutralFL * gain + FLType(dst.isPCChroma() ? 0.499999 : 0.5);

        dst.transform(src, [&](value_type x)
        {
            return static_cast<value_type>(Clip(x * gain + offset, dFloorFL, dCeilFL));
        });
    }

    return *this;
}

Plane& Plane::From(const Plane_FL& src)
{
    if (isChroma() != src.isChroma()) // Plane "src" and "dst" are not both luma planes or chroma planes.
    {
        Neutral_ = (Floor_ + Ceil_ + 1) / 2;
    }

    ReSize(src.Width(), src.Height());

    auto& dst = *this;

    FLType sNeutralFL = src.Neutral();
    FLType sRangeFL = src.ValueRange();
    FLType dNeutralFL = dst.Neutral();
    FLType dRangeFL = dst.ValueRange();
    FLType dFloorFL = dst.Floor();
    FLType dCeilFL = dst.Ceil();

    FLType gain = dRangeFL / sRangeFL;
    FLType offset = dNeutralFL - sNeutralFL * gain + FLType(dst.isPCChroma() ? 0.499999 : 0.5);

    dst.transform(src, [&](Plane_FL::value_type x)
    {
        return static_cast<value_type>(Clip(x * gain + offset, dFloorFL, dCeilFL));
    });

    return *this;
}

Plane& Plane::ConvertFrom(const Plane& src, TransferChar dstTransferChar)
{
    auto& dst = *this;
    LUT<value_type> _LUT(src);
    FLType data;

    FLType src_k0, src_phi, src_alpha, src_power, src_div;
    FLType dst_k0, dst_phi, dst_alpha, dst_power, dst_div;

    value_type srcFloor = src.Floor(), srcCeil = src.Ceil();
    TransferChar srcTransferChar = src.GetTransferChar();
    dst.SetTransferChar(dstTransferChar);

    // Generate conversion LUT
    if (srcTransferChar == TransferChar::linear)
    {
        if (TransferChar_ == TransferChar::linear)
        {
            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                return dst.GetD(data);
            });
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_linear2gamma(data, dst_k0, dst_div);
                return dst.GetD(data);
            });
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
                return dst.GetD(data);
            });
        }
    }
    else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
    {
        TransferChar_Parameter(srcTransferChar, src_k0, src_div);

        if (TransferChar_ == TransferChar::linear)
        {
            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_div);
                return dst.GetD(data);
            });
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_div);
                data = TransferChar_linear2gamma(data, dst_k0, dst_div);
                return dst.GetD(data);
            });
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_div);
                data = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
                return dst.GetD(data);
            });
        }
    }
    else
    {
        TransferChar_Parameter(srcTransferChar, src_k0, src_phi, src_alpha, src_power);

        if (TransferChar_ == TransferChar::linear)
        {
            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_phi, src_alpha, src_power);
                return dst.GetD(data);
            });
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_phi, src_alpha, src_power);
                data = TransferChar_linear2gamma(data, dst_k0, dst_div);
                return dst.GetD(data);
            });
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            _LUT.Set(src, [&](value_type i)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_phi, src_alpha, src_power);
                data = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
                return dst.GetD(data);
            });
        }
    }

    // Conversion
    dst.ReSize(src.Width(), src.Height());
    _LUT.Lookup(dst, src);

    // Output
    return *this;
}

Plane& Plane::YFrom(const Frame& src, ColorMatrix dstColorMatrix)
{
    if (src.isRGB())
    {
        FLType Kr, Kg, Kb;

        auto& dstY = *this;
        auto& srcR = src.R();
        auto& srcG = src.G();
        auto& srcB = src.B();

        ReSize(src.Width(), src.Height());

        ColorMatrix_Parameter(dstColorMatrix, Kr, Kg, Kb);

        FLType gain = static_cast<FLType>(ValueRange_) / srcR.ValueRange_;
        FLType offset = Floor_ - srcR.Floor_ * gain + FLType(0.5);
        Kr *= gain;
        Kg *= gain;
        Kb *= gain;

        dstY.transform(srcR, srcG, srcB, [&](value_type x, value_type y, value_type z)
        {
            return static_cast<value_type>(Kr*x + Kg*y + Kb*z + offset);
        });
    }
    else if (src.isYUV())
    {
        From(src.Y());
    }

    return *this;
}

Plane& Plane::YFrom(const Frame& src)
{
    return YFrom(src, src.GetColorMatrix());
}


Plane& Plane::Binarize(const _Myt& src, value_type lower_thrD, value_type upper_thrD)
{
    double lower_thr = static_cast<double>(lower_thrD - src.Floor()) / src.ValueRange();
    double upper_thr = static_cast<double>(upper_thrD - src.Floor()) / src.ValueRange();

    return Binarize_ratio(src, lower_thr, upper_thr);
}

Plane& Plane::Binarize_ratio(const _Myt& src, double lower_thr, double upper_thr)
{
    auto& dst = *this;

    value_type lower_thrD = static_cast<value_type>(lower_thr * src.ValueRange() + 0.5) + src.Floor();
    value_type upper_thrD = static_cast<value_type>(upper_thr * src.ValueRange() + 0.5) + src.Floor();

    if (upper_thr <= lower_thr || lower_thr >= 1 || upper_thr < 0)
    {
        dst.for_each([&](value_type& x)
        {
            x = dst.Floor();
        });
    }
    else if (lower_thr < 0)
    {
        if (upper_thr >= 1)
        {
            dst.for_each([&](value_type& x)
            {
                x = dst.Ceil();
            });
        }
        else
        {
            dst.transform([&](value_type x)
            {
                return x <= upper_thrD ? dst.Ceil() : dst.Floor();
            });
        }
    }
    else
    {
        if (upper_thr >= 1)
        {
            dst.transform([&](value_type x)
            {
                return x > lower_thrD ? dst.Ceil() : dst.Floor();
            });
        }
        else
        {
            dst.transform([&](value_type x)
            {
                return x > lower_thrD && x <= upper_thrD ? dst.Ceil() : dst.Floor();
            });
        }
    }

    return *this;
}

Plane& Plane::SimplestColorBalance(Plane_FL& flt, const Plane& src, double lower_thr, double upper_thr, int HistBins)
{
    auto& dst = *this;

    Plane_FL::value_type min, max;
    flt.MinMax(min, max);

    if (max <= min)
    {
        dst.From(src);
        return *this;
    }

    if (lower_thr > 0 || upper_thr > 0)
    {
        flt.ReQuantize(min, min, max, false);
        Histogram<Plane_FL::value_type> Histogram(flt, HistBins);
        min = Histogram.Min(lower_thr);
        max = Histogram.Max(upper_thr);
    }

    Plane_FL::value_type gain = dst.ValueRange() / (max - min);
    Plane_FL::value_type offset = dst.Floor() - min * gain + Plane_FL::value_type(0.5);

    if (lower_thr > 0 || upper_thr > 0)
    {
        Plane_FL::value_type FloorFL = static_cast<Plane_FL::value_type>(dst.Floor());
        Plane_FL::value_type CeilFL = static_cast<Plane_FL::value_type>(dst.Ceil());
        
        dst.transform(flt, [&](Plane_FL::value_type x)
        {
            return static_cast<value_type>(Clip(x * gain + offset, FloorFL, CeilFL));
        });
    }
    else
    {
        dst.transform(flt, [&](Plane_FL::value_type x)
        {
            return static_cast<value_type>(x * gain + offset);
        });
    }

    return *this;
}


// Functions of class Plane_FL
void Plane_FL::DefaultPara(bool Chroma, value_type range)
{
    if (Chroma) // Plane "src" is chroma
    {
        Floor_ = -range / 2;
        Neutral_ = 0;
        Ceil_ = range / 2;
    }
    else // Plane "src" is not chroma
    {
        Floor_ = 0;
        Neutral_ = 0;
        Ceil_ = range;
    }
}

void Plane_FL::CopyParaFrom(const _Myt& src)
{
    Width_ = src.Width_;
    Height_ = src.Height_;
    PixelCount_ = Width_ * Height_;
    Floor_ = src.Floor_;
    Neutral_ = src.Neutral_;
    Ceil_ = src.Ceil_;
    TransferChar_ = src.TransferChar_;
}

void Plane_FL::InitValue(value_type Value, bool Init)
{
    if (Init)
    {
        Value = Quantize(Value);
        for (PCType i = 0; i < PixelCount_; i++)
        {
            Data_[i] = Value;
        }
    }
}


Plane_FL::Plane_FL(value_type Value, PCType Width, PCType Height, bool RGB, bool Chroma, bool Init)
    : Width_(Width), Height_(Height), PixelCount_(Width_ * Height_), TransferChar_(!RGB&&Chroma ? TransferChar::linear : TransferChar_Default(Width, Height, RGB))
{
    DefaultPara(!RGB&&Chroma);

    Data_ = new value_type[PixelCount_];

    InitValue(Value, Init);
}

Plane_FL::Plane_FL(value_type Value, PCType Width, PCType Height, value_type Floor, value_type Neutral, value_type Ceil, TransferChar _TransferChar, bool Init)
    : Width_(Width), Height_(Height), PixelCount_(Width_ * Height_),
    Floor_(Floor), Neutral_(Neutral), Ceil_(Ceil), TransferChar_(_TransferChar)
{
    Data_ = new value_type[PixelCount_];

    InitValue(Value, Init);
}

Plane_FL::Plane_FL(const _Myt& src)
    : _Myt(src, false)
{
    memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);
}

Plane_FL::Plane_FL(const _Myt& src, bool Init, value_type Value)
    : _Myt(Value, src.Width_, src.Height_, src.Floor_, src.Neutral_, src.Ceil_, src.TransferChar_, Init)
{}

Plane_FL::Plane_FL(_Myt&& src)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), TransferChar_(src.TransferChar_)
{
    Data_ = src.Data_;

    src.Width_ = 0;
    src.Height_ = 0;
    src.PixelCount_ = 0;
    src.Data_ = nullptr;
}

Plane_FL::Plane_FL(const Plane& src, value_type range)
    : _Myt(src, false, 0, range)
{
    From(src);
}

Plane_FL::Plane_FL(const Plane& src, bool Init, value_type Value, value_type range)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), TransferChar_(src.GetTransferChar())
{
    Data_ = new value_type[PixelCount_];

    if (range > 0)
        DefaultPara(src.isChroma(), range);
    else
    {
        Floor_ = src.Floor();
        Neutral_ = src.Neutral();
        Ceil_ = src.Ceil();
    }

    InitValue(Value, Init);
}

Plane_FL::Plane_FL(const _Myt& src, TransferChar dstTransferChar)
    : _Myt(src, false)
{
    ConvertFrom(src, dstTransferChar);
}

Plane_FL::Plane_FL(const Plane& src, TransferChar dstTransferChar)
    : _Myt(src, false)
{
    ConvertFrom(src, dstTransferChar);
}


Plane_FL::~Plane_FL()
{
    delete[] Data_;
}


Plane_FL& Plane_FL::operator=(const _Myt& src)
{
    if (this == &src)
    {
        return *this;
    }

    CopyParaFrom(src);

    delete[] Data_;
    Data_ = new value_type[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);

    return *this;
}

Plane_FL& Plane_FL::operator=(_Myt&& src)
{
    if (this == &src)
    {
        return *this;
    }

    CopyParaFrom(src);

    delete[] Data_;
    Data_ = src.Data_;

    src.Width_ = 0;
    src.Height_ = 0;
    src.PixelCount_ = 0;
    src.Data_ = nullptr;

    return *this;
}

bool Plane_FL::operator==(const _Myt& b) const
{
    if (this == &b)
    {
        return true;
    }

    if (Width_ != b.Width_ || Height_ != b.Height_ || PixelCount_ != b.PixelCount_ ||
        Floor_ != b.Floor_ || Neutral_ != b.Neutral_ || Ceil_ != b.Ceil_ || TransferChar_ != b.TransferChar_)
    {
        return false;
    }

    PCType i;
    for (i = 0; Data_[i] == b.Data_[i] && i < PixelCount_; i++);
    if (i < PixelCount_)
    {
        return false;
    }
    else
    {
        return true;
    }
}


Plane_FL::value_type Plane_FL::Min() const
{
    value_type min = value_type_MAX;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        min = ::Min(min, Data_[i]);
    }

    return min;
}

Plane_FL::value_type Plane_FL::Max() const
{
    value_type max = -value_type_MAX;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        max = ::Max(max, Data_[i]);
    }

    return max;
}

void Plane_FL::MinMax(reference min, reference max) const
{
    min = value_type_MAX;
    max = -value_type_MAX;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        min = ::Min(min, Data_[i]);
        max = ::Max(max, Data_[i]);
    }
}

Plane_FL::value_type Plane_FL::Mean() const
{
    value_type Sum = 0;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        Sum += Data_[i];
    }

    return Sum / PixelCount_;
}

Plane_FL::value_type Plane_FL::Variance(value_type Mean) const
{
    value_type diff;
    value_type Sum = 0;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        diff = Data_[i] - Mean;
        Sum += diff * diff;
    }

    return Sum / PixelCount_;
}


Plane_FL& Plane_FL::ReSize(PCType Width, PCType Height)
{
    if (Width_ != Width || Height_ != Height)
    {
        if (PixelCount_ != Width * Height)
        {
            delete[] Data_;
            PixelCount_ = Width * Height;
            Data_ = new value_type[PixelCount_];
        }

        Width_ = Width;
        Height_ = Height;
    }

    return *this;
}

Plane_FL& Plane_FL::ReQuantize(value_type Floor, value_type Neutral, value_type Ceil, bool scale, bool clip)
{
    PCType i;

    const char * FunctionName = "Plane_FL::ReQuantize";
    if (Ceil <= Floor)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }
    if (Neutral > Floor && Neutral != (Floor + Ceil) / 2.)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\", \"Neutral=" << Neutral << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }

    if (scale && Data_ && (Floor_ != Floor || Neutral_ != Neutral || Ceil_ != Ceil))
    {
        value_type gain = (Ceil - Floor) / (Ceil_ - Floor_);
        value_type offset = Neutral - Neutral_*gain;

        if (clip)
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = Clip(Data_[i] * gain + offset, Floor, Ceil);
            }
        }
        else
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = Data_[i] * gain + offset;
            }
        }
    }

    Floor_ = Floor;
    Neutral_ = Neutral;
    Ceil_ = Ceil;

    return *this;
}


Plane_FL& Plane_FL::From(const Plane& src)
{
    PCType i, j, upper;
    Plane::value_type sFloor = src.Floor();
    Plane::value_type sNeutral = src.Neutral();
    Plane::value_type sValueRange = src.ValueRange();
    value_type dFloor = Floor_;
    value_type dNeutral = Neutral_;
    value_type dValueRange = Ceil_ - Floor_;
    
    if (isChroma() != src.isChroma()) DefaultPara(src.isChroma());

    ReSize(src.Width(), src.Height());

    PCType height = Height();
    PCType width = Width();
    PCType stride = Stride();

    if (sFloor == dFloor && sNeutral == dNeutral && sValueRange == dValueRange)
    {
        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Data_[i] = static_cast<value_type>(src[i]);
            }
        }
    }
    else
    {
        value_type gain = dValueRange / sValueRange;
        value_type offset = dNeutral - sNeutral * gain;

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Data_[i] = static_cast<value_type>(src[i]) * gain + offset;
            }
        }
    }

    return *this;
}

Plane_FL& Plane_FL::ConvertFrom(const Plane_FL& src, TransferChar dstTransferChar)
{
    PCType i;
    PCType pcount = src.PixelCount();

    value_type src_k0, src_phi, src_alpha, src_power, src_div;
    value_type dst_k0, dst_phi, dst_alpha, dst_power, dst_div;

    value_type data;

    TransferChar srcTransferChar = src.TransferChar_;
    TransferChar_ = dstTransferChar;

    if (TransferChar_ == srcTransferChar)
    {
        *this = src;
        return *this;
    }

    // Conversion
    if (isChroma() != src.isChroma()) DefaultPara(src.isChroma());
    ReSize(src.Width(), src.Height());

    if (srcTransferChar == TransferChar::linear)
    {
        if (TransferChar_ == TransferChar::linear)
        {
            for (i = 0; i < pcount; i++)
            {
                Data_[i] = src.Data_[i];
            }
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            for (i = 0; i < pcount; i++)
            {
                Data_[i] = TransferChar_linear2gamma(src.Data_[i], dst_k0, dst_div);
            }
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            for (i = 0; i < pcount; i++)
            {
                Data_[i] = TransferChar_linear2gamma(src.Data_[i], dst_k0, dst_phi, dst_alpha, dst_power);
            }
        }
    }
    else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
    {
        TransferChar_Parameter(srcTransferChar, src_k0, src_div);

        if (TransferChar_ == TransferChar::linear)
        {
            for (i = 0; i < pcount; i++)
            {
                Data_[i] = TransferChar_gamma2linear(src.Data_[i], src_k0, src_div);
            }
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            for (i = 0; i < pcount; i++)
            {
                data = TransferChar_gamma2linear(src.Data_[i], src_k0, src_div);
                Data_[i] = TransferChar_linear2gamma(data, dst_k0, dst_div);
            }
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            for (i = 0; i < pcount; i++)
            {
                data = TransferChar_gamma2linear(src.Data_[i], src_k0, src_div);
                Data_[i] = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
            }
        }
    }
    else
    {
        TransferChar_Parameter(srcTransferChar, src_k0, src_phi, src_alpha, src_power);

        if (TransferChar_ == TransferChar::linear)
        {
            for (i = 0; i < pcount; i++)
            {
                Data_[i] = TransferChar_gamma2linear(src.Data_[i], src_k0, src_phi, src_alpha, src_power);
            }
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            for (i = 0; i < pcount; i++)
            {
                data = TransferChar_gamma2linear(src.Data_[i], src_k0, src_phi, src_alpha, src_power);
                Data_[i] = TransferChar_linear2gamma(data, dst_k0, dst_div);
            }
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            for (i = 0; i < pcount; i++)
            {
                data = TransferChar_gamma2linear(src.Data_[i], src_k0, src_phi, src_alpha, src_power);
                Data_[i] = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
            }
        }
    }

    // Output
    return *this;
}

Plane_FL& Plane_FL::ConvertFrom(const Plane& src, TransferChar dstTransferChar)
{
    Plane::value_type i;

    value_type src_k0, src_phi, src_alpha, src_power, src_div;
    value_type dst_k0, dst_phi, dst_alpha, dst_power, dst_div;

    value_type data;
    LUT<value_type> LUT(src);

    Plane::value_type srcFloor = src.Floor(), srcCeil = src.Ceil();
    TransferChar srcTransferChar = src.GetTransferChar();
    TransferChar_ = dstTransferChar;

    if (TransferChar_ == srcTransferChar)
    {
        From(src);
        return *this;
    }

    // Generate conversion LUT
    if (srcTransferChar == TransferChar::linear)
    {
        if (TransferChar_ == TransferChar::linear)
        {
            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                LUT.Set(src, i, data);
            }
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_linear2gamma(data, dst_k0, dst_div);
                LUT.Set(src, i, data);
            }
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
                LUT.Set(src, i, data);
            }
        }
    }
    else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
    {
        TransferChar_Parameter(srcTransferChar, src_k0, src_div);

        if (TransferChar_ == TransferChar::linear)
        {
            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_div);
                LUT.Set(src, i, data);
            }
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_div);
                data = TransferChar_linear2gamma(data, dst_k0, dst_div);
                LUT.Set(src, i, data);
            }
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_div);
                data = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
                LUT.Set(src, i, data);
            }
        }
    }
    else
    {
        TransferChar_Parameter(srcTransferChar, src_k0, src_phi, src_alpha, src_power);

        if (TransferChar_ == TransferChar::linear)
        {
            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_phi, src_alpha, src_power);
                LUT.Set(src, i, data);
            }
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_phi, src_alpha, src_power);
                data = TransferChar_linear2gamma(data, dst_k0, dst_div);
                LUT.Set(src, i, data);
            }
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_gamma2linear(data, src_k0, src_phi, src_alpha, src_power);
                data = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
                LUT.Set(src, i, data);
            }
        }
    }

    // Conversion
    if (isChroma() != src.isChroma()) DefaultPara(src.isChroma());
    ReSize(src.Width(), src.Height());
    LUT.Lookup(*this, src);

    // Output
    return *this;
}

Plane_FL& Plane_FL::YFrom(const Frame& src, ColorMatrix dstColorMatrix)
{
    if (src.isRGB())
    {
        PCType i;
        value_type Kr, Kg, Kb;
        PCType pcount = src.PixelCount();

        auto& srcR = src.R();
        auto& srcG = src.G();
        auto& srcB = src.B();

        if (isChroma()) DefaultPara(false);
        ReSize(src.Width(), src.Height());

        ColorMatrix_Parameter(dstColorMatrix, Kr, Kg, Kb);

        value_type gain = (Ceil_ - Floor_) / srcR.ValueRange();
        value_type offset = Floor_ - srcR.Floor() * gain;
        Kr *= gain;
        Kg *= gain;
        Kb *= gain;

        for (i = 0; i < pcount; i++)
        {
            Data_[i] = Kr*srcR[i] + Kg*srcG[i] + Kb*srcB[i] + offset;
        }
    }
    else if (src.isYUV())
    {
        From(src.Y());
    }

    return *this;
}

Plane_FL& Plane_FL::YFrom(const Frame& src)
{
    return YFrom(src, src.GetColorMatrix());
}


Plane_FL& Plane_FL::Binarize(const _Myt& src, value_type lower_thrD, value_type upper_thrD)
{
    PCType i, j, upper;

    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Stride();

    double lower_thr = static_cast<double>(lower_thrD - src.Floor()) / src.ValueRange();
    double upper_thr = static_cast<double>(upper_thrD - src.Floor()) / src.ValueRange();

    if (upper_thr <= lower_thr || lower_thr >= 1 || upper_thr < 0)
    {
        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Data_[i] = Floor_;
            }
        }
    }
    else if (lower_thr < 0)
    {
        if (upper_thr >= 1)
        {
            for (j = 0; j < height; j++)
            {
                i = stride * j;
                for (upper = i + width; i < upper; i++)
                {
                    Data_[i] = Ceil_;
                }
            }
        }
        else
        {
            for (j = 0; j < height; j++)
            {
                i = stride * j;
                for (upper = i + width; i < upper; i++)
                {
                    Data_[i] = src[i] <= upper_thrD ? Ceil_ : Floor_;
                }
            }
        }
    }
    else
    {
        if (upper_thr >= 1)
        {
            for (j = 0; j < height; j++)
            {
                i = stride * j;
                for (upper = i + width; i < upper; i++)
                {
                    Data_[i] = src[i] > lower_thrD ? Ceil_ : Floor_;
                }
            }
        }
        else
        {
            for (j = 0; j < height; j++)
            {
                i = stride * j;
                for (upper = i + width; i < upper; i++)
                {
                    Data_[i] = src[i] > lower_thrD && src[i] <= upper_thrD ? Ceil_ : Floor_;
                }
            }
        }
    }

    return *this;
}

Plane_FL& Plane_FL::Binarize_ratio(const _Myt& src, double lower_thr, double upper_thr)
{
    value_type lower_thrD = static_cast<value_type>(lower_thr * src.ValueRange()) + src.Floor();
    value_type upper_thrD = static_cast<value_type>(upper_thr * src.ValueRange()) + src.Floor();

    return Binarize(src, lower_thrD, upper_thrD);
}

Plane_FL& Plane_FL::SimplestColorBalance(const Plane_FL& flt, const Plane_FL& src, double lower_thr, double upper_thr, int HistBins)
{
    value_type min, max;
    flt.MinMax(min, max);

    if (max <= min)
    {
        *this = src;
        return *this;
    }
    else
    {
        *this = flt;
    }

    if (lower_thr> 0 || upper_thr > 0)
    {
        ReQuantize(min, min, max, false);
        Histogram<value_type> Histogram(*this, HistBins);
        min = Histogram.Min(lower_thr);
        max = Histogram.Max(upper_thr);
    }

    ReQuantize(min, min, max, false);
    ReQuantize(src.Floor(), src.Neutral(), src.Ceil(), true, lower_thr> 0 || upper_thr > 0);

    return *this;
}


// Functions of class Frame
void Frame::InitPlanes(PCType Width, PCType Height, value_type BitDepth, bool Init)
{
    value_type Floor, Neutral, Ceil, ValueRange;

    FreePlanes();

    if (isRGB())
    {
        Quantize_Value(Floor, Neutral, Ceil, ValueRange, BitDepth, QuantRange_, false);

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
        {
            R_ = new _Mysub(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_, Init);

            P_[PlaneCount_++] = R_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
        {
            G_ = new _Mysub(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_, Init);

            P_[PlaneCount_++] = G_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
        {
            B_ = new _Mysub(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_, Init);

            P_[PlaneCount_++] = B_;
        }
    }
    else if (isYUV())
    {
        if (PixelType_ != PixelType::U && PixelType_ != PixelType::V)
        {
            if (PixelType_ == PixelType::YUV422 && Width % 2)
            {
                Width = (Width / 2 + 1) * 2;
            }
            else if (PixelType_ == PixelType::YUV420)
            {
                if (Width % 2)  Width = (Width / 2 + 1) * 2;
                if (Height % 2) Height = (Height / 2 + 1) * 2;
            }
            else if (PixelType_ == PixelType::YUV411 && Width % 4)
            {
                Width = (Width / 4 + 1) * 4;
            }

            Quantize_Value(Floor, Neutral, Ceil, ValueRange, BitDepth, QuantRange_, false);

            Y_ = new _Mysub(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_, Init);

            P_[PlaneCount_++] = Y_;
        }

        if (PixelType_ != PixelType::Y)
        {
            if (PixelType_ == PixelType::YUV422)
            {
                Width = Width / 2;
            }
            else if (PixelType_ == PixelType::YUV420)
            {
                Width = Width / 2;
                Height = Height / 2;
            }
            else if (PixelType_ == PixelType::YUV411)
            {
                Width = Width / 4;
            }

            Quantize_Value(Floor, Neutral, Ceil, ValueRange, BitDepth, QuantRange_, true);

            if (PixelType_ != PixelType::V)
            {
                U_ = new _Mysub(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar::linear, Init);

                P_[PlaneCount_++] = U_;
            }

            if (PixelType_ != PixelType::U)
            {
                V_ = new _Mysub(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar::linear, Init);

                P_[PlaneCount_++] = V_;
            }
        }
    }
}

void Frame::CopyPlanes(const _Myt& src, bool Copy, bool Init)
{
    value_type Floor, Neutral, Ceil, ValueRange;

    FreePlanes();

    if (isRGB())
    {
        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
        {
            if (Copy)
            {
                R_ = new _Mysub(*src.R_);
            }
            else
            {
                Quantize_Value(Floor, Neutral, Ceil, ValueRange, src.R_->BitDepth(), QuantRange_, false);

                R_ = new _Mysub(Floor, src.R_->Width(), src.R_->Height(), src.R_->BitDepth(), Floor, Neutral, Ceil, TransferChar_, Init);
            }
            P_[PlaneCount_++] = R_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
        {
            if (Copy)
            {
                G_ = new _Mysub(*src.G_);
            }
            else
            {
                Quantize_Value(Floor, Neutral, Ceil, ValueRange, src.G_->BitDepth(), QuantRange_, false);

                G_ = new _Mysub(Floor, src.G_->Width(), src.G_->Height(), src.G_->BitDepth(), Floor, Neutral, Ceil, TransferChar_, Init);
            }
            P_[PlaneCount_++] = G_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
        {
            if (Copy)
            {
                B_ = new _Mysub(*src.B_);
            }
            else
            {
                Quantize_Value(Floor, Neutral, Ceil, ValueRange, src.B_->BitDepth(), QuantRange_, false);

                B_ = new _Mysub(Floor, src.B_->Width(), src.B_->Height(), src.B_->BitDepth(), Floor, Neutral, Ceil, TransferChar_, Init);
            }
            P_[PlaneCount_++] = B_;
        }
    }
    else if (isYUV())
    {
        if (PixelType_ != PixelType::U && PixelType_ != PixelType::V)
        {
            if (Copy)
            {
                Y_ = new _Mysub(*src.Y_);
            }
            else
            {
                Quantize_Value(Floor, Neutral, Ceil, ValueRange, src.Y_->BitDepth(), QuantRange_, false);

                Y_ = new _Mysub(Floor, src.Y_->Width(), src.Y_->Height(), src.Y_->BitDepth(), Floor, Neutral, Ceil, TransferChar_, Init);
            }
            P_[PlaneCount_++] = Y_;
        }

        if (PixelType_ != PixelType::Y)
        {
            if (PixelType_ != PixelType::V)
            {
                if (Copy)
                {
                    U_ = new _Mysub(*src.U_);
                }
                else
                {
                    Quantize_Value(Floor, Neutral, Ceil, ValueRange, src.U_->BitDepth(), QuantRange_, true);

                    U_ = new _Mysub(Neutral, src.U_->Width(), src.U_->Height(), src.U_->BitDepth(), Floor, Neutral, Ceil, TransferChar::linear, Init);
                }
                P_[PlaneCount_++] = U_;
            }

            if (PixelType_ != PixelType::U)
            {
                if (Copy)
                {
                    V_ = new _Mysub(*src.V_);
                }
                else
                {
                    Quantize_Value(Floor, Neutral, Ceil, ValueRange, src.V_->BitDepth(), QuantRange_, true);

                    V_ = new _Mysub(Neutral, src.V_->Width(), src.V_->Height(), src.V_->BitDepth(), Floor, Neutral, Ceil, TransferChar::linear, Init);
                }
                P_[PlaneCount_++] = V_;
            }
        }
    }
}

void Frame::MovePlanes(_Myt& src)
{
    PlaneCount_ = src.PlaneCount_;

    P_ = std::move(src.P_);
    src.P_.insert(src.P_.end(), MaxPlaneCount, nullptr);

    R_ = src.R_;
    G_ = src.G_;
    B_ = src.B_;
    Y_ = src.Y_;
    U_ = src.U_;
    V_ = src.V_;
    A_ = src.A_;

    src.R_ = nullptr;
    src.G_ = nullptr;
    src.B_ = nullptr;
    src.Y_ = nullptr;
    src.U_ = nullptr;
    src.V_ = nullptr;
    src.A_ = nullptr;
}

void Frame::FreePlanes()
{
    PlaneCount_ = 0;

    for (auto x : P_)
    {
        delete x;
        x = nullptr;
    }

    R_ = nullptr;
    G_ = nullptr;
    B_ = nullptr;
    Y_ = nullptr;
    U_ = nullptr;
    V_ = nullptr;
    A_ = nullptr;
}


Frame::Frame(FCType FrameNum, PixelType _PixelType, PCType Width, PCType Height, value_type BitDepth, bool Init)
    : _Myt(FrameNum, _PixelType, Width, Height, BitDepth, isYUV(_PixelType) ? QuantRange::TV : QuantRange::PC, ChromaPlacement::MPEG2, Init)
{}

Frame::Frame(FCType FrameNum, PixelType _PixelType, PCType Width, PCType Height, value_type BitDepth,
    QuantRange _QuantRange, ChromaPlacement _ChromaPlacement, bool Init)
    : _Myt(FrameNum, _PixelType, Width, Height, BitDepth, _QuantRange, _ChromaPlacement,
    ColorPrim_Default(Width, Height, isRGB(_PixelType)), TransferChar_Default(Width, Height, isRGB(_PixelType)), ColorMatrix_Default(Width, Height))
{}

Frame::Frame(FCType FrameNum, PixelType _PixelType, PCType Width, PCType Height, value_type BitDepth, QuantRange _QuantRange,
    ChromaPlacement _ChromaPlacement, ColorPrim _ColorPrim, TransferChar _TransferChar, ColorMatrix _ColorMatrix, bool Init)
    : FrameNum_(FrameNum), PixelType_(_PixelType), QuantRange_(_QuantRange), ChromaPlacement_(_ChromaPlacement),
    ColorPrim_(_ColorPrim), TransferChar_(_TransferChar), ColorMatrix_(_ColorMatrix), P_(MaxPlaneCount, nullptr)
{
    const char * FunctionName = "class Frame constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    InitPlanes(Width, Height, BitDepth, Init);
}

Frame::Frame(const _Myt& src, bool Copy, bool Init)
    : FrameNum_(src.FrameNum_), PixelType_(src.PixelType_), QuantRange_(src.QuantRange_), ChromaPlacement_(src.ChromaPlacement_),
    ColorPrim_(src.ColorPrim_), TransferChar_(src.TransferChar_), ColorMatrix_(src.ColorMatrix_), P_(MaxPlaneCount, nullptr)
{
    CopyPlanes(src, Copy, Init);
}

Frame::Frame(_Myt&& src)
    : FrameNum_(src.FrameNum_), PixelType_(src.PixelType_), QuantRange_(src.QuantRange_), ChromaPlacement_(src.ChromaPlacement_),
    ColorPrim_(src.ColorPrim_), TransferChar_(src.TransferChar_), ColorMatrix_(src.ColorMatrix_)
{
    MovePlanes(src);
}


Frame::~Frame()
{
    FreePlanes();
}


Frame& Frame::operator=(const _Myt& src)
{
    if (this == &src)
    {
        return *this;
    }

    FrameNum_ = src.FrameNum_;
    PixelType_ = src.PixelType_;
    QuantRange_ = src.QuantRange_;
    ChromaPlacement_ = src.ChromaPlacement_;
    ColorPrim_ = src.ColorPrim_;
    TransferChar_ = src.TransferChar_;
    ColorMatrix_ = src.ColorMatrix_;

    CopyPlanes(src, true);

    return *this;
}

Frame& Frame::operator=(_Myt&& src)
{
    if (this == &src)
    {
        return *this;
    }

    FrameNum_ = src.FrameNum_;
    PixelType_ = src.PixelType_;
    QuantRange_ = src.QuantRange_;
    ChromaPlacement_ = src.ChromaPlacement_;
    ColorPrim_ = src.ColorPrim_;
    TransferChar_ = src.TransferChar_;
    ColorMatrix_ = src.ColorMatrix_;

    FreePlanes();

    MovePlanes(src);

    return *this;
}

bool Frame::operator==(const _Myt& b) const
{
    if (this == &b)
    {
        return true;
    }

    if (FrameNum_ != b.FrameNum_ || PixelType_ != b.PixelType_ || QuantRange_ != b.QuantRange_ || ChromaPlacement_ != b.ChromaPlacement_
        || ColorPrim_ != b.ColorPrim_ || TransferChar_ != b.TransferChar_ || ColorMatrix_ != b.ColorMatrix_ || PlaneCount_ != b.PlaneCount_)
    {
        return false;
    }

    if (isRGB())
    {
        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
        {
            if (*R_ != *b.R_) return false;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
        {
            if (*G_ != *b.G_) return false;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
        {
            if (*B_ != *b.B_) return false;
        }
    }
    else if (isYUV())
    {
        if (PixelType_ != PixelType::U && PixelType_ != PixelType::V)
        {
            if (*Y_ != *b.Y_) return false;
        }

        if (PixelType_ != PixelType::Y)
        {
            if (PixelType_ != PixelType::V)
            {
                if (*U_ != *b.U_) return false;
            }

            if (PixelType_ != PixelType::U)
            {
                if (*V_ != *b.V_) return false;
            }
        }
    }

    return true;
}


Frame& Frame::ConvertFrom(const Frame& src, TransferChar dstTransferChar)
{
    TransferChar_ = dstTransferChar;

    if (isRGB())
    {
        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
        {
            R_->ConvertFrom(*src.R_, TransferChar_);
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
        {
            G_->ConvertFrom(*src.G_, TransferChar_);
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
        {
            B_->ConvertFrom(*src.B_, TransferChar_);
        }
    }
    else if (isYUV())
    {
        if (PixelType_ != PixelType::U && PixelType_ != PixelType::V)
        {
            Y_->ConvertFrom(*src.Y_, TransferChar_);
        }

        if (PixelType_ != PixelType::Y)
        {
            if (PixelType_ != PixelType::V)
            {
                *U_ = *src.U_;
            }

            if (PixelType_ != PixelType::U)
            {
                *V_ = *src.V_;
            }
        }
    }

    return *this;
}
