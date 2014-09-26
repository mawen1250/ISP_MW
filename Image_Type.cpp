#include <iostream>
#include "include\Image_Type.h"
#include "include\Type_Conv.h"
#include "include\Specification.h"
#include "include\LUT.h"


// Default functions
ResLevel ResLevel_Default(PCType Width, PCType Height)
{
    return
        Width > HD_Width_U || Height > HD_Height_U ? ResLevel::UHD :
        Width > SD_Width_U || Height > SD_Height_U ? ResLevel::HD : ResLevel::SD;
}


ColorPrim ColorPrim_Default(PCType Width, PCType Height, bool RGB)
{
    ResLevel Res_Level = ResLevel_Default(Width, Height);

    if (RGB)
    {
        return ColorPrim::bt709;
    }
    else
    {
        return
            Res_Level == ResLevel::UHD ? ColorPrim::bt2020 :
            Res_Level == ResLevel::HD ? ColorPrim::bt709 :
            Res_Level == ResLevel::SD ? ColorPrim::smpte170m : ColorPrim::bt709;
    }
}


TransferChar TransferChar_Default(PCType Width, PCType Height, bool RGB)
{
    ResLevel Res_Level = ResLevel_Default(Width, Height);

    if (RGB)
    {
        return TransferChar::iec61966_2_1;
    }
    else
    {
        return
            Res_Level == ResLevel::UHD ? TransferChar::bt2020_12 :
            Res_Level == ResLevel::HD ? TransferChar::bt709 :
            Res_Level == ResLevel::SD ? TransferChar::smpte170m : TransferChar::bt709;
    }
}


ColorMatrix ColorMatrix_Default(PCType Width, PCType Height)
{
    ResLevel Res_Level = ResLevel_Default(Width, Height);

    return
        Res_Level == ResLevel::UHD ? ColorMatrix::bt2020nc :
        Res_Level == ResLevel::HD ? ColorMatrix::bt709 :
        Res_Level == ResLevel::SD ? ColorMatrix::smpte170m : ColorMatrix::bt709;
}


// Calculating functions
void Quantize_Value(DType * Floor, DType * Neutral, DType * Ceil, DType * ValueRange, DType BitDepth, QuantRange QuantRange, bool Chroma)
{
    if (Chroma)
    {
        *Floor = QuantRange == QuantRange::PC ? DType(0) : DType(16) << (BitDepth - DType(8));
        *Ceil = QuantRange == QuantRange::PC ? (DType(1) << BitDepth) - DType(1) : DType(240) << (BitDepth - DType(8));
        *Neutral = 1 << (BitDepth - 1);
        *ValueRange = QuantRange == QuantRange::PC ? (DType(1) << BitDepth) - DType(1) : DType(224) << (BitDepth - DType(8));
    }
    else
    {
        *Floor = QuantRange == QuantRange::PC ? DType(0) : DType(16) << (BitDepth - DType(8));
        *Ceil = QuantRange == QuantRange::PC ? (DType(1) << BitDepth) - DType(1) : DType(235) << (BitDepth - DType(8));
        *Neutral = *Floor;
        *ValueRange = QuantRange == QuantRange::PC ? (DType(1) << BitDepth) - DType(1) : DType(219) << (BitDepth - DType(8));
    }
}


// Functions of class Plane
void Plane::CopyParaFrom(const Plane & src)
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

void Plane::InitValue(DType Value, bool Init)
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
            memset(Data_, Value, sizeof(DType)*PixelCount_);
        }
    }
}


Plane::Plane(const Plane & src)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_), BitDepth_(src.BitDepth_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), ValueRange_(src.ValueRange_), TransferChar_(src.TransferChar_)
{
    Data_ = new DType[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(DType) * PixelCount_);
}

Plane::Plane(const Plane & src, bool Init, DType Value)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_), BitDepth_(src.BitDepth_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), ValueRange_(src.ValueRange_), TransferChar_(src.TransferChar_)
{
    Data_ = new DType[PixelCount_];

    InitValue(Value, Init);
}

Plane::Plane(Plane && src)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_), BitDepth_(src.BitDepth_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), ValueRange_(src.ValueRange_), TransferChar_(src.TransferChar_)
{
    Data_ = src.Data_;

    src.Width_ = 0;
    src.Height_ = 0;
    src.PixelCount_ = 0;
    src.Data_ = nullptr;
}

Plane::Plane(const Plane_FL & src, DType BitDepth)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), BitDepth_(BitDepth),
    Floor_(0), Neutral_(DType(1) << (BitDepth_ - 1)), Ceil_((DType(1) << BitDepth_) - 1), ValueRange_(Ceil_ - Floor_), TransferChar_(src.GetTransferChar())
{
    const char * FunctionName = "class Plane constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    PCType i;
    Data_ = new DType[PixelCount_];

    if (isPCChroma())
    {
        for (i = 0; i < PixelCount_; i++)
        {
            Data_[i] = GetD_PCChroma(src[i]);
        }
    }
    else
    {
        for (i = 0; i < PixelCount_; i++)
        {
            Data_[i] = GetD(src[i]);
        }
    }
}

Plane::Plane(const Plane_FL & src, bool Init, DType Value, DType BitDepth)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), BitDepth_(BitDepth),
    Floor_(0), Neutral_(DType(1) << (BitDepth_ - 1)), Ceil_((DType(1) << BitDepth_) - 1), ValueRange_(Ceil_ - Floor_), TransferChar_(src.GetTransferChar())
{
    const char * FunctionName = "class Plane constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    Data_ = new DType[PixelCount_];

    InitValue(Value, Init);
}

Plane::Plane(const Plane_FL & src, DType BitDepth, DType Floor, DType Neutral, DType Ceil)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), BitDepth_(BitDepth),
    Floor_(Floor), Neutral_(Neutral), Ceil_(Ceil), ValueRange_(Ceil_ - Floor_), TransferChar_(src.GetTransferChar())
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
    if (ValueRange_ >= DType(1) << BitDepth)
    {
        std::cerr << FunctionName << ": \"Ceil-Floor=" << ValueRange_ << "\" exceeds \"BitDepth=" << BitDepth << "\" limit.\n";
        exit(EXIT_FAILURE);
    }
    if (Neutral > Floor && Neutral != (Floor + Ceil + 1) / 2)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\", \"Neutral=" << Neutral << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }

    PCType i;
    Data_ = new DType[PixelCount_];

    if (isPCChroma())
    {
        for (i = 0; i < PixelCount_; i++)
        {
            Data_[i] = GetD_PCChroma(src[i]);
        }
    }
    else
    {
        for (i = 0; i < PixelCount_; i++)
        {
            Data_[i] = GetD(src[i]);
        }
    }
}

Plane::Plane(const Plane_FL & src, bool Init, DType Value, DType BitDepth, DType Floor, DType Neutral, DType Ceil)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), BitDepth_(BitDepth),
    Floor_(Floor), Neutral_(Neutral), Ceil_(Ceil), ValueRange_(Ceil_ - Floor_), TransferChar_(src.GetTransferChar())
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
    if (ValueRange_ >= DType(1) << BitDepth)
    {
        std::cerr << FunctionName << ": \"Ceil-Floor=" << ValueRange_ << "\" exceeds \"BitDepth=" << BitDepth << "\" limit.\n";
        exit(EXIT_FAILURE);
    }
    if (Neutral > Floor && Neutral != (Floor + Ceil + 1) / 2)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\", \"Neutral=" << Neutral << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }

    Data_ = new DType[PixelCount_];

    InitValue(Value, Init);
}

Plane::Plane(DType Value, PCType Width, PCType Height, DType BitDepth, bool Init)
    : Width_(Width), Height_(Height), PixelCount_(Width_ * Height_), BitDepth_(BitDepth),
    Floor_(0), Neutral_(DType(1) << (BitDepth_ - 1)), Ceil_((DType(1) << BitDepth_) - 1), ValueRange_(Ceil_ - Floor_), TransferChar_(TransferChar_Default(Width, Height, true))
{
    const char * FunctionName = "class Plane constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    Data_ = new DType[PixelCount_];

    InitValue(Value, Init);
}

Plane::Plane(DType Value, PCType Width, PCType Height, DType BitDepth, DType Floor, DType Neutral, DType Ceil, TransferChar TransferChar, bool Init)
    : Width_(Width), Height_(Height), PixelCount_(Width_ * Height_), BitDepth_(BitDepth),
    Floor_(Floor), Neutral_(Neutral), Ceil_(Ceil), ValueRange_(Ceil_ - Floor_), TransferChar_(TransferChar)
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
    if (ValueRange_ >= DType(1) << BitDepth)
    {
        std::cerr << FunctionName << ": \"Ceil-Floor=" << ValueRange_ << "\" exceeds \"BitDepth=" << BitDepth << "\" limit.\n";
        exit(EXIT_FAILURE);
    }
    if (Neutral > Floor && Neutral != (Floor + Ceil + 1) / 2)
    {
        std::cerr << FunctionName << ": invalid values of \"Floor=" << Floor << "\", \"Neutral=" << Neutral << "\" and \"Ceil=" << Ceil << "\" are set.\n";
        exit(EXIT_FAILURE);
    }

    Data_ = new DType[PixelCount_];

    InitValue(Value, Init);
}

Plane::~Plane()
{
    delete[] Data_;
}

Plane & Plane::operator=(const Plane & src)
{
    if (this == &src)
    {
        return *this;
    }

    CopyParaFrom(src);

    delete[] Data_;
    Data_ = new DType[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(DType) * PixelCount_);

    return *this;
}

Plane & Plane::operator=(Plane && src)
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

bool Plane::operator==(const Plane & b) const
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


DType Plane::Min() const
{
    DType min = ULONG_MAX;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        min = ::Min(min, Data_[i]);
    }

    return min;
}

DType Plane::Max() const
{
    DType max = 0;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        max = ::Max(max, Data_[i]);
    }

    return max;
}

const Plane & Plane::MinMax(DType & min, DType & max) const
{
    min = ULONG_MAX;
    max = 0;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        min = ::Min(min, Data_[i]);
        max = ::Max(max, Data_[i]);
    }

    return *this;
}


Plane & Plane::ReSize(PCType Width, PCType Height)
{
    if (Width_ != Width || Height_ != Height)
    {
        if (PixelCount_ != Width*Height)
        {
            delete[] Data_;
            PixelCount_ = Width*Height;
            Data_ = new DType[PixelCount_];
        }

        Width_ = Width;
        Height_ = Height;
    }

    return *this;
}

Plane & Plane::ReQuantize(DType BitDepth, QuantRange QuantRange, bool scale)
{
    const char * FunctionName = "Plane::ReQuantize";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    DType Floor, Neutral, Ceil, ValueRange;
    Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, BitDepth, QuantRange, isChroma());
    return ReQuantize(BitDepth, Floor, Neutral, Ceil, scale);
}

Plane & Plane::ReQuantize(DType BitDepth, DType Floor, DType Neutral, DType Ceil, bool scale)
{
    PCType i;
    DType ValueRange = Ceil - Floor;

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
    if (ValueRange >= DType(1) << BitDepth)
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
        if (isPCChroma())
        {
            if ((Floor + Ceil - 1) / 2 == Neutral - 1)
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = static_cast<DType>(GetFL_PCChroma(Data_[i])*ValueRange + Neutral + FLType(0.49999999));
                }
            }
            else
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = static_cast<DType>(GetFL_PCChroma(Data_[i])*ValueRange + Neutral + FLType(0.5));
                }
            }
        }
        else
        {
            if ((Floor + Ceil - 1) / 2 == Neutral - 1)
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = static_cast<DType>(GetFL(Data_[i])*ValueRange + Neutral + FLType(0.49999999));
                }
            }
            else
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = static_cast<DType>(GetFL(Data_[i])*ValueRange + Neutral + FLType(0.5));
                }
            }
        }
    }

    BitDepth_ = BitDepth;
    Floor_ = Floor;
    Neutral_ = Neutral;
    Ceil_ = Ceil;
    ValueRange_ = ValueRange;

    return *this;
}


Plane & Plane::From(const Plane & src)
{
    PCType i;

    if (isChroma() != src.isChroma()) // Plane "src" and "dst" are not both luma planes or chroma planes.
    {
        Floor_ = src.Floor_;
        Neutral_ = src.Neutral_;
        Ceil_ = src.Ceil_;
    }

    ReSize(src.Width(), src.Height());

    if (Floor_ == src.Floor_ && Neutral_ == src.Neutral_ && Ceil_ == src.Ceil_)
    {
        memcpy(Data_, src.Data_, sizeof(DType)*PixelCount_);
    }
    else if (src.isPCChroma())
    {
        if (isPCChroma())
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = GetD_PCChroma(src.GetFL_PCChroma(src[i]));
            }
        }
        else
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = GetD(src.GetFL_PCChroma(src[i]));
            }
        }
    }
    else
    {
        if (isPCChroma())
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = GetD_PCChroma(src.GetFL(src[i]));
            }
        }
        else
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = GetD(src.GetFL(src[i]));
            }
        }
    }

    return *this;
}

Plane & Plane::ConvertFrom(const Plane & src, TransferChar dstTransferChar)
{
    DType i;

    FLType src_k0, src_phi, src_alpha, src_power, src_div;
    FLType dst_k0, dst_phi, dst_alpha, dst_power, dst_div;

    FLType data;
    LUT<DType> LUT(src);

    DType srcFloor = src.Floor_, srcCeil = src.Ceil_;
    TransferChar srcTransferChar = src.TransferChar_;
    TransferChar_ = dstTransferChar;

    // Generate conversion LUT
    if (srcTransferChar == TransferChar::linear)
    {
        if (TransferChar_ == TransferChar::linear)
        {
            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                LUT.Set(src, i, GetD(data));
            }
        }
        else if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_div);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_linear2gamma(data, dst_k0, dst_div);
                LUT.Set(src, i, GetD(data));
            }
        }
        else
        {
            TransferChar_Parameter(TransferChar_, dst_k0, dst_phi, dst_alpha, dst_power);

            for (i = srcFloor; i <= srcCeil; i++)
            {
                data = src.GetFL(i);
                data = TransferChar_linear2gamma(data, dst_k0, dst_phi, dst_alpha, dst_power);
                LUT.Set(src, i, GetD(data));
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
                LUT.Set(src, i, GetD(data));
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
                LUT.Set(src, i, GetD(data));
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
                LUT.Set(src, i, GetD(data));
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
                LUT.Set(src, i, GetD(data));
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
                LUT.Set(src, i, GetD(data));
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
                LUT.Set(src, i, GetD(data));
            }
        }
    }

    // Conversion
    ReSize(src.Width(), src.Height());
    LUT.Lookup(*this, src);

    // Output
    return *this;
}

Plane & Plane::YFrom(const Frame & src, ColorMatrix dstColorMatrix)
{
    if (src.isRGB())
    {
        PCType i;
        FLType Kr, Kg, Kb;
        FLType dataR, dataG, dataB;
        PCType pcount = src.PixelCount();

        const Plane & srcR = src.R();
        const Plane & srcG = src.G();
        const Plane & srcB = src.B();

        ReSize(src.Width(), src.Height());

        ColorMatrix_Parameter(dstColorMatrix, Kr, Kg, Kb);

        for (i = 0; i < pcount; i++)
        {
            dataR = srcR.GetFL(srcR[i]);
            dataG = srcG.GetFL(srcG[i]);
            dataB = srcB.GetFL(srcB[i]);
            Data_[i] = GetD(Kr*dataR + Kg*dataB + Kb*dataG);
        }
    }
    else if (src.isYUV())
    {
        From(src.Y());
    }

    return *this;
}

Plane & Plane::YFrom(const Frame & src)
{
    return YFrom(src, src.GetColorMatrix());
}


// Functions of class Plane_FL
void Plane_FL::DefaultPara(bool Chroma)
{
    if (Chroma) // Plane "src" is chroma
    {
        Floor_ = -0.5;
        Neutral_ = 0;
        Ceil_ = 0.5;
    }
    else // Plane "src" is not chroma
    {
        Floor_ = 0;
        Neutral_ = 0;
        Ceil_ = 1;
    }
}

void Plane_FL::CopyParaFrom(const Plane_FL & src)
{
    Width_ = src.Width_;
    Height_ = src.Height_;
    PixelCount_ = Width_ * Height_;
    Floor_ = src.Floor_;
    Neutral_ = src.Neutral_;
    Ceil_ = src.Ceil_;
    TransferChar_ = src.TransferChar_;
}

void Plane_FL::InitValue(FLType Value, bool Init)
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


Plane_FL::Plane_FL(const Plane_FL & src)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), TransferChar_(src.TransferChar_)
{
    Data_ = new FLType[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(FLType) * PixelCount_);
}

Plane_FL::Plane_FL(const Plane_FL & src, bool Init, FLType Value)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), TransferChar_(src.TransferChar_)
{
    Data_ = new FLType[PixelCount_];

    InitValue(Value, Init);
}

Plane_FL::Plane_FL(Plane_FL && src)
    : Width_(src.Width_), Height_(src.Height_), PixelCount_(Width_ * Height_),
    Floor_(src.Floor_), Neutral_(src.Neutral_), Ceil_(src.Ceil_), TransferChar_(src.TransferChar_)
{
    Data_ = src.Data_;

    src.Width_ = 0;
    src.Height_ = 0;
    src.PixelCount_ = 0;
    src.Data_ = nullptr;
}

Plane_FL::Plane_FL(const Plane & src)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), TransferChar_(src.GetTransferChar())
{
    Data_ = new FLType[PixelCount_];

    DefaultPara(src.isChroma());

    From(src);
}

Plane_FL::Plane_FL(const Plane & src, bool Init, FLType Value)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), TransferChar_(src.GetTransferChar())
{
    Data_ = new FLType[PixelCount_];

    DefaultPara(src.isChroma());

    InitValue(Value, Init);
}

Plane_FL::Plane_FL(const Plane_FL & src, TransferChar dstTransferChar)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), TransferChar_(dstTransferChar)
{
    Data_ = new FLType[PixelCount_];

    DefaultPara(src.isChroma());

    ConvertFrom(src, dstTransferChar);
}

Plane_FL::Plane_FL(const Plane & src, TransferChar dstTransferChar)
    : Width_(src.Width()), Height_(src.Height()), PixelCount_(Width_ * Height_), TransferChar_(dstTransferChar)
{
    Data_ = new FLType[PixelCount_];

    DefaultPara(src.isChroma());

    ConvertFrom(src, dstTransferChar);
}

Plane_FL::Plane_FL(FLType Value, PCType Width, PCType Height, bool RGB, bool Chroma, bool Init)
    : Width_(Width), Height_(Height), PixelCount_(Width_ * Height_), TransferChar_(!RGB&&Chroma ? TransferChar::linear : TransferChar_Default(Width, Height, RGB))
{
    Data_ = new FLType[PixelCount_];

    DefaultPara(!RGB&&Chroma);

    InitValue(Value, Init);
}

Plane_FL::Plane_FL(FLType Value, PCType Width, PCType Height, FLType Floor, FLType Neutral, FLType Ceil, TransferChar TransferChar, bool Init)
    : Width_(Width), Height_(Height), PixelCount_(Width_ * Height_),
    Floor_(Floor), Neutral_(Neutral), Ceil_(Ceil), TransferChar_(TransferChar)
{
    Data_ = new FLType[PixelCount_];

    InitValue(Value, Init);
}

Plane_FL::~Plane_FL()
{
    delete[] Data_;
}

Plane_FL & Plane_FL::operator=(const Plane_FL & src)
{
    if (this == &src)
    {
        return *this;
    }

    CopyParaFrom(src);

    delete[] Data_;
    Data_ = new FLType[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(FLType) * PixelCount_);

    return *this;
}

Plane_FL & Plane_FL::operator=(Plane_FL && src)
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

bool Plane_FL::operator==(const Plane_FL & b) const
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


FLType Plane_FL::Min() const
{
    FLType min = FLT_MAX;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        min = ::Min(min, Data_[i]);
    }

    return min;
}

FLType Plane_FL::Max() const
{
    FLType max = FLT_MIN;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        max = ::Max(max, Data_[i]);
    }

    return max;
}

const Plane_FL & Plane_FL::MinMax(FLType & min, FLType & max) const
{
    min = FLT_MAX;
    max = FLT_MIN;

    for (PCType i = 0; i < PixelCount_; i++)
    {
        min = ::Min(min, Data_[i]);
        max = ::Max(max, Data_[i]);
    }

    return *this;
}


Plane_FL & Plane_FL::ReSize(PCType Width, PCType Height)
{
    if (Width_ != Width || Height_ != Height)
    {
        if (PixelCount_ != Width*Height)
        {
            delete[] Data_;
            PixelCount_ = Width*Height;
            Data_ = new FLType[PixelCount_];
        }

        Width_ = Width;
        Height_ = Height;
    }

    return *this;
}

Plane_FL & Plane_FL::ReQuantize(FLType Floor, FLType Neutral, FLType Ceil, bool scale)
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
        FLType gain = (Ceil - Floor) / (Ceil_ - Floor_);
        FLType offset = Neutral - Neutral_*gain;

        for (i = 0; i < PixelCount_; i++)
        {
            Data_[i] = Data_[i] * gain + offset;
        }
    }

    Floor_ = Floor;
    Neutral_ = Neutral;
    Ceil_ = Ceil;

    return *this;
}


Plane_FL & Plane_FL::From(const Plane & src)
{
    PCType i;
    DType iFloor = src.Floor();
    DType iNeutral = src.Neutral();
    DType iValueRange = src.ValueRange();
    FLType oFloor = Floor_;
    FLType oNeutral = Neutral_;
    FLType oValueRange = Ceil_ - Floor_;
    
    if (isChroma() != src.isChroma()) DefaultPara(src.isChroma());

    ReSize(src.Width(), src.Height());

    if (!isChroma()) // Plane_FL "*this" is not chroma
    {
        if (Floor_ == 0 && Neutral_ == 0 && Ceil_ == 1)
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = src.GetFL(src[i]);
            }
        }
        else
        {
            FLType gain = oValueRange / iValueRange;
            FLType offset = oFloor - iFloor*gain;

            for (i = 0; i < PixelCount_; i++)
            {
                Data_[i] = src[i] * gain + offset;
            }
        }
    }
    else
    {
        if (Floor_ == -0.5 && Neutral_ == 0 && Ceil_ == 0.5)
        {
            if (src.isPCChroma())
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = src.GetFL_PCChroma(src[i]);
                }
            }
            else
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = src.GetFL(src[i]);
                }
            }
        }
        else
        {
            if (src.isPCChroma())
            {
                FLType gain = oValueRange / iValueRange;
                FLType offset = oNeutral - iNeutral*gain;

                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = src[i] * gain + offset;
                }
            }
            else
            {
                FLType gain = oValueRange / iValueRange;
                FLType offset = oNeutral - iNeutral*gain;

                for (i = 0; i < PixelCount_; i++)
                {
                    Data_[i] = src[i] * gain + offset;
                }
            }
        }
    }

    return *this;
}

const Plane_FL & Plane_FL::To(Plane & dst) const
{
    PCType i;
    FLType iFloor = Floor_;
    FLType iNeutral = Neutral_;
    FLType iValueRange = Ceil_ - Floor_;
    DType oFloor = dst.Floor();
    DType oNeutral = dst.Neutral();
    DType oValueRange = dst.ValueRange();

    dst.ReSize(Width_, Height_);
    DType * Data = dst.Data();

    if (!isChroma()) // Plane_FL "*this" is not chroma
    {
        if (Floor_ == 0 && Neutral_ == 0 && Ceil_ == 1)
        {
            for (i = 0; i < PixelCount_; i++)
            {
                Data[i] = dst.GetD(Data_[i]);
            }
        }
        else
        {
            FLType gain = oValueRange / iValueRange;
            FLType offset = oFloor - iFloor*gain + FLType(0.5);

            for (i = 0; i < PixelCount_; i++)
            {
                Data[i] = static_cast<DType>(Data_[i] * gain + offset);
            }
        }
    }
    else // Plane_FL "*this" is chroma
    {
        if (Floor_ == -0.5 && Neutral_ == 0 && Ceil_ == 0.5)
        {
            if (dst.isPCChroma())
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data[i] = dst.GetD_PCChroma(Data_[i]);
                }
            }
            else
            {
                for (i = 0; i < PixelCount_; i++)
                {
                    Data[i] = dst.GetD(Data_[i]);
                }
            }
        }
        else
        {
            if (dst.isPCChroma())
            {
                FLType gain = oValueRange / iValueRange;
                FLType offset = oNeutral - iNeutral*gain + FLType(0.49999999);

                for (i = 0; i < PixelCount_; i++)
                {
                    Data[i] = static_cast<DType>(Data_[i] * gain + offset);
                }
            }
            else
            {
                FLType gain = oValueRange / iValueRange;
                FLType offset = oNeutral - iNeutral*gain + FLType(0.5);

                for (i = 0; i < PixelCount_; i++)
                {
                    Data[i] = static_cast<DType>(Data_[i] * gain + offset);
                }
            }
        }
    }

    return *this;
}

Plane_FL & Plane_FL::ConvertFrom(const Plane_FL & src, TransferChar dstTransferChar)
{
    PCType i;
    PCType pcount = src.PixelCount();

    FLType src_k0, src_phi, src_alpha, src_power, src_div;
    FLType dst_k0, dst_phi, dst_alpha, dst_power, dst_div;

    FLType data;

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

Plane_FL & Plane_FL::ConvertFrom(const Plane & src, TransferChar dstTransferChar)
{
    DType i;

    FLType src_k0, src_phi, src_alpha, src_power, src_div;
    FLType dst_k0, dst_phi, dst_alpha, dst_power, dst_div;

    FLType data;
    LUT<FLType> LUT(src);

    DType srcFloor = src.Floor(), srcCeil = src.Ceil();
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

Plane_FL & Plane_FL::YFrom(const Frame & src, ColorMatrix dstColorMatrix)
{
    if (src.isRGB())
    {
        PCType i;
        FLType Kr, Kg, Kb;
        FLType dataR, dataG, dataB;
        PCType pcount = src.PixelCount();

        const Plane & srcR = src.R();
        const Plane & srcG = src.G();
        const Plane & srcB = src.B();

        if (isChroma()) DefaultPara(false);
        ReSize(src.Width(), src.Height());

        ColorMatrix_Parameter(dstColorMatrix, Kr, Kg, Kb);

        for (i = 0; i < pcount; i++)
        {
            dataR = srcR.GetFL(srcR[i]);
            dataG = srcG.GetFL(srcG[i]);
            dataB = srcB.GetFL(srcB[i]);
            Data_[i] = Kr*dataR + Kg*dataB + Kb*dataG;
        }
    }
    else if (src.isYUV())
    {
        From(src.Y());
    }

    return *this;
}

Plane_FL & Plane_FL::YFrom(const Frame & src)
{
    return YFrom(src, src.GetColorMatrix());
}


// Functions of class Frame
void Frame::InitPlanes(PCType Width, PCType Height, DType BitDepth)
{
    DType Floor, Neutral, Ceil, ValueRange;

    PlaneCount_ = 0;
    P_ = new Plane*[MaxPlaneCount];

    if (isRGB())
    {
        Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, BitDepth, QuantRange_, false);

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
        {
            R_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_);

            P_[PlaneCount_++] = R_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
        {
            G_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_);

            P_[PlaneCount_++] = G_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
        {
            B_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_);

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

            Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, BitDepth, QuantRange_, false);

            Y_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar_);

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

            Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, BitDepth, QuantRange_, true);

            if (PixelType_ != PixelType::V)
            {
                U_ = new Plane(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar::linear);

                P_[PlaneCount_++] = U_;
            }

            if (PixelType_ != PixelType::U)
            {
                V_ = new Plane(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, TransferChar::linear);

                P_[PlaneCount_++] = V_;
            }
        }
    }
}

void Frame::CopyPlanes(const Frame & src, bool Copy)
{
    DType Floor, Neutral, Ceil, ValueRange;

    PlaneCount_ = 0;
    P_ = new Plane*[MaxPlaneCount];
    R_ = nullptr;
    G_ = nullptr;
    B_ = nullptr;
    Y_ = nullptr;
    U_ = nullptr;
    V_ = nullptr;
    A_ = nullptr;

    if (isRGB())
    {
        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
        {
            if (Copy)
            {
                R_ = new Plane(*src.R_);
            }
            else
            {
                Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.R_->BitDepth(), QuantRange_, false);

                R_ = new Plane(Floor, src.R_->Width(), src.R_->Height(), src.R_->BitDepth(), Floor, Neutral, Ceil, TransferChar_);
            }
            P_[PlaneCount_++] = R_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
        {
            if (Copy)
            {
                G_ = new Plane(*src.G_);
            }
            else
            {
                Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.G_->BitDepth(), QuantRange_, false);

                G_ = new Plane(Floor, src.G_->Width(), src.G_->Height(), src.G_->BitDepth(), Floor, Neutral, Ceil, TransferChar_);
            }
            P_[PlaneCount_++] = G_;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
        {
            if (Copy)
            {
                B_ = new Plane(*src.B_);
            }
            else
            {
                Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.B_->BitDepth(), QuantRange_, false);

                B_ = new Plane(Floor, src.B_->Width(), src.B_->Height(), src.B_->BitDepth(), Floor, Neutral, Ceil, TransferChar_);
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
                Y_ = new Plane(*src.Y_);
            }
            else
            {
                Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.Y_->BitDepth(), QuantRange_, false);

                Y_ = new Plane(Floor, src.Y_->Width(), src.Y_->Height(), src.Y_->BitDepth(), Floor, Neutral, Ceil, TransferChar_);
            }
            P_[PlaneCount_++] = Y_;
        }

        if (PixelType_ != PixelType::Y)
        {
            if (PixelType_ != PixelType::V)
            {
                if (Copy)
                {
                    U_ = new Plane(*src.U_);
                }
                else
                {
                    Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.U_->BitDepth(), QuantRange_, true);

                    U_ = new Plane(Neutral, src.U_->Width(), src.U_->Height(), src.U_->BitDepth(), Floor, Neutral, Ceil, TransferChar::linear);
                }
                P_[PlaneCount_++] = U_;
            }

            if (PixelType_ != PixelType::U)
            {
                if (Copy)
                {
                    V_ = new Plane(*src.V_);
                }
                else
                {
                    Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.V_->BitDepth(), QuantRange_, true);

                    V_ = new Plane(Neutral, src.V_->Width(), src.V_->Height(), src.V_->BitDepth(), Floor, Neutral, Ceil, TransferChar::linear);
                }
                P_[PlaneCount_++] = V_;
            }
        }
    }
}


Frame::Frame(const Frame & src, bool Copy)
    : FrameNum_(src.FrameNum_), PixelType_(src.PixelType_), QuantRange_(src.QuantRange_), ChromaPlacement_(src.ChromaPlacement_),
    ColorPrim_(src.ColorPrim_), TransferChar_(src.TransferChar_), ColorMatrix_(src.ColorMatrix_)
{
    CopyPlanes(src, Copy);
}

Frame::Frame(Frame && src)
    : FrameNum_(src.FrameNum_), PixelType_(src.PixelType_), QuantRange_(src.QuantRange_), ChromaPlacement_(src.ChromaPlacement_),
    ColorPrim_(src.ColorPrim_), TransferChar_(src.TransferChar_), ColorMatrix_(src.ColorMatrix_), PlaneCount_(src.PlaneCount_)
{
    P_ = src.P_;
    R_ = src.R_;
    G_ = src.G_;
    B_ = src.B_;
    Y_ = src.Y_;
    U_ = src.U_;
    V_ = src.V_;
    A_ = src.A_;

    src.P_ = nullptr;
    src.R_ = nullptr;
    src.G_ = nullptr;
    src.B_ = nullptr;
    src.Y_ = nullptr;
    src.U_ = nullptr;
    src.V_ = nullptr;
    src.A_ = nullptr;
}

Frame::Frame(FCType FrameNum, PixelType PixelType, PCType Width, PCType Height, DType BitDepth)
    : FrameNum_(FrameNum), PixelType_(PixelType), QuantRange_(isYUV() ? QuantRange::TV : QuantRange::PC), ChromaPlacement_(ChromaPlacement::MPEG2),
    ColorPrim_(ColorPrim_Default(Width, Height, isRGB())), TransferChar_(TransferChar_Default(Width, Height, isRGB())), ColorMatrix_(ColorMatrix_Default(Width, Height))
{
    const char * FunctionName = "class Frame constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    InitPlanes(Width, Height, BitDepth);
}

Frame::Frame(FCType FrameNum, PixelType PixelType, PCType Width, PCType Height, DType BitDepth,
    QuantRange QuantRange, ChromaPlacement ChromaPlacement)
    : FrameNum_(FrameNum), PixelType_(PixelType), QuantRange_(QuantRange), ChromaPlacement_(ChromaPlacement),
    ColorPrim_(ColorPrim_Default(Width, Height, isRGB())), TransferChar_(TransferChar_Default(Width, Height, isRGB())), ColorMatrix_(ColorMatrix_Default(Width, Height))
{
    const char * FunctionName = "class Frame constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    InitPlanes(Width, Height, BitDepth);
}

Frame::Frame(FCType FrameNum, PixelType PixelType, PCType Width, PCType Height, DType BitDepth, QuantRange QuantRange,
    ChromaPlacement ChromaPlacement, ColorPrim ColorPrim, TransferChar TransferChar, ColorMatrix ColorMatrix)
    : FrameNum_(FrameNum), PixelType_(PixelType), QuantRange_(QuantRange), ChromaPlacement_(ChromaPlacement),
    ColorPrim_(ColorPrim), TransferChar_(TransferChar), ColorMatrix_(ColorMatrix)
{
    const char * FunctionName = "class Frame constructor";
    if (BitDepth > MaxBitDepth)
    {
        std::cerr << FunctionName << ": \"BitDepth=" << BitDepth << "\" is invalid, maximum allowed bit depth is " << MaxBitDepth << ".\n";
        exit(EXIT_FAILURE);
    }

    InitPlanes(Width, Height, BitDepth);
}

Frame::~Frame()
{
    delete[] P_;
    delete R_;
    delete G_;
    delete B_;
    delete Y_;
    delete U_;
    delete V_;
    delete A_;
}

Frame & Frame::operator=(const Frame & src)
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

    delete[] P_;
    delete R_;
    delete G_;
    delete B_;
    delete Y_;
    delete U_;
    delete V_;
    delete A_;

    CopyPlanes(src, true);

    return *this;
}

Frame & Frame::operator=(Frame && src)
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
    PlaneCount_ = src.PlaneCount_;

    delete[] P_;
    delete R_;
    delete G_;
    delete B_;
    delete Y_;
    delete U_;
    delete V_;
    delete A_;

    P_ = src.P_;
    R_ = src.R_;
    G_ = src.G_;
    B_ = src.B_;
    Y_ = src.Y_;
    U_ = src.U_;
    V_ = src.V_;
    A_ = src.A_;

    src.P_ = nullptr;
    src.R_ = nullptr;
    src.G_ = nullptr;
    src.B_ = nullptr;
    src.Y_ = nullptr;
    src.U_ = nullptr;
    src.V_ = nullptr;
    src.A_ = nullptr;

    return *this;
}

bool Frame::operator==(const Frame & b) const
{
    if (this == &b)
    {
        return true;
    }

    if (FrameNum_ != b.FrameNum_ || PixelType_ != b.PixelType_ || QuantRange_ != b.QuantRange_ || ChromaPlacement_ != b.ChromaPlacement_ ||
        ColorPrim_ != b.ColorPrim_ || TransferChar_ != b.TransferChar_ || ColorMatrix_ != b.ColorMatrix_ || PlaneCount_ != b.PlaneCount_)
    {
        return false;
    }

    if (isRGB())
    {
        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
        {
            if (R_ != b.R_) return false;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
        {
            if (G_ != b.G_) return false;
        }

        if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
        {
            if (B_ != b.B_) return false;
        }
    }
    else if (isYUV())
    {
        if (PixelType_ != PixelType::U && PixelType_ != PixelType::V)
        {
            if (Y_ != b.Y_) return false;
        }

        if (PixelType_ != PixelType::Y)
        {
            if (PixelType_ != PixelType::V)
            {
                if (U_ != b.U_) return false;
            }

            if (PixelType_ != PixelType::U)
            {
                if (V_ != b.V_) return false;
            }
        }
    }

    return true;
}


Frame & Frame::ConvertFrom(const Frame & src, TransferChar dstTransferChar)
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
