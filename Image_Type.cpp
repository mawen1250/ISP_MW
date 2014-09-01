#include <iostream>
#include "include\Image_Type.h"


// Default functions
ResLevel ResLevel_Default(PCType Width, PCType Height)
{
    return
        Width>HD_Width_U || Height>HD_Height_U ? ResLevel::UHD :
        Width>SD_Width_U || Height>SD_Height_U ? ResLevel::HD : ResLevel::SD;
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
STAT Quantize_Value(DType * Floor, DType * Neutral, DType * Ceil, DType * ValueRange, DType BitDepth, QuantRange QuantRange, bool Chroma)
{
    if (Chroma)
    {
        *Floor      = QuantRange==QuantRange::PC ? 0 : 16<<(BitDepth-8);
        *Neutral    = 1<<(BitDepth-1);
        *Ceil = QuantRange == QuantRange::PC ? (1 << BitDepth) - 1 : 240 << (BitDepth - 8);
        *ValueRange = QuantRange == QuantRange::PC ? (1 << BitDepth) - 1 : 224 << (BitDepth - 8);
    }
    else
    {
        *Floor = QuantRange == QuantRange::PC ? 0 : 16 << (BitDepth - 8);
        *Neutral    = 1<<(BitDepth-1);
        *Ceil = QuantRange == QuantRange::PC ? (1 << BitDepth) - 1 : 235 << (BitDepth - 8);
        *ValueRange = QuantRange == QuantRange::PC ? (1 << BitDepth) - 1 : 219 << (BitDepth - 8);
    }

    return STAT::OK;
}


// Functions of class Plane
Plane::Plane(DType Value, PCType Width, PCType Height, DType BitDepth, bool InitValue)
{
    Width_ = Width;
    Height_ = Height;
    PixelCount_ = Width_ * Height_;
    BitDepth_ = BitDepth;
    Floor_ = 0;
    Neutral_ = 1 << (BitDepth_ - 1);
    Ceil_ = (1 << BitDepth_) - 1;
    ValueRange_ = Ceil_ - Floor_;

    Data_ = new DType[PixelCount_];

    if (InitValue)
    {
        for (PCType i = 0; i < PixelCount_; i++)
        {
            Data_[i] = Value;
        }
    }
}

Plane::Plane(DType Value, PCType Width, PCType Height, DType BitDepth, DType Floor, DType Neutral, DType Ceil, DType ValueRange, bool InitValue)
{
    Width_ = Width;
    Height_ = Height;
    PixelCount_ = Width_ * Height_;
    BitDepth_ = BitDepth;
    Floor_ = Floor;
    Neutral_ = Neutral;
    Ceil_ = Ceil;
    ValueRange_ = ValueRange;

    Data_ = new DType[PixelCount_];

    if (InitValue)
    {
        for (PCType i = 0; i < PixelCount_; i++)
        {
            Data_[i] = Value;
        }
    }
}

Plane::Plane(const Plane & src)
{
    Width_ = src.Width_;
    Height_ = src.Height_;
    PixelCount_ = Width_ * Height_;
    BitDepth_ = src.BitDepth_;
    Floor_ = src.Floor_;
    Neutral_ = src.Neutral_;
    Ceil_ = src.Ceil_;
    ValueRange_ = src.ValueRange_;

    Data_ = new DType[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(DType) * PixelCount_);
}

Plane::Plane(const Plane & src, bool InitValue, DType Value)
{
    Width_ = src.Width_;
    Height_ = src.Height_;
    PixelCount_ = Width_ * Height_;
    BitDepth_ = src.BitDepth_;
    Floor_ = src.Floor_;
    Neutral_ = src.Neutral_;
    Ceil_ = src.Ceil_;
    ValueRange_ = src.ValueRange_;

    Data_ = new DType[PixelCount_];

    if (InitValue)
    {
        for (PCType i = 0; i < PixelCount_; i++)
        {
            Data_[i] = Value;
        }
    }
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

    Width_ = src.Width_;
    Height_ = src.Height_;
    PixelCount_ = Width_ * Height_;
    BitDepth_ = src.BitDepth_;
    Floor_ = src.Floor_;
    Neutral_ = src.Neutral_;
    Ceil_ = src.Ceil_;
    ValueRange_ = src.ValueRange_;

    delete[] Data_;
    Data_ = new DType[PixelCount_];

    memcpy(Data_, src.Data_, sizeof(DType) * PixelCount_);

    return *this;
}

bool Plane::operator==(const Plane & b) const
{
    if (this == &b)
    {
        return true;
    }

    if (Width_ != b.Width_ || Height_ != b.Height_ || PixelCount_ != b.PixelCount_ || BitDepth_ != b.BitDepth_ ||
        Floor_ != b.Floor_ || Neutral_ != b.Neutral_ || Ceil_ != b.Ceil_ || ValueRange_ != b.ValueRange_)
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


// Functions of class Frame_YUV
Frame_YUV::Frame_YUV(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange, ChromaPlacement ChromaPlacement)
{
    DType Floor, Neutral, Ceil, ValueRange;

    FrameNum_ = FrameNum;
    PixelType_ = PixelType;
    ChromaPlacement_ = ChromaPlacement;
    QuantRange_ = QuantRange;
    ColorPrim_ = ColorPrim_Default(Width, Height, false);
    TransferChar_ = TransferChar_Default(Width, Height, false);
    ColorMatrix_ = ColorMatrix_Default(Width, Height);

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

        Y_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        Y_ = nullptr;
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
            U_ = new Plane(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
        }
        else
        {
            U_ = nullptr;
        }

        if (PixelType_ != PixelType::U)
        {
            V_ = new Plane(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
        }
        else
        {
            V_ = nullptr;
        }
    }
}

Frame_YUV::Frame_YUV(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange, ChromaPlacement ChromaPlacement, ColorPrim ColorPrim, TransferChar TransferChar, ColorMatrix ColorMatrix)
{
    DType Floor, Neutral, Ceil, ValueRange;

    FrameNum_ = FrameNum;
    PixelType_ = PixelType;
    ChromaPlacement_ = ChromaPlacement;
    QuantRange_ = QuantRange;
    ColorPrim_ = ColorPrim;
    TransferChar_ = TransferChar;
    ColorMatrix_ = ColorMatrix;

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

        Y_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        Y_ = nullptr;
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
            U_ = new Plane(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
        }
        else
        {
            U_ = nullptr;
        }

        if (PixelType_ != PixelType::U)
        {
            V_ = new Plane(Neutral, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
        }
        else
        {
            V_ = nullptr;
        }
    }
}

Frame_YUV::Frame_YUV(const Frame_YUV & src, bool Copy)
{
    DType Floor, Neutral, Ceil, ValueRange;

    FrameNum_ = src.FrameNum_;
    PixelType_ = src.PixelType_;
    ChromaPlacement_ = src.ChromaPlacement_;
    QuantRange_ = src.QuantRange_;
    ColorPrim_ = src.ColorPrim_;
    TransferChar_ = src.TransferChar_;
    ColorMatrix_ = src.ColorMatrix_;

    if (PixelType_ != PixelType::U && PixelType_ != PixelType::V)
    {
        if (Copy)
        {
            Y_ = new Plane(*src.Y_);
        }
        else
        {
            Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.Y_->BitDepth(), QuantRange_, false);

            Y_ = new Plane(Floor, src.Y_->Width(), src.Y_->Height(), src.Y_->BitDepth(), Floor, Neutral, Ceil, ValueRange);
        }
    }
    else
    {
        Y_ = nullptr;
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

                U_ = new Plane(Neutral, src.U_->Width(), src.U_->Height(), src.U_->BitDepth(), Floor, Neutral, Ceil, ValueRange);
            }
        }
        else
        {
            U_ = nullptr;
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

                V_ = new Plane(Neutral, src.V_->Width(), src.V_->Height(), src.V_->BitDepth(), Floor, Neutral, Ceil, ValueRange);
            }
        }
        else
        {
            V_ = nullptr;
        }
    }
}

Frame_YUV::~Frame_YUV()
{
    delete Y_;
    delete U_;
    delete V_;
}

Frame_YUV & Frame_YUV::operator=(const Frame_YUV & src)
{
    if (this == &src)
    {
        return *this;
    }

    FrameNum_ = src.FrameNum_;
    PixelType_ = src.PixelType_;
    ChromaPlacement_ = src.ChromaPlacement_;
    QuantRange_ = src.QuantRange_;
    ColorPrim_ = src.ColorPrim_;
    TransferChar_ = src.TransferChar_;
    ColorMatrix_ = src.ColorMatrix_;

    delete Y_;
    delete U_;
    delete V_;

    if (PixelType_ != PixelType::U && PixelType_ != PixelType::V)
    {
        Y_ = new Plane(*src.Y_);
    }

    if (PixelType_ != PixelType::Y)
    {
        if (PixelType_ != PixelType::V)
        {
            U_ = new Plane(*src.U_);
        }

        if (PixelType_ != PixelType::U)
        {
            V_ = new Plane(*src.V_);
        }
    }

    return *this;
}

bool Frame_YUV::operator==(const Frame_YUV & b) const
{
    if (this == &b)
    {
        return true;
    }

    if (FrameNum_ != b.FrameNum_ || PixelType_ != b.PixelType_ || ChromaPlacement_ != b.ChromaPlacement_ || QuantRange_ != b.QuantRange_ ||
        ColorPrim_ != b.ColorPrim_ || TransferChar_ != b.TransferChar_ || ColorMatrix_ != b.ColorMatrix_)
    {
        return false;
    }

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

    return true;
}


// Functions of class Frame_RGB
Frame_RGB::Frame_RGB(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange, ChromaPlacement ChromaPlacement)
{
    DType Floor, Neutral, Ceil, ValueRange;

    FrameNum_ = FrameNum;
    PixelType_ = PixelType;
    ChromaPlacement_ = ChromaPlacement;
    QuantRange_ = QuantRange;
    ColorPrim_ = ColorPrim_Default(Width, Height, true);
    TransferChar_ = TransferChar_Default(Width, Height, true);
    ColorMatrix_ = ColorMatrix_Default(Width, Height);

    Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, BitDepth, QuantRange_, false);

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
    {
        R_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        R_ = nullptr;
    }

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
    {
        G_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        G_ = nullptr;
    }

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
    {
        B_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        B_ = nullptr;
    }
}

Frame_RGB::Frame_RGB(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange, ChromaPlacement ChromaPlacement, ColorPrim ColorPrim, TransferChar TransferChar, ColorMatrix ColorMatrix)
{
    DType Floor, Neutral, Ceil, ValueRange;

    FrameNum_ = FrameNum;
    PixelType_ = PixelType;
    ChromaPlacement_ = ChromaPlacement;
    QuantRange_ = QuantRange;
    ColorPrim_ = ColorPrim;
    TransferChar_ = TransferChar;
    ColorMatrix_ = ColorMatrix;

    Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, BitDepth, QuantRange_, false);

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
    {
        R_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        R_ = nullptr;
    }

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
    {
        G_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        G_ = nullptr;
    }

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
    {
        B_ = new Plane(Floor, Width, Height, BitDepth, Floor, Neutral, Ceil, ValueRange);
    }
    else
    {
        B_ = nullptr;
    }
}

Frame_RGB::Frame_RGB(const Frame_RGB & src, bool Copy)
{
    DType Floor, Neutral, Ceil, ValueRange;

    FrameNum_ = src.FrameNum_;
    PixelType_ = src.PixelType_;
    ChromaPlacement_ = src.ChromaPlacement_;
    QuantRange_ = src.QuantRange_;
    ColorPrim_ = src.ColorPrim_;
    TransferChar_ = src.TransferChar_;
    ColorMatrix_ = src.ColorMatrix_;

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
    {
        if (Copy)
        {
            R_ = new Plane(*src.R_);
        }
        else
        {
            Quantize_Value(&Floor, &Neutral, &Ceil, &ValueRange, src.R_->BitDepth(), QuantRange_, false);

            R_ = new Plane(Floor, src.R_->Width(), src.R_->Height(), src.R_->BitDepth(), Floor, Neutral, Ceil, ValueRange);
        }
    }
    else
    {
        R_ = nullptr;
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

            G_ = new Plane(Floor, src.G_->Width(), src.G_->Height(), src.G_->BitDepth(), Floor, Neutral, Ceil, ValueRange);
        }
    }
    else
    {
        G_ = nullptr;
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

            B_ = new Plane(Floor, src.B_->Width(), src.B_->Height(), src.B_->BitDepth(), Floor, Neutral, Ceil, ValueRange);
        }
    }
    else
    {
        B_ = nullptr;
    }
}

Frame_RGB::~Frame_RGB()
{
    delete R_;
    delete G_;
    delete B_;
}

Frame_RGB & Frame_RGB::operator=(const Frame_RGB & src)
{
    if (this == &src)
    {
        return *this;
    }

    FrameNum_ = src.FrameNum_;
    PixelType_ = src.PixelType_;
    ChromaPlacement_ = src.ChromaPlacement_;
    QuantRange_ = src.QuantRange_;
    ColorPrim_ = src.ColorPrim_;
    TransferChar_ = src.TransferChar_;
    ColorMatrix_ = src.ColorMatrix_;

    delete R_;
    delete G_;
    delete B_;

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::R)
    {
        R_ = new Plane(*src.R_);
    }
    else
    {
        R_ = nullptr;
    }

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::G)
    {
        G_ = new Plane(*src.G_);
    }
    else
    {
        G_ = nullptr;
    }

    if (PixelType_ == PixelType::RGB || PixelType_ == PixelType::B)
    {
        B_ = new Plane(*src.B_);
    }
    else
    {
        B_ = nullptr;
    }

    return *this;
}

bool Frame_RGB::operator==(const Frame_RGB & b) const
{
    if (this == &b)
    {
        return true;
    }

    if (FrameNum_ != b.FrameNum_ || PixelType_ != b.PixelType_ || ChromaPlacement_ != b.ChromaPlacement_ || QuantRange_ != b.QuantRange_ ||
        ColorPrim_ != b.ColorPrim_ || TransferChar_ != b.TransferChar_ || ColorMatrix_ != b.ColorMatrix_)
    {
        return false;
    }

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

    return true;
}

