#ifndef IMAGE_STRUCT_H_
#define IMAGE_STRUCT_H_


#include "Type.h"


typedef sint32 FCType;
typedef sint32 PCType;
typedef uint32 DType;


const int HD_Width_U = 2048;
const int HD_Height_U = 1536;
const int SD_Width_U = 1024;
const int SD_Height_U = 576;


enum class ResLevel {
    SD = 0,
    HD = 1,
    UHD = 2,
    Unknown = 3
};

enum class ColorPrim {
    bt709 = 1,
    Unspecified = 2,
    bt470m = 4,
    bt470bg = 5,
    smpte170m = 6,
    smpte240m = 7,
    film = 8,
    bt2020 = 9
};

enum class TransferChar {
    bt709 = 1,
    Unspecified = 2,
    bt470m = 4,
    bt470bg = 5,
    smpte170m = 6,
    smpte240m = 7,
    linear = 8,
    log100 = 9,
    log316 = 10,
    iec61966_2_4 = 11,
    bt1361e = 12,
    iec61966_2_1 = 13,
    bt2020_10 = 14,
    bt2020_12 = 15
};

enum class ColorMatrix {
    GBR = 0,
    bt709 = 1,
    Unspecified = 2,
    fcc = 4,
    bt470bg = 5,
    smpte170m = 6,
    smpte240m = 7,
    YCgCo = 8,
    bt2020nc = 9,
    bt2020c = 10
};

enum class PixelType {
    Y = 0,
    U = 1,
    V = 2,
    YUV444 = 3,
    YUV422 = 4,
    YUV420 = 5,
    YUV411 = 6,
    R = 7,
    G = 8,
    B = 9,
    RGB = 10
};

enum class ChromaPlacement {
    MPEG1 = 0,
    MPEG2 = 1,
    DV    = 2
};

enum class QuantRange {
    TV = 0,
    PC = 1
};


ResLevel ResLevel_Default(PCType Width, PCType Height);
ColorPrim ColorPrim_Default(PCType Width, PCType Height, bool RGB);
TransferChar TransferChar_Default(PCType Width, PCType Height, bool RGB);
ColorMatrix ColorMatrix_Default(PCType Width, PCType Height);


STAT Quantize_Value(uint32 * Floor, uint32 * Neutral, uint32 * Ceil, uint32 * ValueRange, uint32 BitDepth, QuantRange QuantRange, bool Chroma);


class Plane {
private:
    PCType Width_;
    PCType Height_;
    PCType PixelCount_;
    DType BitDepth_;
    DType Floor_;
    DType Neutral_;
    DType Ceil_;
    DType ValueRange_;
    DType * Data_;

public:
    Plane(DType Value = 0, PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16, bool InitValue = true);
    Plane(DType Value, PCType Width, PCType Height, DType BitDepth, DType Floor, DType Neutral, DType Ceil, DType ValueRange, bool InitValue = true);
    Plane(const Plane & src);
    Plane(const Plane & src, bool InitValue, DType Value = 0);
    ~Plane();

    Plane & operator=(const Plane & src);
    bool operator==(const Plane & b) const;
    bool operator!=(const Plane & b) const { return !(*this == b); }
    DType & operator[](PCType i) { return Data_[i]; }
    const DType & operator[](PCType i) const { return Data_[i]; }

    PCType Width() const { return Width_; };
    PCType Height() const { return Height_; };
    PCType PixelCount() const { return PixelCount_; };
    DType BitDepth() const { return BitDepth_; };
    DType Floor() const { return Floor_; };
    DType Neutral() const { return Neutral_; };
    DType Ceil() const { return Ceil_; };
    DType ValueRange() const { return ValueRange_; };
    DType * Data() { return Data_; };
    DType * const Data() const { return Data_; };

    template <typename T> friend DType Quantize(T input, const Plane & Plane)
    {
        return input <= Plane.Floor_ ? Plane.Floor_ : input >= Plane.Ceil_ ? Plane.Ceil_ : (DType)(input + 0.5);
    }
};


class Frame_YUV {
private:
    FCType FrameNum_;
    PixelType PixelType_;
    ChromaPlacement ChromaPlacement_;
    QuantRange QuantRange_;
    ColorPrim ColorPrim_;
    TransferChar TransferChar_;
    ColorMatrix ColorMatrix_;
    Plane * Y_;
    Plane * U_;
    Plane * V_;

public:
    Frame_YUV(FCType FrameNum = 0, PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16, PixelType PixelType = PixelType::YUV444, QuantRange QuantRange = QuantRange::TV, ChromaPlacement ChromaPlacement = ChromaPlacement::MPEG2);
    Frame_YUV(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange, ChromaPlacement ChromaPlacement, ColorPrim ColorPrim, TransferChar TransferChar, ColorMatrix ColorMatrix);
    Frame_YUV(const Frame_YUV & src, bool Copy = true);
    ~Frame_YUV();

    Frame_YUV & operator=(const Frame_YUV & src);
    bool operator==(const Frame_YUV & b) const;
    bool operator!=(const Frame_YUV & b) const { return !(*this == b); }

    FCType FrameNum() const { return FrameNum_; };
    PixelType GetPixelType() const { return PixelType_; };
    ChromaPlacement GetChromaPlacement() const { return ChromaPlacement_; };
    QuantRange GetQuantRange() const { return QuantRange_; };
    ColorPrim GetColorPrim() const { return ColorPrim_; };
    TransferChar GetTransferChar() const { return TransferChar_; };
    ColorMatrix GetColorMatrix() const { return ColorMatrix_; };
    Plane & Y() { return *Y_; }
    Plane & U() { return *U_; }
    Plane & V() { return *V_; }
    const Plane & Y() const { return *Y_; }
    const Plane & U() const { return *U_; }
    const Plane & V() const { return *V_; }

    PCType Width() const { return Y_->Width(); };
    PCType Height() const { return Y_->Height(); };
    PCType PixelCount() const { return Y_->PixelCount(); };
};


class Frame_RGB {
private:
    FCType FrameNum_;
    PixelType PixelType_;
    ChromaPlacement ChromaPlacement_;
    QuantRange QuantRange_;
    ColorPrim ColorPrim_;
    TransferChar TransferChar_;
    ColorMatrix ColorMatrix_;
    Plane * R_;
    Plane * G_;
    Plane * B_;

public:
    Frame_RGB(FCType FrameNum = 0, PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16, PixelType PixelType = PixelType::RGB, QuantRange QuantRange = QuantRange::PC, ChromaPlacement ChromaPlacement = ChromaPlacement::MPEG2);
    Frame_RGB(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange, ChromaPlacement ChromaPlacement, ColorPrim ColorPrim, TransferChar TransferChar, ColorMatrix ColorMatrix);
    Frame_RGB(const Frame_RGB & src, bool Copy = true);
    ~Frame_RGB();

    Frame_RGB & operator=(const Frame_RGB & src);
    bool operator==(const Frame_RGB & b) const;
    bool operator!=(const Frame_RGB & b) const { return !(*this == b); }

    FCType FrameNum() const { return FrameNum_; };
    PixelType GetPixelType() const { return PixelType_; };
    ChromaPlacement GetChromaPlacement() const { return ChromaPlacement_; };
    QuantRange GetQuantRange() const { return QuantRange_; };
    ColorPrim GetColorPrim() const { return ColorPrim_; };
    TransferChar GetTransferChar() const { return TransferChar_; };
    ColorMatrix GetColorMatrix() const { return ColorMatrix_; };
    Plane & R() { return *R_; }
    Plane & G() { return *G_; }
    Plane & B() { return *B_; }
    const Plane & R() const { return *R_; }
    const Plane & G() const { return *G_; }
    const Plane & B() const { return *B_; }

    PCType Width() const { return R_->Width(); };
    PCType Height() const { return R_->Height(); };
    PCType PixelCount() const { return R_->PixelCount(); };
};


#endif