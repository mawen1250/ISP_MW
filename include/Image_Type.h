#ifndef IMAGE_STRUCT_H_
#define IMAGE_STRUCT_H_


#include "Type.h"
#include "Type_Conv.h"


typedef sint32 FCType;
typedef sint32 PCType;
typedef uint32 DType;
typedef double FLType;


const DType MaxBitDepth = sizeof(DType) * 8 * 3 / 4;
const PCType HD_Width_U = 2048;
const PCType HD_Height_U = 1536;
const PCType SD_Width_U = 1024;
const PCType SD_Height_U = 576;


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


void Quantize_Value(uint32 * Floor, uint32 * Neutral, uint32 * Ceil, uint32 * ValueRange, uint32 BitDepth, QuantRange QuantRange, bool Chroma);


class Plane;
class Plane_FL;
class Frame_YUV;
class Frame_RGB;


class Plane {
private:
    PCType Width_ = 0;
    PCType Height_ = 0;
    PCType PixelCount_ = 0;
    DType BitDepth_;
    DType Floor_;
    DType Neutral_;
    DType Ceil_;
    DType ValueRange_;
    TransferChar TransferChar_;
    DType * Data_ = nullptr;
public:
    Plane() {} // Default constructor
    Plane(const Plane & src); // Copy constructor
    Plane(const Plane & src, bool InitValue, DType Value = 0);
    Plane(Plane && src); // Move constructor
    Plane(const Plane_FL & src, DType BitDepth = 16); // Convertor/Constructor from Plane_FL
    Plane(const Plane_FL & src, bool InitValue, DType Value = 0, DType BitDepth = 16);
    Plane(const Plane_FL & src, DType BitDepth, DType Floor, DType Neutral, DType Ceil);
    Plane(const Plane_FL & src, bool InitValue, DType Value, DType BitDepth, DType Floor, DType Neutral, DType Ceil);
    explicit Plane(DType Value, PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16, bool InitValue = true); // Convertor/Constructor from DType
    Plane(DType Value, PCType Width, PCType Height, DType BitDepth, DType Floor, DType Neutral, DType Ceil, TransferChar TransferChar, bool InitValue = true);
    ~Plane(); // Destructor

    Plane & operator=(const Plane & src); // Copy assignment operator
    Plane & operator=(Plane && src); // Move assignment operator
    bool operator==(const Plane & b) const;
    bool operator!=(const Plane & b) const { return !(*this == b); }
    DType & operator[](PCType i) { return Data_[i]; }
    DType operator[](PCType i) const { return Data_[i]; }

    PCType Width() const { return Width_; }
    PCType Height() const { return Height_; }
    PCType PixelCount() const { return PixelCount_; }
    DType BitDepth() const { return BitDepth_; }
    DType Floor() const { return Floor_; }
    DType Neutral() const { return Neutral_; }
    DType Ceil() const { return Ceil_; }
    DType ValueRange() const { return ValueRange_; }
    TransferChar GetTransferChar() const { return TransferChar_; }
    DType * Data() { return Data_; }
    const DType * Data() const { return Data_; }
    bool isChroma() const { return Floor_ < Neutral_; }
    bool isPCChroma() const { return (Floor_ + Ceil_ - 1) / 2 == Neutral_ - 1; }

    Plane & Width(PCType Width) { return ReSize(Width, Height_); }
    Plane & Height(PCType Height) { return ReSize(Width_, Height); }
    Plane & ReSize(PCType Width, PCType Height);
    Plane & ReQuantize(DType BitDepth = 16, QuantRange QuantRange = QuantRange::PC, bool scale = true);
    Plane & ReQuantize(DType BitDepth, DType Floor, DType Neutral, DType Ceil, bool scale = true);
    Plane & SetTransferChar(TransferChar TransferChar) { TransferChar_ = TransferChar; return *this; }

    Plane & From(const Plane & src);
    Plane & ConvertFrom(const Plane & src, TransferChar dstTransferChar);
    Plane & ConvertFrom(const Plane & src) { return ConvertFrom(src, TransferChar_); }
    Plane & YFrom(const Frame_RGB & src);
    Plane & YFrom(const Frame_RGB & src, ColorMatrix dstColorMatrix);

    FLType GetFL(DType input) const { return (static_cast<FLType>(input)-Neutral_) / ValueRange_; }
    FLType GetFL_PCChroma(DType input) const { return Clip((static_cast<FLType>(input)-Neutral_) / ValueRange_, -0.5, 0.5); }
    DType GetD(FLType input) const { return static_cast<DType>(input*ValueRange_ + Neutral_ + FLType(0.5)); }
    DType GetD_PCChroma(FLType input) const { return static_cast<DType>(input*ValueRange_ + Neutral_ + FLType(0.49999999)); }

    template <typename T> DType Quantize(T input) const
    {
        T input_up = input + T(0.5);
        return input <= Floor_ ? Floor_ : input_up >= Ceil_ ? Ceil_ : static_cast<DType>(input_up);
    }
};


class Plane_FL {
private:
    PCType Width_ = 0;
    PCType Height_ = 0;
    PCType PixelCount_ = 0;
    FLType Floor_;
    FLType Neutral_;
    FLType Ceil_;
    TransferChar TransferChar_;
    FLType * Data_ = nullptr;
public:
    Plane_FL() {} // Default constructor
    Plane_FL(const Plane_FL & src); // Copy constructor
    Plane_FL(const Plane_FL & src, bool InitValue, FLType Value = 0);
    Plane_FL(Plane_FL && src); // Move constructor
    explicit Plane_FL(const Plane & src); // Convertor/Constructor from Plane
    Plane_FL(const Plane & src, bool InitValue, FLType Value = 0);
    explicit Plane_FL(FLType Value, PCType Width = 1920, PCType Height = 1080, bool InitValue = true); // Convertor/Constructor from FLType
    Plane_FL(FLType Value, PCType Width, PCType Height, FLType Floor, FLType Neutral, FLType Ceil, TransferChar TransferChar, bool InitValue = true);
    ~Plane_FL(); // Destructor

    Plane_FL & operator=(const Plane_FL & src); // Copy assignment operator
    Plane_FL & operator=(Plane_FL && src); // Move assignment operator
    bool operator==(const Plane_FL & b) const;
    bool operator!=(const Plane_FL & b) const { return !(*this == b); }
    FLType & operator[](PCType i) { return Data_[i]; }
    FLType operator[](PCType i) const { return Data_[i]; }

    PCType Width() const { return Width_; }
    PCType Height() const { return Height_; }
    PCType PixelCount() const { return PixelCount_; }
    FLType Floor() const { return Floor_; }
    FLType Neutral() const { return Neutral_; }
    FLType Ceil() const { return Ceil_; }
    TransferChar GetTransferChar() const { return TransferChar_; }
    FLType * Data() { return Data_; }
    const FLType * Data() const { return Data_; }
    bool isChroma() const { return Floor_ < Neutral_; }

    Plane_FL & Width(PCType Width) { return ReSize(Width, Height_); }
    Plane_FL & Height(PCType Height) { return ReSize(Width_, Height); }
    Plane_FL & ReSize(PCType Width, PCType Height);
    Plane_FL & SetTransferChar(TransferChar TransferChar) { TransferChar_ = TransferChar; return *this; }

    Plane_FL & From(const Plane & src);
    const Plane_FL & To(Plane & dst) const;
    //Plane_FL & To(Plane & dst) { return (Plane_FL &)((const Plane_FL *)this)->To(dst); }
    Plane_FL & ConvertFrom(const Plane & src, TransferChar dstTransferChar);
    Plane_FL & ConvertFrom(const Plane & src) { return ConvertFrom(src, TransferChar_); }
    Plane_FL & ConvertFrom(const Plane_FL & src, TransferChar dstTransferChar);
    Plane_FL & ConvertFrom(const Plane_FL & src) { return ConvertFrom(src, TransferChar_); }
    Plane_FL & YFrom(const Frame_RGB & src);
    Plane_FL & YFrom(const Frame_RGB & src, ColorMatrix dstColorMatrix);

    template <typename T> FLType Quantize(T input)
    {
        return input <= Floor_ ? Floor_ : input >= Ceil_ ? Ceil_ : input;
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
    Plane * Y_ = nullptr;
    Plane * U_ = nullptr;
    Plane * V_ = nullptr;
public:
    Frame_YUV() {} // Default constructor
    Frame_YUV(const Frame_YUV & src, bool Copy = true); // Copy constructor
    Frame_YUV(Frame_YUV && src); // Move constructor
    explicit Frame_YUV(FCType FrameNum, PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16,
        PixelType PixelType = PixelType::YUV444, QuantRange QuantRange = QuantRange::TV, ChromaPlacement ChromaPlacement = ChromaPlacement::MPEG2); // Convertor/Constructor from FCType
    Frame_YUV(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange,
        ChromaPlacement ChromaPlacement, ColorPrim ColorPrim, TransferChar TransferChar, ColorMatrix ColorMatrix);
    ~Frame_YUV(); // Destructor

    Frame_YUV & operator=(const Frame_YUV & src); // Copy assignment operator
    Frame_YUV & operator=(Frame_YUV && src); // Move assignment operator
    bool operator==(const Frame_YUV & b) const;
    bool operator!=(const Frame_YUV & b) const { return !(*this == b); }

    FCType FrameNum() const { return FrameNum_; }
    PixelType GetPixelType() const { return PixelType_; }
    ChromaPlacement GetChromaPlacement() const { return ChromaPlacement_; }
    QuantRange GetQuantRange() const { return QuantRange_; }
    ColorPrim GetColorPrim() const { return ColorPrim_; }
    TransferChar GetTransferChar() const { return TransferChar_; }
    ColorMatrix GetColorMatrix() const { return ColorMatrix_; }
    Plane & Y() { return *Y_; }
    Plane & U() { return *U_; }
    Plane & V() { return *V_; }
    const Plane & Y() const { return *Y_; }
    const Plane & U() const { return *U_; }
    const Plane & V() const { return *V_; }

    PCType Width() const { return Y_->Width(); }
    PCType Height() const { return Y_->Height(); }
    PCType PixelCount() const { return Y_->PixelCount(); }

    Frame_YUV & ConvertFrom(const Frame_YUV & src, TransferChar dstTransferChar);
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
    Plane * R_ = nullptr;
    Plane * G_ = nullptr;
    Plane * B_ = nullptr;
public:
    Frame_RGB() {} // Default constructor
    Frame_RGB(const Frame_RGB & src, bool Copy = true); // Copy constructor
    Frame_RGB(Frame_RGB && src); // Move constructor
    explicit Frame_RGB(FCType FrameNum, PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16,
        PixelType PixelType = PixelType::RGB, QuantRange QuantRange = QuantRange::PC, ChromaPlacement ChromaPlacement = ChromaPlacement::MPEG2); // Convertor/Constructor from FCType
    Frame_RGB(FCType FrameNum, PCType Width, PCType Height, DType BitDepth, PixelType PixelType, QuantRange QuantRange,
        ChromaPlacement ChromaPlacement, ColorPrim ColorPrim, TransferChar TransferChar, ColorMatrix ColorMatrix);
    ~Frame_RGB(); // Destructor

    Frame_RGB & operator=(const Frame_RGB & src); // Copy assignment operator
    Frame_RGB & operator=(Frame_RGB && src); // Move assignment operator
    bool operator==(const Frame_RGB & b) const;
    bool operator!=(const Frame_RGB & b) const { return !(*this == b); }

    FCType FrameNum() const { return FrameNum_; }
    PixelType GetPixelType() const { return PixelType_; }
    ChromaPlacement GetChromaPlacement() const { return ChromaPlacement_; }
    QuantRange GetQuantRange() const { return QuantRange_; }
    ColorPrim GetColorPrim() const { return ColorPrim_; }
    TransferChar GetTransferChar() const { return TransferChar_; }
    ColorMatrix GetColorMatrix() const { return ColorMatrix_; }
    Plane & R() { return *R_; }
    Plane & G() { return *G_; }
    Plane & B() { return *B_; }
    const Plane & R() const { return *R_; }
    const Plane & G() const { return *G_; }
    const Plane & B() const { return *B_; }

    PCType Width() const { return R_->Width(); }
    PCType Height() const { return R_->Height(); }
    PCType PixelCount() const { return R_->PixelCount(); }

    Frame_RGB & ConvertFrom(const Frame_RGB & src, TransferChar dstTransferChar);
};


#endif