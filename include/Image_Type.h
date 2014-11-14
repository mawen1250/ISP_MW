#ifndef IMAGE_STRUCT_H_
#define IMAGE_STRUCT_H_


#include <vector>
#include "Type.h"
#include "Helper.h"
#include "Specification.h"


const DType MaxBitDepth = sizeof(DType) * 8 * 3 / 4;


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


void Quantize_Value(uint32 * Floor, uint32 * Neutral, uint32 * Ceil, uint32 * ValueRange, uint32 BitDepth, QuantRange _QuantRange, bool Chroma);


class Plane;
class Plane_FL;
class Frame;


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

protected:
    void DefaultPara(bool Chroma, DType BitDepth = 16, QuantRange _QuantRange = QuantRange::PC);
    void CopyParaFrom(const Plane & src);
    void InitValue(DType Value, bool Init = true);

public:
    explicit Plane(DType Value = 0, PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16, bool Init = true); // Default constructor and Convertor/Constructor from DType
    Plane(DType Value, PCType Width, PCType Height, DType BitDepth, DType Floor, DType Neutral, DType Ceil, TransferChar _TransferChar, bool Init = true);

    Plane(const Plane & src); // Copy constructor
    Plane(const Plane & src, bool Init, DType Value = 0);
    Plane(Plane && src); // Move constructor
    explicit Plane(const Plane_FL & src, DType BitDepth = 16); // Convertor/Constructor from Plane_FL
    Plane(const Plane_FL & src, DType BitDepth, DType Floor, DType Neutral, DType Ceil);
    Plane(const Plane_FL & src, bool Init, DType Value = 0, DType BitDepth = 16);
    Plane(const Plane_FL & src, bool Init, DType Value, DType BitDepth, DType Floor, DType Neutral, DType Ceil);

    ~Plane(); // Destructor

    Plane & operator=(const Plane & src); // Copy assignment operator
    Plane & operator=(Plane && src); // Move assignment operator
    bool operator==(const Plane & b) const;
    bool operator!=(const Plane & b) const { return !(*this == b); }
    DType & operator[](PCType i) { return Data_[i]; }
    DType operator[](PCType i) const { return Data_[i]; }

    PCType Height() const { return Height_; }
    PCType Width() const { return Width_; }
    PCType Stride() const { return Width_; }
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
    bool isPCChroma() const { return Floor_ < Neutral_ && (Floor_ + Ceil_) % 2 == 1; }
    DType Min() const;
    DType Max() const;
    const Plane & MinMax(DType & min, DType & max) const;
    FLType Mean() const;
    FLType Variance(FLType Mean) const;
    FLType Variance() const { return Variance(Mean()); }

    Plane & Width(PCType Width) { return ReSize(Width, Height_); }
    Plane & Height(PCType Height) { return ReSize(Width_, Height); }
    Plane & ReSize(PCType Width, PCType Height);
    Plane & ReQuantize(DType BitDepth = 16, QuantRange _QuantRange = QuantRange::PC, bool scale = true, bool clip = false);
    Plane & ReQuantize(DType BitDepth, DType Floor, DType Neutral, DType Ceil, bool scale = true, bool clip = false);
    Plane & SetTransferChar(TransferChar _TransferChar) { TransferChar_ = _TransferChar; return *this; }

    Plane & From(const Plane & src);
    Plane & From(const Plane_FL & src);
    Plane & ConvertFrom(const Plane & src, TransferChar dstTransferChar);
    Plane & ConvertFrom(const Plane & src) { return ConvertFrom(src, TransferChar_); }
    Plane & YFrom(const Frame & src);
    Plane & YFrom(const Frame & src, ColorMatrix dstColorMatrix);

    FLType GetFL(DType input) const { return static_cast<FLType>(input - Neutral_) / ValueRange_; }
    FLType GetFL_PCChroma(DType input) const { return Clip(static_cast<FLType>(input - Neutral_) / ValueRange_, -0.5, 0.5); }
    DType GetD(FLType input) const { return static_cast<DType>(input*ValueRange_ + Neutral_ + FLType(0.5)); }
    DType GetD_PCChroma(FLType input) const { return static_cast<DType>(input*ValueRange_ + Neutral_ + FLType(0.499999)); }

    Plane & Binarize(const Plane &src, DType lower_thrD, DType upper_thrD);
    Plane & Binarize(DType lower_thrD, DType upper_thrD) { return Binarize(*this, lower_thrD, upper_thrD); }
    Plane & Binarize_ratio(const Plane &src, double lower_thr = 0, double upper_thr = 1);
    Plane & Binarize_ratio(double lower_thr = 0, double upper_thr = 1) { return Binarize_ratio(*this, lower_thr, upper_thr); }
    Plane & SimplestColorBalance(Plane_FL & flt, const Plane & src, double lower_thr = 0, double upper_thr = 0, int HistBins = 4096);

    template < typename T > DType Quantize(T input) const;

    template < typename _Fn1 > void for_each(_Fn1 _Func) const;
    template < typename _Fn1 > void transform(_Fn1 _Func);
    template < typename _St1, typename _Fn1 > void transform(const _St1& src, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _Fn1 > void transform(const _St1& src1, const _St2& src2, _Fn1 _Func);
    template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > void convolute(const _St1& src, _Fn1 _Func);

    template < typename _Fn1 > void for_each_PPL(_Fn1 _Func) const;
    template < typename _Fn1 > void transform_PPL(_Fn1 _Func);
    template < typename _St1, typename _Fn1 > void transform_PPL(const _St1& src, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _Fn1 > void transform_PPL(const _St1& src1, const _St2& src2, _Fn1 _Func);
    template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > void convolute_PPL(const _St1& src, _Fn1 _Func);
};


class Plane_FL {
private:
    PCType Width_ = 0;
    PCType Height_ = 0;
    PCType PixelCount_ = 0;
    FLType Floor_ = 0;
    FLType Neutral_ = 0;
    FLType Ceil_ = 1;
    TransferChar TransferChar_;
    FLType * Data_ = nullptr;

protected:
    void DefaultPara(bool Chroma, FLType range = 1);
    void CopyParaFrom(const Plane_FL & src);
    void InitValue(FLType Value, bool Init = true);

public:
    explicit Plane_FL(FLType Value = 0, PCType Width = 1920, PCType Height = 1080, bool RGB = true, bool Chroma = false, bool Init = true); // Default constructor and Convertor/Constructor from FLType
    Plane_FL(FLType Value, PCType Width, PCType Height, FLType Floor, FLType Neutral, FLType Ceil, TransferChar _TransferChar, bool Init = true);

    Plane_FL(const Plane_FL & src); // Copy constructor
    Plane_FL(const Plane_FL & src, bool Init, FLType Value = 0);
    Plane_FL(Plane_FL && src); // Move constructor
    explicit Plane_FL(const Plane & src, FLType range = 1.); // Convertor/Constructor from Plane
    Plane_FL(const Plane & src, bool Init, FLType Value = 0, FLType range = 1.);
    Plane_FL(const Plane_FL & src, TransferChar dstTransferChar);
    Plane_FL(const Plane & src, TransferChar dstTransferChar);

    ~Plane_FL(); // Destructor

    Plane_FL & operator=(const Plane_FL & src); // Copy assignment operator
    Plane_FL & operator=(Plane_FL && src); // Move assignment operator
    bool operator==(const Plane_FL & b) const;
    bool operator!=(const Plane_FL & b) const { return !(*this == b); }
    FLType & operator[](PCType i) { return Data_[i]; }
    FLType operator[](PCType i) const { return Data_[i]; }

    PCType Height() const { return Height_; }
    PCType Width() const { return Width_; }
    PCType Stride() const { return Width_; }
    PCType PixelCount() const { return PixelCount_; }
    FLType Floor() const { return Floor_; }
    FLType Neutral() const { return Neutral_; }
    FLType Ceil() const { return Ceil_; }
    FLType ValueRange() const { return Ceil_ - Floor_; }
    TransferChar GetTransferChar() const { return TransferChar_; }

    FLType * Data() { return Data_; }
    const FLType * Data() const { return Data_; }

    bool isChroma() const { return Floor_ < Neutral_; }
    FLType Min() const;
    FLType Max() const;
    const Plane_FL & MinMax(FLType & min, FLType & max) const;
    FLType Mean() const;
    FLType Variance(FLType Mean) const;
    FLType Variance() const { return Variance(Mean()); }

    Plane_FL & Width(PCType Width) { return ReSize(Width, Height_); }
    Plane_FL & Height(PCType Height) { return ReSize(Width_, Height); }
    Plane_FL & ReSize(PCType Width, PCType Height);
    Plane_FL & ReQuantize(FLType Floor, FLType Neutral, FLType Ceil, bool scale = true, bool clip = false);
    Plane_FL & SetTransferChar(TransferChar _TransferChar) { TransferChar_ = _TransferChar; return *this; }

    Plane_FL & From(const Plane & src);
    const Plane_FL & To(Plane & dst) const;
    //Plane_FL & To(Plane & dst) { return const_cast<Plane_FL &>(const_cast<const Plane_FL *>(this)->To(dst)); }
    Plane_FL & ConvertFrom(const Plane_FL & src, TransferChar dstTransferChar);
    Plane_FL & ConvertFrom(const Plane_FL & src) { return ConvertFrom(src, TransferChar_); }
    Plane_FL & ConvertFrom(const Plane & src, TransferChar dstTransferChar);
    Plane_FL & ConvertFrom(const Plane & src) { return ConvertFrom(src, TransferChar_); }
    Plane_FL & YFrom(const Frame & src);
    Plane_FL & YFrom(const Frame & src, ColorMatrix dstColorMatrix);

    Plane_FL & Binarize(const Plane_FL &src, FLType lower_thrD, FLType upper_thrD);
    Plane_FL & Binarize(FLType lower_thrD, FLType upper_thrD) { return Binarize(*this, lower_thrD, upper_thrD); }
    Plane_FL & Binarize_ratio(const Plane_FL &src, double lower_thr = 0, double upper_thr = 1);
    Plane_FL & Binarize_ratio(double lower_thr = 0, double upper_thr = 1) { return Binarize_ratio(*this, lower_thr, upper_thr); }
    Plane_FL & SimplestColorBalance(const Plane_FL & flt, const Plane_FL & src, double lower_thr = 0, double upper_thr = 0, int HistBins = 4096);

    template < typename T > FLType Quantize(T input) const;
};


class Frame {
public:
    typedef sint32 PlaneCountType;
    static const PlaneCountType MaxPlaneCount = 7;

private:
    FCType FrameNum_;
    PixelType PixelType_;
    QuantRange QuantRange_;
    ChromaPlacement ChromaPlacement_;
    ColorPrim ColorPrim_;
    TransferChar TransferChar_;
    ColorMatrix ColorMatrix_;

    PlaneCountType PlaneCount_ = 0;
    std::vector<Plane *> P_;

    Plane * R_ = nullptr;
    Plane * G_ = nullptr;
    Plane * B_ = nullptr;
    Plane * Y_ = nullptr;
    Plane * U_ = nullptr;
    Plane * V_ = nullptr;
    Plane * A_ = nullptr;

protected:
    bool isYUV(PixelType _PixelType) const { return _PixelType >= PixelType::Y && _PixelType < PixelType::R; }
    bool isRGB(PixelType _PixelType) const { return _PixelType >= PixelType::R && _PixelType <= PixelType::RGB; }

    void InitPlanes(PCType Width = 1920, PCType Height = 1080, DType BitDepth = 16, bool Init = true);
    void CopyPlanes(const Frame & src, bool Copy = true, bool Init = false);
    void MovePlanes(Frame & src);
    void FreePlanes();

public:
    explicit Frame(FCType FrameNum = 0, PixelType _PixelType = PixelType::RGB, PCType Width = 1920, PCType Height = 1080,
        DType BitDepth = 16, bool Init = true); // Default constructor and Convertor/Constructor from FCType
    Frame(FCType FrameNum, PixelType _PixelType, PCType Width, PCType Height, DType BitDepth,
        QuantRange _QuantRange, ChromaPlacement _ChromaPlacement = ChromaPlacement::MPEG2, bool Init = true);
    Frame(FCType FrameNum, PixelType _PixelType, PCType Width, PCType Height, DType BitDepth, QuantRange _QuantRange,
        ChromaPlacement _ChromaPlacement, ColorPrim _ColorPrim, TransferChar _TransferChar, ColorMatrix _ColorMatrix, bool Init = true);

    Frame(const Frame & src, bool Copy = true, bool Init = false); // Copy constructor
    Frame(Frame && src); // Move constructor

    ~Frame(); // Destructor

    Frame & operator=(const Frame & src); // Copy assignment operator
    Frame & operator=(Frame && src); // Move assignment operator
    bool operator==(const Frame & b) const;
    bool operator!=(const Frame & b) const { return !(*this == b); }

    FCType FrameNum() const { return FrameNum_; }
    PixelType GetPixelType() const { return PixelType_; }
    ChromaPlacement GetChromaPlacement() const { return ChromaPlacement_; }
    QuantRange GetQuantRange() const { return QuantRange_; }
    ColorPrim GetColorPrim() const { return ColorPrim_; }
    TransferChar GetTransferChar() const { return TransferChar_; }
    ColorMatrix GetColorMatrix() const { return ColorMatrix_; }
    PlaneCountType PlaneCount() const { return PlaneCount_; }

    Plane & P(PlaneCountType PlaneNum) { return *P_[PlaneNum]; }
    Plane & R() { return *R_; }
    Plane & G() { return *G_; }
    Plane & B() { return *B_; }
    Plane & Y() { return *Y_; }
    Plane & U() { return *U_; }
    Plane & V() { return *V_; }
    Plane & A() { return *A_; }
    const Plane & P(PlaneCountType PlaneNum) const { return *P_[PlaneNum]; }
    const Plane & R() const { return *R_; }
    const Plane & G() const { return *G_; }
    const Plane & B() const { return *B_; }
    const Plane & Y() const { return *Y_; }
    const Plane & U() const { return *U_; }
    const Plane & V() const { return *V_; }
    const Plane & A() const { return *A_; }

    bool isYUV() const { return PixelType_ >= PixelType::Y && PixelType_ < PixelType::R; }
    bool isRGB() const { return PixelType_ >= PixelType::R && PixelType_ <= PixelType::RGB; }

    PCType Height() const { return P_[0]->Height(); }
    PCType Width() const { return P_[0]->Width(); }
    PCType Stride() const { return P_[0]->Stride(); }
    PCType PixelCount() const { return P_[0]->PixelCount(); }
    DType BitDepth() const { return P_[0]->BitDepth(); }

    Frame & ConvertFrom(const Frame & src, TransferChar dstTransferChar);
};


#include "Image_Type.hpp"


#endif