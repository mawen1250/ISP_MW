#ifndef IMAGE_STRUCT_H_
#define IMAGE_STRUCT_H_


#include <vector>
#include <array>
#include <crtdefs.h>
#include "Type.h"
#include "Helper.h"
#include "Specification.h"


const DType MaxBitDepth = sizeof(DType) * 8 * 3 / 4;


enum class PixelType
{
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

enum class ChromaPlacement
{
    MPEG1 = 0,
    MPEG2 = 1,
    DV    = 2
};

enum class QuantRange
{
    TV = 0,
    PC = 1
};


struct Pos
{
    PCType y = 0;
    PCType x = 0;

    explicit Pos(PCType _y = 0, PCType _x = 0)
        : y(_y), x(_x) {}

    bool operator==(const Pos &right) const
    {
        return y == right.y && x == right.x;
    }

    bool operator<(const Pos &right) const
    {
        if (y < right.y)
        {
            return true;
        }
        else if (y > right.y)
        {
            return false;
        }
        else if (x < right.x)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool operator>(const Pos &right) const
    {
        return !(*this < right);
    }

    friend std::ostream &operator<<(std::ostream &out, const Pos &src)
    {
        out << "(" << src.y << ", " << src.x << ")";

        return out;
    }
};


class Plane;
class Plane_FL;
class Frame;


class Plane
{
    typedef DType _Ty;

public:
    typedef Plane _Myt;
    typedef _Ty value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Ty *pointer;
    typedef const _Ty *const_pointer;
    typedef _Ty &reference;
    typedef const _Ty &const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

public:
    const value_type value_type_MIN = value_type(0);
    const value_type value_type_MAX = value_type_MIN - 1;
    
private:
    PCType Width_ = 0;
    PCType Height_ = 0;
    PCType PixelCount_ = 0;
    value_type BitDepth_;
    value_type Floor_;
    value_type Neutral_;
    value_type Ceil_;
    TransferChar TransferChar_;
    pointer Data_ = nullptr;

protected:
    void DefaultPara(bool Chroma, value_type _BitDepth = 16, QuantRange _QuantRange = QuantRange::PC);
    void CopyParaFrom(const _Myt &src);

public:
    void InitValue(value_type Value, bool Init = true);

    Plane() {} // Default constructor
    explicit Plane(value_type Value, PCType _Width = 1920, PCType _Height = 1080, value_type _BitDepth = 16, bool Init = true); // Convertor/Constructor from value_type
    Plane(value_type Value, PCType _Width, PCType _Height, value_type _BitDepth, value_type _Floor, value_type _Neutral, value_type _Ceil, TransferChar _TransferChar, bool Init = true);

    Plane(const _Myt &src); // Copy constructor
    Plane(const _Myt &src, bool Init, value_type Value = 0);
    Plane(_Myt &&src); // Move constructor
    explicit Plane(const Plane_FL &src, value_type _BitDepth = 16); // Convertor/Constructor from Plane_FL
    Plane(const Plane_FL &src, value_type _BitDepth, value_type _Floor, value_type _Neutral, value_type _Ceil);
    Plane(const Plane_FL &src, bool Init, value_type Value = 0, value_type _BitDepth = 16);
    Plane(const Plane_FL &src, bool Init, value_type Value, value_type _BitDepth, value_type _Floor, value_type _Neutral, value_type _Ceil);

    ~Plane(); // Destructor

    _Myt &operator=(const _Myt &src); // Copy assignment operator
    _Myt &operator=(_Myt &&src); // Move assignment operator
    bool operator==(const _Myt &b) const;
    bool operator!=(const _Myt &b) const { return !(*this == b); }
    reference operator[](PCType i) { return Data_[i]; }
    const_reference operator[](PCType i) const { return Data_[i]; }
    reference operator()(PCType j, PCType i) { return Data_[j * Stride() + i]; }
    const_reference operator()(PCType j, PCType i) const { return Data_[j * Stride() + i]; }

    iterator begin() { return Data_; }
    const_iterator begin() const { return Data_; }
    iterator end() { return Data_ + PixelCount_; }
    const_iterator end() const { return Data_ + PixelCount_; }
    size_type size() const { return PixelCount_; }
    pointer data() { return Data_; }
    const_pointer data() const { return Data_; }
    value_type value(PCType i) { return Data_[i]; }
    const value_type value(PCType i) const { return Data_[i]; }

    PCType Height() const { return Height_; }
    PCType Width() const { return Width_; }
    PCType Stride() const { return Width_; }
    PCType PixelCount() const { return PixelCount_; }
    value_type BitDepth() const { return BitDepth_; }
    value_type Floor() const { return Floor_; }
    value_type Neutral() const { return Neutral_; }
    value_type Ceil() const { return Ceil_; }
    value_type ValueRange() const { return Ceil_ - Floor_; }
    TransferChar GetTransferChar() const { return TransferChar_; }

    pointer Data() { return Data_; }
    const_pointer Data() const { return Data_; }

    bool isChroma() const;
    bool isPCChroma() const;
    value_type Min() const;
    value_type Max() const;
    void MinMax(reference min, reference max) const;
    FLType Mean() const;
    FLType Variance(FLType Mean) const;
    FLType Variance() const { return Variance(Mean()); }
    void ValidRange(reference min, reference max, double lower_thr = 0., double upper_thr = 0., int HistBins = 1024, bool protect = false) const;

    _Myt &Width(PCType _Width) { return ReSize(_Width, Height()); }
    _Myt &Height(PCType _Height) { return ReSize(Width(), _Height); }
    _Myt &ReSize(PCType _Width, PCType _Height);
    void ReSetChroma(bool Chroma = false);
    _Myt &ReQuantize(value_type _BitDepth = 16, QuantRange _QuantRange = QuantRange::PC, bool scale = true, bool clip = false);
    _Myt &ReQuantize(value_type _BitDepth, value_type _Floor, value_type _Neutral, value_type _Ceil, bool scale = true, bool clip = false);
    _Myt &SetTransferChar(TransferChar _TransferChar) { TransferChar_ = _TransferChar; return *this; }

    FLType GetFL(value_type input) const { return static_cast<FLType>(input - Neutral()) / ValueRange(); }
    FLType GetFL_PCChroma(value_type input) const { return Clip(static_cast<FLType>(input - Neutral()) / ValueRange(), -0.5, 0.5); }
    value_type GetD(FLType input) const { return static_cast<value_type>(input * ValueRange() + Neutral_ + FLType(0.5)); }
    value_type GetD_PCChroma(FLType input) const { return static_cast<value_type>(input * ValueRange() + Neutral_ + FLType(0.499999)); }

    _Myt &Binarize(const _Myt &src, value_type lower_thrD, value_type upper_thrD);
    _Myt &Binarize(value_type lower_thrD, value_type upper_thrD) { return Binarize(*this, lower_thrD, upper_thrD); }
    _Myt &Binarize_ratio(const _Myt &src, double lower_thr = 0., double upper_thr = 1.);
    _Myt &Binarize_ratio(double lower_thr = 0., double upper_thr = 1.) { return Binarize_ratio(*this, lower_thr, upper_thr); }

    template < typename T > value_type Quantize(T input) const;

    template < typename _Fn1 > void for_each(_Fn1 _Func) const;
    template < typename _Fn1 > void for_each(_Fn1 _Func);
    template < typename _Fn1 > void transform(_Fn1 _Func);
    template < typename _St1, typename _Fn1 > void transform(const _St1 &src, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _Fn1 > void transform(const _St1 &src1, const _St2 &src2, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _St3, typename _Fn1 > void transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > void transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func);
    template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > void convolute(const _St1 &src, _Fn1 _Func);
};


class Plane_FL
{
    typedef FLType _Ty;

public:
    typedef Plane_FL _Myt;
    typedef _Ty value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Ty *pointer;
    typedef const _Ty *const_pointer;
    typedef _Ty &reference;
    typedef const _Ty &const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

public:
    const value_type value_type_MIN = sizeof(value_type) < 8 ? FLT_MIN : DBL_MIN;
    const value_type value_type_MAX = sizeof(value_type) < 8 ? FLT_MAX : DBL_MAX;

private:
    PCType Width_ = 0;
    PCType Height_ = 0;
    PCType PixelCount_ = 0;
    value_type Floor_ = 0;
    value_type Neutral_ = 0;
    value_type Ceil_ = 1;
    TransferChar TransferChar_;
    pointer Data_ = nullptr;

protected:
    void DefaultPara(bool Chroma, value_type range = 1);
    void CopyParaFrom(const _Myt &src);

public:
    void InitValue(value_type Value, bool Init = true);

    Plane_FL() {} // Default constructor
    explicit Plane_FL(value_type Value, PCType _Width = 1920, PCType _Height = 1080, bool RGB = true, bool Chroma = false, bool Init = true); // Convertor/Constructor from value_type
    Plane_FL(value_type Value, PCType _Width, PCType _Height, value_type _Floor, value_type _Neutral, value_type _Ceil, TransferChar _TransferChar, bool Init = true);

    Plane_FL(const _Myt &src); // Copy constructor
    Plane_FL(const _Myt &src, bool Init, value_type Value = 0);
    Plane_FL(_Myt &&src); // Move constructor
    explicit Plane_FL(const Plane &src, value_type range = 1.); // Convertor/Constructor from Plane
    Plane_FL(const Plane &src, bool Init, value_type Value = 0, value_type range = 1.);
    template < typename _St1 > Plane_FL(const _St1 &src, TransferChar dstTransferChar);

    ~Plane_FL(); // Destructor

    _Myt &operator=(const _Myt &src); // Copy assignment operator
    _Myt &operator=(_Myt &&src); // Move assignment operator
    bool operator==(const _Myt &b) const;
    bool operator!=(const _Myt &b) const { return !(*this == b); }
    reference operator[](PCType i) { return Data_[i]; }
    const_reference operator[](PCType i) const { return Data_[i]; }
    reference operator()(PCType j, PCType i) { return Data_[j * Stride() + i]; }
    const_reference operator()(PCType j, PCType i) const { return Data_[j * Stride() + i]; }

    iterator begin() { return Data_; }
    const_iterator begin() const { return Data_; }
    iterator end() { return Data_ + PixelCount_; }
    const_iterator end() const { return Data_ + PixelCount_; }
    size_type size() const { return PixelCount_; }
    pointer data() { return Data_; }
    const_pointer data() const { return Data_; }
    value_type value(PCType i) { return Data_[i]; }
    const value_type value(PCType i) const { return Data_[i]; }

    PCType Height() const { return Height_; }
    PCType Width() const { return Width_; }
    PCType Stride() const { return Width_; }
    PCType PixelCount() const { return PixelCount_; }
    value_type Floor() const { return Floor_; }
    value_type Neutral() const { return Neutral_; }
    value_type Ceil() const { return Ceil_; }
    value_type ValueRange() const { return Ceil_ - Floor_; }
    TransferChar GetTransferChar() const { return TransferChar_; }

    pointer Data() { return Data_; }
    const_pointer Data() const { return Data_; }

    bool isChroma() const;
    bool isPCChroma() const;
    value_type Min() const;
    value_type Max() const;
    void MinMax(reference min, reference max) const;
    value_type Mean() const;
    value_type Variance(value_type Mean) const;
    value_type Variance() const { return Variance(Mean()); }
    void ValidRange(reference min, reference max, double lower_thr = 0., double upper_thr = 0., int HistBins = 1024, bool protect = false) const;

    _Myt &Width(PCType _Width) { return ReSize(_Width, Height()); }
    _Myt &Height(PCType _Height) { return ReSize(Width(), _Height); }
    _Myt &ReSize(PCType _Width, PCType _Height);
    void ReSetChroma(bool Chroma = false);
    _Myt &ReQuantize(value_type _Floor, value_type _Neutral, value_type _Ceil, bool scale = true, bool clip = false);
    _Myt &SetTransferChar(TransferChar _TransferChar) { TransferChar_ = _TransferChar; return *this; }

    _Myt &Binarize(const _Myt &src, value_type lower_thrD, value_type upper_thrD);
    _Myt &Binarize(value_type lower_thrD, value_type upper_thrD) { return Binarize(*this, lower_thrD, upper_thrD); }
    _Myt &Binarize_ratio(const _Myt &src, double lower_thr = 0., double upper_thr = 1.);
    _Myt &Binarize_ratio(double lower_thr = 0., double upper_thr = 1.) { return Binarize_ratio(*this, lower_thr, upper_thr); }

    template < typename T > value_type Quantize(T input) const;

    template < typename _Fn1 > void for_each(_Fn1 _Func) const;
    template < typename _Fn1 > void for_each(_Fn1 _Func);
    template < typename _Fn1 > void transform(_Fn1 _Func);
    template < typename _St1, typename _Fn1 > void transform(const _St1 &src, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _Fn1 > void transform(const _St1 &src1, const _St2 &src2, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _St3, typename _Fn1 > void transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func);
    template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > void transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func);
    template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > void convolute(const _St1 &src, _Fn1 _Func);
};


class Frame
{
    typedef DType _Ty;

public:
    typedef Frame _Myt;
    typedef Plane _Mysub;
    typedef _Ty value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Ty *pointer;
    typedef const _Ty *const_pointer;
    typedef _Ty &reference;
    typedef const _Ty &const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

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
    std::vector<_Mysub *> P_;

    _Mysub *R_ = nullptr;
    _Mysub *G_ = nullptr;
    _Mysub *B_ = nullptr;
    _Mysub *Y_ = nullptr;
    _Mysub *U_ = nullptr;
    _Mysub *V_ = nullptr;
    _Mysub *A_ = nullptr;

protected:
    bool isYUV(PixelType _PixelType) const { return _PixelType >= PixelType::Y && _PixelType < PixelType::R; }
    bool isRGB(PixelType _PixelType) const { return _PixelType >= PixelType::R && _PixelType <= PixelType::RGB; }

    void InitPlanes(PCType _Width = 1920, PCType _Height = 1080, value_type _BitDepth = 16, bool Init = true);
    void CopyPlanes(const _Myt &src, bool Copy = true, bool Init = false);
    void MovePlanes(_Myt &src);
    void FreePlanes();

public:
    Frame() {} // Default constructor
    explicit Frame(FCType _FrameNum, PixelType _PixelType = PixelType::RGB, PCType _Width = 1920, PCType _Height = 1080,
        value_type _BitDepth = 16, bool Init = true); // Convertor/Constructor from FCType
    Frame(FCType _FrameNum, PixelType _PixelType, PCType _Width, PCType _Height, value_type _BitDepth,
        QuantRange _QuantRange, ChromaPlacement _ChromaPlacement = ChromaPlacement::MPEG2, bool Init = true);
    Frame(FCType _FrameNum, PixelType _PixelType, PCType _Width, PCType _Height, value_type _BitDepth, QuantRange _QuantRange,
        ChromaPlacement _ChromaPlacement, ColorPrim _ColorPrim, TransferChar _TransferChar, ColorMatrix _ColorMatrix, bool Init = true);

    Frame(const _Myt &src, bool Copy = true, bool Init = false); // Copy constructor
    Frame(_Myt &&src); // Move constructor

    ~Frame(); // Destructor

    _Myt &operator=(const _Myt &src); // Copy assignment operator
    _Myt &operator=(_Myt &&src); // Move assignment operator
    bool operator==(const _Myt &b) const;
    bool operator!=(const _Myt &b) const { return !(*this == b); }

    FCType FrameNum() const { return FrameNum_; }
    PixelType GetPixelType() const { return PixelType_; }
    ChromaPlacement GetChromaPlacement() const { return ChromaPlacement_; }
    QuantRange GetQuantRange() const { return QuantRange_; }
    ColorPrim GetColorPrim() const { return ColorPrim_; }
    TransferChar GetTransferChar() const { return TransferChar_; }
    ColorMatrix GetColorMatrix() const { return ColorMatrix_; }
    PlaneCountType PlaneCount() const { return PlaneCount_; }

    _Mysub &P(PlaneCountType PlaneNum) { return *P_[PlaneNum]; }
    _Mysub &R() { return *R_; }
    _Mysub &G() { return *G_; }
    _Mysub &B() { return *B_; }
    _Mysub &Y() { return *Y_; }
    _Mysub &U() { return *U_; }
    _Mysub &V() { return *V_; }
    _Mysub &A() { return *A_; }
    const _Mysub &P(PlaneCountType PlaneNum) const { return *P_[PlaneNum]; }
    const _Mysub &R() const { return *R_; }
    const _Mysub &G() const { return *G_; }
    const _Mysub &B() const { return *B_; }
    const _Mysub &Y() const { return *Y_; }
    const _Mysub &U() const { return *U_; }
    const _Mysub &V() const { return *V_; }
    const _Mysub &A() const { return *A_; }

    bool isYUV() const { return PixelType_ >= PixelType::Y && PixelType_ < PixelType::R; }
    bool isRGB() const { return PixelType_ >= PixelType::R && PixelType_ <= PixelType::RGB; }

    PCType Height() const { return P_[0]->Height(); }
    PCType Width() const { return P_[0]->Width(); }
    PCType Stride() const { return P_[0]->Stride(); }
    PCType PixelCount() const { return P_[0]->PixelCount(); }
    value_type BitDepth() const { return P_[0]->BitDepth(); }

    _Myt &SetTransferChar(TransferChar _TransferChar) { TransferChar_ = _TransferChar; return *this; }
};


#include "Image_Type.hpp"


// Inline functions for class Plane
inline bool Plane::isChroma() const { return ::isChroma(Floor(), Neutral()); }
inline bool Plane::isPCChroma() const { return ::isPCChroma(Floor(), Neutral(), Ceil()); }
inline void Plane::ValidRange(reference min, reference max, double lower_thr, double upper_thr, int HistBins, bool protect) const
{
    ::ValidRange(*this, min, max, lower_thr, upper_thr, HistBins, protect);
}
inline void Plane::ReSetChroma(bool Chroma) { ::ReSetChroma(Floor_, Neutral_, Ceil_, Chroma); }


// Inline functions for class Plane_FL
inline bool Plane_FL::isChroma() const { return ::isChroma(Floor(), Neutral()); }
inline bool Plane_FL::isPCChroma() const { return ::isPCChroma(Floor(), Neutral(), Ceil()); }
inline void Plane_FL::ValidRange(reference min, reference max, double lower_thr, double upper_thr, int HistBins, bool protect) const
{
    ::ValidRange(*this, min, max, lower_thr, upper_thr, HistBins, protect);
}
inline void Plane_FL::ReSetChroma(bool Chroma) { ::ReSetChroma(Floor_, Neutral_, Ceil_, Chroma); }


// Template functions for class Plane
template < typename T > inline
Plane::value_type Plane::Quantize(T input) const
{
    T input_up = input + T(0.5);
    return input <= Floor_ ? Floor_ : input_up >= Ceil_ ? Ceil_ : static_cast<Plane::value_type>(input_up);
}


template < typename _Fn1 > inline
void Plane::for_each(_Fn1 _Func) const
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::for_each(_Fn1 _Func)
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane::transform(_Fn1 _Func)
{
    TRANSFORM(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane::transform(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane::transform(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane::convolute(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE<VRad, HRad>(*this, src, _Func);
}


// Template functions for class Plane_FL
template < typename _St1 > inline
Plane_FL::Plane_FL(const _St1 &src, TransferChar dstTransferChar)
    : _Myt(src, false)
{
    TransferConvert(*this, src, dstTransferChar);
}


template < typename T > inline
Plane_FL::value_type Plane_FL::Quantize(T input) const
{
    return input <= Floor_ ? Floor_ : input >= Ceil_ ? Ceil_ : input;
}


template < typename _Fn1 > inline
void Plane_FL::for_each(_Fn1 _Func) const
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::for_each(_Fn1 _Func)
{
    FOR_EACH(*this, _Func);
}

template < typename _Fn1 > inline
void Plane_FL::transform(_Fn1 _Func)
{
    TRANSFORM(*this, _Func);
}

template < typename _St1, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src, _Fn1 _Func)
{
    TRANSFORM(*this, src, _Func);
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src1, const _St2 &src2, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, _Func);
}

template < typename _St1, typename _St2, typename _St3, typename _St4, typename _Fn1 > inline
void Plane_FL::transform(const _St1 &src1, const _St2 &src2, const _St3 &src3, const _St4 &src4, _Fn1 _Func)
{
    TRANSFORM(*this, src1, src2, src3, src4, _Func);
}

template < PCType VRad, PCType HRad, typename _St1, typename _Fn1 > inline
void Plane_FL::convolute(const _St1 &src, _Fn1 _Func)
{
    CONVOLUTE<VRad, HRad>(*this, src, _Func);
}


#endif
