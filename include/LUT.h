#ifndef LUT_H_
#define LUT_H_


#include "Image_Type.h"


template < typename T = DType >
class LUT {
public:
    typedef uint32 LevelType;
private:
    LevelType Levels_ = 0;
    T * Table_ = nullptr;
public:
    LUT() {} // Default constructor
    LUT(const LUT & src); // Copy constructor
    LUT(LUT && src); // Move constructor
    explicit LUT(LevelType Levels); // Convertor/Constructor from LevelType
    explicit LUT(const Plane & src); // Convertor/Constructor from Plane
    ~LUT(); // Destructor

    LUT & operator=(const LUT & src); // Copy assignment operator
    LUT & operator=(LUT && src); // Move assignment operator
    T & operator[](LevelType i) { return Table_[i]; }
    const T & operator[](LevelType i) const { return Table_[i]; }

    LevelType Levels() const { return Levels_; }
    T * Table() { return Table_; }
    const T * Table() const { return Table_; }

    LUT & Set(const Plane & src, LevelType i, T o);
    LUT & SetRange(const Plane & src, T o = 0) { return SetRange(src, o, src.Floor(), src.Ceil()); }
    LUT & SetRange(const Plane & src, T o, LevelType start, LevelType end);

    T Lookup(const Plane & src, LevelType Value) const { return Table_[Value - src.Floor()]; }
    const LUT & Lookup(Plane & dst, const Plane & src) const;
    //LUT & Lookup(Plane & dst, const Plane & src) { return const_cast<LUT &>(const_cast<const LUT *>(this)->Lookup(dst, src)); }
    const LUT & Lookup(Plane_FL & dst, const Plane & src) const;
    //LUT & Lookup(Plane_FL & dst, const Plane & src) { return const_cast<LUT &>(const_cast<const LUT *>(this)->Lookup(dst, src)); }
    const LUT & Lookup(Plane_FL & dst, const Plane_FL & src) const;

    const LUT & Lookup_Gain(Plane & dst, const Plane & src, const Plane & ref) const;
    //LUT & Lookup_Gain(Plane & dst, const Plane & src, const Plane & ref) { return const_cast<LUT_FL &>(const_cast<const LUT_FL *>(this)->Lookup(dst, src, ref)); }
    const LUT & Lookup_Gain(Plane_FL & dst, const Plane_FL & src, const Plane & ref) const;
    //LUT & Lookup_Gain(Plane_FL & dst, const Plane_FL & src, const Plane & ref) { return const_cast<LUT_FL &>(const_cast<const LUT_FL *>(this)->Lookup(dst, src, ref)); }
    const LUT & Lookup_Gain(Frame & dst, const Frame & src, const Plane & ref) const;
};


// Template functions of template class LUT
template < typename T >
LUT<T>::LUT(const LUT<T> & src)
    : Levels_(src.Levels_)
{
    Table_ = new T[Levels_];

    memcpy(Table_, src.Table_, sizeof(T)*Levels_);
}

template < typename T >
LUT<T>::LUT(LUT<T> && src)
    : Levels_(src.Levels_)
{
    Table_ = src.Table_;

    src.Levels_ = 0;
    src.Table_ = nullptr;
}

template < typename T >
LUT<T>::LUT(LevelType Levels)
    : Levels_(Levels)
{
    Table_ = new T[Levels_];
}

template < typename T >
LUT<T>::LUT(const Plane & src)
    : Levels_(src.ValueRange() + 1)
{
    Table_ = new T[Levels_];
}

template < typename T >
LUT<T>::~LUT()
{
    delete[] Table_;
}

template < typename T >
LUT<T> & LUT<T>::operator=(const LUT<T> & src)
{
    if (this == &src)
    {
        return *this;
    }

    Levels_ = src.Levels_;

    delete[] Table_;
    Table_ = new T[Levels_];

    memcpy(Table_, src.Table_, sizeof(T)*Levels_);

    return *this;
}

template < typename T >
LUT<T> & LUT<T>::operator=(LUT<T> && src)
{
    if (this == &src)
    {
        return *this;
    }

    Levels_ = src.Levels_;

    delete[] Table_;
    Table_ = src.Table_;

    src.Levels_ = 0;
    src.Table_ = nullptr;

    return *this;
}


template < typename T >
LUT<T> & LUT<T>::Set(const Plane & src, LevelType i, T o)
{
    if (i >= src.Floor() && i <= src.Ceil()) Table_[i - src.Floor()] = o;
    return *this;
}

template < typename T >
LUT<T> & LUT<T>::SetRange(const Plane & src, T o, LevelType start, LevelType end)
{
    sint64 length = static_cast<sint64>(end) - static_cast<sint64>(start) + 1;
    start = Max(LevelType(0), start - src.Floor());
    end = start + static_cast<LevelType>(length) - 1;

    if (length >= 1 && end < Levels_)
    {
        for (LevelType i = start; i <= end; i++)
        {
            Table_[i] = o;
        }
    }

    return *this;
}


template < typename T >
const LUT<T> & LUT<T>::Lookup_Gain(Plane & dst, const Plane & src, const Plane & ref) const
{
    PCType i, j, upper;
    PCType height = ref.Height();
    PCType width = ref.Width();
    PCType stride = ref.Width();
    DType rFloor = ref.Floor();
    DType sNeutral = src.Neutral();
    DType dNeutral = dst.Neutral();

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            dst[i] = dst.Quantize((src[i] - sNeutral) * Table_[ref[i] - rFloor] + dNeutral);
        }
    }

    return *this;
}

template < typename T >
const LUT<T> & LUT<T>::Lookup_Gain(Plane_FL & dst, const Plane_FL & src, const Plane & ref) const
{
    PCType i, j;
    PCType height = ref.Height();
    PCType width = ref.Width();
    PCType ref_stride = ref.Width();
    PCType src_stride = src.Width();
    PCType dst_stride = dst.Width();
    DType rFloor = ref.Floor();

    const DType *refp = ref.Data();
    const DType *srcp = src.Data();
    FLType *dstp = dst.Data();

    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i++)
        {
            dstp[i] = dst.Quantize(srcp[i] * Table_[refp[i] - rFloor]);
        }

        refp += ref_stride;
        srcp += src_stride;
        dstp += dst_stride;
    }

    return *this;
}

template < typename T >
const LUT<T> & LUT<T>::Lookup_Gain(Frame & dst, const Frame & src, const Plane & ref) const
{
    PCType i, j, upper;
    PCType height = ref.Height();
    PCType width = ref.Width();
    PCType stride = ref.Width();
    DType rFloor = ref.Floor();

    if (src.isYUV())
    {
        const Plane &srcY = src.Y();
        const Plane &srcU = src.U();
        const Plane &srcV = src.V();
        Plane &dstY = dst.Y();
        Plane &dstU = dst.U();
        Plane &dstV = dst.V();

        sint64 sNeutral = srcU.Neutral();
        FLType sRangeC2FL = static_cast<FLType>(srcU.ValueRange()) / 2.;

        FLType gain;

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                gain = Min(sRangeC2FL / Max(Abs(srcU[i] - sNeutral), Abs(srcV[i] - sNeutral)), Table_[ref[i] - rFloor]);
                dstY[i] = dstY.GetD(srcY.GetFL(srcY[i])*gain);
                dstU[i] = dstU.GetD(srcU.GetFL(srcU[i])*gain);
                dstV[i] = dstV.GetD(srcV.GetFL(srcV[i])*gain);
            }
        }
    }
    else if (src.isRGB())
    {
        const Plane & srcR = src.R();
        const Plane & srcG = src.G();
        const Plane & srcB = src.B();
        Plane & dstR = dst.R();
        Plane & dstG = dst.G();
        Plane & dstB = dst.B();

        DType sRange = srcR.ValueRange();
        FLType sRangeFL = static_cast<FLType>(sRange);

        FLType gain;

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                gain = Min(sRangeFL / Max(srcR[i], Max(srcG[i], srcB[i])), Table_[ref[i] - rFloor]);
                dstR[i] = dstR.GetD(srcR.GetFL(srcR[i])*gain);
                dstG[i] = dstG.GetD(srcG.GetFL(srcG[i])*gain);
                dstB[i] = dstB.GetD(srcB.GetFL(srcB[i])*gain);
            }
        }
    }

    return *this;
}


// Functions of template class LUT instantiation
inline const LUT<DType> & LUT<DType>::Lookup(Plane & dst, const Plane & src) const
{
    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();
    DType iFloor = src.Floor();

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            dst[i] = Table_[src[i] - iFloor];
        }
    }

    return *this;
}

inline const LUT<FLType> & LUT<FLType>::Lookup(Plane_FL & dst, const Plane & src) const
{
    PCType i, j;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType src_stride = src.Width();
    PCType dst_stride = dst.Width();
    DType iFloor = src.Floor();

    const DType *srcp = src.Data();
    FLType *dstp = dst.Data();

    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i++)
        {
            dstp[i] = Table_[srcp[i] - iFloor];
        }

        srcp += src_stride;
        dstp += dst_stride;
    }

    return *this;
}

inline const LUT<FLType> & LUT<FLType>::Lookup(Plane_FL & dst, const Plane_FL & src) const
{
    PCType i, j, upper;
    PCType height = src.Height();
    PCType width = src.Width();
    PCType stride = src.Width();
    FLType iFloor = src.Floor();
    FLType iValueRange = src.ValueRange();

    LevelType _lower, _upper;
    LevelType LevelRange = Levels_ - 1;
    FLType value;
    FLType scale;

    scale = static_cast<FLType>(LevelRange) / iValueRange;

    for (j = 0; j < height; j++)
    {
        i = stride * j;
        for (upper = i + width; i < upper; i++)
        {
            value = (src[i] - iFloor) * scale;
            _lower = static_cast<LevelType>(value);
            if (_lower >= LevelRange) _lower = LevelRange - 1;
            _upper = _lower + 1;
            dst[i] = Table_[_lower] * (_upper - value) + Table_[_upper] * (value - _lower);
        }
    }

    return *this;
}


#endif