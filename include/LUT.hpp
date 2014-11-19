#ifndef LUT_HPP_
#define LUT_HPP_


// Template functions of template class LUT
template < typename T > inline
LUT<T>::LUT(const _Myt& src)
: Levels_(src.Levels_)
{
    Table_ = new T[Levels_];

    memcpy(Table_, src.Table_, sizeof(T)*Levels_);
}

template < typename T > inline
LUT<T>::LUT(_Myt&& src)
: Levels_(src.Levels_)
{
    Table_ = src.Table_;

    src.Levels_ = 0;
    src.Table_ = nullptr;
}

template < typename T > inline
LUT<T>::LUT(LevelType Levels)
: Levels_(Levels)
{
    Table_ = new T[Levels_];
}

template < typename T > inline
LUT<T>::LUT(const Plane& src)
: Levels_(src.ValueRange() + 1)
{
    Table_ = new T[Levels_];
}

template < typename T > inline
LUT<T>::~LUT()
{
    delete[] Table_;
}

template < typename T > inline
LUT<T>& LUT<T>::operator=(const _Myt& src)
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

template < typename T > inline
LUT<T>& LUT<T>::operator=(_Myt&& src)
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


template < typename T > inline
LUT<T>& LUT<T>::Set(const Plane& src, LevelType i, T o)
{
    if (i >= src.Floor() && i <= src.Ceil()) Table_[i - src.Floor()] = o;
    return *this;
}

template < typename T > inline
LUT<T>& LUT<T>::SetRange(const Plane& src, T o, LevelType start, LevelType end)
{
    sint64 length = static_cast<sint64>(end)-static_cast<sint64>(start)+1;
    start = Max(LevelType(0), start - src.Floor());
    end = start + static_cast<LevelType>(length)-1;

    if (length >= 1 && end < Levels_)
    {
        for (LevelType i = start; i <= end; ++i)
        {
            Table_[i] = o;
        }
    }

    return *this;
}

template < typename T >
template < typename _St1, typename _Fn1 > inline
LUT<T>& LUT<T>::Set(const _St1 src, _Fn1 _Func)
{
    for (_St1::value_type i = src.Floor(); i <= src.Ceil(); ++i)
    {
        Table_[i - src.Floor()] = _Func(i);
    }

    return *this;
}


template < typename T >
template < typename _Dt1, typename _St1, typename _Rt1 > inline
const LUT<T>& LUT<T>::Lookup_Gain(_Dt1& dst, const _St1& src, const _Rt1& ref) const
{
    dst.transform(src, ref, [&](_St1::value_type s, _Rt1::value_type r)
    {
        return dst.Quantize((s - src.Neutral()) * Table_[r - ref.Floor()] + dst.Neutral());
    });

    return *this;
}

template < typename T >
template < typename _Rt1 > inline
const LUT<T>& LUT<T>::Lookup_Gain(Frame& dst, const Frame& src, const _Rt1& ref) const
{
    PCType i, j, upper;
    PCType height = ref.Height();
    PCType width = ref.Width();
    PCType stride = ref.Width();
    DType rFloor = ref.Floor();

    FLType gain, offset;

    if (src.isYUV())
    {
        auto& srcY = src.Y();
        auto& srcU = src.U();
        auto& srcV = src.V();
        auto& dstY = dst.Y();
        auto& dstU = dst.U();
        auto& dstV = dst.V();

        DType sFloor = srcY.Floor();
        sint32 sNeutral = srcU.Neutral();
        FLType sRangeFL = static_cast<FLType>(srcY.ValueRange());
        FLType sRangeC2FL = static_cast<FLType>(srcU.ValueRange()) / 2.;

        DType Yval;
        sint32 Uval, Vval;
        if (dstU.isPCChroma())
            offset = dstU.Neutral() + FLType(0.499999);
        else
            offset = dstU.Neutral() + FLType(0.5);
        FLType offsetY = dstY.Floor() + FLType(0.5);

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Yval = srcY[i] - sFloor;
                Uval = srcU[i] - sNeutral;
                Vval = srcV[i] - sNeutral;
                gain = Table_[ref[i] - rFloor];
                gain = Min(sRangeFL / Yval, Min(sRangeC2FL / Max(Abs(Uval), Abs(Vval)), gain));
                dstY[i] = static_cast<DType>(Yval * gain + offsetY);
                dstU[i] = static_cast<DType>(Uval * gain + offset);
                dstV[i] = static_cast<DType>(Vval * gain + offset);
            }
        }
    }
    else if (src.isRGB())
    {
        auto& srcR = src.R();
        auto& srcG = src.G();
        auto& srcB = src.B();
        auto& dstR = dst.R();
        auto& dstG = dst.G();
        auto& dstB = dst.B();

        DType sFloor = srcR.Floor();
        FLType sRangeFL = static_cast<FLType>(srcR.ValueRange());

        DType Rval, Gval, Bval;
        offset = dstR.Floor() + FLType(0.5);

        for (j = 0; j < height; j++)
        {
            i = stride * j;
            for (upper = i + width; i < upper; i++)
            {
                Rval = srcR[i] - sFloor;
                Gval = srcG[i] - sFloor;
                Bval = srcB[i] - sFloor;
                gain = Table_[ref[i] - rFloor];
                gain = Min(sRangeFL / Max(Rval, Max(Gval, Bval)), gain);
                dstR[i] = static_cast<DType>(Rval * gain + offset);
                dstG[i] = static_cast<DType>(Gval * gain + offset);
                dstB[i] = static_cast<DType>(Bval * gain + offset);
            }
        }
    }

    return *this;
}


// Functions of template class LUT instantiation
/*template < typename T >
template < typename _Dt1, typename _St1 > inline
const LUT<T>& LUT<T>::Lookup(_Dt1& dst, const _St1 src) const
{
    dst.transform(src, [&](_St1::value_type x)
    {
        return Table_[x - src.Floor()];
    });

    return *this;
}*/

inline const LUT<DType> & LUT<DType>::Lookup(Plane& dst, const Plane& src) const
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

inline const LUT<FLType> & LUT<FLType>::Lookup(Plane_FL& dst, const Plane& src) const
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

template < typename T >
inline const LUT<T> & LUT<T>::Lookup(Plane_FL& dst, const Plane_FL& src) const
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