#ifndef LUT_H_
#define LUT_H_


#include "image_Type.h"


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
    explicit LUT(const Plane & input); // Convertor/Constructor from Plane
    ~LUT(); // Destructor

    LUT & operator=(const LUT & src); // Copy assignment operator
    LUT & operator=(LUT && src); // Move assignment operator
    T & operator[](LevelType i) { return Table_[i]; }
    const T & operator[](LevelType i) const { return Table_[i]; }

    LevelType Levels() const { return Levels_; }
    T * Table() { return Table_; }
    const T * Table() const { return Table_; }

    LUT & Set(const Plane & input, LevelType i, T o);
    LUT & SetRange(const Plane & input, T o = 0) { return SetRange(input, o, input.Floor(), input.Ceil()); }
    LUT & SetRange(const Plane & input, T o, LevelType start, LevelType end);

    T Lookup(const Plane & input, LevelType Value) const { return Table_[Value - input.Floor()]; }
    const LUT & Lookup(Plane & output, const Plane & input) const;
    //LUT & Lookup(Plane & output, const Plane & input) { return const_cast<LUT &>(const_cast<const LUT *>(this)->Lookup(output, input)); }
    const LUT & Lookup(Plane_FL & output, const Plane & input) const;
    //LUT & Lookup(Plane_FL & output, const Plane & input) { return const_cast<LUT &>(const_cast<const LUT *>(this)->Lookup(output, input)); }

    const LUT & Lookup_Gain(Plane & data, const Plane & ref) const;
    //LUT & Lookup_Gain(Plane & data, const Plane & ref) { return const_cast<LUT_FL &>(const_cast<const LUT_FL *>(this)->Lookup(data, ref)); }
    const LUT & Lookup_Gain(Plane_FL & data, const Plane & ref) const;
    //LUT & Lookup_Gain(Plane_FL & data, const Plane & ref) { return const_cast<LUT_FL &>(const_cast<const LUT_FL *>(this)->Lookup(data, ref)); }
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
LUT<T>::LUT(const Plane & input)
    : Levels_(input.ValueRange() + 1)
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
LUT<T> & LUT<T>::Set(const Plane & input, LevelType i, T o)
{
    if (i >= input.Floor() && i <= input.Ceil()) Table_[i - input.Floor()] = o;
    return *this;
}

template < typename T >
LUT<T> & LUT<T>::SetRange(const Plane & input, T o, LevelType start, LevelType end)
{
    sint64 length = (sint64)end - (sint64)start + 1;
    start = Max((LevelType)0, start - input.Floor());
    end = start + (LevelType)length - 1;

    if (length >= 1 && end < Levels_)
    {
        for (LevelType i = start; i <= end; i++)
        {
            Table_[i] = o;
        }
    }

    return *this;
}


inline const LUT<DType> & LUT<DType>::Lookup(Plane & output, const Plane & input) const
{
    PCType i;
    PCType pcount = output.PixelCount();
    DType iFloor = input.Floor();

    for (i = 0; i < pcount; i++)
    {
        output[i] = Table_[input[i] - iFloor];
    }

    return *this;
}

inline const LUT<FLType> & LUT<FLType>::Lookup(Plane_FL & output, const Plane & input) const
{
    PCType i;
    PCType pcount = output.PixelCount();
    DType iFloor = input.Floor();

    for (i = 0; i < pcount; i++)
    {
        output[i] = Table_[input[i] - iFloor];
    }

    return *this;
}


template < typename T >
const LUT<T> & LUT<T>::Lookup_Gain(Plane & data, const Plane & ref) const
{
    PCType i;
    PCType pcount = data.PixelCount();
    DType rFloor = ref.Floor();
    DType dNeutral = data.Neutral();

    for (i = 0; i < pcount; i++)
    {
        data[i] = data.Quantize((data[i] - dNeutral) * Table_[ref[i] - rFloor] + dNeutral);
    }

    return *this;
}

template < typename T >
const LUT<T> & LUT<T>::Lookup_Gain(Plane_FL & data, const Plane & ref) const
{
    PCType i;
    PCType pcount = data.PixelCount();
    DType rFloor = ref.Floor();

    for (i = 0; i < pcount; i++)
    {
        data[i] = data.Quantize(data[i] * Table_[ref[i] - rFloor]);
    }

    return *this;
}


#endif