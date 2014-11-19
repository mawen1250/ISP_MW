#ifndef LUT_H_
#define LUT_H_


#include "Image_Type.h"


template < typename T = DType >
class LUT {
public:
    typedef LUT<T> _Myt;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

    typedef uint32 LevelType;

private:
    LevelType Levels_ = 0;
    pointer Table_ = nullptr;

public:
    _Myt() {} // Default constructor
    _Myt(const _Myt& src); // Copy constructor
    _Myt(_Myt&& src); // Move constructor
    explicit _Myt(LevelType Levels); // Convertor/Constructor from LevelType
    explicit _Myt(const Plane& src); // Convertor/Constructor from Plane
    ~LUT(); // Destructor

    _Myt& operator=(const _Myt& src); // Copy assignment operator
    _Myt& operator=(_Myt&& src); // Move assignment operator
    reference operator[](LevelType i) { return Table_[i]; }
    const_reference operator[](LevelType i) const { return Table_[i]; }

    LevelType Levels() const { return Levels_; }
    pointer Table() { return Table_; }
    const_pointer Table() const { return Table_; }

    _Myt& Set(const Plane& src, LevelType i, T o);
    _Myt& SetRange(const Plane& src, T o = 0) { return SetRange(src, o, src.Floor(), src.Ceil()); }
    _Myt& SetRange(const Plane& src, T o, LevelType start, LevelType end);
    template < typename _St1, typename _Fn1 > _Myt& Set(const _St1 src, _Fn1 _Func);

    T Lookup(const Plane& src, LevelType Value) const { return Table_[Value - src.Floor()]; }
    //template < typename _Dt1, typename _St1 > const _Myt& Lookup(_Dt1& dst, const _St1 src) const;
    const _Myt& Lookup(Plane& dst, const Plane& src) const;
    const _Myt& Lookup(Plane_FL& dst, const Plane& src) const;
    const _Myt& Lookup(Plane_FL& dst, const Plane_FL& src) const;

    template < typename _Dt1, typename _St1, typename _Rt1 > const _Myt& Lookup_Gain(_Dt1& dst, const _St1& src, const _Rt1& ref) const;
    template < typename _Rt1 > const _Myt& Lookup_Gain(Frame& dst, const Frame& src, const _Rt1& ref) const;
};


#include "LUT.hpp"


#endif