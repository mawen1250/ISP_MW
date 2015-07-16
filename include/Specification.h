#ifndef SPECIFICATION_H_
#define SPECIFICATION_H_


#include <cmath>
#include "Type.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


const PCType HD_Width_U = 2048;
const PCType HD_Height_U = 1536;
const PCType SD_Width_U = 1024;
const PCType SD_Height_U = 576;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


enum class ResLevel
{
    SD = 0,
    HD,
    UHD,
    Unknown,
};


enum class ColorPrim
{
    bt709 = 1,
    Unspecified = 2,
    bt470m = 4,
    bt470bg = 5,
    smpte170m = 6,
    smpte240m = 7,
    film = 8,
    bt2020 = 9
};


enum class TransferChar
{
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


enum class ColorMatrix
{
    GBR = 0,
    bt709 = 1,
    Unspecified = 2,
    fcc = 4,
    bt470bg = 5,
    smpte170m = 6,
    smpte240m = 7,
    YCgCo = 8,
    bt2020nc = 9,
    bt2020c = 10,
    OPP = 100, // opponent colorspace
    Minimum,
    Maximum
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Parameter functions
template < typename T >
void ColorPrim_Parameter(ColorPrim _ColorPrim, T &green_x, T &green_y, T &blue_x, T &blue_y, T &red_x, T &red_y, T &white_x, T &white_y)
{
    switch (_ColorPrim)
    {
    case ColorPrim::bt709:
        green_x = T(0.300);
        green_y = T(0.600);
        blue_x = T(0.150);
        blue_y = T(0.060);
        red_x = T(0.640);
        red_y = T(0.330);
        white_x = T(0.3127);
        white_y = T(0.3290);
        break;
    case ColorPrim::bt470m:
        green_x = T(0.21);
        green_y = T(0.71);
        blue_x = T(0.14);
        blue_y = T(0.08);
        red_x = T(0.67);
        red_y = T(0.33);
        white_x = T(0.310);
        white_y = T(0.316);
        break;
    case ColorPrim::bt470bg:
        green_x = T(0.29);
        green_y = T(0.60);
        blue_x = T(0.15);
        blue_y = T(0.06);
        red_x = T(0.64);
        red_y = T(0.33);
        white_x = T(0.3127);
        white_y = T(0.3290);
        break;
    case ColorPrim::smpte170m:
        green_x = T(0.310);
        green_y = T(0.595);
        blue_x = T(0.155);
        blue_y = T(0.070);
        red_x = T(0.630);
        red_y = T(0.340);
        white_x = T(0.3127);
        white_y = T(0.3290);
        break;
    case ColorPrim::smpte240m:
        green_x = T(0.310);
        green_y = T(0.595);
        blue_x = T(0.155);
        blue_y = T(0.070);
        red_x = T(0.630);
        red_y = T(0.340);
        white_x = T(0.3127);
        white_y = T(0.3290);
        break;
    case ColorPrim::film:
        green_x = T(0.243);
        green_y = T(0.692);
        blue_x = T(0.145);
        blue_y = T(0.049);
        red_x = T(0.681);
        red_y = T(0.319);
        white_x = T(0.310);
        white_y = T(0.316);
        break;
    case ColorPrim::bt2020:
        green_x = T(0.170);
        green_y = T(0.797);
        blue_x = T(0.131);
        blue_y = T(0.046);
        red_x = T(0.708);
        red_y = T(0.292);
        white_x = T(0.3127);
        white_y = T(0.3290);
        break;
    default:
        green_x = T(0.300);
        green_y = T(0.600);
        blue_x = T(0.150);
        blue_y = T(0.060);
        red_x = T(0.640);
        red_y = T(0.330);
        white_x = T(0.3127);
        white_y = T(0.3290);
        break;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename T >
void TransferChar_Parameter(TransferChar _TransferChar, T &k0, T &phi, T &alpha, T &power, T &div)
{
    k0 = T(1);
    phi = T(1);
    alpha = T(0);
    power = T(1);
    div = T(1);

    switch (_TransferChar)
    {
    case TransferChar::bt709:
        k0 = T(0.018);
        phi = T(4.500);
        alpha = T(0.099);
        power = T(0.45);
        break;
    case TransferChar::bt470m:
        k0 = T(0);
        phi = T(0);
        alpha = T(0);
        power = T(1. / 2.2);
        break;
    case TransferChar::bt470bg:
        k0 = T(0);
        phi = T(0);
        alpha = T(0);
        power = T(1. / 2.8);
        break;
    case TransferChar::smpte170m:
        k0 = T(0.018);
        phi = T(4.500);
        alpha = T(0.099);
        power = T(0.45);
        break;
    case TransferChar::smpte240m:
        k0 = T(0.0228);
        phi = T(4.0);
        alpha = T(0.1115);
        power = T(0.45);
        break;
    case TransferChar::linear:
        k0 = T(1);
        phi = T(1);
        alpha = T(0);
        power = T(1);
        break;
    case TransferChar::log100:
        k0 = T(0.01);
        div = T(2);
        break;
    case TransferChar::log316:
        k0 = T(sqrt(10.) / 1000.);
        div = T(2.5);
        break;
    case TransferChar::iec61966_2_4:
        k0 = T(0.018);
        phi = T(4.500);
        alpha = T(0.099);
        power = T(0.45);
        break;
    case TransferChar::bt1361e:
        k0 = T(0.018);
        phi = T(4.500);
        alpha = T(0.099);
        power = T(0.45);
        break;
    case TransferChar::iec61966_2_1:
        k0 = T(0.0031308);
        phi = T(12.92);
        alpha = T(0.055);
        power = T(1. / 2.4);
        break;
    case TransferChar::bt2020_10:
        k0 = T(0.018);
        phi = T(4.500);
        alpha = T(0.099);
        power = T(0.45);
        break;
    case TransferChar::bt2020_12:
        k0 = T(0.0181);
        phi = T(4.500);
        alpha = T(0.0993);
        power = T(0.45);
        break;
    default:
        k0 = T(1);
        phi = T(1);
        alpha = T(0);
        power = T(1);
        div = T(1);
        break;
    }
}


template < typename T >
void TransferChar_Parameter(TransferChar _TransferChar, T &k0, T &phi, T &alpha, T &power)
{
    T temp;
    TransferChar_Parameter(_TransferChar, k0, phi, alpha, power, temp);
}

template < typename T >
void TransferChar_Parameter(TransferChar _TransferChar, T &k0, T &div)
{
    T temp;
    TransferChar_Parameter(_TransferChar, k0, temp, temp, temp, div);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename T >
void ColorMatrix_Parameter(ColorMatrix _ColorMatrix, T &Kr, T &Kg, T &Kb)
{
    switch (_ColorMatrix)
    {
    case ColorMatrix::GBR:
        Kr = T(0);
        Kg = T(1);
        Kb = T(0);
        break;
    case ColorMatrix::bt709:
        Kr = T(0.2126);
        Kg = T(0.7152);
        Kb = T(0.0722);
        break;
    case ColorMatrix::fcc:
        Kr = T(0.30);
        Kg = T(0.59);
        Kb = T(0.11);
        break;
    case ColorMatrix::bt470bg:
        Kr = T(0.299);
        Kg = T(0.587);
        Kb = T(0.114);
        break;
    case ColorMatrix::smpte170m:
        Kr = T(0.299);
        Kg = T(0.587);
        Kb = T(0.114);
        break;
    case ColorMatrix::smpte240m:
        Kr = T(0.212);
        Kg = T(0.701);
        Kb = T(0.087);
        break;
    case ColorMatrix::YCgCo:
        Kr = T(0.25);
        Kg = T(0.50);
        Kb = T(0.25);
        break;
    case ColorMatrix::bt2020nc:
        Kr = T(0.2627);
        Kg = T(0.6780);
        Kb = T(0.0593);
        break;
    case ColorMatrix::bt2020c:
        Kr = T(0.2627);
        Kg = T(0.6780);
        Kb = T(0.0593);
        break;
    case ColorMatrix::OPP:
        Kr = T(1. / 3.);
        Kg = T(1. / 3.);
        Kb = T(1. / 3.);
        break;
    default:
        Kr = T(0.2126);
        Kg = T(0.7152);
        Kb = T(0.0722);
        break;
    }
}


template < typename T >
void ColorMatrix_RGB2YUV_Parameter(ColorMatrix _ColorMatrix, T &Yr, T &Yg, T &Yb, T &Ur, T &Ug, T &Ub, T &Vr, T &Vg, T &Vb)
{
    if (_ColorMatrix == ColorMatrix::GBR)
    {
        // E'Y = 1 * E'G
        Yr = static_cast<T>(0.0L);
        Yg = static_cast<T>(1.0L);
        Yb = static_cast<T>(0.0L);

        // E'Pb = 1 * E'B
        Ur = static_cast<T>(0.0L);
        Ug = static_cast<T>(0.0L);
        Ub = static_cast<T>(1.0L);

        // E'Pr = 1 * E'R
        Vr = static_cast<T>(1.0L);
        Vg = static_cast<T>(0.0L);
        Vb = static_cast<T>(0.0L);
    }
    else if (_ColorMatrix == ColorMatrix::YCgCo)
    {
        // E'Y  =   1 / 4 * E'R + 1 / 2 * E'G + 1 / 4 * E'B
        Yr = static_cast<T>(1.0L / 4.0L);
        Yg = static_cast<T>(1.0L / 2.0L);
        Yb = static_cast<T>(1.0L / 4.0L);

        // E'Pg = - 1 / 4 * E'R + 1 / 2 * E'G - 1 / 4 * E'B
        Ur = static_cast<T>(-1.0L / 4.0L);
        Ug = static_cast<T>(1.0L / 2.0L);
        Ub = static_cast<T>(-1.0L / 4.0L);

        // E'Po = 1 / 2 * E'R                 - 1 / 2 * E'B
        Vr = static_cast<T>(1.0L / 2.0L);
        Vg = static_cast<T>(0.0L);
        Vb = static_cast<T>(-1.0L / 2.0L);
    }
    else if (_ColorMatrix == ColorMatrix::OPP)
    {
        // E'Y  = 1 / 3 * E'R + 1 / 3 * E'G + 1 / 3 * E'B
        Yr = static_cast<T>(1.0L / 3.0L);
        Yg = static_cast<T>(1.0L / 3.0L);
        Yb = static_cast<T>(1.0L / 3.0L);

        // E'Pb = 1 / 2 * E'R               - 1 / 2 * E'B
        Ur = static_cast<T>(1.0L / 2.0L);
        Ug = static_cast<T>(0.0L);
        Ub = static_cast<T>(-1.0L / 2.0L);

        // E'Pr = 1 / 4 * E'R - 1 / 2 * E'G + 1 / 4 * E'B
        Vr = static_cast<T>(1.0L / 4.0L);
        Vg = static_cast<T>(-1.0L / 2.0L);
        Vb = static_cast<T>(1.0L / 4.0L);
    }
    else
    {
        ldbl Kr, Kg, Kb;
        ColorMatrix_Parameter(_ColorMatrix, Kr, Kg, Kb);

        // E'Y = Kr * E'R + ( 1 - Kr - Kg ) * E'G + Kb * E'B
        Yr = static_cast<T>(Kr);
        Yg = static_cast<T>(Kg);
        Yb = static_cast<T>(Kb);

        // E'Pb = 0.5 * ( E'B - E'Y ) / ( 1 - Kb )
        Ur = static_cast<T>(-Kr * 0.5L / (1.0L - Kb));
        Ug = static_cast<T>(-Kg * 0.5L / (1.0L - Kb));
        Ub = static_cast<T>(0.5L);

        // E'Pr = 0.5 * ( E'R - E'Y ) / ( 1 - Kr )
        Vr = static_cast<T>(0.5L);
        Vg = static_cast<T>(-Kg * 0.5L / (1.0L - Kr));
        Vb = static_cast<T>(-Kb * 0.5L / (1.0L - Kr));
    }
}


template < typename T >
void ColorMatrix_YUV2RGB_Parameter(ColorMatrix _ColorMatrix, T &Ry, T &Ru, T &Rv, T &Gy, T &Gu, T &Gv, T &By, T &Bu, T &Bv)
{
    if (_ColorMatrix == ColorMatrix::GBR)
    {
        // E'R = 1 * E'Pr
        Ry = static_cast<T>(0.0L);
        Ru = static_cast<T>(0.0L);
        Rv = static_cast<T>(1.0L);

        // E'G = 1 * E'Y
        Gy = static_cast<T>(1.0L);
        Gu = static_cast<T>(0.0L);
        Gv = static_cast<T>(0.0L);

        // E'B = 1 * E'Pb
        By = static_cast<T>(0.0L);
        Bu = static_cast<T>(1.0L);
        Bv = static_cast<T>(0.0L);
    }
    else if (_ColorMatrix == ColorMatrix::YCgCo)
    {
        // E'R = E'Y - E'Pg + E'Po
        Ry = static_cast<T>(1.0L);
        Ru = static_cast<T>(-1.0L);
        Rv = static_cast<T>(1.0L);

        // E'G = E'Y + E'Pg
        Gy = static_cast<T>(1.0L);
        Gu = static_cast<T>(1.0L);
        Gv = static_cast<T>(0.0L);

        // E'B = E'Y - E'Pg - E'Po
        By = static_cast<T>(1.0L);
        Bu = static_cast<T>(-1.0L);
        Bv = static_cast<T>(-1.0L);
    }
    else if (_ColorMatrix == ColorMatrix::OPP)
    {
        // E'R = E'Y + E'Pb + 2 / 3 * E'Pr
        Ry = static_cast<T>(1.0L);
        Ru = static_cast<T>(1.0L);
        Rv = static_cast<T>(2.0L / 3.0L);

        // E'G = E'Y        - 4 / 3 * E'Pr
        Gy = static_cast<T>(1.0L);
        Gu = static_cast<T>(0.0L);
        Gv = static_cast<T>(-4.0L / 3.0L);

        // E'B = E'Y - E'Pb + 2 / 3 * E'Pr
        By = static_cast<T>(1.0L);
        Bu = static_cast<T>(-1.0L);
        Bv = static_cast<T>(2.0L / 3.0L);
    }
    else
    {
        ldbl Kr, Kg, Kb;
        ColorMatrix_Parameter(_ColorMatrix, Kr, Kg, Kb);

        // E'R = E'Y + 2 * ( 1 - Kr ) * E'Pr
        Ry = static_cast<T>(1.0L);
        Ru = static_cast<T>(0.0L);
        Rv = static_cast<T>(2.0L * (1.0L - Kr));

        // E'G = E'Y - 2 * Kb * ( 1 - Kb ) / ( 1 - Kr - Kb ) * E'Pb - 2 * Kr * ( 1 - Kr ) / ( 1 - Kr - Kb ) * E'Pr
        Gy = static_cast<T>(1.0L);
        Gu = static_cast<T>(-2.0L * Kb * (1.0L - Kb) / Kg);
        Gv = static_cast<T>(-2.0L * Kr * (1.0L - Kr) / Kg);

        // E'B = E'Y + 2 * ( 1 - Kb ) * E'Pb
        By = static_cast<T>(1.0L);
        Bu = static_cast<T>(2.0L * (1.0L - Kb));
        Bv = static_cast<T>(0.0L);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Default functions
inline ResLevel ResLevel_Default(int Width, int Height)
{
    if (Width > HD_Width_U || Height > HD_Height_U) return ResLevel::UHD;
    if (Width > SD_Width_U || Height > SD_Height_U) return ResLevel::HD;
    return ResLevel::SD;
}


inline ColorPrim ColorPrim_Default(int Width, int Height, bool RGB)
{
    ResLevel _ResLevel = ResLevel_Default(Width, Height);

    if (RGB) return ColorPrim::bt709;
    if (_ResLevel == ResLevel::UHD) return ColorPrim::bt2020;
    if (_ResLevel == ResLevel::HD) return ColorPrim::bt709;
    if (_ResLevel == ResLevel::SD) return ColorPrim::smpte170m;
    return ColorPrim::bt709;
}


inline TransferChar TransferChar_Default(int Width, int Height, bool RGB)
{
    ResLevel _ResLevel = ResLevel_Default(Width, Height);

    if (RGB) return TransferChar::iec61966_2_1;
    if (_ResLevel == ResLevel::UHD) return TransferChar::bt2020_12;
    if (_ResLevel == ResLevel::HD) return TransferChar::bt709;
    if (_ResLevel == ResLevel::SD) return TransferChar::smpte170m;
    return TransferChar::bt709;
}


inline ColorMatrix ColorMatrix_Default(int Width, int Height)
{
    ResLevel _ResLevel = ResLevel_Default(Width, Height);

    if (_ResLevel == ResLevel::UHD) return ColorMatrix::bt2020nc;
    if (_ResLevel == ResLevel::HD) return ColorMatrix::bt709;
    if (_ResLevel == ResLevel::SD) return ColorMatrix::smpte170m;
    return ColorMatrix::bt709;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Conversion classes
template < typename T = FLType >
struct TransferChar_Conv_Sub
{
    typedef TransferChar_Conv_Sub<T> _Myt;

    // General Parameters
    T k0 = 1;
    T k0Mphi;

    // Parameters for gamma function
    T phi = 1;
    T recPhi;
    T alpha = 0;
    T beta;
    T recBeta;
    T power = 1;
    T gamma;

    // Parameters for log function
    T div = 1;
    T recDiv;

    TransferChar_Conv_Sub(TransferChar _TransferChar)
    {
        ldbl _k0 = 1.0L, _phi = 1.0L, _alpha = 0.0L, _power = 1.0L, _div = 1.0L;
        TransferChar_Parameter(_TransferChar, _k0, _phi, _alpha, _power, _div);
        ldbl _beta = _alpha + 1.0L;

        k0 = static_cast<T>(_k0);
        phi = static_cast<T>(_phi);
        alpha = static_cast<T>(_alpha);
        power = static_cast<T>(_power);
        div = static_cast<T>(_div);
        k0Mphi = static_cast<T>(_k0 * _phi);
        recPhi = static_cast<T>(1 / _phi);
        beta = static_cast<T>(_beta);
        recBeta = static_cast<T>(1 / _beta);
        gamma = static_cast<T>(1 / _power);
        recDiv = static_cast<T>(1 / _div);
    }

    bool operator==(const _Myt &right) const
    {
        auto &left = *this;

        if (left.k0 == right.k0 && left.phi == right.phi && left.alpha == right.alpha
            && left.power == right.power && left.div == right.div)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    T gamma2linear(T x) const
    {
        return x < k0Mphi ? x * recPhi : pow((x + alpha) * recBeta, gamma);
    }

    T linear2gamma(T x) const
    {
        return x < k0 ? x * phi : pow(x, power) * beta - alpha;
    }

    T log2linear(T x) const
    {
        return x == 0 ? 0 : pow(T(10), (x - 1) * div);
    }

    T linear2log(T x) const
    {
        return x < k0 ? 0 : T(1) + log10(x) * recDiv;
    }
};


class TransferChar_Conv_Base
{
public:
    typedef TransferChar_Conv_Base _Myt;

    enum class TransferType
    {
        linear = 0,
        gamma,
        log
    };

    enum class ConvType
    {
        none = 0,
        linear2gamma,
        linear2log,
        gamma2linear,
        gamma2gamma,
        gamma2log,
        log2linear,
        log2gamma,
        log2log
    };

protected:
    TransferType srcType_;
    TransferType dstType_;
    ConvType type_;

protected:
    TransferType TransferTypeDecision(TransferChar _TransferChar) const
    {
        if (_TransferChar == TransferChar::linear)
        {
            return TransferType::linear;
        }
        else if (_TransferChar == TransferChar::log100 || _TransferChar == TransferChar::log316)
        {
            return TransferType::log;
        }
        else
        {
            return TransferType::gamma;
        }
    }

    void ConvTypeDecision(bool same = false)
    {
        if (same)
        {
            type_ = ConvType::none;
        }
        else if (srcType_ == TransferType::linear)
        {
            if (dstType_ == TransferType::gamma)
            {
                type_ = ConvType::linear2gamma;
            }
            else
            {
                type_ = ConvType::linear2log;
            }
        }
        else if (srcType_ == TransferType::gamma)
        {
            if (dstType_ == TransferType::linear)
            {
                type_ = ConvType::gamma2linear;
            }
            else if (dstType_ == TransferType::gamma)
            {
                type_ = ConvType::gamma2gamma;
            }
            else
            {
                type_ = ConvType::gamma2log;
            }
        }
        else
        {
            if (dstType_ == TransferType::linear)
            {
                type_ = ConvType::log2linear;
            }
            else if (dstType_ == TransferType::gamma)
            {
                type_ = ConvType::log2gamma;
            }
            else
            {
                type_ = ConvType::log2log;
            }
        }
    }

public:
    TransferChar_Conv_Base(TransferChar dst, TransferChar src)
        : srcType_(TransferTypeDecision(src)), dstType_(TransferTypeDecision(dst))
    {}

    ConvType Type() const
    {
        return type_;
    }
};


template < typename T = FLType >
class TransferChar_Conv
    : public TransferChar_Conv_Base
{
public:
    typedef TransferChar_Conv<T> _Myt;
    typedef TransferChar_Conv_Base _Mybase;

protected:
    TransferChar_Conv_Sub<T> ToLinear;
    TransferChar_Conv_Sub<T> LinearTo;

public:
    TransferChar_Conv(TransferChar dst, TransferChar src)
        : _Mybase(dst, src), ToLinear(src), LinearTo(dst)
    {
        ConvTypeDecision(ToLinear == LinearTo);
    }

    T operator()(T x) const
    {
        switch (type_)
        {
        case ConvType::none:
            return none(x);
        case ConvType::linear2gamma:
            return linear2gamma(x);
        case ConvType::linear2log:
            return linear2log(x);
        case ConvType::gamma2linear:
            return gamma2linear(x);
        case ConvType::gamma2gamma:
            return gamma2gamma(x);
        case ConvType::gamma2log:
            return gamma2log(x);
        case ConvType::log2linear:
            return log2linear(x);
        case ConvType::log2gamma:
            return log2gamma(x);
        case ConvType::log2log:
            return log2log(x);
        default:
            return none(x);
        }
    }

    T none(T x) const
    {
        return x;
    }

    T linear2gamma(T x) const
    {
        return LinearTo.linear2gamma(x);
    }

    T linear2log(T x) const
    {
        return LinearTo.linear2log(x);
    }

    T gamma2linear(T x) const
    {
        return ToLinear.gamma2linear(x);
    }

    T gamma2gamma(T x) const
    {
        return LinearTo.linear2gamma(ToLinear.gamma2linear(x));
    }

    T gamma2log(T x) const
    {
        return LinearTo.linear2log(ToLinear.gamma2linear(x));
    }

    T log2linear(T x) const
    {
        return ToLinear.log2linear(x);
    }

    T log2gamma(T x) const
    {
        return LinearTo.linear2gamma(ToLinear.log2linear(x));
    }

    T log2log(T x) const
    {
        return LinearTo.linear2log(ToLinear.log2linear(x));
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif
