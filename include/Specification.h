#ifndef SPECIFICATION_H_
#define SPECIFICATION_H_


#include <cmath>


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


const int HD_Width_U = 2048;
const int HD_Height_U = 1536;
const int SD_Width_U = 1024;
const int SD_Height_U = 576;


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
    Average = 100,
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
        green_x = 0.300;
        green_y = 0.600;
        blue_x = 0.150;
        blue_y = 0.060;
        red_x = 0.640;
        red_y = 0.330;
        white_x = 0.3127;
        white_y = 0.3290;
        break;
    case ColorPrim::bt470m:
        green_x = 0.21;
        green_y = 0.71;
        blue_x = 0.14;
        blue_y = 0.08;
        red_x = 0.67;
        red_y = 0.33;
        white_x = 0.310;
        white_y = 0.316;
        break;
    case ColorPrim::bt470bg:
        green_x = 0.29;
        green_y = 0.60;
        blue_x = 0.15;
        blue_y = 0.06;
        red_x = 0.64;
        red_y = 0.33;
        white_x = 0.3127;
        white_y = 0.3290;
        break;
    case ColorPrim::smpte170m:
        green_x = 0.310;
        green_y = 0.595;
        blue_x = 0.155;
        blue_y = 0.070;
        red_x = 0.630;
        red_y = 0.340;
        white_x = 0.3127;
        white_y = 0.3290;
        break;
    case ColorPrim::smpte240m:
        green_x = 0.310;
        green_y = 0.595;
        blue_x = 0.155;
        blue_y = 0.070;
        red_x = 0.630;
        red_y = 0.340;
        white_x = 0.3127;
        white_y = 0.3290;
        break;
    case ColorPrim::film:
        green_x = 0.243;
        green_y = 0.692;
        blue_x = 0.145;
        blue_y = 0.049;
        red_x = 0.681;
        red_y = 0.319;
        white_x = 0.310;
        white_y = 0.316;
        break;
    case ColorPrim::bt2020:
        green_x = 0.170;
        green_y = 0.797;
        blue_x = 0.131;
        blue_y = 0.046;
        red_x = 0.708;
        red_y = 0.292;
        white_x = 0.3127;
        white_y = 0.3290;
        break;
    default:
        green_x = 0.300;
        green_y = 0.600;
        blue_x = 0.150;
        blue_y = 0.060;
        red_x = 0.640;
        red_y = 0.330;
        white_x = 0.3127;
        white_y = 0.3290;
        break;
    }
}

template < typename T >
void TransferChar_Parameter(TransferChar _TransferChar, T &k0, T &phi, T &alpha, T &power, T &div)
{
    switch (_TransferChar)
    {
    case TransferChar::bt709:
        k0 = 0.018;
        phi = 4.500;
        alpha = 0.099;
        power = 0.45;
        break;
    case TransferChar::bt470m:
        k0 = 0;
        phi = 0;
        alpha = 0;
        power = 1 / 2.2;
        break;
    case TransferChar::bt470bg:
        k0 = 0;
        phi = 0;
        alpha = 0;
        power = 1 / 2.8;
        break;
    case TransferChar::smpte170m:
        k0 = 0.018;
        phi = 4.500;
        alpha = 0.099;
        power = 0.45;
        break;
    case TransferChar::smpte240m:
        k0 = 0.0228;
        phi = 4.0;
        alpha = 0.1115;
        power = 0.45;
        break;
    case TransferChar::linear:
        k0 = 1;
        phi = 1;
        alpha = 0;
        power = 1;
        break;
    case TransferChar::log100:
        k0 = 0.01;
        div = 2;
        break;
    case TransferChar::log316:
        k0 = sqrt(10.) / 1000.;
        div = 2.5;
        break;
    case TransferChar::iec61966_2_4:
        k0 = 0.018;
        phi = 4.500;
        alpha = 0.099;
        power = 0.45;
        break;
    case TransferChar::bt1361e:
        k0 = 0.018;
        phi = 4.500;
        alpha = 0.099;
        power = 0.45;
        break;
    case TransferChar::iec61966_2_1:
        k0 = 0.0031308;
        phi = 12.92;
        alpha = 0.055;
        power = 1 / 2.4;
        break;
    case TransferChar::bt2020_10:
        k0 = 0.018;
        phi = 4.500;
        alpha = 0.099;
        power = 0.45;
        break;
    case TransferChar::bt2020_12:
        k0 = 0.0181;
        phi = 4.500;
        alpha = 0.0993;
        power = 0.45;
        break;
    default:
        k0 = 0.0031308;
        phi = 12.92;
        alpha = 0.055;
        power = 1 / 2.4;
        break;
    }
}

template < typename T >
void ColorMatrix_Parameter(ColorMatrix _ColorMatrix, T &Kr, T &Kg, T &Kb)
{
    switch (_ColorMatrix)
    {
    case ColorMatrix::GBR:
        Kr = 0;
        Kg = 1;
        Kb = 0;
        break;
    case ColorMatrix::bt709:
        Kr = 0.2126;
        Kg = 0.7152;
        Kb = 0.0722;
        break;
    case ColorMatrix::fcc:
        Kr = 0.30;
        Kg = 0.59;
        Kb = 0.11;
        break;
    case ColorMatrix::bt470bg:
        Kr = 0.299;
        Kg = 0.587;
        Kb = 0.114;
        break;
    case ColorMatrix::smpte170m:
        Kr = 0.299;
        Kg = 0.587;
        Kb = 0.114;
        break;
    case ColorMatrix::smpte240m:
        Kr = 0.212;
        Kg = 0.701;
        Kb = 0.087;
        break;
    case ColorMatrix::YCgCo:
        Kr = 0.25;
        Kg = 0.50;
        Kb = 0.25;
        break;
    case ColorMatrix::bt2020nc:
        Kr = 0.2627;
        Kg = 0.6780;
        Kb = 0.0593;
        break;
    case ColorMatrix::bt2020c:
        Kr = 0.2627;
        Kg = 0.6780;
        Kb = 0.0593;
        break;
    case ColorMatrix::Average:
        Kr = 1. / 3.;
        Kg = 1. / 3.;
        Kb = 1. / 3.;
        break;
    default:
        Kr = 0.2126;
        Kg = 0.7152;
        Kb = 0.0722;
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
class TransferChar_Conv_Sub
{
public:
    typedef TransferChar_Conv_Sub<T> _Myt;

private:
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
    T recPower;

    // Parameters for log function
    T div = 1;
    T recDiv;

public:
    TransferChar_Conv_Sub(TransferChar _TransferChar)
    {
        TransferChar_Parameter(_TransferChar, k0, phi, alpha, power, div);

        k0Mphi = k0 * phi;
        recPhi = 1 / phi;
        beta = alpha + 1;
        recBeta = 1 / beta;
        recPower = 1 / power;
        recDiv = 1 / div;
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
        return x < k0Mphi ? x * recPhi : pow((x + alpha) * recBeta, recPower);
    }

    T linear2gamma(T x) const
    {
        return x < k0 ? x * phi : pow(x, power) * beta - alpha;
    }

    T log2linear(T x) const
    {
        return x == 0 ? 0 : pow(10, (x - 1) * div);
    }

    T linear2log(T x) const
    {
        return x < k0 ? 0 : 1 + log10(x) * recDiv;
    }
};


template < typename T = FLType >
class TransferChar_Conv
{
public:
    typedef TransferChar_Conv<T> _Myt;

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

private:
    TransferChar_Conv_Sub<T> ToLinear;
    TransferChar_Conv_Sub<T> LinearTo;
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

    void ConvTypeDecision()
    {
        if (ToLinear == LinearTo)
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
    TransferChar_Conv(TransferChar dst, TransferChar src)
        : ToLinear(src), LinearTo(dst), srcType_(TransferTypeDecision(src)), dstType_(TransferTypeDecision(dst))
    {
        ConvTypeDecision();
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

    ConvType Type() const
    {
        return type_;
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif
