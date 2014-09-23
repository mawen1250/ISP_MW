#ifndef SPECIFICATION_H_
#define SPECIFICATION_H_


#include "Image_Type.h"
#include "cmath"


void ColorPrim_Parameter(ColorPrim ColorPrim, FLType & green_x, FLType & green_y, FLType & blue_x, FLType & blue_y, FLType & red_x, FLType & red_y, FLType & white_x, FLType & white_y);


void TransferChar_Parameter(TransferChar TransferChar, FLType & k0, FLType & phi, FLType & alpha, FLType & power, FLType & div);
inline void TransferChar_Parameter(TransferChar TransferChar, FLType & k0, FLType & phi, FLType & alpha, FLType & power)
{
    FLType temp;
    TransferChar_Parameter(TransferChar, k0, phi, alpha, power, temp);
}
inline void TransferChar_Parameter(TransferChar TransferChar, FLType & k0, FLType & div)
{
    FLType temp;
    TransferChar_Parameter(TransferChar, k0, temp, temp, temp, div);
}


void ColorMatrix_Parameter(ColorMatrix ColorMatrix, FLType & Kr, FLType & Kg, FLType & Kb);


// Conversion functions
inline FLType TransferChar_gamma2linear(FLType data, FLType k0, FLType phi, FLType alpha, FLType power)
{
    return data < k0*phi ? data / phi : std::pow((data + alpha) / (1 + alpha), 1 / power);
}

inline FLType TransferChar_linear2gamma(FLType data, FLType k0, FLType phi, FLType alpha, FLType power)
{
    return data < k0 ? phi*data : (1 + alpha)*std::pow(data, power) - alpha;
}

inline FLType TransferChar_gamma2linear(FLType data, FLType k0, FLType div)
{
    return data == 0 ? 0 : std::pow(10, (data - 1)*div);
}

inline FLType TransferChar_linear2gamma(FLType data, FLType k0, FLType div)
{
    return data < k0 ? 0 : 1 + std::log10(data) / div;
}


#endif