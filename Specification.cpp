#include "include/Specification.h"
#include "include/ISP_MW.h"


STAT ColorPrim_Parameter(ColorPrim ColorPrim, double * green_x, double * green_y, double * blue_x, double * blue_y, double * red_x, double * red_y, double * white_x, double * white_y)
{
    switch(ColorPrim)
    {
    case ColorPrim::bt709:
        *green_x = 0.300;
        *green_y = 0.600;
        *blue_x  = 0.150;
        *blue_y  = 0.060;
        *red_x   = 0.640;
        *red_y   = 0.330;
        *white_x = 0.3127;
        *white_y = 0.3290;
        break;
    case ColorPrim::bt470m:
        *green_x = 0.21;
        *green_y = 0.71;
        *blue_x  = 0.14;
        *blue_y  = 0.08;
        *red_x   = 0.67;
        *red_y   = 0.33;
        *white_x = 0.310;
        *white_y = 0.316;
        break;
    case ColorPrim::bt470bg:
        *green_x = 0.29;
        *green_y = 0.60;
        *blue_x  = 0.15;
        *blue_y  = 0.06;
        *red_x   = 0.64;
        *red_y   = 0.33;
        *white_x = 0.3127;
        *white_y = 0.3290;
        break;
    case ColorPrim::smpte170m:
        *green_x = 0.310;
        *green_y = 0.595;
        *blue_x  = 0.155;
        *blue_y  = 0.070;
        *red_x   = 0.630;
        *red_y   = 0.340;
        *white_x = 0.3127;
        *white_y = 0.3290;
        break;
    case ColorPrim::smpte240m:
        *green_x = 0.310;
        *green_y = 0.595;
        *blue_x  = 0.155;
        *blue_y  = 0.070;
        *red_x   = 0.630;
        *red_y   = 0.340;
        *white_x = 0.3127;
        *white_y = 0.3290;
        break;
    case ColorPrim::film:
        *green_x = 0.243;
        *green_y = 0.692;
        *blue_x  = 0.145;
        *blue_y  = 0.049;
        *red_x   = 0.681;
        *red_y   = 0.319;
        *white_x = 0.310;
        *white_y = 0.316;
        break;
    case ColorPrim::bt2020:
        *green_x = 0.170;
        *green_y = 0.797;
        *blue_x  = 0.131;
        *blue_y  = 0.046;
        *red_x   = 0.708;
        *red_y   = 0.292;
        *white_x = 0.3127;
        *white_y = 0.3290;
        break;
    default:
        *green_x = 0.300;
        *green_y = 0.600;
        *blue_x  = 0.150;
        *blue_y  = 0.060;
        *red_x   = 0.640;
        *red_y   = 0.330;
        *white_x = 0.3127;
        *white_y = 0.3290;
        break;
    }

    return STAT::OK;
}


STAT TransferChar_Parameter(TransferChar TransferChar, double * k0, double * phi, double * alpha, double * power, double * div)
{
    switch(TransferChar)
    {
    case TransferChar::bt709:
        *k0    = 0.018;
        *phi   = 4.500;
        *alpha = 0.099;
        *power = 0.45;
        break;
    case TransferChar::bt470m:
        *k0    = 0;
        *phi   = 0;
        *alpha = 0;
        *power = 1/2.2;
        break;
    case TransferChar::bt470bg:
        *k0    = 0;
        *phi   = 0;
        *alpha = 0;
        *power = 1/2.8;
        break;
    case TransferChar::smpte170m:
        *k0    = 0.018;
        *phi   = 4.500;
        *alpha = 0.099;
        *power = 0.45;
        break;
    case TransferChar::smpte240m:
        *k0    = 0.0228;
        *phi   = 4.0;
        *alpha = 0.1115;
        *power = 0.45;
        break;
    case TransferChar::linear:
        *k0    = 1;
        *phi   = 1;
        *alpha = 0;
        *power = 1;
        break;
    case TransferChar::log100:
        *k0    = 0.01;
        *div   = 2;
        break;
    case TransferChar::log316:
        *k0    = 0.0031622776601683793319988935444327;    // sqrt(10)/1000
        *div   = 2.5;
        break;
    case TransferChar::iec61966_2_4:
        *k0    = 0.018;
        *phi   = 4.500;
        *alpha = 0.099;
        *power = 0.45;
        break;
    case TransferChar::bt1361e:
        *k0    = 0.018;
        *phi   = 4.500;
        *alpha = 0.099;
        *power = 0.45;
        break;
    case TransferChar::iec61966_2_1:
        *k0    = 0.0031308;
        *phi   = 12.92;
        *alpha = 0.055;
        *power = 1/2.4;
        break;
    case TransferChar::bt2020_10:
        *k0    = 0.018;
        *phi   = 4.500;
        *alpha = 0.099;
        *power = 0.45;
        break;
    case TransferChar::bt2020_12:
        *k0    = 0.0181;
        *phi   = 4.500;
        *alpha = 0.0993;
        *power = 0.45;
        break;
    default:
        *k0    = 0.0031308;
        *phi   = 12.92;
        *alpha = 0.055;
        *power = 1/2.4;
        break;
    }

    return STAT::OK;
}


STAT ColorMatrix_Parameter(ColorMatrix ColorMatrix, double * Kr, double * Kg, double * Kb)
{
    switch(ColorMatrix)
    {
    case ColorMatrix::GBR:
        *Kr = 0;
        *Kg = 1;
        *Kb = 0;
        break;
    case ColorMatrix::bt709:
        *Kr = 0.2126;
        *Kg = 0.7152;
        *Kb = 0.0722;
        break;
    case ColorMatrix::fcc:
        *Kr = 0.30;
        *Kg = 0.59;
        *Kb = 0.11;
        break;
    case ColorMatrix::bt470bg:
        *Kr = 0.299;
        *Kg = 0.587;
        *Kb = 0.114;
        break;
    case ColorMatrix::smpte170m:
        *Kr = 0.299;
        *Kg = 0.587;
        *Kb = 0.114;
        break;
    case ColorMatrix::smpte240m:
        *Kr = 0.212;
        *Kg = 0.701;
        *Kb = 0.087;
        break;
    case ColorMatrix::YCgCo:
        *Kr = 0.25;
        *Kg = 0.50;
        *Kb = 0.25;
        break;
    case ColorMatrix::bt2020nc:
        *Kr = 0.2627;
        *Kg = 0.6780;
        *Kb = 0.0593;
        break;
    case ColorMatrix::bt2020c:
        *Kr = 0.2627;
        *Kg = 0.6780;
        *Kb = 0.0593;
        break;
    default:
        *Kr = 0.2126;
        *Kg = 0.7152;
        *Kb = 0.0722;
        break;
    }

    return STAT::OK;
}