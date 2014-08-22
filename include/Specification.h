#ifndef SPECIFICATION_H_
#define SPECIFICATION_H_


#include "Image_Type.h"


STAT ColorPrim_Parameter(ColorPrim ColorPrim, double * green_x, double * green_y, double * blue_x, double * blue_y, double * red_x, double * red_y, double * white_x, double * white_y);


STAT Transfer_Parameter(TransferChar TransferChar, double * k0, double * phi, double * alpha, double * power, double * div);


STAT ColorMatrix_Parameter(ColorMatrix ColorMatrix, double * Kr, double * Kg, double * Kb);


#endif