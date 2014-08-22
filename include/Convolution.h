#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_


#include "Image_Type.h"


Plane & Convolution3V(Plane & Plane, bool norm, float K0, float K1, float K2);

Plane & Convolution3H(Plane & Plane, bool norm, float K0, float K1, float K2);

Plane & Convolution3(Plane & Plane, bool norm, float K0, float K1, float K2, float K3, float K4, float K5, float K6, float K7, float K8);


#endif