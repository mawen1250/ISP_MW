#ifndef GAUSSIAN_CUH_
#define GAUSSIAN_CUH_


#include "Gaussian.h"


// Implementation of recursive Gaussian algorithm from "Ian T. Young, Lucas J. van Vliet - Recursive implementation of the Gaussian filter"
// For 32bit float type, the maximum sigma with valid result is about 120.
// CUDA version
class CUDA_RecursiveGaussian
    : public RecursiveGaussian
{
public:
    typedef CUDA_RecursiveGaussian _Myt;
    typedef RecursiveGaussian _Mybase;

public:
    _Myt(long double sigma)
        : _Mybase(sigma)
    {}


};


#endif
