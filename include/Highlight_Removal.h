#ifndef HIGHLIGHT_REMOVAL_H
#define HIGHLIGHT_REMOVAL_H


#include "Image_Type.h"


const struct Highlight_Removal_Default {
    const double thr = 0.03;
    const double sigmaS = 3.0;
    const double sigmaR = 0.1;
    const DType PBFICnum = 8;
} Highlight_Removal_Default;


Frame & Specular_Highlight_Removal(Frame & output, const Frame & input, const double thr = Highlight_Removal_Default.thr,
    const double sigmaS = Highlight_Removal_Default.sigmaS, const double sigmaR = Highlight_Removal_Default.sigmaR, const DType PBFICnum = Highlight_Removal_Default.PBFICnum);
inline Frame Specular_Highlight_Removal(const Frame & input, const double thr = Highlight_Removal_Default.thr,
    const double sigmaS = Highlight_Removal_Default.sigmaS, const double sigmaR = Highlight_Removal_Default.sigmaR, const DType PBFICnum = Highlight_Removal_Default.PBFICnum)
{
    Frame output(input, false);
    return Specular_Highlight_Removal(output, input, thr, sigmaS, sigmaR, PBFICnum);
}


#endif