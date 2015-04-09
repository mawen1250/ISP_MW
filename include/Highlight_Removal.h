#ifndef HIGHLIGHT_REMOVAL_H
#define HIGHLIGHT_REMOVAL_H


#include "Filter.h"
#include "Image_Type.h"


const struct Highlight_Removal_Para
{
    double thr = 0.03;
    double sigmaS = 3.0;
    double sigmaR = 0.1;
    DType PBFICnum = 0;
} Highlight_Removal_Default;


Frame &Specular_Highlight_Removal(Frame &dst, const Frame &src, const double thr = Highlight_Removal_Default.thr,
    const double sigmaS = Highlight_Removal_Default.sigmaS, const double sigmaR = Highlight_Removal_Default.sigmaR, const DType PBFICnum = Highlight_Removal_Default.PBFICnum);
inline Frame Specular_Highlight_Removal(const Frame &src, const double thr = Highlight_Removal_Default.thr,
    const double sigmaS = Highlight_Removal_Default.sigmaS, const double sigmaR = Highlight_Removal_Default.sigmaR, const DType PBFICnum = Highlight_Removal_Default.PBFICnum)
{
    Frame dst(src, false);
    return Specular_Highlight_Removal(dst, src, thr, sigmaS, sigmaR, PBFICnum);
}


#endif
