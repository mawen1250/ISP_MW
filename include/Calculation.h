#ifndef CALCULATION_H_
#define CALCULATION_H_


#include "Type.h"
#include "Type_Conv.h"


template <typename T>
inline T Spline2(T x, T x1, T y1, T x2, T y2)
{
    return (y2-y1) / (x2-x1) * (x-x1) + y1;
}


template <typename T>
T limit_dif_abs(T value, T ref, bool smooth, T dthr, T uthr, T elast)
{
    T diff, abdiff, thr, alpha, beta, output;
    
    dthr = Max(dthr, (T)0.0);
    uthr = Max(uthr, (T)0.0);
    elast = Max(elast, (T)1.0);
    smooth = elast==1 ? false : smooth;
    
    diff     = value - ref;
    abdiff   = Abs(diff);
    thr      = diff > 0 ? uthr : dthr;
    alpha    = 1. / (thr * (elast - 1));
    beta     = elast * thr;
    
    if(smooth) {
        output   = abdiff <= thr  ? value :
                   abdiff >= beta ? ref   :
                                    ref + alpha * diff * (beta - abdiff);
    } else {
        output   = abdiff <= thr  ? value :
                   abdiff >= beta ? ref   :
                                    ref + thr * (diff / abdiff);
    }
    
    return output;
}


template <typename T>
T limit_dif_ratio(T value, T ref, bool smooth, T dthr, T uthr, T elast)
{
    T ratio, abratio, negative, inverse, nratio, thr, lratio, output;
    
    dthr = Max(dthr, (T)0.0);
    uthr = Max(uthr, (T)0.0);
    elast = Max(elast, (T)1.0);
    smooth   = elast==1 ? false : smooth;
    
    ratio    = value / ref;
    abratio  = Abs(ratio);
    negative = ratio*abratio < 0;
    inverse  = abratio < 1;
    nratio   = inverse ? 1./abratio : abratio;
    thr      = inverse ? dthr : uthr;
    thr      = thr < 1 ? 1./thr-1 : thr-1;
    
    lratio   = limit_dif_abs(nratio, 1., smooth, thr, thr, elast);
    lratio   = inverse ? 1./lratio : lratio;
    output   = negative ? -lratio*ref : lratio*ref;
    
    return output;
}


template <typename T>
T damp_ratio(T ratio, T damp)
{
    T abratio, negative, inverse, nratio, dratio, output;
    
    abratio  = Abs(ratio);
    negative = ratio*abratio < 0;
    inverse  = abratio < 1;
    nratio   = inverse ? 1./abratio : abratio;
    
    dratio   = (nratio-1)*damp+1;
    dratio   = inverse ? 1./dratio : dratio;
    output   = negative ? -dratio : dratio;
    
    return output;
}


#endif