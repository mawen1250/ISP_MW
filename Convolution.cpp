#include <iostream>
#include "include/Convolution.h"
#include "include/Image_Type.h"

 
Plane & Convolution3V(Plane & Plane, bool norm, float K0, float K1, float K2)
{
    PCType i, j, k, l, m, n;
    PCType length = 1;
    float sum;
    float P0=0, P1=0, P2=0;
    DType * index, * upper;

    if(norm)
    {
        sum = K0 + K1 + K2;
        if(sum!=0)
        {
            K0 /= sum;
            K1 /= sum;
            K2 /= sum;
        }
    }
    
    j = length * 2;
    k = Plane.Width();
    l = Plane.Height();
    m = Plane.PixelCount() - 1;
    n = Plane.Width() * length;
    
    for (i = 0, index = Plane.Data(), upper = Plane.Data() + Plane.Width();;)
    {
        P0 = P1;
        P1 = P2;
        P2 = (float)(*index);
        
        if(i<j)
        {
            i++;
            index  += k;
        }
        else
        {
            *(index-n) = Quantize(K0*P0 + K1*P1 + K2*P2, Plane);
            
            i++;
            index  += k;

            if(i>=l)
            {
                i = 0;
                index  -= m;
                if(index>=upper)
                {
                    break;
                }
            }
        }
    }

    return Plane;
}


Plane & Convolution3H(Plane & Plane, bool norm, float K0, float K1, float K2)
{
    PCType i, j, k;
    PCType length = 1;
    float sum;
    float P0=0, P1=0, P2=0;
    DType * index, * upper;

    if(norm)
    {
        sum = K0 + K1 + K2;
        if(sum!=0)
        {
            K0 /= sum;
            K1 /= sum;
            K2 /= sum;
        }
    }
    
    j = length * 2;
    k = Plane.Width();
    
    for (i = 0, index = Plane.Data(), upper = Plane.Data() + Plane.PixelCount();;)
    {
        P0 = P1;
        P1 = P2;
        P2 = (float)(*index);
        
        if(i<j)
        {
            i++;
            index++;
        }
        else
        {
            *(index-length) = Quantize(K0*P0 + K1*P1 + K2*P2, Plane);
            
            i++;
            index++;

            if(i>=k)
            {
                i = 0;
                if(index>=upper)
                {
                    break;
                }
            }
        }
    }

    return Plane;
}


Plane & Convolution3(Plane & Plane, bool norm, float K0, float K1, float K2, float K3, float K4, float K5, float K6, float K7, float K8)
{
    PCType i, j, k;
    PCType length = 1;
    float sum;
    float P0=0, P1=0, P2=0, P3=0, P4=0, P5=0, P6=0, P7=0, P8=0;
    DType * index0, * index1, * upper1, * index2;

    DType * pp = new DType[Plane.PixelCount()];

    if(norm)
    {
        sum = K0 + K1 + K2 + K3 + K4 + K5 + K6 + K7 + K8;
        if(sum!=0)
        {
            K0 /= sum;
            K1 /= sum;
            K2 /= sum;
            K3 /= sum;
            K4 /= sum;
            K5 /= sum;
            K6 /= sum;
            K7 /= sum;
            K8 /= sum;
        }
    }
    
    memcpy(pp, Plane.Data(), Plane.PixelCount() * sizeof(DType));

    j = length * 2;
    k = Plane.Width();
    
    for (i = 0, index0 = pp, index1 = Plane.Data() + Plane.Width(), index2 = pp + Plane.Width() * 2, upper1 = Plane.Data() + Plane.PixelCount() - Plane.Width()*length;;)
    {
        P0 = P1; P1 = P2; P2 = (float)(*index0);
        P3 = P4; P4 = P5; P5 = (float)(*index1);
        P6 = P7; P7 = P8; P8 = (float)(*index2);
        
        if(i<j)
        {
            i++;
            index0++;
            index1++;
            index2++;
        }
        else
        {
            *(index1-length) = Quantize(K0*P0 + K1*P1 + K2*P2 + K3*P3 + K4*P4 + K5*P5 + K6*P6 + K7*P7 + K8*P8, Plane);
            
            i++;
            index0++;
            index1++;
            index2++;

            if(i>=k)
            {
                i = 0;
                if(index1>=upper1)
                {
                    break;
                }
            }
        }
    }

    delete[] pp;

    return Plane;
}