#include <cmath>
#include "Highlight_Removal.h"
#include "Bilateral.h"


// Implementation of specular highilght removal algorithm from "Qingxiong Yang, ShengnanWang, and Narendra Ahuja - Real-Time Specular Highlight Removal Using Bilateral Filtering"
Frame &Specular_Highlight_Removal(Frame &dst, const Frame &src, const double thr, const double sigmaS, const double sigmaR, const DType PBFICnum)
{
    if (!(src.isRGB() && dst.isRGB()))
    {
        DEBUG_FAIL("Specular_Highlight_Removal: only Frame of RGB PixelType is supported.");
    }

    const Plane &srcR = src.R();
    const Plane &srcG = src.G();
    const Plane &srcB = src.B();
    Plane &dstR = dst.R();
    Plane &dstG = dst.G();
    Plane &dstB = dst.B();

    PCType i;
    PCType pcount = srcR.PixelCount();
    DType ValueRange = srcR.ValueRange();

    DType R, G, B;
    FLType RGBsum, sigma_R, sigma_G, sigma_B, sigmaMax, sigmaMin;
    Plane PsigmaMax(srcR, false);
    Plane PlambdaMax(srcR, false);
    Plane PsigmaMaxF(srcR, false);

    PsigmaMax.ReQuantize(srcR.BitDepth(), 0, 0, ValueRange);
    PlambdaMax.ReQuantize(srcR.BitDepth(), 0, 0, ValueRange);
    PsigmaMaxF.ReQuantize(srcR.BitDepth(), 0, 0, ValueRange);

    DType diffthr = static_cast<DType>(thr * ValueRange + 0.5);
    FLType specular_component;

    // Compute σ_max at every pixel using the src image and store it as a grayscale image.
    // Compute λ_max at every pixel using the src image and store it as a grayscale image.
    for (i = 0; i < pcount; i++)
    {
        R = srcR[i];
        G = srcG[i];
        B = srcB[i];

        RGBsum = (FLType)(R + G + B);

        if (RGBsum == 0)
        {
            sigma_R = sigma_G = sigma_B = FLType(1. / 3.);
        }
        else
        {
            sigma_R = (FLType)R / RGBsum;
            sigma_G = (FLType)G / RGBsum;
            sigma_B = (FLType)B / RGBsum;
        }

        sigmaMax = Max(Max(sigma_R, sigma_G), sigma_B);
        sigmaMin = Min(Min(sigma_R, sigma_G), sigma_B);

        PsigmaMax[i] = (DType)(sigmaMax * ValueRange + 0.5);
        PlambdaMax[i] = sigmaMax == sigmaMin ? 0 : (DType)((sigmaMax - sigmaMin) / (1 - 3 * sigmaMin) * ValueRange + 0.5);
    }

    // repeat until σ_maxF − σ_max < 0.03 at every pixel.
    PCType flag = 1;

    while (flag)
    {
        // Apply joint bilateral filter to image σmax using λ_max as the guidance image, store the filtered image as σF σ_maxF
        Bilateral2D_Para blPara;
        blPara.sigmaS = sigmaS;
        blPara.sigmaR = sigmaR;
        Bilateral2D_Data blData(PlambdaMax, blPara);
        Bilateral2D(PsigmaMaxF, PsigmaMax, PlambdaMax, blData);

        // For each pixel p, σ_max(p) = max(σ_max(p), σ_maxF(p))
        for (i = 0, flag = 0; i < pcount; i++)
        {
            if (PsigmaMaxF[i] > PsigmaMax[i])
            {
                if (PsigmaMaxF[i] - PsigmaMax[i] > diffthr)
                {
                    flag++;
                }

                PsigmaMax[i] = PsigmaMaxF[i];
            }
        }
    }

    // Compute diffuse component
    for (i = 0; i < pcount; i++)
    {
        R = srcR[i];
        G = srcG[i];
        B = srcB[i];
        sigmaMax = (FLType)PsigmaMax[i] / ValueRange;

        if (PsigmaMax[i] * 3 <= ValueRange)
        {
            dstR[i] = R;
            dstG[i] = G;
            dstB[i] = B;
        }
        else
        {
            specular_component = (Max(Max(R, G), B) - sigmaMax*(R + G + B)) / (1 - 3 * sigmaMax);
            dstR[i] = dstR.Quantize(R - specular_component);
            dstG[i] = dstG.Quantize(G - specular_component);
            dstB[i] = dstB.Quantize(B - specular_component);
        }
        /*dst.R()[i] = PsigmaMax[i];
        dst.G()[i] = PsigmaMax[i];
        dst.B()[i] = PsigmaMax[i];*/
        /*dst.R()[i] = PlambdaMax[i];
        dst.G()[i] = PlambdaMax[i];
        dst.B()[i] = PlambdaMax[i];*/
    }

    // Output
    return dst;
}
