#include <cmath>
#include "Tone_Mapping.h"
#include "Helper.h"
#include "Specification.h"


Frame &Adaptive_Global_Tone_Mapping(Frame &dst, const Frame &src)
{
    if (src.isYUV())
    {
        const Plane &srcY = src.Y();

        LUT<FLType> _LUT = Adaptive_Global_Tone_Mapping_Gain_LUT_Generation(srcY);

        _LUT.Lookup_Gain(dst, src, srcY);
    }
    else if (src.isRGB())
    {
        Plane srcY(src.R(), false);
        ConvertToY(srcY, src, ColorMatrix::Average);

        LUT<FLType> _LUT = Adaptive_Global_Tone_Mapping_Gain_LUT_Generation(srcY);

        _LUT.Lookup_Gain(dst, src, srcY);
    }

    return dst;
}


LUT<FLType> Adaptive_Global_Tone_Mapping_Gain_LUT_Generation(const Plane &src)
{
    PCType pcount = src.PixelCount();

    // Convert src plane to linear scale
    Plane ilinear(src, false);

    TransferConvert(ilinear, src, TransferChar::linear);

    // Generate histogram of linear scale
    Histogram<DType>::BinType HistogramBins = Min(static_cast<Histogram<DType>::BinType>(src.ValueRange() + 1), AGTM_Default.HistBins);

    Histogram<DType> Histogram(ilinear, HistogramBins);

    // Calculate parameters for sigmoidal curve
    PCType thr0l = (pcount * 20 + 32) / 64;
    PCType thr0h = (pcount * 40 + 32) / 64;
    PCType thr0hh = (pcount * 50 + 32) / 64;
    PCType thr0hhh = (pcount * 60 + 32) / 64;
    PCType thr1l = (pcount * 10 + 32) / 64;
    PCType thr1h = (pcount * 30 + 32) / 64;
    PCType thr6l = (pcount * 1 + 32) / 64;
    PCType thr7l = (pcount * 1 + 64) / 128;
    PCType thr01l = (pcount * 8 + 32) / 64;
    PCType thr01h = (pcount * 28 + 32) / 64;
    PCType thr01hh = (pcount * 48 + 32) / 64;
    PCType thr23ll = (pcount * 3 + 32) / 64;
    PCType thr23l = (pcount * 10 + 32) / 64;
    PCType thr23h = (pcount * 14 + 32) / 64;
    PCType thr45ll = (pcount * 1 + 32) / 64;
    PCType thr45l = (pcount * 6 + 32) / 64;
    PCType thr45h = (pcount * 12 + 32) / 64;
    PCType thr67l = (pcount * 2 + 32) / 64;
    PCType thr67h = (pcount * 6 + 32) / 64;
    PCType thr67hh = (pcount * 12 + 32) / 64;
    PCType thr67hhh = (pcount * 24 + 32) / 64;

    FLType Alpha, Beta;
    PCType H0 = Histogram[0];
    PCType H1 = Histogram[1];
    PCType H6 = Histogram[6];
    PCType H7 = Histogram[7];
    PCType H01 = Histogram[0] + Histogram[1];
    PCType H23 = Histogram[2] + Histogram[3];
    PCType H45 = Histogram[4] + Histogram[5];
    PCType H67 = Histogram[6] + Histogram[7];

    //std::cout << H0 * 64. / pcount << " " << H1 * 64. / pcount << " " << H6 * 64. / pcount << " " << H7 * 64. / pcount << std::endl;
    //std::cout << H01 * 64. / pcount << " " << H23 * 64. / pcount << " " << H45 * 64. / pcount << " " << H67 * 64. / pcount << std::endl;

    if (H0 >= thr0h && H1 < thr1l && H45 < thr45l && H6 < thr6l && H7 < thr7l) // Dark scene
    {
        if (H0 >= thr0hhh) // Dark scene 3
        {
            Alpha = 1.0;
            Beta = -4.5;
            std::cout << "Mode 1-3\n";
        }
        else if (H0 >= thr0hh) // Dark scene 2
        {
            Alpha = 0.8;
            Beta = -4.0;
            std::cout << "Mode 1-2\n";
        }
        else // Dark scene 1
        {
            Alpha = 0.6;
            Beta = -3.5;
            std::cout << "Mode 1-1\n";
        }
    }
    else if (H01 < thr01l && H45 + H67 >= thr45h + thr67h) // Bright scene
    {
        if (H67 >= thr67hhh) // Bright scene 3
        {
            Alpha = 1.0;
            Beta = 4.5;
            std::cout << "Mode 4-3\n";
        }
        else if (H67 >= thr67hh) // Bright scene 2
        {
            Alpha = 0.9;
            Beta = 4.0;
            std::cout << "Mode 4-2\n";
        }
        else // Bright scene 1
        {
            Alpha = 0.8;
            Beta = 3.5;
            std::cout << "Mode 4-1\n";
        }
    }
    else if (H67 < thr67l) // Little bright light
    {
        if (H01 < thr01l && H7 < thr7l) // Little dark area - contrast enhancement
        {
            Alpha = 0.4;
            Beta = 3;
            std::cout << "Mode 2-1\n";
        }
        else if (H01 >= thr01h && H67 < thr67l) // Mainly dark area and no bright light
        {
            Alpha = 0.55;
            Beta = -3;
            std::cout << "Mode 2-3\n";
        }
        else
        {
            Alpha = 0.4;
            Beta = -3;
            std::cout << "Mode 2-2\n";
        }
    }
    else if (H01 >= thr01l && H7 >= thr7l) // Dark area and bright light
    {
        if (H67 >= thr67hh) // Bright scene 2
        {
            Alpha = 0.35;
            Beta = -5;
            std::cout << "Mode 3-5\n";
        }
        else if (H67 >= thr67h) // Bright scene 1
        {
            Alpha = 0.35;
            Beta = -4.5;
            std::cout << "Mode 3-4\n";
        }
        else if (H01 >= thr01hh) // Dark scene 2
        {
            Alpha = 0.4;
            Beta = -5;
            std::cout << "Mode 3-1\n";
        }
        else if (H01 >= thr01h) // Dark scene 1
        {
            Alpha = 0.4;
            Beta = -4.5;
            std::cout << "Mode 3-2\n";
        }
        else // Normal
        {
            Alpha = 0.375;
            Beta = -4.5;
            std::cout << "Mode 3-3\n";
        }
    }
    else
    {
        Alpha = 0.5;
        Beta = 0;
        std::cout << "Mode 0\n";
    }

    // Generate LUT for tone mapping
    LUT<FLType> LUT_Gain(src);
    
    if (Beta == 0)
    {
        LUT_Gain.SetRange(src, 1);
    }
    else
    {
        TransferChar_Conv<FLType> ToLinear(TransferChar::linear, src.GetTransferChar());
        TransferChar_Conv<FLType> LinearTo(src.GetTransferChar(), TransferChar::linear);

        const FLType Const1 = 1 / (1 + exp(Beta * Alpha));
        const FLType Const2 = 1 / (1 + exp(Beta * (Alpha - 1))) - Const1;
        
        LUT_Gain.Set(src, [&](DType i)
        {
            FLType x = src.GetFL(i);

            FLType y = ToLinear(x);

            if (Beta > 0)
            {
                y = (1 / (1 + exp(Beta * (Alpha - y))) - Const1) / Const2;
            }
            else
            {
                y = Alpha - log(1 / (Const2 * y + Const1) - 1) / Beta;
            }

            y = LinearTo(y);

            return x == 0 ? 1 : y / x;
        });
    }

    // Output
    return LUT_Gain;
}
