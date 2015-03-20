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
        srcY.YFrom(src);

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

    ilinear.ConvertFrom(src, TransferChar::linear);

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
    DType i;
    FLType data, idata;
    FLType k0, phi, alpha, power, div;
    FLType Const1, Const2;
    TransferChar srcTransferChar = src.GetTransferChar();
    LUT<FLType> LUT_Gain(src);
    
    if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
    {
        TransferChar_Parameter(srcTransferChar, k0, div);
    }
    else if (srcTransferChar != TransferChar::linear)
    {
        TransferChar_Parameter(srcTransferChar, k0, phi, alpha, power);
    }

    if (Beta == 0)
    {
        LUT_Gain.SetRange(src, 1);
    }
    else
    {
        Const1 = 1 / (1 + exp(Beta*Alpha));
        Const2 = 1 / (1 + exp(Beta*(Alpha - 1))) - Const1;
        
        for (i = src.Floor(); i <= src.Ceil(); i++)
        {
            data = idata = src.GetFL(i);

            if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
            {
                data = TransferChar_gamma2linear(data, k0, div);
            }
            else if (srcTransferChar != TransferChar::linear)
            {
                data = TransferChar_gamma2linear(data, k0, phi, alpha, power);
            }

            if (Beta > 0)
                data = (1 / (1 + exp(Beta*(Alpha - data))) - Const1) / Const2;
            else
                data = Alpha - log(1 / (Const2*data + Const1) - 1) / Beta;

            if (srcTransferChar == TransferChar::log100 || srcTransferChar == TransferChar::log316)
            {
                data = TransferChar_linear2gamma(data, k0, div);
            }
            else if (srcTransferChar != TransferChar::linear)
            {
                data = TransferChar_linear2gamma(data, k0, phi, alpha, power);
            }

            LUT_Gain.Set(src, i, idata == 0 ? 1 : data / idata);
        }
    }

    // Output
    return LUT_Gain;
}
