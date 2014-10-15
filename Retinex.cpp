#include <iostream>
#include <cmath>
#include "include\Retinex.h"
#include "include\Gaussian.h"
#include "include\Histogram.h"
#include "include\Specification.h"


int Retinex_MSRCP_IO(const int argc, const std::vector<std::string> &args)
{
    using namespace std;
    using namespace mw;

    int i;
    int Flag = 0;

    char Drive[DRIVELEN];
    char Dir[PATHLEN];
    char FileName[PATHLEN];
    char Ext[EXTLEN];

    // Default Parameters
    string IPath;
    double sigma;
    std::vector<double> sigmaVector;
    double lower_thr = Retinex_Default.lower_thr;
    double upper_thr = Retinex_Default.upper_thr;
    string Tag = ".Retinex_MSRCP";
    string Format = ".png";

    // Arguments Process
    for (i = 0; i < argc; i++)
    {
        if (args[i] == "-T" || args[i] == "--tag")
        {
            Flag |= arg2para(i, argc, args, Tag);
            continue;
        }
        if (args[i] == "-F" || args[i] == "--format")
        {
            Flag |= arg2para(i, argc, args, Format);
            continue;
        }
        if (args[i] == "-S" || args[i] == "--sigma")
        {
            Flag |= arg2para(i, argc, args, sigma);
            sigmaVector.push_back(sigma);
            continue;
        }
        if (args[i] == "-L" || args[i] == "--lower_thr")
        {
            Flag |= arg2para(i, argc, args, lower_thr);
            continue;
        }
        if (args[i] == "-U" || args[i] == "--upper_thr")
        {
            Flag |= arg2para(i, argc, args, upper_thr);
            continue;
        }

        IPath = args[i];
    }

    if (sigmaVector.size() <= 0)
    {
        sigmaVector = Retinex_Default.sigmaVector;
    }

    Frame SFrame = ImageReader(IPath);
    Frame PFrame = Retinex_MSRCP(SFrame, sigmaVector, lower_thr, upper_thr);

    _splitpath_s(IPath.c_str(), Drive, PATHLEN, Dir, PATHLEN, FileName, PATHLEN, Ext, PATHLEN);
    string OPath = string(Drive) + string(Dir) + string(FileName) + Tag + Format;

    ImageWriter(PFrame, OPath);

    return 0;
}


Plane & Retinex_SSR(Plane & output, const Plane & input, const double sigma, const double lower_thr, const double upper_thr)
{
    if (sigma <= 0)
    {
        output = input;
        return output;
    }

    PCType i;
    PCType pcount = input.PixelCount();

    FLType B, B1, B2, B3;
    Recursive_Gaussian_Parameters(sigma, B, B1, B2, B3);

    Plane_FL data(input);
    Plane_FL gauss(data, false);

    Recursive_Gaussian2D_Horizontal(gauss, data, B, B1, B2, B3);
    Recursive_Gaussian2D_Vertical(gauss, B, B1, B2, B3);

    for (i = 0; i < pcount; i++)
    {
        data[i] = gauss[i] <= 0 ? 0 : log(data[i] / gauss[i] + 1);
    }
    
    FLType min, max;
    data.MinMax(min, max);

    if (max <= min)
    {
        output = input;
        return output;
    }

    if (lower_thr> 0 || upper_thr > 0)
    {
        data.ReQuantize(min, min, max, false);
        Histogram<FLType> Histogram(data, Retinex_Default.HistBins);
        min = Histogram.Min(lower_thr);
        max = Histogram.Max(upper_thr);
    }

    data.ReQuantize(min, min, max, false);

    FLType gain = output.ValueRange() / (max - min);
    FLType offset = output.Floor() - min*gain + FLType(0.5);

    for (i = 0; i < pcount; i++)
    {
        output[i] = static_cast<DType>(data.Quantize(data[i]) * gain + offset);
    }
    
    return output;
}


Plane_FL Retinex_MSR(const Plane_FL & idata, const std::vector<double> & sigmaVector, const double lower_thr, const double upper_thr)
{
    size_t s, scount = sigmaVector.size();

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        Plane_FL odata(idata);
        return odata;
    }

    PCType i;
    PCType pcount = idata.PixelCount();

    Plane_FL odata(idata, true, 1);
    Plane_FL gauss(idata, false);

    FLType B, B1, B2, B3;

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0)
        {
            Recursive_Gaussian_Parameters(sigmaVector[s], B, B1, B2, B3);
            Recursive_Gaussian2D_Horizontal(gauss, idata, B, B1, B2, B3);
            Recursive_Gaussian2D_Vertical(gauss, B, B1, B2, B3);

            for (i = 0; i < pcount; i++)
            {
                odata[i] *= gauss[i] <= 0 ? 1 : idata[i] / gauss[i] + 1;
            }
        }
        else
        {
            for (i = 0; i < pcount; i++)
            {
                odata[i] *= FLType(2);
            }
        }
    }

    for (i = 0; i < pcount; i++)
    {
        odata[i] = log(odata[i]) / static_cast<FLType>(scount);
    }

    FLType min, max;
    odata.MinMax(min, max);

    if (max <= min)
    {
        odata = idata;
        return odata;
    }

    if (lower_thr> 0 || upper_thr > 0)
    {
        odata.ReQuantize(min, min, max, false);
        Histogram<FLType> Histogram(odata, Retinex_Default.HistBins);
        min = Histogram.Min(lower_thr);
        max = Histogram.Max(upper_thr);
    }

    odata.ReQuantize(min, min, max, false);
    odata.ReQuantize(idata.Floor(), idata.Neutral(), idata.Ceil(), true, lower_thr> 0 || upper_thr > 0);

    return odata;
}

Plane & Retinex_MSR(Plane & output, const Plane & input, const std::vector<double> & sigmaVector, const double lower_thr, const double upper_thr)
{
    size_t s, scount = sigmaVector.size();

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        output = input;
        return output;
    }

    Plane_FL idata(input);
    Plane_FL odata = Retinex_MSR(idata, sigmaVector, lower_thr, upper_thr);

    odata.To(output);

    return output;
}

Frame & Retinex_MSRCP(Frame & output, const Frame & input, const std::vector<double> & sigmaVector, const double lower_thr, const double upper_thr)
{
    size_t s, scount = sigmaVector.size();

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0) break;
    }
    if (s >= scount)
    {
        output = input;
        return output;
    }

    PCType i;
    PCType pcount = input.PixelCount();

    if (input.isYUV())
    {
        const Plane & inputY = input.Y();
        const Plane & inputU = input.U();
        const Plane & inputV = input.V();
        Plane & outputY = output.Y();
        Plane & outputU = output.U();
        Plane & outputV = output.V();

        sint64 iNeutral = inputU.Neutral();
        FLType iRangeC2FL = static_cast<FLType>(inputU.ValueRange()) / 2.;

        Plane_FL idata(inputY);
        Plane_FL odata = Retinex_MSR(idata, sigmaVector, lower_thr, upper_thr);

        odata.To(outputY);

        FLType gain;

        if (outputU.isPCChroma())
        {
            for (i = 0; i < pcount; i++)
            {
                gain = Min(iRangeC2FL / Max(Abs(inputU[i] - iNeutral), Abs(inputV[i] - iNeutral)), idata[i] <= 0 ? 1 : odata[i] / idata[i]);
                outputU[i] = outputU.GetD_PCChroma(inputU.GetFL(inputU[i])*gain);
                outputV[i] = outputV.GetD_PCChroma(inputV.GetFL(inputV[i])*gain);
            }
        }
        else
        {
            for (i = 0; i < pcount; i++)
            {
                gain = Min(iRangeC2FL / Max(Abs(inputU[i] - iNeutral), Abs(inputV[i] - iNeutral)), idata[i] <= 0 ? 1 : odata[i] / idata[i]);
                outputU[i] = outputU.GetD(inputU.GetFL(inputU[i])*gain);
                outputV[i] = outputV.GetD(inputV.GetFL(inputV[i])*gain);
            }
        }
    }
    else if (input.isRGB())
    {
        const Plane & inputR = input.R();
        const Plane & inputG = input.G();
        const Plane & inputB = input.B();
        Plane & outputR = output.R();
        Plane & outputG = output.G();
        Plane & outputB = output.B();

        DType iRange = inputR.ValueRange();
        FLType iRangeFL = static_cast<FLType>(iRange);

        Plane_FL idata(inputR, false);
        idata.YFrom(input);
        Plane_FL odata = Retinex_MSR(idata, sigmaVector, lower_thr, upper_thr);

        FLType gain;

        for (i = 0; i < pcount; i++)
        {
            gain = Min(iRangeFL / Max(inputR[i], Max(inputG[i], inputB[i])), idata[i] <= 0 ? 1 : odata[i] / idata[i]);
            outputR[i] = outputR.GetD(inputR.GetFL(inputR[i])*gain);
            outputG[i] = outputG.GetD(inputG.GetFL(inputG[i])*gain);
            outputB[i] = outputB.GetD(inputB.GetFL(inputB[i])*gain);
        }
    }

    return output;
}
