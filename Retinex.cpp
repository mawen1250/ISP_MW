#include <iostream>
#include <cmath>
#include <vector>
#include "include\Retinex.h"
#include "include\Gaussian.h"
#include "include\Histogram.h"
#include "include\Specification.h"
#include "include\Image_Type.h"
#include "include\IO.h"


int Retinex_MSR_IO(const int argc, const std::string * args)
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
    string Tag = ".Retinex_MSR";
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
    Frame PFrame = Retinex_MSR(SFrame, sigmaVector, lower_thr, upper_thr);

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

    double B, B1, B2, B3;
    Recursive_Gaussian_Parameters(sigma, B, B1, B2, B3);

    Plane_FL data(input);
    Plane_FL gauss(data, false);

    Recursive_Gaussian2D_Horizontal(gauss, data, B, B1, B2, B3);
    Recursive_Gaussian2D_Vertical(gauss, B, B1, B2, B3);

    for (i = 0; i < pcount; i++)
    {
        data[i] = gauss[i] < DBL_MIN ? 0 : data[i] / gauss[i];
    }
    
    FLType min, max;
    data.MinMax(min, max);
    data.ReQuantize(min, min, max, false);
    Histogram<FLType> Histogram(data, Retinex_Default.HistBins);
    min = Histogram.Min(lower_thr);
    max = Histogram.Max(upper_thr);
    data.ReQuantize(min, min, max, false);

    FLType gain = output.ValueRange() / (max - min);
    FLType offset = output.Floor() - min*gain + FLType(0.5);

    for (i = 0; i < pcount; i++)
    {
        output[i] = static_cast<DType>(data.Quantize(data[i]) * gain + offset);
    }
    
    return output;
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

    PCType i;
    PCType pcount = input.PixelCount();

    Plane_FL idata(input);
    Plane_FL odata(idata, true, 0);
    Plane_FL gauss(idata, false);

    double B, B1, B2, B3;

    for (s = 0; s < scount; s++)
    {
        if (sigmaVector[s] > 0)
        {
            Recursive_Gaussian_Parameters(sigmaVector[s], B, B1, B2, B3);
            Recursive_Gaussian2D_Horizontal(gauss, idata, B, B1, B2, B3);
            Recursive_Gaussian2D_Vertical(gauss, B, B1, B2, B3);

            for (i = 0; i < pcount; i++)
            {
                odata[i] += gauss[i] < DBL_MIN ? 0 : idata[i] / gauss[i];
            }
        }
        else
        {
            for (i = 0; i < pcount; i++)
            {
                odata[i] += 1;
            }
        }
    }

    for (i = 0; i < pcount; i++)
    {
        odata[i] /= FLType(scount);
    }

    FLType min, max;
    odata.MinMax(min, max);
    odata.ReQuantize(min, min, max, false);
    Histogram<FLType> Histogram(odata, Retinex_Default.HistBins);
    min = Histogram.Min(lower_thr);
    max = Histogram.Max(upper_thr);
    odata.ReQuantize(min, min, max, false);

    FLType gain = output.ValueRange() / (max - min);
    FLType offset = output.Floor() - min*gain + FLType(0.5);

    for (i = 0; i < pcount; i++)
    {
        output[i] = static_cast<DType>(odata.Quantize(odata[i]) * gain + offset);
    }

    return output;
}
