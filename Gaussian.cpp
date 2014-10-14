#include <iostream>
#include <cmath>
#include "include\Gaussian.h"


int Gaussian2D_IO(const int argc, const std::vector<std::string> &args)
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
    double sigma = Gaussian2D_Default.sigma;
    string Tag = ".Gaussian";
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
            continue;
        }

        if (Flag)
        {
            return Flag;
        }

        IPath = args[i];
    }

    Frame SFrame = ImageReader(IPath);
    Frame PFrame = Gaussian2D(SFrame, sigma);

    _splitpath_s(IPath.c_str(), Drive, PATHLEN, Dir, PATHLEN, FileName, PATHLEN, Ext, PATHLEN);
    string OPath = string(Drive) + string(Dir) + string(FileName) + Tag + Format;

    ImageWriter(PFrame, OPath);

    return 0;
}


// Implementation of recursive Gaussian algorithm from "Ian T. Young, Lucas J. van Vliet - Recursive implementation of the Gaussian filter"
Plane & Gaussian2D(Plane & output, const Plane & input, const double sigma)
{
    if (sigma <= 0)
    {
        output = input;
        return output;
    }

    FLType B, B1, B2, B3;
    Recursive_Gaussian_Parameters(sigma, B, B1, B2, B3);

    Plane_FL data(input);

    Recursive_Gaussian2D_Horizontal(data, B, B1, B2, B3);
    Recursive_Gaussian2D_Vertical(data, B, B1, B2, B3);
    
    data.To(output);
    
    return output;
}


void Recursive_Gaussian_Parameters(const double sigma, FLType & B, FLType & B1, FLType & B2, FLType & B3)
{
    const double q = sigma < 2.5 ? 3.97156 - 4.14554*sqrt(1 - 0.26891*sigma) : 0.98711*sigma - 0.96330;

    const double b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
    const double b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
    const double b2 = -(1.4281*q*q + 1.26661*q*q*q);
    const double b3 = 0.422205*q*q*q;

    B = static_cast<FLType>(1 - (b1 + b2 + b3) / b0);
    B1 = static_cast<FLType>(b1 / b0);
    B2 = static_cast<FLType>(b2 / b0);
    B3 = static_cast<FLType>(b3 / b0);
}

void Recursive_Gaussian2D_Vertical(Plane_FL & output, const Plane_FL & input, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    PCType i, j, lower, upper;
    PCType sw = input.Width();
    PCType sh = input.Height();
    PCType pcount = sw*sh;
    FLType P0, P1, P2, P3;

    for (j = 0; j < sw; j++)
    {
        lower = j;
        upper = pcount;

        i = lower;
        output[i] = P3 = P2 = P1 = input[i];

        for (i += sw; i < upper; i += sw)
        {
            P0 = B*input[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        i -= sw;
        P3 = P2 = P1 = output[i];
        
        for (i -= sw; i >= lower; i -= sw)
        {
            P0 = B*output[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
    }
}

void Recursive_Gaussian2D_Horizontal(Plane_FL & output, const Plane_FL & input, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    PCType i, j, lower, upper;
    PCType sw = input.Width();
    PCType sh = input.Height();
    FLType P0, P1, P2, P3;

    for (j = 0; j < sh; j++)
    {
        lower = sw*j;
        upper = lower + sw;

        i = lower;
        output[i] = P3 = P2 = P1 = input[i];

        for (i++; i < upper; i++)
        {
            P0 = B*input[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        i--;
        P3 = P2 = P1 = output[i];

        for (i--; i >= lower; i--)
        {
            P0 = B*output[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
    }
}
