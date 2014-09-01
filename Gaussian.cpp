#include <iostream>
#include <cmath>
#include "include\Gaussian.h"
#include "include\Image_Type.h"
#include "include\IO.h"


int Gaussian2D_IO(int argc, char ** argv)
{
    using namespace std;
    using namespace mw;

    int i;
    int Flag = 0;

    char * Drive = new char[DRIVELEN];
    char * Dir = new char[PATHLEN];
    char * FileName = new char[PATHLEN];
    char * Ext = new char[EXTLEN];

    // Default Parameters
    double sigma = 1.0;
    string Tag = ".Gaussian";
    string Format = ".png";

    // Arguments Process
    if (argc <= 1)
    {
        return 0;
    }

    string * args = new string[argc];
    for (i = 0; i < argc; i++)
    {
        args[i] = argv[i];
    }

    for (i = 1; i < argc; i++)
    {
        if (args[i] == "-T" || args[i] == "--tag")
        {
            Flag |= args2arg(i, argc, args, Tag);
            continue;
        }
        if (args[i] == "-F" || args[i] == "--format")
        {
            Flag |= args2arg(i, argc, args, Format);
            continue;
        }
        if (args[i] == "-S" || args[i] == "--sigma")
        {
            Flag |= args2arg(i, argc, args, sigma);
            continue;
        }

        if (Flag)
        {
            return Flag;
        }
        
        Frame_RGB SFrame = ImageReader(args[i]);
        Frame_RGB PFrame = Gaussian2D(SFrame, sigma);

        _splitpath_s(argv[i], Drive, PATHLEN, Dir, PATHLEN, FileName, PATHLEN, Ext, PATHLEN);
        string OPath = string(Drive) + string(Dir) + string(FileName) + Tag + Format;

        ImageWriter(PFrame, OPath);
    }

    // Clean
    delete[] args;
    delete[] Drive, Dir, FileName, Ext;

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

    PCType i;

    double B, B1, B2, B3;
    Recursive_Gaussian_Parameters(sigma, B, B1, B2, B3);

    const PCType pcount = input.PixelCount();

    FLType * data = new FLType[pcount];
    for (i = 0; i < pcount; i++)
    {
        data[i] = (FLType)input[i];
    }

    Recursive_Gaussian2D_Horizontal(data, input.Width(), input.Height(), B, B1, B2, B3);
    Recursive_Gaussian2D_Vertical(data, input.Width(), input.Height(), B, B1, B2, B3);

    for (i = 0; i < pcount; i++)
    {
        output[i] = Quantize(data[i], output);
    }
    
    delete[] data;

    return output;
}


void Recursive_Gaussian_Parameters(const double sigma, double & B, double & B1, double & B2, double & B3)
{
    const double q = sigma < 2.5 ? 3.97156 - 4.14554*sqrt(1 - 0.26891*sigma) : 0.98711*sigma - 0.96330;

    const double b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
    const double b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
    const double b2 = -(1.4281*q*q + 1.26661*q*q*q);
    const double b3 = 0.422205*q*q*q;

    B = 1 - (b1 + b2 + b3) / b0;
    B1 = b1 / b0;
    B2 = b2 / b0;
    B3 = b3 / b0;
}

void Recursive_Gaussian2D_Vertical(FLType * data, const PCType sw, const PCType sh, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    PCType i, j, lower, upper;
    PCType pcount = sw*sh;
    FLType P1, P2, P3;

    for (j = 0; j < sw; j++)
    {
        lower = j;
        upper = pcount;

        i = lower;
        P3 = P2 = P1 = data[i];

        for (i += sw; i < upper; i += sw)
        {
            data[i] = B*data[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = data[i];
        }

        i -= sw;
        P3 = P2 = P1 = data[i];

        for (i -= sw; i >= lower; i -= sw)
        {
            data[i] = B*data[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = data[i];
        }
    }
}

void Recursive_Gaussian2D_Horizontal(FLType * data, const PCType sw, const PCType sh, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    PCType i, j, lower, upper;
    FLType P1, P2, P3;

    for (j = 0; j < sh; j++)
    {
        lower = sw*j;
        upper = lower + sw;

        i = lower;
        P3 = P2 = P1 = data[i];

        for (i++; i < upper; i++)
        {
            data[i] = B*data[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = data[i];
        }

        i--;
        P3 = P2 = P1 = data[i];

        for (i--; i >= lower; i--)
        {
            data[i] = B*data[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = data[i];
        }
    }
}

