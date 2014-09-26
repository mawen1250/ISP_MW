#include <iostream>
#include <cmath>
#include "include\Bilateral.h"
#include "include\Gaussian.h"
#include "include\LUT.h"
#include "include\Type_Conv.h"
#include "include\Image_Type.h"
#include "include\IO.h"


int Bilateral2D_IO(const int argc, const std::string * args)
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
    double sigmaS = Bilateral2D_Default.sigmaS;
    double sigmaR = Bilateral2D_Default.sigmaR;
    string Tag = ".Bilateral";
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
        if (args[i] == "-S" || args[i] == "--sigmaS")
        {
            Flag |= arg2para(i, argc, args, sigmaS);
            continue;
        }
        if (args[i] == "-R" || args[i] == "--sigmaR")
        {
            Flag |= arg2para(i, argc, args, sigmaR);
            continue;
        }
        
        IPath = args[i];
    }

    Frame SFrame = ImageReader(IPath);
    Frame PFrame = Bilateral2D(SFrame, sigmaS, sigmaR);

    _splitpath_s(IPath.c_str(), Drive, PATHLEN, Dir, PATHLEN, FileName, PATHLEN, Ext, PATHLEN);
    string OPath = string(Drive) + string(Dir) + string(FileName) + Tag + Format;

    ImageWriter(PFrame, OPath);

    return 0;
}


Plane & Bilateral2D(Plane & output, const Plane & input, const Plane & ref, const double sigmaS, const double sigmaR, const DType PBFICnum)
{
    // Skip processing if either sigma is not positive
    if (sigmaS <= 0 || sigmaR <= 0)
    {
        output = input;
        return output;
    }

    const PCType radius = Max((PCType)(sigmaS*sigmaSMul+0.5), (PCType)1);
    const PCType radiusx = radius, radiusy = radius;
    const PCType xUpper = radiusx + 1, yUpper = radiusy + 1;

    // Generate LUT
    LUT<FLType> GR_LUT = Gaussian_Function_Range_LUT_Generation(ref.ValueRange(), sigmaR*ref.ValueRange());
    LUT<FLType> GS_LUT;

    // Choose the appropriate algorithm to apply bilateral filter
    int algorithm = (radiusx * 2 + 1)*(radiusy * 2 + 1) * 8 > (PCType)(PBFICnum * 96) ? 1 : 0;

    switch (algorithm)
    {
    case 0:
        GS_LUT = Gaussian_Function_Spatial_LUT_Generation(xUpper, yUpper, sigmaS);
        Bilateral2D_0(output, input, ref, GS_LUT, GR_LUT, radiusx, radiusy);
        break;
    case 1:
        Bilateral2D_1(output, input, ref, GR_LUT, sigmaS, PBFICnum);
        break;
    }
    
    // Clear and output
    return output;
}


LUT<FLType> Gaussian_Function_Spatial_LUT_Generation(const PCType xUpper, const PCType yUpper, const double sigmaS)
{
    PCType x, y;
    LUT<FLType> GS_LUT(xUpper*yUpper);

    for (y = 0; y < yUpper; y++)
    {
        for (x = 0; x < xUpper; x++)
        {
            GS_LUT[y*xUpper + x] = Gaussian_Function_sqr_x((FLType)(x*x + y*y), sigmaS);
        }
    }

    return GS_LUT;
}

LUT<FLType> Gaussian_Function_Range_LUT_Generation(const DType ValueRange, const double sigmaR)
{
    DType i;
    const DType upper = Min(ValueRange, (DType)(sigmaR*sigmaRMul + 0.5));
    LUT<FLType> GR_LUT(ValueRange + 1);

    for (i = 0; i <= upper; i++)
    {
        GR_LUT[i] = Gaussian_Function((FLType)i, sigmaR);
    }
    // For unknown reason, when more range weights equal 0, the runtime speed gets lower - mainly in function Recursive_Gaussian2D_Horizontal.
    // To avoid this problem, we set range weights whose range values are larger than sigmaR*sigmaRMul to the Gaussian distribution value at sigmaR*sigmaRMul.
    const FLType upperLUTvalue = GR_LUT[upper];
    for (; i < ValueRange; i++)
    {
        GR_LUT[i] = upperLUTvalue;
    }

    return GR_LUT;
}


// Implementation of Bilateral filter with truncated spatial window
Plane & Bilateral2D_0(Plane & output, const Plane & input, const Plane & ref, const LUT<FLType> & GS_LUT, const LUT<FLType> & GR_LUT, const PCType radiusx, const PCType radiusy)
{
    const PCType xUpper = radiusx + 1, yUpper = radiusy + 1;

    PCType index0, i, j, index1, x, y;
    FLType Weight, WeightSum;
    FLType Sum;

    index0 = radiusy*input.Width();
    memcpy(output.Data(), input.Data(), index0*sizeof(DType));
    for (j = radiusy; j < input.Height() - radiusy; j++)
    {
        for (i = 0; i < radiusx; i++, index0++)
        {
            output[index0] = input[index0];
        }
        for (; i < input.Width() - radiusx; i++, index0++)
        {
            WeightSum = 0;
            Sum = 0;
            for (y = -radiusy; y < yUpper; y++)
            {
                index1 = index0 + y*input.Width() - radiusx;
                for (x = -radiusx; x < xUpper; x++, index1++)
                {
                    Weight = Gaussian_Distribution2D_Spatial_LUT_Lookup(GS_LUT, xUpper, x, y) * Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, ref[index0], ref[index1]);
                    WeightSum += Weight;
                    Sum += input[index1] * Weight;
                }
            }
            output[index0] = output.Quantize(Sum / WeightSum);
        }
        for (; i < input.Width(); i++, index0++)
        {
            output[index0] = input[index0];
        }
    }
    memcpy(output.Data() + index0, input.Data() + index0, radiusy*input.Width()*sizeof(DType));

    return output;
}


// Implementation of O(1) Bilateral filter algorithm from "Qingxiong Yang, Kar-Han Tan, Narendra Ahuja - Real-Time O(1) Bilateral Filtering"
Plane & Bilateral2D_1(Plane & output, const Plane & input, const Plane & ref, const LUT<FLType> & GR_LUT, const double sigmaS, const DType PBFICnum)
{
    DType i;
    PCType j;
    PCType sw = ref.Width();
    PCType sh = ref.Height();
    PCType pcount = ref.PixelCount();
    
    // Get the minimum and maximum pixel value of Plane "ref"
    DType rLower = ref.Ceil(), rUpper = ref.Floor(), rRange; // First set rLower to the highest number and rUpper to the lowest number

    for (j = 0; j < pcount; j++)
    {
        if (rLower > ref[j]) rLower = ref[j];
        if (rUpper < ref[j]) rUpper = ref[j];
    }
    rRange = rUpper - rLower;

    // Generate quantized PBFICs' parameters
    DType * PBFICk = new DType[PBFICnum];

    for (i = 0; i < PBFICnum; i++)
    {
        PBFICk[i] = Round_Div(rRange*i, PBFICnum - 1) + rLower;
    }

    // Generate recursive Gaussian parameters
    double B, B1, B2, B3;
    Recursive_Gaussian_Parameters(sigmaS, B, B1, B2, B3);

    // Generate quantized PBFICs
    Plane_FL Wk(ref, false);
    Plane_FL Jk(ref, false);
    Plane_FL * PBFIC = new Plane_FL[PBFICnum];
    
    for (i = 0; i < PBFICnum; i++)
    {
        PBFIC[i] = Plane_FL(ref, false);

        if (input.isPCChroma())
        {
            for (j = 0; j < pcount; j++)
            {
                Wk[j] = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, PBFICk[i], ref[j]);
                Jk[j] = Wk[j] * input.GetFL_PCChroma(input[j]);
            }
        }
        else
        {
            for (j = 0; j < pcount; j++)
            {
                Wk[j] = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, PBFICk[i], ref[j]);
                Jk[j] = Wk[j] * input.GetFL(input[j]);
            }
        }

        Recursive_Gaussian2D_Horizontal(Wk, B, B1, B2, B3);
        Recursive_Gaussian2D_Vertical(Wk, B, B1, B2, B3);
        Recursive_Gaussian2D_Horizontal(Jk, B, B1, B2, B3);
        Recursive_Gaussian2D_Vertical(Jk, B, B1, B2, B3);

        for (j = 0; j < pcount; j++)
        {
            PBFIC[i][j] = Jk[j] / Wk[j];
        }
    }

    // Generate filtered result from PBFICs using bilinear interpolation
    if (output.isPCChroma())
    {
        for (j = 0; j < pcount; j++)
        {
            for (i = 0; i < PBFICnum - 2; i++)
            {
                if (ref[j] < PBFICk[i + 1] && ref[j] >= PBFICk[i]) break;
            }

            output[j] = output.GetD_PCChroma(((PBFICk[i + 1] - ref[j])*PBFIC[i][j] + (ref[j] - PBFICk[i])*PBFIC[i + 1][j]) / (PBFICk[i + 1] - PBFICk[i]));
        }
    }
    else
    {
        for (j = 0; j < pcount; j++)
        {
            for (i = 0; i < PBFICnum - 2; i++)
            {
                if (ref[j] < PBFICk[i + 1] && ref[j] >= PBFICk[i]) break;
            }

            output[j] = output.GetD(((PBFICk[i + 1] - ref[j])*PBFIC[i][j] + (ref[j] - PBFICk[i])*PBFIC[i + 1][j]) / (PBFICk[i + 1] - PBFICk[i]));
        }
    }

    // Clear and output
    delete[] PBFIC;
    delete[] PBFICk;

    return output;
}