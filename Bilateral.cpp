#include <iostream>
#include <cmath>
#include "include\Bilateral.h"
#include "include\Gaussian.h"
#include "include\Type_Conv.h"
#include "include\Image_Type.h"
#include "include\IO.h"


int Bilateral2D_IO(int argc, char ** argv)
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
    double sigmaS = 1.0;
    double sigmaR = 0.1;
    string Tag = ".Bilateral";
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
        if (args[i] == "-S" || args[i] == "--sigmaS")
        {
            Flag |= args2arg(i, argc, args, sigmaS);
            continue;
        }
        if (args[i] == "-R" || args[i] == "--sigmaR")
        {
            Flag |= args2arg(i, argc, args, sigmaR);
            continue;
        }
        
        Frame_RGB SFrame = ImageReader(args[i]);
        Frame_RGB PFrame = Bilateral2D(SFrame, sigmaS, sigmaR);

        _splitpath_s(argv[i], Drive, PATHLEN, Dir, PATHLEN, FileName, PATHLEN, Ext, PATHLEN);
        string OPath = string(Drive) + string(Dir) + string(FileName) + Tag + Format;

        ImageWriter(PFrame, OPath);
    }

    // Clean
    delete[] args;
    delete[] Drive, Dir, FileName, Ext;

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
    FLType * GS_LUT = nullptr;
    FLType * GR_LUT = new FLType[ref.ValueRange()];

    Gaussian_Distribution2D_Range_LUT_Generation(GR_LUT, ref.ValueRange(), sigmaR*ref.ValueRange());

    // Choose the appropriate algorithm to apply bilateral filter
    int algorithm = (radiusx * 2 + 1)*(radiusy * 2 + 1) * 8 > (PCType)(PBFICnum * 96) ? 1 : 0;

    switch (algorithm)
    {
    case 0:
        GS_LUT = new FLType[xUpper*yUpper];
        Gaussian_Distribution2D_Spatial_LUT_Generation(GS_LUT, xUpper, yUpper, sigmaS);
        Bilateral2D_0(output, input, ref, GS_LUT, GR_LUT, radiusx, radiusy);
        break;
    case 1:
        Bilateral2D_1(output, input, ref, GR_LUT, sigmaS, PBFICnum);
        break;
    }
    
    // Clear and output
    delete[] GR_LUT;
    delete[] GS_LUT;

    return output;
}


void Gaussian_Distribution2D_Spatial_LUT_Generation(FLType * GS_LUT, const PCType xUpper, const PCType yUpper, const double sigmaS)
{
    PCType x, y;

    for (y = 0; y < yUpper; y++)
    {
        for (x = 0; x < xUpper; x++)
        {
            GS_LUT[y*xUpper + x] = (FLType)Gaussian_Distribution2D_x2((double)(x*x + y*y), sigmaS);
        }
    }
}

void Gaussian_Distribution2D_Range_LUT_Generation(FLType * GR_LUT, const DType ValueRange, const double sigmaR)
{
    DType i;
    const DType upper = Min(ValueRange, (DType)(sigmaR*sigmaRMul + 0.5));

    for (i = 0; i < upper; i++)
    {
        GR_LUT[i] = (FLType)Gaussian_Distribution2D((double)i, sigmaR);
    }
    // For unknown reason, when more range weights equal 0, the runtime speed gets lower - mainly in function Recursive_Gaussian2D_Horizontal.
    // To avoid this problem, we set range weights whose range values are larger than sigmaR*sigmaRMul to the Gaussian distribution value at sigmaR*sigmaRMul.
    const FLType upperLUTvalue = GR_LUT[upper - 1];
    for (; i < ValueRange; i++)
    {
        GR_LUT[i] = upperLUTvalue;
    }
}


// Implementation of Bilateral filter with truncated spatial window
Plane & Bilateral2D_0(Plane & output, const Plane & input, const Plane & ref, FLType * GS_LUT, FLType * GR_LUT, const PCType radiusx, const PCType radiusy)
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
            output[index0] = Quantize(Sum / WeightSum, output);
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
Plane & Bilateral2D_1(Plane & output, const Plane & input, const Plane & ref, FLType * GR_LUT, const double sigmaS, const DType PBFICnum)
{
    DType i;
    PCType j;
    PCType sw = ref.Width();
    PCType sh = ref.Height();
    PCType pcount = ref.PixelCount();
    
    // Get the minimum and maximum pixel value of Plane "ref"
    DType rLower = ref.Ceil(), rUpper = ref.Floor(), rRange;

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
    FLType * Wk = new FLType[pcount];
    FLType * Jk = new FLType[pcount];
    FLType ** PBFIC = new FLType * [PBFICnum];
    
    for (i = 0; i < PBFICnum; i++)
    {
        PBFIC[i] = new FLType[pcount];

        for (j = 0; j < pcount; j++)
        {
            Wk[j] = Gaussian_Distribution2D_Range_LUT_Lookup(GR_LUT, PBFICk[i], ref[j]);
            Jk[j] = Wk[j] * input[j];
        }

        Recursive_Gaussian2D_Horizontal(Wk, input.Width(), input.Height(), B, B1, B2, B3);
        Recursive_Gaussian2D_Vertical(Wk, input.Width(), input.Height(), B, B1, B2, B3);
        Recursive_Gaussian2D_Horizontal(Jk, input.Width(), input.Height(), B, B1, B2, B3);
        Recursive_Gaussian2D_Vertical(Jk, input.Width(), input.Height(), B, B1, B2, B3);

        for (j = 0; j < pcount; j++)
        {
            PBFIC[i][j] = Jk[j] / Wk[j];
        }
    }

    // Generate filtered result from PBFICs using bilinear interpolation
    for (j = 0; j < pcount; j++)
    {
        for (i = 0; i < PBFICnum - 2; i++)
        {
            if (ref[j] < PBFICk[i + 1] && ref[j] >= PBFICk[i]) break;
        }

        output[j] = Quantize(((PBFICk[i + 1] - ref[j])*PBFIC[i][j] + (ref[j] - PBFICk[i])*PBFIC[i + 1][j]) / (PBFICk[i + 1] - PBFICk[i]), output);
    }

    // Clear and output
    for (i = 0; i < PBFICnum; i++) delete[] PBFIC[i];
    delete[] PBFIC;
    delete[] Jk;
    delete[] Wk;
    delete[] PBFICk;

    return output;
}