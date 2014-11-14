#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include "ISP_MW.h"


//#define Test
//#define Test_Other
//#define Test_Write

//#define Convolution
//#define EdgeDetect_
//#define Gaussian
//#define Bilateral
//#define Transpose_
//#define Specular_Highlight_Removal_
//#define Tone_Mapping
//#define Retinex_MSRCP_
//#define Retinex_MSRCR_
//#define Retinex_MSRCR_GIMP_
//#define Histogram_Equalization_
//#define AWB_


int main(int argc, char ** argv)
{
#if defined(Test)
    Frame IFrame = ImageReader("D:\\Test Images\\01.bmp");
#ifdef Test_Other
    for (double sigmaS = 0; sigmaS <= 25; sigmaS += 0.5)
    {
        Bilateral2D_2_Para para(sigmaS);
        std::cout << Max(static_cast<PCType>(sigmaS*sigmaSMul + 0.5), PCType(1)) << " "
            << para.radius << " " << para.samples << " " << para.step << std::endl;
    }
    system("pause");
#elif defined(Test_Write) // Test_Other
    Frame PFrame;
#if defined(Convolution)
    PFrame = Convolution3(IFrame, 1, 2, 1, 2, 4, 2, 1, 2, 1);
#elif defined(EdgeDetect_)
    PFrame = EdgeDetect(IFrame, EdgeKernel::Laplace1);
#elif defined(Gaussian)
    PFrame = Gaussian2D(IFrame, 5.0);
#elif defined(Bilateral)
    Bilateral2D_Data bldata(IFrame, 3.0, 0.02, 0);
    PFrame = Bilateral2D(IFrame, bldata);
#elif defined(Transpose_)
    PFrame = Transpose(IFrame);
#elif defined(Specular_Highlight_Removal_)
    PFrame = Specular_Highlight_Removal(IFrame);
#elif defined(Tone_Mapping)
    PFrame = Adaptive_Global_Tone_Mapping(IFrame);
#elif defined(Retinex_MSRCP_)
    PFrame = Retinex_MSRCP(IFrame);
#elif defined(Retinex_MSRCR_)
    PFrame = Retinex_MSRCR(IFrame);
#elif defined(Retinex_MSRCR_GIMP_)
    PFrame = Retinex_MSRCR_GIMP(IFrame);
#elif defined(AWB_)
    AWB1 AWBObj(IFrame);
    PFrame = AWBObj.process();
#else
    PFrame = IFrame;
#endif
    ImageWriter(PFrame, "D:\\Test Images\\01.Test.png");
    system("pause");
#else // Test_Write
    PCType i;
    const int Loop = 2000;
    const Plane& srcR = IFrame.R();
    Plane dstR(srcR, false);
    for (i = 0; i < Loop; i++)
    {
#if defined(Convolution)
        Convolution3V(IFrame, 1, 2, 1);
        //Convolution3(IFrame, 1, 2, 1, 2, 4, 2, 1, 2, 1);
#elif defined(EdgeDetect_)
        EdgeDetect(IFrame, EdgeKernel::Laplace2);
#elif defined(Gaussian)
        Gaussian2D(IFrame, 5.0);
#elif defined(Bilateral)
        Bilateral2D_Data bldata(IFrame, 3.0, 0.08, 1);
        Bilateral2D(IFrame, bldata);
#elif defined(Transpose_)
        Transpose(IFrame);
#elif defined(Specular_Highlight_Removal_)
        Specular_Highlight_Removal(IFrame);
#elif defined(Retinex_MSRCP_)
        Retinex_MSRCP(IFrame);
#elif defined(Retinex_MSRCR_)
        Retinex_MSRCR(IFrame);
#elif defined(Retinex_MSRCR_GIMP_)
        Retinex_MSRCR_GIMP(IFrame);
#elif defined(Histogram_Equalization_)
        Histogram_Equalization(IFrame, 1.0, false);
#else
        //DType count = 0;
        //srcR.for_each([&count](DType x){ count += x; });

        //dstR.transform(srcR, [](DType x){ return static_cast<DType>(log(x) + 0.5); });
        //dstR.transform_PPL(srcR, [](DType x){ return static_cast<DType>(log(x) + 0.5); });

        dstR.convolute_PPL<1, 1>(srcR, [](DType (*srcb2D)[3])->DType
        {
            return (srcb2D[0][0] + 2 * srcb2D[0][1] + srcb2D[0][2]
                + 2 * srcb2D[1][0] + 4 * srcb2D[1][1] + 2 * srcb2D[1][2]
                + srcb2D[2][0] + 2 * srcb2D[2][1] + srcb2D[2][2]) / 16;
        });
#endif
    }
#endif // Test_Write
#else // Test

    return Filtering(argc, argv);
    
#endif // Test
}


int Filtering(const int argc, char ** argv)
{
    int i;

    if (argc <= 2)
    {
        std::cout << "Not enough arguments specified.\n";
        return 0;
    }

    std::string FilterName = argv[1];
    std::transform(FilterName.begin(), FilterName.end(), FilterName.begin(), tolower);

    int argc2 = argc - 2;
    std::vector<std::string> args(argc2);

    for (i = 0; i < argc2; i++)
    {
        args[i] = argv[i + 2];
    }

    FilterIO *FilterObj = nullptr;

    if (FilterName == "--gaussian")
    {
        FilterObj = new Gaussian2D_IO(argc2, args);
    }
    else if (FilterName == "--bilateral")
    {
        FilterObj = new Bilateral2D_IO(argc2, args);
    }
    else if (FilterName == "--agtm" || FilterName == "--adaptive_global_tone_mapping")
    {
        FilterObj = new Adaptive_Global_Tone_Mapping_IO(argc2, args);
    }
    else if (FilterName == "--retinex_msrcp" || FilterName == "--msrcp" || FilterName == "--retinex_msr" || FilterName == "--msr" || FilterName == "--retinex")
    {
        FilterObj = new Retinex_MSRCP_IO(argc2, args);
    }
    else if (FilterName == "--retinex_msrcr" || FilterName == "--msrcr")
    {
        FilterObj = new Retinex_MSRCR_IO(argc2, args);
    }
    else if (FilterName == "--retinex_msrcr_gimp" || FilterName == "--msrcr_gimp")
    {
        FilterObj = new Retinex_MSRCR_GIMP_IO(argc2, args);
    }
    else if (FilterName == "--he" || FilterName == "--histogram_equalization")
    {
        FilterObj = new Histogram_Equalization_IO(argc2, args);
    }
    else if (FilterName == "--awb1")
    {
        FilterObj = new AWB1_IO(argc2, args);
    }
    else if (FilterName == "--awb2")
    {
        FilterObj = new AWB2_IO(argc2, args);
    }
    else if (FilterName == "--ed" || FilterName == "--edgedetect")
    {
        FilterObj = new EdgeDetect_IO(argc2, args);
    }
    else
    {
        return 1;
    }

    if (FilterObj)
    {
        FilterObj->process();
        delete FilterObj;
    }

    return 0;
}
