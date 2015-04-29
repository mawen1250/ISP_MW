#include <iostream>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include "ISP_MW.h"


//#define Test
#define Test_Func Test_Speed
//#define Test_Func Test_Write
//#define Test_Func Test_Other

//#define Convolution_
//#define EdgeDetect_
//#define Gaussian_
//#define Bilateral_
//#define Transpose_
//#define Specular_Highlight_Removal_
//#define Tone_Mapping_
//#define Retinex_MSRCP_
//#define Retinex_MSRCR_
//#define Retinex_MSRCR_GIMP_
//#define Histogram_Equalization_
//#define AWB_
//#define NLMeans_
#define Haze_Removal_

#ifdef _CUDA_
//#define CUDA_Gaussian_
//#define CUDA_Transpose_
//#define CUDA_Haze_Removal_
#endif


int Test_Speed()
{
    const int Loop = 20;

    Frame IFrame = ImageReader("D:\\Test Images\\Haze\\20150202_132636.jpg");
    //Frame IFrame = ImageReader("D:\\Test Images\\Haze\\2\\1 tz WDR=on 2.png");
    Frame PFrame(IFrame, false);
    const Plane &srcR = IFrame.R();
    Plane dstR(srcR, false);

    for (int l = 0; l < Loop; l++)
    {
#if defined(Convolution_)
        Convolution3V(IFrame, 1, 2, 1);
        //Convolution3(IFrame, 1, 2, 1, 2, 4, 2, 1, 2, 1);
#elif defined(EdgeDetect_)
        EdgeDetect(IFrame, EdgeKernel::Laplace2);
#elif defined(Gaussian_)
        Gaussian2D filter;
        filter(IFrame);
#elif defined(CUDA_Gaussian_)
        CUDA_Gaussian2D filter;
        filter(IFrame);
#elif defined(Bilateral_)
        Bilateral2D_Para para;
        para.sigmaS = 3.0;
        para.sigmaR = 0.08;
        para.algorithm = 1;
        Bilateral2D_Data bldata(IFrame, para);
        Bilateral2D(IFrame, bldata);
#elif defined(Transpose_)
        Transpose(PFrame, IFrame);
#elif defined(CUDA_Transpose_)
        CUDA_Transpose(PFrame, IFrame);
#elif defined(Specular_Highlight_Removal_)
        Specular_Highlight_Removal(IFrame);
#elif defined(Retinex_MSRCP_)
        Retinex_MSRCP filter;
        filter(IFrame);
#elif defined(Retinex_MSRCR_)
        Retinex_MSRCR filter;
        filter(IFrame);
#elif defined(Retinex_MSRCR_GIMP_)
        Retinex_MSRCR_GIMP filter;
        filter(IFrame);
#elif defined(Histogram_Equalization_)
        Histogram_Equalization(IFrame, 1.0, false);
#elif defined(NLMeans_)
        NLMeans filter;
        filter(IFrame);
#elif defined(Haze_Removal_)
        Haze_Removal_Retinex filter;
        filter(IFrame);
#elif defined(CUDA_Haze_Removal_)
        CUDA_Haze_Removal_Retinex filter;
        filter(IFrame);
#else
        const PCType pzero = 0;
        const PCType pcount = dstR.PixelCount();
        const PCType height = dstR.Height();
        const PCType width = dstR.Width();
        const PCType stride = dstR.Stride();
#endif
    }

    return 0;
}

int Test_Write()
{
    Frame IFrame = ImageReader("D:\\Test Images\\Haze\\20150202_132636.jpg");
    Frame PFrame(IFrame, false);
#if defined(Convolution_)
    PFrame = Convolution3(IFrame, 1, 2, 1, 2, 4, 2, 1, 2, 1);
#elif defined(EdgeDetect_)
    PFrame = EdgeDetect(IFrame, EdgeKernel::Laplace1);
#elif defined(Gaussian_)
    Gaussian2D filter;
    PFrame = filter(IFrame);
#elif defined(CUDA_Gaussian_)
    CUDA_Gaussian2D filter;
    PFrame = filter(IFrame);
#elif defined(Bilateral_)
    Bilateral2D_Para para;
    para.sigmaS = 3.0;
    para.sigmaR = 0.02;
    para.algorithm = 0;
    Bilateral2D_Data bldata(IFrame, para);
    PFrame = Bilateral2D(IFrame, bldata);
#elif defined(Transpose_)
    Transpose(PFrame, IFrame);
#elif defined(CUDA_Transpose_)
    CUDA_Transpose(PFrame, IFrame);
#elif defined(Specular_Highlight_Removal_)
    PFrame = Specular_Highlight_Removal(IFrame);
#elif defined(Tone_Mapping_)
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
#elif defined(NLMeans_)
    NLMeans filter;
    PFrame = filter(IFrame);
#elif defined(Haze_Removal_)
    Haze_Removal_Retinex filter;
    PFrame = filter(IFrame);
#elif defined(CUDA_Haze_Removal_)
    CUDA_Haze_Removal_Retinex filter;
    PFrame = filter(IFrame);
#else
    PFrame = IFrame;
#endif
    ImageWriter(PFrame, "D:\\Test Images\\Haze\\20150202_132636.0.png");

    system("pause");
    return 0;
}

int Test_Other()
{
    const int Loop = 1000;

    Frame IFrame = ImageReader("D:\\Test Images\\Haze\\20150202_132636.jpg");
    Plane &IR = IFrame.R();

    system("pause");
    return 0;
}


int main(int argc, char ** argv)
{
    srand(static_cast<unsigned int>(time(0)));

    std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
    std::cout.setf(std::ios_base::showpos);
    std::cout.setf(std::ios_base::showpoint);
    std::cout.precision(7);

#ifdef _CUDA_
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif

#ifdef Test
    return Test_Func();
#else
    return Filtering(argc, argv);
#endif
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

    FilterIO *filterIOPtr = nullptr;

    if (FilterName == "--gaussian")
    {
        filterIOPtr = new _Gaussian2D_IO;
    }
    else if (FilterName == "--bilateral")
    {
        filterIOPtr = new Bilateral2D_IO;
    }
    else if (FilterName == "--agtm" || FilterName == "--adaptive_global_tone_mapping")
    {
        filterIOPtr = new Adaptive_Global_Tone_Mapping_IO;
    }
    else if (FilterName == "--retinex_msrcp" || FilterName == "--msrcp" || FilterName == "--retinex_msr" || FilterName == "--msr" || FilterName == "--retinex")
    {
        filterIOPtr = new Retinex_MSRCP_IO;
    }
    else if (FilterName == "--retinex_msrcr" || FilterName == "--msrcr")
    {
        filterIOPtr = new Retinex_MSRCR_IO;
    }
    else if (FilterName == "--retinex_msrcr_gimp" || FilterName == "--msrcr_gimp")
    {
        filterIOPtr = new Retinex_MSRCR_GIMP_IO;
    }
    else if (FilterName == "--he" || FilterName == "--histogram_equalization")
    {
        filterIOPtr = new Histogram_Equalization_IO;
    }
    else if (FilterName == "--awb1")
    {
        filterIOPtr = new AWB1_IO;
    }
    else if (FilterName == "--awb2")
    {
        filterIOPtr = new AWB2_IO;
    }
    else if (FilterName == "--ed" || FilterName == "--edgedetect")
    {
        filterIOPtr = new EdgeDetect_IO;
    }
    else if (FilterName == "--nlm" || FilterName == "--nlmeans" || FilterName == "--nonlocalmeans")
    {
        filterIOPtr = new NLMeans_IO;
    }
    else if (FilterName == "--hrr" || FilterName == "--haze_removal" || FilterName == "--haze_removal_retinex")
    {
        filterIOPtr = new _Haze_Removal_Retinex_IO;
    }
    else
    {
        return 1;
    }

    if (filterIOPtr)
    {
        filterIOPtr->SetArgs(argc2, args);
        filterIOPtr->operator()();
        delete filterIOPtr;
    }

    return 0;
}
