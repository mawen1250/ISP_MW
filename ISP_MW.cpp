#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include "ISP_MW.h"


//#define Test
//#define Test_Other
//#define Test_Write

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


#if defined(Test)
int main()
{
#ifdef Test_Other
    const int Loop = 1;

    const PCType BlockSize = 16;
    typedef Block<double, double> BlockT;

    Frame IFrame = ImageReader("D:\\Project\\Retinex test\\_DSC8176.JPG");
    Plane &IR = IFrame.R();
    
    /*std::vector<Pos> matchedPos;
    
    BlockT ref(IR, BlockSize, BlockSize, Pos(1060, 765));
    //std::cout << ref;

    //matchedPos.push_back(ref.BlockMatching(IR, true, 48, 2));
    //BlockT matched(IR, BlockSize, BlockSize, matchedPos);
    //std::cout << matched;

    auto matchedPosCode = ref.BlockMatchingMulti(IR, 48, 2);
    for (auto i = matchedPosCode.begin(); i != matchedPosCode.end(); ++i)
    {
        std::cout << i->first << " " << i->second << std::endl;
    }

    system("pause");*/

    /*concurrency::array_view<float, 1> srcv(1280 * 720, src);
    concurrency::array_view<float, 1> dstv(1280 * 720, dst);
    dstv.discard_data();

    for (int i = 0; i < Loop; i++)
    {
        concurrency::parallel_for_each(srcv.extent, [=](concurrency::index<1> idx) restrict(amp)
        {
            dstv[idx] = concurrency::fast_math::exp(concurrency::fast_math::log(srcv[idx]));
        });
    }*/
#elif defined(Test_Write) // Test_Other
    Frame IFrame = ImageReader("D:\\Test Images\\Haze\\20150126_131557.jpg");
    Frame PFrame;
#if defined(Convolution_)
    PFrame = Convolution3(IFrame, 1, 2, 1, 2, 4, 2, 1, 2, 1);
#elif defined(EdgeDetect_)
    PFrame = EdgeDetect(IFrame, EdgeKernel::Laplace1);
#elif defined(Gaussian_)
    PFrame = Gaussian2D(IFrame, 5.0);
#elif defined(Bilateral_)
    Bilateral2D_Para para;
    para.sigmaS = 3.0;
    para.sigmaR = 0.02;
    para.algorithm = 0;
    Bilateral2D_Data bldata(IFrame, para);
    PFrame = Bilateral2D(IFrame, bldata);
#elif defined(Transpose_)
    PFrame = Transpose(IFrame);
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
#else
    PFrame = IFrame;
#endif
    ImageWriter(PFrame, "D:\\Test Images\\Haze\\20150126_131557.0.png");
    system("pause");
#else // Test_Write
    const int Loop = 50;

    Frame IFrame = ImageReader("D:\\Test Images\\Haze\\2\\1 tz WDR=on 2.png");
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
        Gaussian2D(IFrame, 5.0);
#elif defined(Bilateral_)
        Bilateral2D_Para para;
        para.sigmaS = 3.0;
        para.sigmaR = 0.08;
        para.algorithm = 1;
        Bilateral2D_Data bldata(IFrame, para);
        Bilateral2D(IFrame, bldata);
#elif defined(Transpose_)
        Transpose(IFrame);
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
#else
        const PCType pzero = 0;
        const PCType pcount = dstR.PixelCount();
        const PCType height = dstR.Height();
        const PCType width = dstR.Width();
        const PCType stride = dstR.Stride();

        /*for (PCType i = pzero; i < pcount; ++i)
        {
            dstR[i] = static_cast<DType>(srcR[i] * srcR[i] * 0.1 + 0.5);
        }*/

        /*concurrency::parallel_for(pzero, pcount, [&](PCType i)
        {
            dstR[i] = static_cast<DType>(srcR[i] * srcR[i] * 0.1 + 0.5);
        });*/

        /*for (PCType j = pzero; j < height; ++j)
        {
            const PCType lower = j * stride;
            const PCType upper = lower + width;

            for (PCType i = lower; i < upper; ++i)
            {
                dstR[i] = static_cast<DType>(srcR[i] * srcR[i] * 0.1 + 0.5);
            }
        }*/

        /*for (PCType j = pzero; j < height; ++j)
        {
            const PCType lower = j * stride;
            const PCType upper = lower + width;

            concurrency::parallel_for(lower, upper, [&](PCType i)
            {
                dstR[i] = static_cast<DType>(srcR[i] * srcR[i] * 0.1 + 0.5);
            });
        }*/

        /*concurrency::parallel_for(pzero, height, [&](PCType j)
        {
            const PCType lower = j * stride;
            const PCType upper = lower + width;

            for (PCType i = lower; i < upper; ++i)
            {
                dstR[i] = static_cast<DType>(srcR[i] * srcR[i] * 0.1 + 0.5);
            }
        });*/

        /*const PCType pNum = (height + PPL_HP - 1) / PPL_HP;

        concurrency::parallel_for(PCType(0), pNum, [&](PCType p)
        {
            const PCType lower = p * PPL_HP;
            const PCType upper = Min(height, lower + PPL_HP);

            for (PCType j = lower; j < upper; ++j)
            {
                PCType i = j * stride;

                for (const PCType upper = i + width; i < upper; ++i)
                {
                    dstR[i] = static_cast<DType>(srcR[i] * srcR[i] * 0.1 + 0.5);
                }
            }
        });*/
#endif
    }
#endif // Test_Write
    return 0;
}
#else // Test
int main(int argc, char ** argv)
{
    return Filtering(argc, argv);
}
#endif // Test


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
        filterIOPtr = new Gaussian2D_IO;
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
        filterIOPtr = new Haze_Removal_Retinex_IO;
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
