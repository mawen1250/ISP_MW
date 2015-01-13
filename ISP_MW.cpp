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
#define NLMeans_


#if defined(Test)
int main()
{
#ifdef Test_Other
    const int Loop = 1;// 81600;

    const PCType BlockSize = 16;
    typedef Block<double, double> BlockT;

    Frame IFrame = ImageReader("D:\\Project\\Retinex test\\_DSC8176.JPG");
    Plane &IR = IFrame.R();
    
    /*std::vector<Pos> matchedPos;
    for (PCType l = 0; l < Loop; ++l)
    {
        BlockT ref(IR, BlockSize, BlockSize, Pos(1060, 765));
        //std::cout << ref;

        matchedPos.push_back(ref.BlockMatching(IR, true, 48, 2));
        //BlockT matched(IR, BlockSize, BlockSize, matchedPos);
        //std::cout << matched;
    }

    //auto matchedPosCode = ref.BlockMatchingMulti(IR, 48, 2);
    //for (auto i = matchedPosCode.begin(); i != matchedPosCode.end(); ++i)
    //{
    //    std::cout << i->first << " " << i->second << std::endl;
    //}

    system("pause");*/

    //std::vector<Pos> matchedPos;
    for (PCType j = 0; j < IR.Height() - BlockSize + 1; j += BlockSize)
    {
        for (PCType i = 0; i < IR.Width() - BlockSize + 1; i += BlockSize)
        {
            BlockT ref(IR, BlockSize, BlockSize, Pos(j, i));

            //matchedPos.push_back(ref.BlockMatching(IR, 48, 2));
            auto matchedPosCode = ref.BlockMatchingMulti(IR, 48, 2);
        }
    }

    /*std::vector<float> src(1280 * 720, 5.f);
    std::vector<float> dst(1280 * 720);

    for (int l = 0; l < Loop; l++)
    {
        concurrency::parallel_for(0, 720, [&](int j)
        {
            int i = 720 * j;
            for (int upper = i + 1280; i < upper; i++)
            {
                dst[i] = exp(log(src[i]));
            }
        });
    }*/

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
    Frame IFrame = ImageReader("D:\\Test Images\\Noise\\03.0Source.png");
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
#elif defined(NLMeans_)
    PFrame = NLMeans(IFrame);
#else
    PFrame = IFrame;
#endif
    ImageWriter(PFrame, "D:\\Test Images\\Noise\\03.0.png");
    system("pause");
#else // Test_Write
    const int Loop = 10;

    Frame IFrame = ImageReader("D:\\Test Images\\01.bmp");
    const Plane& srcR = IFrame.R();
    Plane dstR(srcR, false);

    for (int l = 0; l < Loop; l++)
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
#elif defined(NLMeans_)
        NLMeans filter;
        filter.process(IFrame);
#else
        //DType count = 0;
        //srcR.for_each([&count](DType x){ count += x; });

        //dstR.transform(srcR, [](DType x){ return static_cast<DType>(log(x) + 0.5); });
        //dstR.transform_PPL(srcR, [](DType x){ return static_cast<DType>(log(x) + 0.5); });
        dstR.transform_AMP(srcR, [](DType x) restrict(amp)
        {
            return static_cast<DType>(concurrency::fast_math::log(static_cast<float>(x)) + 0.5f);
        });

        /*dstR.convolute<1, 1>(srcR, [](DType (*srcb2D)[3])->DType
        {
            return (srcb2D[0][0] + 2 * srcb2D[0][1] + srcb2D[0][2]
                + 2 * srcb2D[1][0] + 4 * srcb2D[1][1] + 2 * srcb2D[1][2]
                + srcb2D[2][0] + 2 * srcb2D[2][1] + srcb2D[2][2]) / 16;
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
    else if (FilterName == "--nlm" || FilterName == "--nlmeans" || FilterName == "--nonlocalmeans")
    {
        FilterObj = new NLMeans_IO(argc2, args);
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
