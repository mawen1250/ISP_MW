#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include "include\ISP_MW.h"


//#define Test
//#define Test_Other
//#define Test_Write

//#define Gaussian
//#define Bilateral
//#define Transpose_
//#define Specular_Highlight_Removal_
//#define Tone_Mapping
#define Retinex


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
#if defined(Gaussian)
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
#elif defined(Retinex)
    PFrame = Retinex_MSRCP(IFrame);
#endif
    ImageWriter(PFrame, "D:\\Test Images\\01.Test.png");
    system("pause");
#else // Test_Write
    PCType i;
    const int Loop = 20;

    for (i = 0; i < Loop; i++)
    {
#if defined(Gaussian)
        Gaussian2D(IFrame, 5.0);
#elif defined(Bilateral)
        Bilateral2D_Data bldata(IFrame, 3.0, 0.02, 0);
        Bilateral2D(IFrame, bldata);
#elif defined(Transpose_)
        Transpose(IFrame);
#elif defined(Specular_Highlight_Removal_)
        Specular_Highlight_Removal(IFrame);
#elif defined(Retinex)
        Retinex_MSRCP(IFrame);
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
    //std::string * args = new std::string[argc2];

    for (i = 0; i < argc2; i++)
    {
        args[i] = argv[i + 2];
    }

    if (FilterName == "--gaussian")
    {
        return Gaussian2D_IO(argc2, args);
    }
    if (FilterName == "--bilateral")
    {
        return Bilateral2D_IO(argc2, args);
    }
    if (FilterName == "--agtm" || FilterName == "--adaptiveglobaltonemapping" || FilterName == "--adaptive_global_tone_mapping")
    {
        return Adaptive_Global_Tone_Mapping_IO(argc2, args);
    }
    if (FilterName == "--retinex" || FilterName == "--retinex_msr" || FilterName == "--msr")
    {
        return Retinex_MSRCP_IO(argc2, args);
    }

    return 0;
}
