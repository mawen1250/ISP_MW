#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>
#include "include\ISP_MW.h"


//#define Test
#define Test_Write

//#define Gaussian
//#define Bilateral
//#define Transpose_
//#define Specular_Highlight_Removal_
//#define Tone_Mapping
#define Retinex


int main(int argc, char ** argv)
{
#ifdef Test
    Frame IFrame = ImageReader("D:\\Project\\4_WDR\\Test\\4\\off-B40.bmp");
#ifdef Test_Write
    Frame PFrame;
#if defined(Gaussian)
    PFrame = Gaussian2D(IFrame, 5.0);
#elif defined(Bilateral)
    PFrame = Bilateral2D(IFrame, 3.0, 0.1);
#elif defined(Transpose_)
    PFrame = Transpose(IFrame);
#elif defined(Specular_Highlight_Removal_)
    PFrame = Specular_Highlight_Removal(IFrame);
#elif defined(Tone_Mapping)
    PFrame = Adaptive_Global_Tone_Mapping(IFrame);
#elif defined(Retinex)
    PFrame = Retinex_MSR(IFrame);
#endif
    ImageWriter(PFrame, "D:\\Project\\4_WDR\\Test\\4\\off-B40.Test.png");
    system("pause");
#else // Test_Write
    PCType i;
    const int Loop = 15;

    for (i = 0; i<Loop; i++)
    {
#if defined(Gaussian)
        Gaussian2D(Frame, 5.0);
#elif defined(Bilateral)
        Bilateral2D(Frame, 3.0, 0.1);
#elif defined(Transpose_)
        Transpose(Frame);
#elif defined(Highlight_Removal_)
        Specular_Highlight_Removal(Frame);
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
    std::string * args = new std::string[argc2];

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
        return Retinex_MSR_IO(argc2, args);
    }

    return 0;
}
