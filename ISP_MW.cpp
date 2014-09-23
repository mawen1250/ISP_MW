#include <iostream>
#include <string>
#include <cstdlib>

#include "include\ISP_MW.h"


//#define Test
#define Test_Write

//#define Gaussian
//#define Bilateral
//#define Transpose_
//#define Specular_Highlight_Removal_
#define Tone_Mapping


using namespace std;


int main(int argc, char ** argv)
{
#ifdef Test
    PCType i;
    const int Loop = 15;

    Frame_RGB Frame = ImageReader("D:\\Test Images\\04.bmp");

#ifdef Test_Write
    Frame_RGB PFrame;
#if defined(Gaussian)
    PFrame = Gaussian2D(Frame, 5.0);
#elif defined(Bilateral)
    PFrame = Bilateral2D(Frame, 3.0, 0.1);
#elif defined(Transpose_)
    PFrame = Transpose(Frame);
#elif defined(Specular_Highlight_Removal_)
    PFrame = Specular_Highlight_Removal(Frame);
#elif defined(Tone_Mapping)
    PFrame = Adaptive_Global_Tone_Mapping(Frame);
#endif
    ImageWriter(PFrame, "D:\\Test Images\\04.Test.png");
    system("pause");
#else // Test_Write
    for (i = 0; i<Loop; i++) {
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

#if defined(Gaussian)
    return Gaussian2D_IO(argc, argv);
#elif defined(Bilateral)
    return Bilateral2D_IO(argc, argv);
#elif defined(Tone_Mapping)
    return Adaptive_Global_Tone_Mapping_IO(argc, argv);
#endif
    
#endif // Test
}
