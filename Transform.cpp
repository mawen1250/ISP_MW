#include "include/Transform.h"
#include "include/Image_Type.h"


Plane & Transpose(Plane & Plane)
{
    PCType i, j;
    DType * index0, * upper0, * index1, * upper1;
    
    DType * pp = new DType[Plane.PixelCount()];

    i = Plane.Width();
    j = Plane.PixelCount() - 1;
    
    for (index0 = Plane.Data(), upper0 = Plane.Data() + Plane.PixelCount(), index1 = pp, upper1 = pp + Plane.PixelCount(); index0 < upper0; index0++)
    {
        *index0 = *index1;
        index1 += i;

        if(index1>=upper1)
        {
            index1 -= j;
        }
    }
    
    delete[] pp;

    return Plane;
}
