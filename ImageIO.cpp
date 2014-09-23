#include <iostream>
#include <string>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "include\Image_Type.h"
#include "include\Type_Conv.h"


using namespace cv;


Frame_RGB ImageReader(const std::string & filename, const FCType FrameNum, const DType BitDepth)
{
    Mat image = imread(filename, IMREAD_COLOR);

    if (!image.data) // Check for invalid input
    {
        std::cout << "Could not open or find the image file: " << filename << std::endl;

        Frame_RGB Frame(FrameNum, 1920, 1080, BitDepth);
        return Frame;
    }

    PCType sw = image.cols;
    PCType sh = image.rows;
    PCType pcount = sw * sh;
    PCType nCols = sw * image.channels();
    PCType nRows = sh;

    Frame_RGB Frame(FrameNum, sw, sh, BitDepth);

    PCType i, j;
    Plane & R = Frame.R();
    Plane & G = Frame.G();
    Plane & B = Frame.B();

    if (image.isContinuous())
    {
        uchar * p = image.ptr<uchar>(0);
        for (i = 0, j = 0; i < pcount; i++)
        {
            B[i] = p[j++] * 257;
            G[i] = p[j++] * 257;
            R[i] = p[j++] * 257;
        }
    }
    else
    {
        std::cout << "Memory is not continuous for image file: " << filename << std::endl;
    }

    return Frame;
}


bool ImageWriter(const Frame_RGB & Frame, const std::string & filename, int _type)
{
    Mat image(Frame.Height(), Frame.Width(), _type);

    PCType i, j;
    const Plane & R = Frame.R();
    const Plane & G = Frame.G();
    const Plane & B = Frame.B();

    if (image.isContinuous()) {
        uchar * p = image.ptr<uchar>(0);
        for (i = 0, j = 0 ; i < Frame.PixelCount(); i++)
        {
            p[j++] = (uchar)Round_Div(B[i], (DType)257);
            p[j++] = (uchar)Round_Div(G[i], (DType)257);
            p[j++] = (uchar)Round_Div(R[i], (DType)257);
        }
        
        imwrite(filename, image);

        return true;
    }
    else
    {
        return false;
    }
}