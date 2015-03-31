#include "ImageIO.h"
#include "Helper.h"
#include "LUT.h"


Frame ImageReader(const std::string &filename, const FCType FrameNum, const DType BitDepth)
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    if (!image.data) // Check for invalid input
    {
        std::cout << "Could not open or find the image file: " << filename << std::endl;

        Frame src(FrameNum, PixelType::RGB, 1920, 1080, BitDepth, true);
        return src;
    }

    PCType sw = image.cols;
    PCType sh = image.rows;
    PCType pcount = sw * sh;
    PCType nCols = sw * image.channels();
    PCType nRows = sh;

    Frame src(FrameNum, PixelType::RGB, sw, sh, BitDepth, false);

    PCType i, j;
    Plane &R = src.R();
    Plane &G = src.G();
    Plane &B = src.B();

    if (image.isContinuous())
    {
        uchar *p = image.ptr<uchar>(0);

        if (R.Floor() == 0 && R.Ceil() == 65535)
        {
            for (i = 0, j = 0; i < pcount; i++)
            {
                B[i] = static_cast<DType>(p[j++]) * DType(257);
                G[i] = static_cast<DType>(p[j++]) * DType(257);
                R[i] = static_cast<DType>(p[j++]) * DType(257);
            }
        }
        else if (R.Floor() == 0 && R.Ceil() == 65535)
        {
            for (i = 0, j = 0; i < pcount; i++)
            {
                B[i] = static_cast<DType>(p[j++]);
                G[i] = static_cast<DType>(p[j++]);
                R[i] = static_cast<DType>(p[j++]);
            }
        }
        else
        {
            LUT<DType>::LevelType k;
            const LUT<DType>::LevelType iLevels = 256;
            LUT<DType> ConvertLUT(iLevels);

            for (k = 0; k < iLevels; k++)
            {
                ConvertLUT[k] = R.GetD(static_cast<FLType>(k) / FLType(255));
            }

            for (i = 0, j = 0; i < pcount; i++)
            {
                B[i] = ConvertLUT[p[j++]];
                G[i] = ConvertLUT[p[j++]];
                R[i] = ConvertLUT[p[j++]];
            }
        }
    }
    else
    {
        std::cout << "Memory is not continuous for image file: " << filename << std::endl;
    }

    return src;
}


bool ImageWriter(const Frame &src, const std::string &filename, int _type)
{
    cv::Mat image(src.Height(), src.Width(), _type);

    PCType i, j;
    const Plane &R = src.R();
    const Plane &G = src.G();
    const Plane &B = src.B();
    PCType pcount = src.PixelCount();

    if (image.isContinuous()) {
        uchar *p = image.ptr<uchar>(0);

        if (R.Floor() == 0 && R.Ceil() == 65535)
        {
            for (i = 0, j = 0; i < pcount; i++)
            {
                p[j++] = static_cast<uchar>(Round_Div(B[i], DType(257)));
                p[j++] = static_cast<uchar>(Round_Div(G[i], DType(257)));
                p[j++] = static_cast<uchar>(Round_Div(R[i], DType(257)));
            }
        }
        else if (R.Floor() == 0 && R.Ceil() == 255)
        {
            for (i = 0, j = 0; i < pcount; i++)
            {
                p[j++] = static_cast<uchar>(B[i]);
                p[j++] = static_cast<uchar>(G[i]);
                p[j++] = static_cast<uchar>(R[i]);
            }
        }
        else
        {
            LUT<uchar>::LevelType k;
            LUT<uchar> ConvertLUT(R);

            for (k = R.Floor(); k < R.Ceil(); k++)
            {
                ConvertLUT.Set(R, k, static_cast<uchar>(R.GetFL(k) * 255. + 0.5));
            }

            for (i = 0, j = 0; i < pcount; i++)
            {
                p[j++] = ConvertLUT.Lookup(R, B[i]);
                p[j++] = ConvertLUT.Lookup(R, G[i]);
                p[j++] = ConvertLUT.Lookup(R, R[i]);
            }
        }
        
        cv::imwrite(filename, image);

        return true;
    }
    else
    {
        return false;
    }
}
