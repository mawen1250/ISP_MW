#ifndef IMAGEIO_H_
#define IMAGEIO_H_


#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Image_Type.h"


Frame ImageReader(const std::string & filename, const FCType FrameNum = 0, const DType BitDepth = 16);
bool ImageWriter(const Frame & Frame, const std::string & filename, int _type = CV_8UC3);


#endif