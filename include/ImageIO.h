#ifndef IMAGEIO_H_
#define IMAGEIO_H_


#include <string>
#include "Image_Type.h"


Frame ImageReader(const std::string &filename, const FCType FrameNum = 0, const DType BitDepth = 16);

bool ImageWriter(const Frame &src, const std::string &filename);
bool ImageWriter(const Frame &src, const std::string &filename, int _type);


#endif
