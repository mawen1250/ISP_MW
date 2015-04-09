#ifndef ISP_MW_H_
#define ISP_MW_H_


//#define ENABLE_PPL
//#define ENABLE_AMP


#include "Type.h"
#include "Helper.h"
#include "Image_Type.h"
#include "Specification.h"
#include "LUT.h"
#include "Histogram.h"
#include "Block.h"
#include "ImageIO.h"
#include "Filter.h"

#include "Transform.h"
#include "Convolution.h"
#include "Gaussian.h"
#include "Bilateral.h"
#include "Highlight_Removal.h"
#include "Tone_Mapping.h"
#include "Retinex.h"
#include "Histogram_Equalization.h"
#include "AWB.h"
#include "NLMeans.h"
#include "BM3D.h"
#include "Haze_Removal.h"

#ifdef _CUDA_
#include "Gaussian.cuh"

typedef CUDA_Gaussian2D_IO _Gaussian2D_IO;
#else
typedef Gaussian2D_IO _Gaussian2D_IO;
#endif


int Filtering(const int argc, char ** argv);


#endif
