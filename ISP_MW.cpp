#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "include/ISP_MW.h"


using namespace cv;
using namespace std;


int main(int argc, char ** argv)
{
    int i;
    char * Drive = new char[DRIVELEN];
    char * Dir = new char[PATHLEN];
    char * FileName = new char[PATHLEN];
    char * Ext = new char[EXTLEN];

	// Default Parameters
    double sigmaS = 1.0;
    double sigmaR = 0.1;
    string Flag = ".Bilateral";
    string Format = ".png";

	// Arguments Process
    if (argc <= 1)
    {
        return 0;
    }

    string * args = new string[argc];
    for (i = 0; i < argc; i++)
    {
        args[i] = argv[i];
    }
    
    for (i = 1; i < argc; i++)
    {
        if (args[i] == "--sigmaS")
        {
            if (++i >= argc) return 0;
            sigmaS = stod(args[i]);
        }
        else if (args[i] == "--sigmaR")
        {
            if (++i >= argc) return 0;
            sigmaR = stod(args[i]);
        }
        else
        {
            Frame_RGB SFrame = ImageReader(args[i]);
            Frame_RGB PFrame = Bilateral2D(SFrame, sigmaS, sigmaR);

            _splitpath_s(argv[i], Drive, PATHLEN, Dir, PATHLEN, FileName, PATHLEN, Ext, PATHLEN);
            string OPath = string(Drive) + string(Dir) + string(FileName) + Flag + Format;

            ImageWriter(PFrame, OPath);
        }
    }
    
	// Clean
    delete[] args;
    delete[] Drive, Dir, FileName, Ext;

    return 0;
}
