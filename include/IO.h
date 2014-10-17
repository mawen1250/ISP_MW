#ifndef IO_H_
#define IO_H_


#include <cstdlib>
#include "Args.h"
#include "ImageIO.h"


class FilterIO
{
private:
    static const int DRIVELEN = 16;
    static const int PATHLEN = 256;
    static const int EXTLEN = 64;

    std::string IPath;
    std::string OPath;
    std::string Tag;
    std::string Format = ".png";

    void generate_OPath()
    {
        char Drive[DRIVELEN];
        char Dir[PATHLEN];
        char FileName[PATHLEN];
        char Ext[EXTLEN];

        _splitpath_s(IPath.c_str(), Drive, PATHLEN, Dir, PATHLEN, FileName, PATHLEN, Ext, PATHLEN);
        OPath = std::string(Drive) + std::string(Dir) + std::string(FileName) + Tag + Format;
    }

    void processIO()
    {
        const Frame src = ImageReader(IPath);
        Frame dst = processFrame(src);
        ImageWriter(dst, OPath);
    }

protected:
    int argc = 0;
    std::vector<std::string> args;

    virtual void arguments_process()
    {
        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-T" || args[i] == "--tag")
            {
                ArgsObj.GetPara(i, Tag);
                continue;
            }
            if (args[i] == "-F" || args[i] == "--format")
            {
                ArgsObj.GetPara(i, Format);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }

            IPath = args[i];
        }

        ArgsObj.Check();
    }

    virtual Frame processFrame(const Frame &src) = 0;

public:
    FilterIO(int _argc, const std::vector<std::string> &_args, std::string _Tag = "")
        : argc(_argc), args(_args), Tag(std::move(_Tag)) {}

    virtual ~FilterIO() {}

    void process(std::string _IPath = "", std::string _OPath = "")
    {
        arguments_process();
        if(_IPath != "") IPath = std::move(_IPath);
        if(_OPath != "") OPath = std::move(_OPath);
        else generate_OPath();
        processIO();
    }

private:
    FilterIO(FilterIO &&src);
    FilterIO & operator=(FilterIO &&src);
};


#endif