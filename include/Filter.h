#ifndef FILTER_H_
#define FILTER_H_


#include <cstdlib>
#include "Args.h"
#include "ImageIO.h"


template < typename _Ty = FLType >
struct FilterData
{
    typedef FilterData _Myt;

    PCType height = 0;
    PCType width = 0;
    PCType stride = 0;
    size_t count = 0;

    _Ty *dst_data = nullptr;
    const _Ty *src_data = nullptr;

    template < typename _St1 >
    void InitSize(const _St1 &dst)
    {
        height = dst.Height();
        width = dst.Width();
        stride = dst.Stride();
        count = dst.size();
    }

    void InitSize(PCType _height, PCType _width, PCType _stride)
    {
        height = _height;
        width = _width;
        stride = _stride;
        count = _height * _stride;
    }

    void EndSize()
    {
        height = 0;
        width = 0;
        stride = 0;
        count = 0;
    }

    template < typename _St1 >
    void Init(_St1 &dst, const _St1 &src)
    {
        if (typeid(_Ty) != typeid(typename _St1::value_type))
        {
            std::cerr << "FilterData::Init: _Ty and _St1::value_type don't match!\n";
            exit(EXIT_FAILURE);
        }

        InitSize(dst);

        dst_data = dst.data();
        src_data = src.data();
    }

    void Init(_Ty *dst, const _Ty *src, PCType _height, PCType _width, PCType _stride)
    {
        InitSize(_height, _width, _stride);

        dst_data = dst;
        src_data = src;
    }

    void End()
    {
        dst_data = nullptr;
        src_data = nullptr;

        EndSize();
    }
};


class FilterIF
{
public:
    typedef FilterIF _Myt;

private:
    template < typename _St1 >
    _St1 &processT(_St1 &dst, const _St1 &src)
    {
        std::cerr << "FilterIF::processT: This filter does nothing except copying!\n";
        dst = src;
        return dst;
    }

protected:
    virtual Plane_FL &process_Plane_FL(Plane_FL &dst, const Plane_FL &src)
    {
        return processT(dst, src);
    }

    virtual Plane &process_Plane(Plane &dst, const Plane &src)
    {
        return processT(dst, src);
    }

    virtual Frame &process_Frame(Frame &dst, const Frame &src)
    {
        for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
        {
            process_Plane(dst.P(i), src.P(i));
        }

        return dst;
    }

public:
    Plane_FL &process(Plane_FL &dst, const Plane_FL &src)
    {
        return process_Plane_FL(dst, src);
    }

    Plane &process(Plane &dst, const Plane &src)
    {
        return process_Plane(dst, src);
    }

    Frame &process(Frame &dst, const Frame &src)
    {
        return process_Frame(dst, src);
    }

    template < typename _St1 >
    _St1 operator()(const _St1 &src)
    {
        _St1 dst(src, false);
        return process(dst, src);
    }
};


class FilterIF2
{
public:
    typedef FilterIF2 _Myt;

private:
    template < typename _St1 >
    _St1 &processT(_St1 &dst, const _St1 &src, const _St1 &ref)
    {
        std::cerr << "FilterIF2::processT: This filter does nothing except copying!\n";
        dst = src;
        return dst;
    }

protected:
    virtual Plane_FL &process_Plane_FL(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref)
    {
        return processT(dst, src, ref);
    }

    virtual Plane &process_Plane(Plane &dst, const Plane &src, const Plane &ref)
    {
        return processT(dst, src, ref);
    }

    virtual Frame &process_Frame(Frame &dst, const Frame &src, const Frame &ref)
    {
        for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
        {
            process_Plane(dst.P(i), src.P(i), ref.P(i));
        }

        return dst;
    }

public:
    Plane_FL &process(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref)
    {
        return process_Plane_FL(dst, src, ref);
    }

    Plane &process(Plane &dst, const Plane &src, const Plane &ref)
    {
        return process_Plane(dst, src, ref);
    }

    Frame &process(Frame &dst, const Frame &src, const Frame &ref)
    {
        return process_Frame(dst, src, ref);
    }

    template < typename _St1 >
    _St1 operator()(const _St1 &src, const _St1 &ref)
    {
        _St1 dst(src, false);
        return process(dst, src, ref);
    }

    Plane_FL &process(Plane_FL &dst, const Plane_FL &src)
    {
        return process_Plane_FL(dst, src, src);
    }

    Plane &process(Plane &dst, const Plane &src)
    {
        return process_Plane(dst, src, src);
    }

    Frame &process(Frame &dst, const Frame &src)
    {
        return process_Frame(dst, src, src);
    }

    template < typename _St1 >
    _St1 operator()(const _St1 &src)
    {
        _St1 dst(src, false);
        return process(dst, src);
    }
};


class FilterIO
{
public:
    typedef FilterIO _Myt;

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
        Frame dst = process(src);
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

    virtual Frame process(const Frame &src) = 0;

public:
    _Myt(std::string _Tag = "")
        : Tag(std::move(_Tag)) {}

    virtual ~FilterIO() {}

    void SetArgs(int _argc, const std::vector<std::string> &_args)
    {
        argc = _argc;
        args = _args;
    }

    void SetTag(std::string _Tag)
    {
        Tag = std::move(_Tag);
    }

    void operator()(std::string _IPath = "", std::string _OPath = "")
    {
        arguments_process();
        if(_IPath != "") IPath = std::move(_IPath);
        if(_OPath != "") OPath = std::move(_OPath);
        else generate_OPath();
        processIO();
    }

    _Myt(const _Myt &src) = default;
    _Myt &operator=(const _Myt &src) = default;
    _Myt(_Myt &&src) = delete;
    _Myt &operator=(_Myt &&src) = delete;
};


#endif
