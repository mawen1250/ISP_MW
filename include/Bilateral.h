#ifndef BILATERAL_H_
#define BILATERAL_H_


#include "IO.h"
#include "Image_Type.h"
#include "Helper.h"
#include "LUT.h"
#include "Gaussian.h"


const struct Bilateral2D_Para
{
    double sigmaS = 3.0;
    double sigmaR = 0.02;
    int algorithm = 0;
    int PBFICnum = 0;
} Bilateral2D_Default;


class Bilateral2D_Data
{
private:
    int PlaneCount = 1;
    bool isChroma = false;
    bool isYUV = false;
    int BPS = 16;

public:
    std::vector<double> sigmaS;
    std::vector<double> sigmaR;
    std::vector<int> process;
    std::vector<int> algorithm;

    std::vector<int> radius0;

    std::vector<int> PBFICnum;

    std::vector<int> radius;
    std::vector<int> samples;
    std::vector<int> step;

    std::vector<LUT<FLType>> GS_LUT;
    std::vector<LUT<FLType>> GR_LUT;

public:
    Bilateral2D_Data(const Plane &src, const Bilateral2D_Para &para = Bilateral2D_Default)
        : PlaneCount(1), isChroma(src.isChroma()), BPS(src.BitDepth()),
        sigmaS(PlaneCount, para.sigmaS), sigmaR(PlaneCount, para.sigmaR), process(PlaneCount),
        algorithm(PlaneCount, para.algorithm), radius0(PlaneCount), PBFICnum(PlaneCount, para.PBFICnum),
        radius(PlaneCount), samples(PlaneCount), step(PlaneCount), GS_LUT(PlaneCount), GR_LUT(PlaneCount)
    {
        process_define();

        Bilateral2D_0_Paras();
        Bilateral2D_1_Paras();
        Bilateral2D_2_Paras();

        algorithm_select();

        GS_LUT_Init();
        GR_LUT_Init();
    }

    Bilateral2D_Data(const Frame &src, const Bilateral2D_Para &para = Bilateral2D_Default)
        : PlaneCount(src.PlaneCount()), isYUV(src.isYUV()), BPS(src.BitDepth()),
        sigmaS(PlaneCount, para.sigmaS), sigmaR(PlaneCount, para.sigmaR), process(PlaneCount),
        algorithm(PlaneCount, para.algorithm), radius0(PlaneCount), PBFICnum(PlaneCount, para.PBFICnum),
        radius(PlaneCount), samples(PlaneCount), step(PlaneCount), GS_LUT(PlaneCount), GR_LUT(PlaneCount)
    {
        process_define();

        Bilateral2D_0_Paras();
        Bilateral2D_1_Paras();
        Bilateral2D_2_Paras();

        algorithm_select();

        GS_LUT_Init();
        GR_LUT_Init();
    }

    Bilateral2D_Data &Bilateral2D_0_Paras()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (process[i])
            {
                radius0[i] = Max(static_cast<int>(sigmaS[i] * sigmaSMul + 0.5), 1);
            }
        }

        return *this;
    }

    Bilateral2D_Data &Bilateral2D_1_Paras()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (process[i] && PBFICnum[i] == 0)
            {
                if (sigmaR[i] >= 0.08)
                {
                    PBFICnum[i] = 4;
                }
                else if (sigmaR[i] >= 0.015)
                {
                    PBFICnum[i] = Min(16, static_cast<int>(4 * 0.08 / sigmaR[i] + 0.5));
                }
                else
                {
                    PBFICnum[i] = Min(32, static_cast<int>(16 * 0.015 / sigmaR[i] + 0.5));
                }

                if ((isChroma || i > 0 && isYUV) && PBFICnum[i] % 2 == 0 && PBFICnum[i] < 256) // Set odd PBFIC number to chroma planes by default
                    PBFICnum[i]++;
            }
        }

        return *this;
    }

    Bilateral2D_Data &Bilateral2D_2_Paras()
    {
        int orad[3];

        for (int i = 0; i < 3; i++)
        {
            if (process[i])
            {
                orad[i] = Max(static_cast<int>(sigmaS[i] * sigmaSMul + 0.5), 1);

                step[i] = orad[i] < 4 ? 1 : orad[i] < 8 ? 2 : 3;

                samples[i] = 1;
                radius[i] = 1 + (samples[i] - 1)*step[i];

                while (orad[i] * 2 > radius[i] * 3)
                {
                    samples[i]++;
                    radius[i] = 1 + (samples[i] - 1)*step[i];
                    if (radius[i] >= orad[i] && samples[i] > 2)
                    {
                        samples[i]--;
                        radius[i] = 1 + (samples[i] - 1)*step[i];
                        break;
                    }
                }
            }
        }

        return *this;
    }

    Bilateral2D_Data &process_define()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (sigmaS[i] > 0 && sigmaR[i] > 0)
                process[i] = 1;
            else
                process[i] = 0;
        }

        return *this;
    }

    Bilateral2D_Data &algorithm_select()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (algorithm[i] <= 0)
                algorithm[i] = step[i] == 1 ? 2 : sigmaR[i] < 0.08 && samples[i] < 5 ? 2
                : 4 * samples[i] * samples[i] <= 15 * PBFICnum[i] ? 2 : 1;
        }

        return *this;
    }

    Bilateral2D_Data &GS_LUT_Init()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (process[i] && algorithm[i] == 2)
            {
                GS_LUT[i] = Gaussian_Function_Spatial_LUT_Generation(radius[i] + 1, radius[i] + 1, sigmaS[i]);
            }
            else if (process[i] && algorithm[i] > 2)
            {
                GS_LUT[i] = Gaussian_Function_Spatial_LUT_Generation(radius0[i] + 1, radius0[i] + 1, sigmaS[i]);
            }
        }

        return *this;
    }

    Bilateral2D_Data &GR_LUT_Init()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (process[i])
            {
                GR_LUT[i] = Gaussian_Function_Range_LUT_Generation((1 << BPS) - 1, sigmaR[i]);
            }
        }

        return *this;
    }
};


Plane &Bilateral2D(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane = 0);
Plane &Bilateral2D_0(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane = 0);
Plane &Bilateral2D_1(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane = 0);
Plane &Bilateral2D_2(Plane &dst, const Plane &src, const Plane &ref, const Bilateral2D_Data &d, int plane = 0);
Plane &Bilateral2D_2(Plane &dst, const Plane &src, const Bilateral2D_Data &d, int plane = 0);


inline Plane Bilateral2D(const Plane &src, const Plane &ref, const Bilateral2D_Data &d)
{
    Plane dst(src, false);

    return Bilateral2D(dst, src, ref, d, 0);
}

inline Plane Bilateral2D(const Plane &src, const Bilateral2D_Data &d)
{
    Plane dst(src, false);

    return Bilateral2D(dst, src, src, d, 0);
}

inline Frame Bilateral2D(const Frame &src, const Frame &ref, const Bilateral2D_Data &d)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
    {
        Bilateral2D(dst.P(i), src.P(i), ref.P(i), d, i);
    }

    return dst;
}

inline Frame Bilateral2D(const Frame &src, const Bilateral2D_Data &d)
{
    Frame dst(src, false);

    for (Frame::PlaneCountType i = 0; i < src.PlaneCount(); i++)
    {
        Bilateral2D(dst.P(i), src.P(i), src.P(i), d, i);
    }

    return dst;
}


class Bilateral2D_IO
    : public FilterIO
{
public:
    typedef Bilateral2D_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    Bilateral2D_Para para;
    std::string RPath;

    virtual void arguments_process()
    {
        FilterIO::arguments_process();

        Args ArgsObj(argc, args);

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "--ref")
            {
                ArgsObj.GetPara(i, RPath);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--sigmaS")
            {
                ArgsObj.GetPara(i, para.sigmaS);
                continue;
            }
            if (args[i] == "-R" || args[i] == "--sigmaR")
            {
                ArgsObj.GetPara(i, para.sigmaR);
                continue;
            }
            if (args[i] == "-A" || args[i] == "--algorithm")
            {
                ArgsObj.GetPara(i, para.algorithm);
                continue;
            }
            if (args[i] == "-N" || args[i] == "--PBFICnum")
            {
                ArgsObj.GetPara(i, para.PBFICnum);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        ArgsObj.Check();
    }

    virtual Frame process(const Frame &src)
    {
        if (RPath.size() == 0)
        {
            Bilateral2D_Data data(src, para);
            return Bilateral2D(src, data);
        }
        else
        {
            const Frame ref = ImageReader(RPath);
            Bilateral2D_Data data(ref, para);
            return Bilateral2D(src, ref, data);
        }
    }

public:
    _Myt(std::string _Tag = ".Bilateral")
        : _Mybase(std::move(_Tag)) {}
};


#endif
