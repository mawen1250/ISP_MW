#ifndef BILATERAL_H_
#define BILATERAL_H_


#include "IO.h"
#include "Image_Type.h"
#include "Type_Conv.h"
#include "LUT.h"
#include "Gaussian.h"


const struct Bilateral2D_Para {
    double sigmaS = 3.0;
    double sigmaR = 0.02;
    int algorithm = 0;
    int PBFICnum = 0;
} Bilateral2D_Default;


class Bilateral2D_Data {
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
    Bilateral2D_Data(const Plane & input, double _sigmaS = Bilateral2D_Default.sigmaS, double _sigmaR = Bilateral2D_Default.sigmaR,
        int _algorithm = Bilateral2D_Default.algorithm, int _PBFICnum = Bilateral2D_Default.PBFICnum)
        : PlaneCount(1), isChroma(input.isChroma()), BPS(input.BitDepth()),
        sigmaS(PlaneCount), sigmaR(PlaneCount), process(PlaneCount), algorithm(PlaneCount), radius0(PlaneCount), PBFICnum(PlaneCount),
        radius(PlaneCount), samples(PlaneCount), step(PlaneCount), GS_LUT(PlaneCount), GR_LUT(PlaneCount)
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            sigmaS[i] = _sigmaS;
            sigmaR[i] = _sigmaR;
            algorithm[i] = _algorithm;
            PBFICnum[i] = _PBFICnum;
        }

        process_define();

        Bilateral2D_0_Paras();
        Bilateral2D_1_Paras();
        Bilateral2D_2_Paras();

        algorithm_select();

        GS_LUT_Init();
        GR_LUT_Init();
    }

    Bilateral2D_Data(const Frame & input, double _sigmaS = Bilateral2D_Default.sigmaS, double _sigmaR = Bilateral2D_Default.sigmaR,
        int _algorithm = Bilateral2D_Default.algorithm, int _PBFICnum = Bilateral2D_Default.PBFICnum)
        : PlaneCount(input.PlaneCount()), isYUV(input.isYUV()), BPS(input.BitDepth()),
        sigmaS(PlaneCount), sigmaR(PlaneCount), process(PlaneCount), algorithm(PlaneCount), radius0(PlaneCount), PBFICnum(PlaneCount),
        radius(PlaneCount), samples(PlaneCount), step(PlaneCount), GS_LUT(PlaneCount), GR_LUT(PlaneCount)
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            sigmaS[i] = _sigmaS;
            sigmaR[i] = _sigmaR;
            algorithm[i] = _algorithm;
            PBFICnum[i] = _PBFICnum;
        }

        process_define();

        Bilateral2D_0_Paras();
        Bilateral2D_1_Paras();
        Bilateral2D_2_Paras();

        algorithm_select();

        GS_LUT_Init();
        GR_LUT_Init();
    }

    Bilateral2D_Data & Bilateral2D_0_Paras()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (process[i])
            {
                radius0[i] = Max(static_cast<int>(sigmaS[i]*sigmaSMul + 0.5), 1);
            }
        }

        return *this;
    }

    Bilateral2D_Data & Bilateral2D_1_Paras()
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

                if (i > 0 && (isChroma || isYUV) && PBFICnum[i] % 2 == 0 && PBFICnum[i] < 256) // Set odd PBFIC number to chroma planes by default
                    PBFICnum[i]++;
            }
        }

        return *this;
    }

    Bilateral2D_Data & Bilateral2D_2_Paras()
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

    Bilateral2D_Data & process_define()
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

    Bilateral2D_Data & algorithm_select()
    {
        for (int i = 0; i < PlaneCount; i++)
        {
            if (algorithm[i] <= 0)
                algorithm[i] = step[i] == 1 ? 2 : sigmaR[i] < 0.08 && samples[i] < 5 ? 2
                : 4 * samples[i] * samples[i] <= 15 * PBFICnum[i] ? 2 : 1;
        }

        return *this;
    }

    Bilateral2D_Data & GS_LUT_Init()
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

    Bilateral2D_Data & GR_LUT_Init()
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


int Bilateral2D_IO(const int argc, const std::vector<std::string> &args);


Plane & Bilateral2D(Plane & output, const Plane & input, const Plane & ref, const Bilateral2D_Data &d, int plane = 0);
Plane & Bilateral2D_0(Plane & dst, const Plane & src, const Plane & ref, const Bilateral2D_Data &d, int plane = 0);
Plane & Bilateral2D_1(Plane & dst, const Plane & src, const Plane & ref, const Bilateral2D_Data &d, int plane = 0);
Plane & Bilateral2D_2(Plane & dst, const Plane & src, const Plane & ref, const Bilateral2D_Data &d, int plane = 0);
Plane & Bilateral2D_2(Plane & dst, const Plane & src, const Bilateral2D_Data &d, int plane = 0);


inline Plane Bilateral2D(const Plane & input, const Plane & ref, const Bilateral2D_Data &d)
{
    Plane output(input, false);

    return Bilateral2D(output, input, ref, d, 0);
}

inline Plane Bilateral2D(const Plane & input, const Bilateral2D_Data &d)
{
    Plane output(input, false);

    return Bilateral2D(output, input, input, d, 0);
}

inline Frame Bilateral2D(const Frame & input, const Frame & ref, const Bilateral2D_Data &d)
{
    Frame output(input, false);

    for (Frame::PlaneCountType i = 0; i < input.PlaneCount(); i++)
    {
        Bilateral2D(output.P(i), input.P(i), ref.P(i), d, i);
    }

    return output;
}

inline Frame Bilateral2D(const Frame & input, const Bilateral2D_Data &d)
{
    Frame output(input, false);

    for (Frame::PlaneCountType i = 0; i < input.PlaneCount(); i++)
    {
        Bilateral2D(output.P(i), input.P(i), input.P(i), d, i);
    }

    return output;
}


#endif