#ifndef BM3D_H_
#define BM3D_H_


#include "fftw3_helper.hpp"
#include "Filter.h"
#include "Image_Type.h"
#include "Helper.h"
#include "Block.h"


struct BM3D_Para_Base
{
    std::string profile;
    std::vector<double> sigma;
    PCType BlockSize;
    PCType BlockStep;
    PCType GroupSize;
    PCType BMrange;
    PCType BMstep;
    double thMSE;
    double lambda;

    BM3D_Para_Base(std::string _profile = "lc")
        : profile(_profile), sigma({ 10.0, 10.0, 10.0 })
    {
        BMrange = 16;
        BMstep = 1;

        if (profile == "lc")
        {
            BMrange = 9;
        }
    }

    virtual void thMSE_Default() {}
};

const struct BM3D_Basic_Para
    : public BM3D_Para_Base
{
    typedef BM3D_Basic_Para _Myt;
    typedef BM3D_Para_Base _Mybase;

    _Myt(std::string _profile = "lc")
        : _Mybase(_profile)
    {
        BlockSize = 8;
        BlockStep = 3;
        GroupSize = 16;
        lambda = 2.5;

        if (profile == "lc")
        {
            BlockStep = 6;
        }
        else if (profile == "vn")
        {
            BlockStep = 4;
            GroupSize = 32;
        }
        else if (profile == "high")
        {
            BlockStep = 2;
        }
        else if (profile != "np")
        {
            std::cerr << "BM3D_Para: unrecognized profile \"" << profile << "\","
                " should be \"lc\", \"np\", \"vn\" or \"high\"\n";
        }

        thMSE_Default();
    }

    virtual void thMSE_Default()
    {
        thMSE = 400 + sigma[0] * 80;

        if (profile == "vn")
        {
            thMSE = 1000 + sigma[0] * 150;
        }
    }
} BM3D_Basic_Default;

const struct BM3D_Final_Para
    : public BM3D_Para_Base
{
    typedef BM3D_Final_Para _Myt;
    typedef BM3D_Para_Base _Mybase;

    _Myt(std::string _profile = "lc")
        : _Mybase(_profile)
    {
        BlockSize = 8;
        BlockStep = 3;
        GroupSize = 32;

        if (profile == "lc")
        {
            BlockStep = 5;
            GroupSize = 16;
        }
        else if (profile == "vn")
        {
            BlockSize = 11;
            BlockStep = 6;
        }
        else if (profile == "high")
        {
            BlockStep = 2;
        }
        else if (profile != "np")
        {
            std::cerr << "BM3D_Para: unrecognized profile \"" << profile << "\","
                " should be \"lc\", \"np\", \"vn\" or \"high\"\n";
        }

        thMSE_Default();
    }

    virtual void thMSE_Default()
    {
        thMSE = 200 + sigma[0] * 10;

        if (profile == "vn")
        {
            thMSE = 400 + sigma[0] * 40;
        }
    }
} BM3D_Final_Default;

const struct BM3D_Para
{
    BM3D_Basic_Para basic;
    BM3D_Final_Para final;

    BM3D_Para(std::string _profile = "lc")
        : basic(_profile), final(_profile)
    {}

    void thMSE_Default()
    {
        basic.thMSE_Default();
        final.thMSE_Default();
    }
} BM3D_Default;


struct BM3D_Data
{
    typedef BM3D_Data _Myt;

    typedef fftwh<FLType> fftw;

    std::vector<fftw::plan> fp;
    std::vector<fftw::plan> bp;
    std::vector<double> finalAMP;
    std::vector<std::vector<FLType>> thrTable;
    std::vector<FLType> wienerSigmaSqr;

    _Myt(bool wiener, double sigma, PCType GroupSize, PCType BlockSize, double lambda,
        unsigned int flags = FFTW_PATIENT,
        fftw::r2r_kind fkind = FFTW_REDFT01, fftw::r2r_kind bkind = FFTW_REDFT10)
        : fp(GroupSize), bp(GroupSize), finalAMP(GroupSize), thrTable(wiener ? 0 : GroupSize),
        wienerSigmaSqr(wiener ? GroupSize : 0)
    {
        FLType *temp = nullptr;

        for (PCType i = 1; i <= GroupSize; ++i)
        {
            AlignedMalloc(temp, i * BlockSize * BlockSize);
            fp[i - 1].r2r_3d(i, BlockSize, BlockSize, temp, temp, fkind, fkind, fkind, flags);
            bp[i - 1].r2r_3d(i, BlockSize, BlockSize, temp, temp, bkind, bkind, bkind, flags);
            AlignedFree(temp);

            finalAMP[i - 1] = 2 * i * 2 * BlockSize * 2 * BlockSize;
            double forwardAMP = sqrt(finalAMP[i - 1]);

            if (wiener)
            {
                wienerSigmaSqr[i - 1] = static_cast<FLType>(sigma * forwardAMP * sigma * forwardAMP);
            }
            else
            {
                double thrBase = sigma * lambda * forwardAMP;
                std::vector<double> thr(4);
                thr[0] = thrBase;
                thr[1] = thrBase;
                thr[2] = thrBase / 2.;
                thr[3] = 0;

                thrTable[i - 1] = std::vector<FLType>(i * BlockSize * BlockSize);
                auto thr_d = thrTable[i - 1].data();

                for (PCType z = 0; z < i; ++z)
                {
                    for (PCType y = 0; y < BlockSize; ++y)
                    {
                        for (PCType x = 0; x < BlockSize; ++x, ++thr_d)
                        {
                            int flag = 0;
                            double scale = 1;

                            if (x == 0)
                            {
                                ++flag;
                            }
                            else if (x >= BlockSize / 2)
                            {
                                scale *= 1.07;
                            }
                            else if (x >= BlockSize / 4)
                            {
                                scale *= 1.01;
                            }

                            if (y == 0)
                            {
                                ++flag;
                            }
                            else if (y >= BlockSize / 2)
                            {
                                scale *= 1.07;
                            }
                            else if (y >= BlockSize / 4)
                            {
                                scale *= 1.01;
                            }

                            if (z == 0)
                            {
                                ++flag;
                            }
                            else if (z >= i / 2)
                            {
                                scale *= 1.07;
                            }
                            else if (z >= i / 4)
                            {
                                scale *= 1.01;
                            }

                            *thr_d = static_cast<FLType>(thr[flag] * scale);
                        }
                    }
                }
            }
        }
    }

    _Myt(const _Myt &right) = delete;

    _Myt(_Myt &&right)
        : fp(std::move(right.fp)), bp(std::move(right.bp)),
        finalAMP(std::move(right.finalAMP)), thrTable(std::move(right.thrTable)),
        wienerSigmaSqr(std::move(right.wienerSigmaSqr))
    {}

    _Myt &operator=(const _Myt &right) = delete;

    _Myt &operator=(_Myt &&right)
    {
        fp = std::move(right.fp);
        bp = std::move(right.bp);
        finalAMP = std::move(right.finalAMP);
        thrTable = std::move(right.thrTable);
        wienerSigmaSqr = std::move(right.wienerSigmaSqr);

        return *this;
    }
};


// BM3D denoising algorithm based on block matching and collaborative filtering of grouped blocks
class BM3D_Base
    : public FilterIF2
{
public:
    typedef BM3D_Base _Myt;
    typedef FilterIF2 _Mybase;

    typedef Block<FLType, FLType> block_type;
    typedef block_type::KeyType KeyType;
    typedef block_type::PosType PosType;
    typedef block_type::PosPair PosPair;
    typedef block_type::PosPairCode PosPairCode;
    typedef block_type::KeyCode KeyCode;
    typedef block_type::PosCode PosCode;

protected:
    BM3D_Para_Base para;
    std::vector<BM3D_Data> data;

public:
    _Myt(const BM3D_Para_Base &_para, bool wiener = false)
        : para(_para), data()
    {
        // Adjust sigma and thMSE to fit for the unnormalized YUV color space
        double Yr, Yg, Yb, Ur, Ug, Ub, Vr, Vg, Vb;
        ColorMatrix_RGB2YUV_Parameter(ColorMatrix::OPP, Yr, Yg, Yb, Ur, Ug, Ub, Vr, Vg, Vb);

        para.thMSE *= sqrt(Yr * Yr + Yg * Yg + Yb * Yb);

        // Initialize BM3D data - FFTW plans, unnormalized transform amplification factor, hard threshold table, etc.
        data.push_back(BM3D_Data(wiener, para.sigma[0] * sqrt(Yr * Yr + Yg * Yg + Yb * Yb) / double(255),
            para.GroupSize, para.BlockSize, para.lambda));
        data.push_back(BM3D_Data(wiener, para.sigma[1] * sqrt(Ur * Ur + Ug * Ug + Ub * Ub) / double(255),
            para.GroupSize, para.BlockSize, para.lambda));
        data.push_back(BM3D_Data(wiener, para.sigma[2] * sqrt(Vr * Vr + Vg * Vg + Vb * Vb) / double(255),
            para.GroupSize, para.BlockSize, para.lambda));
    }

    void Kernel(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref) const;

    void Kernel(Plane_FL &dstY, Plane_FL &dstU, Plane_FL &dstV,
        const Plane_FL &srcY, const Plane_FL &srcU, const Plane_FL &srcV,
        const Plane_FL &refY, const Plane_FL &refU, const Plane_FL &refV) const;

    virtual bool RGB2YUV(Plane_FL &srcY, Plane_FL &srcU, Plane_FL &srcV,
        Plane_FL &refY, Plane_FL &refU, Plane_FL &refV,
        const Plane &srcR, const Plane &srcG, const Plane &srcB,
        const Plane &refR, const Plane &refG, const Plane &refB) const = 0;

protected:
    virtual Plane_FL &process_Plane_FL(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref);
    virtual Plane &process_Plane(Plane &dst, const Plane &src, const Plane &ref);
    virtual Frame &process_Frame(Frame &dst, const Frame &src, const Frame &ref);

protected:
    virtual void CollaborativeFilter(const BM3D_Data &d,
        Plane_FL &ResNum, Plane_FL &ResDen,
        const Plane_FL &src, const Plane_FL &ref,
        const PosPairCode &posPairCode) const = 0;
};


class BM3D_Basic
    : public BM3D_Base
{
public:
    typedef BM3D_Basic _Myt;
    typedef BM3D_Base _Mybase;

public:
    _Myt(const BM3D_Basic_Para &_para = BM3D_Basic_Default)
        : _Mybase(_para, false)
    {}

    virtual bool RGB2YUV(Plane_FL &srcY, Plane_FL &srcU, Plane_FL &srcV,
        Plane_FL &refY, Plane_FL &refU, Plane_FL &refV,
        const Plane &srcR, const Plane &srcG, const Plane &srcB,
        const Plane &refR, const Plane &refG, const Plane &refB) const;

protected:
    virtual void CollaborativeFilter(const BM3D_Data &d,
        Plane_FL &ResNum, Plane_FL &ResDen,
        const Plane_FL &src, const Plane_FL &ref,
        const PosPairCode &posPairCode) const;
};


class BM3D_Final
    : public BM3D_Base
{
public:
    typedef BM3D_Final _Myt;
    typedef BM3D_Base _Mybase;

public:
    _Myt(const BM3D_Final_Para &_para = BM3D_Final_Default)
        : _Mybase(_para, true)
    {}

    virtual bool RGB2YUV(Plane_FL &srcY, Plane_FL &srcU, Plane_FL &srcV,
        Plane_FL &refY, Plane_FL &refU, Plane_FL &refV,
        const Plane &srcR, const Plane &srcG, const Plane &srcB,
        const Plane &refR, const Plane &refG, const Plane &refB) const;

protected:
    virtual void CollaborativeFilter(const BM3D_Data &d,
        Plane_FL &ResNum, Plane_FL &ResDen,
        const Plane_FL &src, const Plane_FL &ref,
        const PosPairCode &posPairCode) const;
};


class BM3D
    : public FilterIF2
{
public:
    typedef BM3D _Myt;
    typedef FilterIF2 _Mybase;

protected:
    BM3D_Basic basic;
    BM3D_Final final;

public:
    _Myt(const BM3D_Para &_para = BM3D_Default)
        : basic(_para.basic), final(_para.final)
    {}

protected:
    virtual Plane_FL &process_Plane_FL(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref);
    virtual Plane &process_Plane(Plane &dst, const Plane &src, const Plane &ref);
    virtual Frame &process_Frame(Frame &dst, const Frame &src, const Frame &ref);
};


class BM3D_IO
    : public FilterIO
{
public:
    typedef BM3D_IO _Myt;
    typedef FilterIO _Mybase;

protected:
    BM3D_Para para;
    std::string RPath;

    virtual void arguments_process()
    {
        _Mybase::arguments_process();

        Args ArgsObj(argc, args);

        std::string profile;

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "-P" || args[i] == "--profile")
            {
                ArgsObj.GetPara(i, profile, 1);
                para.basic = BM3D_Basic_Para(profile);
                para.final = BM3D_Final_Para(profile);
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        bool thMSE1_def = false;
        bool thMSE2_def = false;
        para.basic.sigma.clear();

        for (int i = 0; i < argc; i++)
        {
            if (args[i] == "--ref")
            {
                ArgsObj.GetPara(i, RPath);
                continue;
            }
            if (args[i] == "-S" || args[i] == "--sigma")
            {
                double sigma;
                ArgsObj.GetPara(i, sigma);
                para.basic.sigma.push_back(sigma);
                continue;
            }
            if (args[i] == "-BS1" || args[i] == "--BlockSize1")
            {
                ArgsObj.GetPara(i, para.basic.BlockSize);
                continue;
            }
            if (args[i] == "-BSP1" || args[i] == "--BlockStep1")
            {
                ArgsObj.GetPara(i, para.basic.BlockStep);
                continue;
            }
            if (args[i] == "-GS1" || args[i] == "--GroupSize1")
            {
                ArgsObj.GetPara(i, para.basic.GroupSize);
                continue;
            }
            if (args[i] == "-MR1" || args[i] == "--BMrange1")
            {
                ArgsObj.GetPara(i, para.basic.BMrange);
                continue;
            }
            if (args[i] == "-MS1" || args[i] == "--BMstep1")
            {
                ArgsObj.GetPara(i, para.basic.BMstep);
                continue;
            }
            if (args[i] == "-TH1" || args[i] == "--thMSE1")
            {
                ArgsObj.GetPara(i, para.basic.thMSE);
                thMSE1_def = true;
                continue;
            }
            if (args[i] == "-LD" || args[i] == "--lambda")
            {
                ArgsObj.GetPara(i, para.basic.lambda);
                continue;
            }
            if (args[i] == "-BS2" || args[i] == "--BlockSize2")
            {
                ArgsObj.GetPara(i, para.final.BlockSize);
                continue;
            }
            if (args[i] == "-BSP2" || args[i] == "--BlockStep2")
            {
                ArgsObj.GetPara(i, para.final.BlockStep);
                continue;
            }
            if (args[i] == "-GS2" || args[i] == "--GroupSize2")
            {
                ArgsObj.GetPara(i, para.final.GroupSize);
                continue;
            }
            if (args[i] == "-MR2" || args[i] == "--BMrange2")
            {
                ArgsObj.GetPara(i, para.final.BMrange);
                continue;
            }
            if (args[i] == "-MS2" || args[i] == "--BMstep2")
            {
                ArgsObj.GetPara(i, para.final.BMstep);
                continue;
            }
            if (args[i] == "-TH2" || args[i] == "--thMSE2")
            {
                ArgsObj.GetPara(i, para.final.thMSE);
                thMSE2_def = true;
                continue;
            }
            if (args[i][0] == '-')
            {
                i++;
                continue;
            }
        }

        ArgsObj.Check();

        if (para.basic.sigma.size() == 0)
        {
            para.basic.sigma = BM3D_Basic_Default.sigma;
        }
        else while (para.basic.sigma.size() < 3)
        {
            para.basic.sigma.push_back(para.basic.sigma.end()[-1]);
        }

        para.final.sigma = para.basic.sigma;

        if (!thMSE1_def) para.basic.thMSE_Default();
        if (!thMSE2_def) para.final.thMSE_Default();
    }

    virtual Frame process(const Frame &src)
    {
        BM3D filter(para);

        if (RPath.size() == 0)
        {
            return filter(src);
        }
        else
        {
            const Frame ref = ImageReader(RPath);
            return filter(src, ref);
        }
    }

public:
    _Myt(std::string _Tag = ".BM3D")
        : _Mybase(std::move(_Tag))
    {}
};


#endif
