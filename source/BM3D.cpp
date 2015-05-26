#include "BM3D.h"
#include "Conversion.hpp"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions of struct BM3D_FilterData


BM3D_FilterData::BM3D_FilterData(bool wiener, double sigma, PCType GroupSize, PCType BlockSize, double lambda)
    : fp(GroupSize), bp(GroupSize), finalAMP(GroupSize), thrTable(wiener ? 0 : GroupSize),
    wienerSigmaSqr(wiener ? GroupSize : 0)
{
    const unsigned int flags = FFTW_PATIENT;
    const fftw::r2r_kind fkind = FFTW_REDFT10;
    const fftw::r2r_kind bkind = FFTW_REDFT01;

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
            thr[1] = thrBase * sqrt(double(2));
            thr[2] = thrBase * double(2);
            thr[3] = thrBase * sqrt(double(8));

            thrTable[i - 1] = std::vector<FLType>(i * BlockSize * BlockSize);
            auto thr_d = thrTable[i - 1].data();

            for (PCType z = 0; z < i; ++z)
            {
                for (PCType y = 0; y < BlockSize; ++y)
                {
                    for (PCType x = 0; x < BlockSize; ++x, ++thr_d)
                    {
                        int flag = 0;

                        if (x == 0)
                        {
                            ++flag;
                        }
                        if (y == 0)
                        {
                            ++flag;
                        }
                        if (z == 0)
                        {
                            ++flag;
                        }

                        *thr_d = static_cast<FLType>(thr[flag]);
                    }
                }
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions of class BM3D_Base


void BM3D_Base::Kernel(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref) const
{
    if (para.sigma[0] <= 0 || para.GroupSize == 0 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return;
    }

    Plane_FL ResNum(dst, true, 0);
    Plane_FL ResDen(dst, true, 0);

    PCType height = src.Height();
    PCType width = src.Width();

    PCType BlockPosRight = width - para.BlockSize;
    PCType BlockPosBottom = height - para.BlockSize;

    for (PCType j = 0;; j += para.BlockStep)
    {
        // Handle scan of reference block - vertical
        if (j >= BlockPosBottom + para.BlockStep)
        {
            break;
        }
        else if (j > BlockPosBottom)
        {
            j = BlockPosBottom;
        }

        for (PCType i = 0;; i += para.BlockStep)
        {
            // Handle scan of reference block - horizontal
            if (i >= BlockPosRight + para.BlockStep)
            {
                break;
            }
            else if (i > BlockPosRight)
            {
                i = BlockPosRight;
            }

            PosPairCode matchCode = BlockMatching(ref, j, i);

            // Get the filtered result through collaborative filtering and aggregation of matched blocks
            CollaborativeFilter(0, ResNum, ResDen, src, ref, matchCode);
        }
    }

    // The filtered blocks are sumed and averaged to form the final filtered image
    dst.ReSize(width, height);

    _Transform(dst, ResNum, ResDen, [](FLType num, FLType den)
    {
        return num / den;
    });
}


void BM3D_Base::Kernel(Plane_FL &dstY, Plane_FL &dstU, Plane_FL &dstV,
    const Plane_FL &srcY, const Plane_FL &srcU, const Plane_FL &srcV,
    const Plane_FL &refY, const Plane_FL &refU, const Plane_FL &refV) const
{
    if ((para.sigma[0] <= 0 && para.sigma[1] <= 0 && para.sigma[2] <= 0)
        || para.GroupSize == 0 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dstY = srcY;
        dstU = srcU;
        dstV = srcV;
        return;
    }

    Plane_FL ResNumY(srcY, true, 0);
    Plane_FL ResDenY(srcY, true, 0);
    Plane_FL ResNumU(srcU, true, 0);
    Plane_FL ResDenU(srcU, true, 0);
    Plane_FL ResNumV(srcV, true, 0);
    Plane_FL ResDenV(srcV, true, 0);

    PCType height = srcY.Height();
    PCType width = srcY.Width();

    PCType BlockPosRight = width - para.BlockSize;
    PCType BlockPosBottom = height - para.BlockSize;

    for (PCType j = 0;; j += para.BlockStep)
    {
        // Handle scan of reference block - vertical
        if (j >= BlockPosBottom + para.BlockStep)
        {
            break;
        }
        else if (j > BlockPosBottom)
        {
            j = BlockPosBottom;
        }

        for (PCType i = 0;; i += para.BlockStep)
        {
            // Handle scan of reference block - horizontal
            if (i >= BlockPosRight + para.BlockStep)
            {
                break;
            }
            else if (i > BlockPosRight)
            {
                i = BlockPosRight;
            }

            PosPairCode matchCode = BlockMatching(refY, j, i);

            // Get the filtered result through collaborative filtering and aggregation of matched blocks
            if (para.sigma[0] > 0) CollaborativeFilter(0, ResNumY, ResDenY, srcY, refY, matchCode);
            if (para.sigma[1] > 0) CollaborativeFilter(1, ResNumU, ResDenU, srcU, refU, matchCode);
            if (para.sigma[2] > 0) CollaborativeFilter(2, ResNumV, ResDenV, srcV, refV, matchCode);
        }
    }

    // The filtered blocks are sumed and averaged to form the final filtered image
    dstY.ReSize(width, height);
    dstU.ReSize(width, height);
    dstV.ReSize(width, height);

    if (para.sigma[0] > 0) _Transform(dstY, ResNumY, ResDenY, [](FLType num, FLType den)
    {
        return num / den;
    });

    if (para.sigma[1] > 0) _Transform(dstU, ResNumU, ResDenU, [](FLType num, FLType den)
    {
        return num / den;
    });

    if (para.sigma[2] > 0) _Transform(dstV, ResNumV, ResDenV, [](FLType num, FLType den)
    {
        return num / den;
    });
}


BM3D_Base::PosPairCode BM3D_Base::BlockMatching(
    const Plane_FL &ref, PCType j, PCType i) const
{
    // Skip block matching if GroupSize is 1 or thMSE is not positive,
    // and take the reference block as the only element in the group
    if (para.GroupSize == 1 || para.thMSE <= 0)
    {
        return PosPairCode(1, PosPair(KeyType(0), PosType(j, i)));
    }

    // Get reference block from the reference plane
    block_type refBlock(ref, para.BlockSize, para.BlockSize, PosType(j, i));

    // Block matching
    return refBlock.BlockMatchingMulti(ref, para.BMrange, para.BMstep, para.thMSE, 1, para.GroupSize, true);
}


Plane_FL &BM3D_Base::process_Plane_FL(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref)
{
    // Execute kernel
    Kernel(dst, src, ref);

    return dst;
}


Plane &BM3D_Base::process_Plane(Plane &dst, const Plane &src, const Plane &ref)
{
    if (para.sigma[0] <= 0 || para.GroupSize == 1 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return dst;
    }

    Plane_FL dst_data(dst, false);
    Plane_FL src_data(src, false);

    RangeConvert(src_data, src, false);

    // Execute kernel
    if (src.data() == ref.data())
    {
        process_Plane_FL(dst_data, src_data, src_data);
    }
    else
    {
        Plane_FL ref_data(ref, false);

        RangeConvert(ref_data, ref, false);

        process_Plane_FL(dst_data, src_data, ref_data);
    }

    RangeConvert(dst, dst_data, true);

    return dst;
}


Frame &BM3D_Base::process_Frame(Frame &dst, const Frame &src, const Frame &ref)
{
    if ((para.sigma[0] <= 0 && para.sigma[1] <= 0 && para.sigma[2] <= 0)
        || para.GroupSize == 1 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return dst;
    }

    Plane_FL srcY, srcU, srcV;
    Plane_FL refY, refU, refV;

    // Convert source image and reference image from RGB to YUV
    bool ref_equal_src = RGB2YUV(srcY, srcU, srcV, refY, refU, refV,
        src.R(), src.G(), src.B(), ref.R(), ref.G(), ref.B());

    // Execute kernel
    if (ref_equal_src)
    {
        Kernel(srcY, srcU, srcV, srcY, srcU, srcV, srcY, srcU, srcV);
    }
    else
    {
        Kernel(srcY, srcU, srcV, srcY, srcU, srcV, refY, refU, refV);
    }

    // Convert filtered image from YUV to RGB
    MatrixConvert_YUV2RGB(dst.R(), dst.G(), dst.B(), srcY, srcU, srcV, ColorMatrix::OPP, true);

    return dst;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions of class BM3D_Basic


bool BM3D_Basic::RGB2YUV(Plane_FL &srcY, Plane_FL &srcU, Plane_FL &srcV,
    Plane_FL &refY, Plane_FL &refU, Plane_FL &refV,
    const Plane &srcR, const Plane &srcG, const Plane &srcB,
    const Plane &refR, const Plane &refG, const Plane &refB) const
{
    MatrixConvert_RGB2YUV(srcY, srcU, srcV, srcR, srcG, srcB, ColorMatrix::OPP, false);

    if (srcR.data() == refR.data() && srcG.data() == refG.data() && srcB.data() == refB.data())
    {
        return true;
    }
    else
    {
        ConvertToY(refY, refR, refG, refB, ColorMatrix::OPP, false);
        return false;
    }
}


void BM3D_Basic::CollaborativeFilter(int plane,
    Plane_FL &ResNum, Plane_FL &ResDen,
    const Plane_FL &src, const Plane_FL &ref,
    const PosPairCode &code) const
{
    PCType GroupSize = static_cast<PCType>(code.size());
    // When para.GroupSize > 0, limit GroupSize up to para.GroupSize
    if (para.GroupSize > 0 && GroupSize > para.GroupSize)
    {
        GroupSize = para.GroupSize;
    }

    // Construct source group guided by matched pos code
    BlockGroup<FLType, FLType> srcGroup(src, code, GroupSize, para.BlockSize, para.BlockSize);

    // Initialize retianed coefficients of hard threshold filtering
    int retainedCoefs = 0;

    // Apply forward 3D transform to the source group
    f[plane].fp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Apply hard-thresholding to the source group
    Block_For_each(srcGroup, f[plane].thrTable[GroupSize - 1], [&](FLType &x, FLType y)
    {
        if (Abs(x) <= y)
        {
            x = 0;
        }
        else
        {
            ++retainedCoefs;
        }
    });

    // Apply backward 3D transform to the filtered group
    f[plane].bp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Calculate weight for the filtered group
    // Also include the normalization factor to compensate for the amplification introduced in 3D transform
    FLType denWeight = retainedCoefs < 1 ? 1 : FLType(1) / static_cast<FLType>(retainedCoefs);
    FLType numWeight = static_cast<FLType>(denWeight / f[plane].finalAMP[GroupSize - 1]);

    // Store the weighted filtered group to the numerator part of the final estimation
    // Store the weight to the denominator part of the final estimation
    srcGroup.AddTo(ResNum, numWeight);
    srcGroup.CountTo(ResDen, denWeight);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions of class BM3D_Final


bool BM3D_Final::RGB2YUV(Plane_FL &srcY, Plane_FL &srcU, Plane_FL &srcV,
    Plane_FL &refY, Plane_FL &refU, Plane_FL &refV,
    const Plane &srcR, const Plane &srcG, const Plane &srcB,
    const Plane &refR, const Plane &refG, const Plane &refB) const
{
    MatrixConvert_RGB2YUV(srcY, srcU, srcV, srcR, srcG, srcB, ColorMatrix::OPP, false);

    if (srcR.data() == refR.data() && srcG.data() == refG.data() && srcB.data() == refB.data())
    {
        return true;
    }
    else
    {
        MatrixConvert_RGB2YUV(refY, refU, refV, refR, refG, refB, ColorMatrix::OPP, false);
        return false;
    }
}


void BM3D_Final::CollaborativeFilter(int plane,
    Plane_FL &ResNum, Plane_FL &ResDen,
    const Plane_FL &src, const Plane_FL &ref,
    const PosPairCode &code) const
{
    PCType GroupSize = static_cast<PCType>(code.size());
    // When para.GroupSize > 0, limit GroupSize up to para.GroupSize
    if (para.GroupSize > 0 && GroupSize > para.GroupSize)
    {
        GroupSize = para.GroupSize;
    }

    // Construct source group and reference group guided by matched pos code
    BlockGroup<FLType, FLType> srcGroup(src, code, GroupSize, para.BlockSize, para.BlockSize);
    BlockGroup<FLType, FLType> refGroup(ref, code, GroupSize, para.BlockSize, para.BlockSize);

    // Initialize L2-norm of Wiener coefficients
    FLType L2Wiener = 0;

    // Apply forward 3D transform to the source group and the reference group
    f[plane].fp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());
    f[plane].fp[GroupSize - 1].execute_r2r(refGroup.data(), refGroup.data());

    // Apply empirical Wiener filtering to the source group guided by the reference group
    const FLType sigmaSquare = f[plane].wienerSigmaSqr[GroupSize - 1];

    Block_For_each(srcGroup, refGroup, [&](FLType &x, FLType y)
    {
        FLType ySquare = y * y;
        FLType wienerCoef = ySquare / (ySquare + sigmaSquare);
        x *= wienerCoef;
        L2Wiener += wienerCoef * wienerCoef;
    });

    // Apply backward 3D transform to the filtered group
    f[plane].bp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Calculate weight for the filtered group
    // Also include the normalization factor to compensate for the amplification introduced in 3D transform
    FLType denWeight = L2Wiener <= 0 ? 1 : FLType(1) / L2Wiener;
    FLType numWeight = static_cast<FLType>(denWeight / f[plane].finalAMP[GroupSize - 1]);

    // Store the weighted filtered group to the numerator part of the final estimation
    // Store the weight to the denominator part of the final estimation
    srcGroup.AddTo(ResNum, numWeight);
    srcGroup.CountTo(ResDen, denWeight);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions of class BM3D


Plane_FL &BM3D::process_Plane_FL(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref)
{
    Plane_FL tmp;

    basic.Kernel(tmp, src, ref);
    final.Kernel(dst, src, tmp);

    return dst;
}


Plane &BM3D::process_Plane(Plane &dst, const Plane &src, const Plane &ref)
{
    Plane_FL dst_data(dst, false);
    Plane_FL src_data(src, false);

    RangeConvert(src_data, src, false);

    if (src.data() == ref.data())
    {
        process_Plane_FL(dst_data, src_data, src_data);
    }
    else
    {
        Plane_FL ref_data(ref, false);

        RangeConvert(ref_data, ref, false);

        process_Plane_FL(dst_data, src_data, ref_data);
    }

    RangeConvert(dst, dst_data, true);

    return dst;
}


Frame &BM3D::process_Frame(Frame &dst, const Frame &src, const Frame &ref)
{
    Plane_FL tmpY, tmpU, tmpV;
    Plane_FL srcY, srcU, srcV;
    Plane_FL refY, refU, refV;

    // Convert source image and reference image from RGB to YUV
    bool ref_equal_src = basic.RGB2YUV(srcY, srcU, srcV, refY, refU, refV,
        src.R(), src.G(), src.B(), ref.R(), ref.G(), ref.B());

    // Execute kernel
    if (ref_equal_src)
    {
        basic.Kernel(tmpY, tmpU, tmpV, srcY, srcU, srcV, srcY, srcU, srcV);
    }
    else
    {
        basic.Kernel(tmpY, tmpU, tmpV, srcY, srcU, srcV, refY, refU, refV);
    }

    final.Kernel(srcY, srcU, srcV, srcY, srcU, srcV, tmpY, tmpU, tmpV);

    // Convert filtered image from YUV to RGB
    MatrixConvert_YUV2RGB(dst.R(), dst.G(), dst.B(), srcY, srcU, srcV, ColorMatrix::OPP, true);

    return dst;
}
