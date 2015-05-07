#include "BM3D.h"
#include "Conversion.hpp"


// Functions of BM3D_Base
void BM3D_Base::Kernel(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref) const
{
    if (para.sigma[0] <= 0 || para.GroupSize == 0 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return;
    }

    PCType height = src.Height();
    PCType width = src.Width();

    PCType BlockPosRight = width - para.BlockSize;
    PCType BlockPosBottom = height - para.BlockSize;

    block_type refBlock(para.BlockSize, para.BlockSize, Pos(0, 0), false);

    Plane_FL ResNum(dst, true, 0);
    Plane_FL ResDen(dst, true, 0);

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

            PosPairCode matchCode;

            if (para.GroupSize == 1)
            {
                // Skip block matching if GroupSize is 1, and take the reference block as the only element in the group
                matchCode = PosPairCode(1, PosPair(KeyType(0), PosType(j, i)));
            }
            else
            {
                // Get reference block from reference plane
                refBlock.From(ref, Pos(j, i));

                // Form a group by block matching between reference block and its neighborhood in reference plane
                matchCode = refBlock.BlockMatchingMulti(ref, para.BMrange, para.BMstep, para.thMSE);
            }

            // Get the filtered result through collaborative filtering and aggregation of matched blocks
            CollaborativeFilter(data[0], ResNum, ResDen, src, ref, matchCode);
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

    PCType height = srcY.Height();
    PCType width = srcY.Width();

    PCType BlockPosRight = width - para.BlockSize;
    PCType BlockPosBottom = height - para.BlockSize;

    block_type refBlock(para.BlockSize, para.BlockSize, Pos(0, 0), false);

    Plane_FL ResNumY(srcY, true, 0);
    Plane_FL ResDenY(srcY, true, 0);
    Plane_FL ResNumU(srcU, true, 0);
    Plane_FL ResDenU(srcU, true, 0);
    Plane_FL ResNumV(srcV, true, 0);
    Plane_FL ResDenV(srcV, true, 0);

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

            PosPairCode matchCode;

            if (para.GroupSize == 1)
            {
                // Skip block matching if GroupSize is 1, and take the reference block as the only element in the group
                matchCode = PosPairCode(1, PosPair(KeyType(0), PosType(j, i)));
            }
            else
            {
                // Get reference block from reference plane
                refBlock.From(refY, Pos(j, i));

                // Form a group by block matching between reference block and its neighborhood in reference plane
                matchCode = refBlock.BlockMatchingMulti(refY, para.BMrange, para.BMstep, para.thMSE);
            }

            // Get the filtered result through collaborative filtering and aggregation of matched blocks
            if (para.sigma[0] > 0) CollaborativeFilter(data[0], ResNumY, ResDenY, srcY, refY, matchCode);
            if (para.sigma[1] > 0) CollaborativeFilter(data[1], ResNumU, ResDenU, srcU, refU, matchCode);
            if (para.sigma[2] > 0) CollaborativeFilter(data[2], ResNumV, ResDenV, srcV, refV, matchCode);
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
    Plane_FL src_data(src);

    // Execute kernel
    if (src.data() == ref.data())
    {
        process_Plane_FL(dst_data, src_data, src_data);
    }
    else
    {
        Plane_FL ref_data(ref);
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
    MatrixConvert_YUV2RGB(dst.R(), dst.G(), dst.B(), srcY, srcU, srcV, ColorMatrix::OPP);

    return dst;
}


// Functions of BM3D_Basic
bool BM3D_Basic::RGB2YUV(Plane_FL &srcY, Plane_FL &srcU, Plane_FL &srcV,
    Plane_FL &refY, Plane_FL &refU, Plane_FL &refV,
    const Plane &srcR, const Plane &srcG, const Plane &srcB,
    const Plane &refR, const Plane &refG, const Plane &refB) const
{
    MatrixConvert_RGB2YUV(srcY, srcU, srcV, srcR, srcG, srcB, ColorMatrix::OPP);

    if (srcR.data() == refR.data() && srcG.data() == refG.data() && srcB.data() == refB.data())
    {
        return true;
    }
    else
    {
        ConvertToY(refY, refR, refG, refB, ColorMatrix::OPP);
        return false;
    }
}


void BM3D_Basic::CollaborativeFilter(const BM3D_Data &d,
    Plane_FL &ResNum, Plane_FL &ResDen,
    const Plane_FL &src, const Plane_FL &ref,
    const PosPairCode &posPairCode) const
{
    PCType GroupSize = static_cast<PCType>(posPairCode.size());
    // When para.GroupSize > 0, limit GroupSize up to para.GroupSize
    if (para.GroupSize > 0 && GroupSize > para.GroupSize)
    {
        GroupSize = para.GroupSize;
    }

    // Construct source group guided by matched pos code
    BlockGroup<FLType, FLType> srcGroup(src, posPairCode, GroupSize, para.BlockSize, para.BlockSize);

    // Initialize retianed coefficients of hard threshold filtering
    int retainedCoefs = 0;

    // Apply forward 3D transform to the source group
    d.fp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Apply hard threshold filtering to the source group
    Block_For_each(srcGroup, d.thrTable[GroupSize - 1], [&](FLType &x, FLType y)
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
    d.bp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Calculate weight for the filtered group
    // Also include the normalization factor to compensate for the amplification introduced in 3D transform
    FLType denWeight = retainedCoefs < 1 ? 1 : FLType(1) / static_cast<FLType>(retainedCoefs);
    FLType numWeight = static_cast<FLType>(denWeight / d.finalAMP[GroupSize - 1]);

    // Store the weighted filtered group to the numerator part of the final estimation
    // Store the weight to the denominator part of the final estimation
    srcGroup.AddTo(ResNum, numWeight);
    srcGroup.CountTo(ResDen, denWeight);
}


// Functions of BM3D_Final
bool BM3D_Final::RGB2YUV(Plane_FL &srcY, Plane_FL &srcU, Plane_FL &srcV,
    Plane_FL &refY, Plane_FL &refU, Plane_FL &refV,
    const Plane &srcR, const Plane &srcG, const Plane &srcB,
    const Plane &refR, const Plane &refG, const Plane &refB) const
{
    MatrixConvert_RGB2YUV(srcY, srcU, srcV, srcR, srcG, srcB, ColorMatrix::OPP);

    if (srcR.data() == refR.data() && srcG.data() == refG.data() && srcB.data() == refB.data())
    {
        return true;
    }
    else
    {
        MatrixConvert_RGB2YUV(refY, refU, refV, refR, refG, refB, ColorMatrix::OPP);
        return false;
    }
}


void BM3D_Final::CollaborativeFilter(const BM3D_Data &d,
    Plane_FL &ResNum, Plane_FL &ResDen,
    const Plane_FL &src, const Plane_FL &ref,
    const PosPairCode &posPairCode) const
{
    PCType GroupSize = static_cast<PCType>(posPairCode.size());
    // When para.GroupSize > 0, limit GroupSize up to para.GroupSize
    if (para.GroupSize > 0 && GroupSize > para.GroupSize)
    {
        GroupSize = para.GroupSize;
    }

    // Construct source group and reference group guided by matched pos code
    BlockGroup<FLType, FLType> srcGroup(src, posPairCode, GroupSize, para.BlockSize, para.BlockSize);
    BlockGroup<FLType, FLType> refGroup(ref, posPairCode, GroupSize, para.BlockSize, para.BlockSize);

    // Initialize L2-norm of Wiener coefficients
    FLType L2Wiener = 0;

    // Apply forward 3D transform to the source group and the reference group
    d.fp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());
    d.fp[GroupSize - 1].execute_r2r(refGroup.data(), refGroup.data());

    // Apply empirical Wiener filtering to the source group guided by the reference group
    const FLType sigmaSquare = d.wienerSigmaSqr[GroupSize - 1];

    Block_For_each(srcGroup, refGroup, [&](FLType &x, FLType y)
    {
        FLType ySquare = y * y;
        FLType wienerCoef = ySquare / (ySquare + sigmaSquare);
        x *= wienerCoef;
        L2Wiener += wienerCoef * wienerCoef;
    });

    // Apply backward 3D transform to the filtered group
    d.bp[GroupSize - 1].execute_r2r(srcGroup.data(), srcGroup.data());

    // Calculate weight for the filtered group
    // Also include the normalization factor to compensate for the amplification introduced in 3D transform
    FLType denWeight = L2Wiener <= 0 ? 1 : FLType(1) / L2Wiener;
    FLType numWeight = static_cast<FLType>(denWeight / d.finalAMP[GroupSize - 1]);

    // Store the weighted filtered group to the numerator part of the final estimation
    // Store the weight to the denominator part of the final estimation
    srcGroup.AddTo(ResNum, numWeight);
    srcGroup.CountTo(ResDen, denWeight);
}


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
    Plane_FL src_data(src);

    if (src.data() == ref.data())
    {
        process_Plane_FL(dst_data, src_data, src_data);
    }
    else
    {
        Plane_FL ref_data(ref);
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
    MatrixConvert_YUV2RGB(dst.R(), dst.G(), dst.B(), srcY, srcU, srcV, ColorMatrix::OPP);

    return dst;
}
