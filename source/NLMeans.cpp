#include "NLMeans.h"
#include "Conversion.hpp"


// Get the filtered block through weighted averaging of matched blocks in Plane src
template < typename _St1 >
void NLMeans::WeightedAverage(block_type &dstBlock, const block_type &refBlock, const _St1 &src,
    const PosPairCode &code)
{
    PCType GroupSize = static_cast<PCType>(code.size());
    // When para.GroupSize > 0, limit GroupSize up to para.GroupSize
    if (para.GroupSize > 0 && GroupSize > para.GroupSize)
    {
        GroupSize = para.GroupSize;
    }

    if (GroupSize < 2)
    {
        dstBlock.From(src, refBlock.GetPos());
        return;
    }
    else
    {
        dstBlock.SetPos(refBlock.GetPos());
    }

    block_type sumX1(refBlock, true, 0);

    FLType exponentMul = static_cast<FLType>(-1 / (para.strength * para.strength));
    FLType weightSum = 0;
    FLType weight;

    for (PCType k = 0; k < GroupSize; ++k)
    {
        weight = exp(code[k].first * exponentMul);
        weightSum += weight;

        Pos pos = code[k].second;
        auto sumX1p = sumX1.data();
        auto srcp = src.data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < refBlock.Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + refBlock.Width(); x < upper; ++x, ++sumX1p)
            {
                *sumX1p += static_cast<FLType>(srcp[x]) * weight;
            }
        }
    }

    FLType weightSumRec = FLType(1) / weightSum;

    auto dstp = dstBlock.data();
    auto sumX1p = sumX1.data();

    for (auto upper = dstp + dstBlock.PixelCount(); dstp != upper; ++dstp, ++sumX1p)
    {
        *dstp = static_cast<FLType>(*sumX1p * weightSumRec);
    }
}


// Get the filtered block through weighted averaging of matched blocks in Plane src
// A soft threshold optimal correction is applied by testing staionarity, to improve the NL-means algorithm
template < typename _St1 >
void NLMeans::WeightedAverage_Correction(block_type &dstBlock, const block_type &refBlock, const _St1 &src,
    const PosPairCode &code)
{
    PCType GroupSize = static_cast<PCType>(code.size());
    // When para.GroupSize > 0, limit GroupSize up to para.GroupSize
    if (para.GroupSize > 0 && GroupSize > para.GroupSize)
    {
        GroupSize = para.GroupSize;
    }

    if (GroupSize < 2)
    {
        dstBlock.From(src, refBlock.GetPos());
        return;
    }
    else
    {
        dstBlock.SetPos(refBlock.GetPos());
    }

    block_type sumX1(refBlock, true, 0);
    block_type sumX2(refBlock, true, 0);

    FLType exponentMul = static_cast<FLType>(-1 / (para.strength * para.strength));
    FLType weightSum = 0;
    FLType weight;
    FLType temp;

    for (PCType k = 0; k < GroupSize; ++k)
    {
        weight = exp(code[k].first * exponentMul);
        weightSum += weight;

        Pos pos = code[k].second;
        auto sumX1p = sumX1.data();
        auto sumX2p = sumX2.data();
        auto srcp = src.data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < refBlock.Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + refBlock.Width(); x < upper; ++x, ++sumX1p, ++sumX2p)
            {
                temp = static_cast<FLType>(srcp[x]) * weight;
                *sumX1p += temp;
                *sumX2p += static_cast<FLType>(srcp[x]) * temp;
            }
        }
    }

    FLType sigma = static_cast<FLType>(para.sigma * src.ValueRange() / 255); // sigma is converted from 8bit-scale to fit the src range

    FLType X, EX, VarX;
    FLType VarN = static_cast<FLType>(sigma * sigma);
    FLType weightSumRec = FLType(1) / weightSum;

    auto dstp = dstBlock.data();
    auto refp = refBlock.data();
    auto sumX1p = sumX1.data();
    auto sumX2p = sumX2.data();

    for (auto upper = dstp + dstBlock.PixelCount(); dstp != upper; ++dstp, ++refp, ++sumX1p, ++sumX2p)
    {
        X = static_cast<FLType>(*refp);
        EX = *sumX1p * weightSumRec;
        VarX = *sumX2p * weightSumRec - EX * EX;

        // estimated_Y = EX + max(0, 1 - VarN / VarX) * (X - EX);
        if (VarX > VarN)
        {
            *dstp = static_cast<FLType>(X - (VarN / VarX) * (X - EX));
        }
        else
        {
            *dstp = static_cast<FLType>(EX);
        }
    }
}


// Non-local Means denoising algorithm based on block matching and weighted average of grouped blocks
Plane_FL &NLMeans::process_Plane_FL(Plane_FL &dst, const Plane_FL &src, const Plane_FL &ref)
{
    if (para.strength <= 0 || para.GroupSize == 1 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return dst;
    }

    PCType height = src.Height();
    PCType width = src.Width();

    PCType BlockPosRight = width - para.BlockSize;
    PCType BlockPosBottom = height - para.BlockSize;

    block_type dstBlock(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    block_type srcBlock(para.BlockSize, para.BlockSize, Pos(0, 0), false);
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

            // Get source block from src
            srcBlock.From(src, Pos(j, i));

            // Get reference block from ref
            refBlock.From(ref, Pos(j, i));

            // Form a group by block matching between reference block and its neighborhood in reference plane
            PosPairCode matchCode = refBlock.BlockMatchingMulti(ref, para.BMrange, para.BMstep, para.thMSE, 1, para.GroupSize, true);

            // Get the filtered block through weighted averaging of matched blocks in Plane src
            // A soft threshold optimal correction is applied by testing staionarity, to improve the NL-means algorithm
            if (para.correction)
            {
                WeightedAverage_Correction(dstBlock, srcBlock, src, matchCode);
            }
            else
            {
                WeightedAverage(dstBlock, srcBlock, src, matchCode);
            }

            // The filtered blocks are sumed and averaged to form the final filtered image
            dstBlock.AddTo(ResNum);
            dstBlock.CountTo(ResDen);
        }
    }

    // The filtered blocks are sumed and averaged to form the final filtered image
    _Transform(dst, ResNum, ResDen, [](FLType num, FLType den)
    {
        return num / den;
    });

    return dst;
}


Plane &NLMeans::process_Plane(Plane &dst, const Plane &src, const Plane &ref)
{
    if (para.strength <= 0 || para.GroupSize == 1 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return dst;
    }

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

    RangeConvert(dst, dst_data, false);

    return dst;
}


// Non-local Means denoising algorithm based on block matching and weighted average of grouped blocks
Frame &NLMeans::process_Frame(Frame &dst, const Frame &src, const Frame &ref)
{
    if (para.strength <= 0 || para.GroupSize == 1 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return dst;
    }

    PCType height = src.Height();
    PCType width = src.Width();

    Plane &dst0 = dst.P(0);
    Plane &dst1 = dst.P(1);
    Plane &dst2 = dst.P(2);
    const Plane &src0 = src.P(0);
    const Plane &src1 = src.P(1);
    const Plane &src2 = src.P(2);

    Plane_FL refY(ref.P(0), false);
    ConvertToY(refY, ref, ColorMatrix::OPP);

    PCType BlockPosRight = width - para.BlockSize;
    PCType BlockPosBottom = height - para.BlockSize;

    block_type dstBlock0(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    block_type dstBlock1(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    block_type dstBlock2(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    block_type srcBlock0(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    block_type srcBlock1(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    block_type srcBlock2(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    block_type refBlockY(para.BlockSize, para.BlockSize, Pos(0, 0), false);

    Plane_FL ResNum0(dst0, true, 0);
    Plane_FL ResNum1(dst1, true, 0);
    Plane_FL ResNum2(dst2, true, 0);
    Plane_FL ResDen(dst0, true, 0);

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

            // Get reference block from ref
            refBlockY.From(refY, Pos(j, i));

            // Get source block from src
            srcBlock0.From(src0, Pos(j, i));
            srcBlock1.From(src1, Pos(j, i));
            srcBlock2.From(src2, Pos(j, i));

            // Form a group by block matching between reference block and its neighborhood in reference plane
            PosPairCode matchCode = refBlockY.BlockMatchingMulti(refY, para.BMrange, para.BMstep, para.thMSE, 1, para.GroupSize, true);

            // Get the filtered block through weighted averaging of matched blocks in Plane src
            // A soft threshold optimal correction is applied by testing staionarity, to improve the NL-means algorithm
            if (para.correction)
            {
                WeightedAverage_Correction(dstBlock0, srcBlock0, src0, matchCode);
                WeightedAverage_Correction(dstBlock1, srcBlock1, src1, matchCode);
                WeightedAverage_Correction(dstBlock2, srcBlock2, src2, matchCode);
            }
            else
            {
                WeightedAverage(dstBlock0, srcBlock0, src0, matchCode);
                WeightedAverage(dstBlock1, srcBlock1, src1, matchCode);
                WeightedAverage(dstBlock2, srcBlock2, src2, matchCode);
            }

            // The filtered blocks are sumed and averaged to form the final filtered image
            dstBlock0.AddTo(ResNum0);
            dstBlock1.AddTo(ResNum1);
            dstBlock2.AddTo(ResNum2);

            dstBlock0.CountTo(ResDen);
        }
    }

    // The filtered blocks are sumed and averaged to form the final filtered image
    _Transform(dst0, ResNum0, ResDen, [](Plane_FL::value_type num, Plane_FL::value_type den)
    {
        return static_cast<Plane::value_type>(num / den + Plane_FL::value_type(0.5));
    });

    _Transform(dst1, ResNum1, ResDen, [](Plane_FL::value_type num, Plane_FL::value_type den)
    {
        return static_cast<Plane::value_type>(num / den + Plane_FL::value_type(0.5));
    });

    _Transform(dst2, ResNum2, ResDen, [](Plane_FL::value_type num, Plane_FL::value_type den)
    {
        return static_cast<Plane::value_type>(num / den + Plane_FL::value_type(0.5));
    });

    return dst;
}
