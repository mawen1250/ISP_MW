#include "NLMeans.h"
#include "Conversion.hpp"


// Get the filtered block through weighted averaging of matched blocks in Plane src
template < typename _St1, typename _Ty, typename _FTy >
void NLMeans::WeightedAverage(Block<_Ty, _FTy> &dstBlock, const Block<_Ty, _FTy> &refBlock, const _St1 &src,
    const typename Block<_Ty, _FTy>::PosPairCode &posPairCode)
{
    PCType GroupSize = static_cast<PCType>(posPairCode.size());
    // When GroupSizeMax > 0, limit GroupSize up to GroupSizeMax
    if (para.GroupSizeMax > 0 && GroupSize > para.GroupSizeMax)
    {
        GroupSize = para.GroupSizeMax;
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

    Block<_Ty, _FTy> sumX1(refBlock, true, 0);

    double exponentMul = -1 / (para.strength * para.strength);
    double weightSum = 0;
    double weight;

    for (PCType k = 0; k < GroupSize; ++k)
    {
        weight = exp(posPairCode[k].first * exponentMul);
        weightSum += weight;

        Pos pos = posPairCode[k].second;
        auto sumX1p = sumX1.Data();
        auto srcp = src.Data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < refBlock.Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + refBlock.Width(); x < upper; ++x, ++sumX1p)
            {
                *sumX1p += static_cast<double>(srcp[x]) * weight;
            }
        }
    }

    double weightSumRec = 1 / weightSum;

    auto dstp = dstBlock.Data();
    auto sumX1p = sumX1.Data();

    for (auto upper = dstp + dstBlock.PixelCount(); dstp != upper; ++dstp, ++sumX1p)
    {
        *dstp = *sumX1p * weightSumRec;
    }
}


// Get the filtered block through weighted averaging of matched blocks in Plane src
// A soft threshold optimal correction is applied by testing staionarity, to improve the NL-means algorithm
template < typename _St1, typename _Ty, typename _FTy >
void NLMeans::WeightedAverage_Correction(Block<_Ty, _FTy> &dstBlock, const Block<_Ty, _FTy> &refBlock, const _St1 &src,
    const typename Block<_Ty, _FTy>::PosPairCode &posPairCode)
{
    PCType GroupSize = static_cast<PCType>(posPairCode.size());
    // When GroupSizeMax > 0, limit GroupSize up to GroupSizeMax
    if (para.GroupSizeMax > 0 && GroupSize > para.GroupSizeMax)
    {
        GroupSize = para.GroupSizeMax;
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

    Block<_Ty, _FTy> sumX1(refBlock, true, 0);
    Block<_Ty, _FTy> sumX2(refBlock, true, 0);

    double exponentMul = -1 / (para.strength * para.strength);
    double weightSum = 0;
    double weight;
    double temp;

    for (PCType k = 0; k < GroupSize; ++k)
    {
        weight = exp(posPairCode[k].first * exponentMul);
        weightSum += weight;

        Pos pos = posPairCode[k].second;
        auto sumX1p = sumX1.Data();
        auto sumX2p = sumX2.Data();
        auto srcp = src.Data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < refBlock.Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + refBlock.Width(); x < upper; ++x, ++sumX1p, ++sumX2p)
            {
                temp = static_cast<double>(srcp[x]) * weight;
                *sumX1p += temp;
                *sumX2p += static_cast<double>(srcp[x]) * temp;
            }
        }
    }

    double sigma = para.sigma * src.ValueRange() / 255.; // sigma is converted from 8bit-scale to fit the src range

    double X, EX, VarX;
    double VarN = sigma * sigma;
    double weightSumRec = 1 / weightSum;

    auto dstp = dstBlock.Data();
    auto refp = refBlock.Data();
    auto sumX1p = sumX1.Data();
    auto sumX2p = sumX2.Data();

    for (auto upper = dstp + dstBlock.PixelCount(); dstp != upper; ++dstp, ++refp, ++sumX1p, ++sumX2p)
    {
        X = *refp;
        EX = *sumX1p * weightSumRec;
        VarX = *sumX2p * weightSumRec - EX * EX;

        // estimated_Y = EX + max(0, 1 - VarN / VarX) * (X - EX);
        if (VarX > VarN)
        {
            *dstp = X - (VarN / VarX) * (X - EX);
        }
        else
        {
            *dstp = EX;
        }
    }
}


// Non-local Means denoising algorithm based on block matching and weighted average of grouped blocks
Plane &NLMeans::process(Plane &dst, const Plane &src, const Plane &ref)
{
    if (para.strength <= 0 || para.GroupSizeMax == 1 || para.BlockSize <= 0
        || para.BMrange <= 0 || para.BMrange < para.BMstep || para.thMSE <= 0)
    {
        dst = src;
        return dst;
    }

    PCType height = src.Height();
    PCType width = src.Width();

    typedef Block<double, double> BlockT;

    PCType BlockStep = para.BlockSize - para.Overlap;
    PCType RightBlockPos = width - para.BlockSize;
    PCType BottomBlockPos = height - para.BlockSize;

    BlockT dstSlidingWindow(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT srcSlidingWindow(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT refSlidingWindow(para.BlockSize, para.BlockSize, Pos(0, 0), false);

    Plane_FL ResNum(dst, true, 0);
    Plane_FL ResDen(dst, true, 0);

    for (PCType j = 0;;)
    {
        for (PCType i = 0;;)
        {
            // Get source block from src
            srcSlidingWindow.From(src, Pos(j, i));

            // Get reference block from ref
            refSlidingWindow.From(ref, Pos(j, i));

            // Form a group by block matching between reference block and its neighborhood in Plane ref
            auto matchedPosPairGroup = refSlidingWindow.BlockMatchingMulti(ref, para.BMrange, para.BMstep, para.thMSE);

            // Get the filtered block through weighted averaging of matched blocks in Plane src
            // A soft threshold optimal correction is applied by testing staionarity, to improve the NL-means algorithm
            if (para.correction)
            {
                WeightedAverage_Correction(dstSlidingWindow, srcSlidingWindow, src, matchedPosPairGroup);
            }
            else
            {
                WeightedAverage(dstSlidingWindow, srcSlidingWindow, src, matchedPosPairGroup);
            }

            // The filtered blocks are sumed and averaged to form the final filtered image
            dstSlidingWindow.AddTo(ResNum);
            dstSlidingWindow.CountTo(ResDen);

            // Block loop controlling - horizontal
            i += BlockStep;

            if (i <= RightBlockPos)
            {
                continue;
            }
            else if (i < RightBlockPos + BlockStep)
            {
                i = RightBlockPos;
                continue;
            }
            else
            {
                break;
            }
        }

        // Block loop controlling - vertical
        j += BlockStep;

        if (j <= BottomBlockPos)
        {
            continue;
        }
        else if (j < BottomBlockPos + BlockStep)
        {
            j = BottomBlockPos;
            continue;
        }
        else
        {
            break;
        }
    }

    // The filtered blocks are sumed and averaged to form the final filtered image
    dst.transform(ResNum, ResDen, [](Plane_FL::value_type num, Plane_FL::value_type den)
    {
        return static_cast<Plane::value_type>(num / den + Plane_FL::value_type(0.5));
    });

    return dst;
}


// Non-local Means denoising algorithm based on block matching and weighted average of grouped blocks
Frame &NLMeans::process(Frame &dst, const Frame &src, const Frame &ref)
{
    if (para.strength <= 0 || para.GroupSizeMax == 1 || para.BlockSize <= 0
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
    ConvertToY(refY, ref, ColorMatrix::Average);

    typedef Block<double, double> BlockT;

    PCType BlockStep = para.BlockSize - para.Overlap;
    PCType RightBlockPos = width - para.BlockSize;
    PCType BottomBlockPos = height - para.BlockSize;

    BlockT dstSlidingWindow0(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT dstSlidingWindow1(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT dstSlidingWindow2(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT srcSlidingWindow0(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT srcSlidingWindow1(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT srcSlidingWindow2(para.BlockSize, para.BlockSize, Pos(0, 0), false);
    BlockT refSlidingWindowY(para.BlockSize, para.BlockSize, Pos(0, 0), false);

    Plane_FL ResNum0(dst0, true, 0);
    Plane_FL ResNum1(dst1, true, 0);
    Plane_FL ResNum2(dst2, true, 0);
    Plane_FL ResDen(dst0, true, 0);

    for (PCType j = 0;;)
    {
        for (PCType i = 0;;)
        {
            // Get reference block from ref
            refSlidingWindowY.From(refY, Pos(j, i));

            // Get source block from src
            srcSlidingWindow0.From(src0, Pos(j, i));
            srcSlidingWindow1.From(src1, Pos(j, i));
            srcSlidingWindow2.From(src2, Pos(j, i));

            // Form a group by block matching between reference block and its neighborhood in Plane ref
            auto matchedPosPairGroup = refSlidingWindowY.BlockMatchingMulti(refY, para.BMrange, para.BMstep, para.thMSE);

            // Get the filtered block through weighted averaging of matched blocks in Plane src
            // A soft threshold optimal correction is applied by testing staionarity, to improve the NL-means algorithm
            if (para.correction)
            {
                WeightedAverage_Correction(dstSlidingWindow0, srcSlidingWindow0, src0, matchedPosPairGroup);
                WeightedAverage_Correction(dstSlidingWindow1, srcSlidingWindow1, src1, matchedPosPairGroup);
                WeightedAverage_Correction(dstSlidingWindow2, srcSlidingWindow2, src2, matchedPosPairGroup);
            }
            else
            {
                WeightedAverage(dstSlidingWindow0, srcSlidingWindow0, src0, matchedPosPairGroup);
                WeightedAverage(dstSlidingWindow1, srcSlidingWindow1, src1, matchedPosPairGroup);
                WeightedAverage(dstSlidingWindow2, srcSlidingWindow2, src2, matchedPosPairGroup);
            }

            // The filtered blocks are sumed and averaged to form the final filtered image
            dstSlidingWindow0.AddTo(ResNum0);
            dstSlidingWindow1.AddTo(ResNum1);
            dstSlidingWindow2.AddTo(ResNum2);

            dstSlidingWindow0.CountTo(ResDen);

            // Block loop controlling - horizontal
            i += BlockStep;

            if (i <= RightBlockPos)
            {
                continue;
            }
            else if (i < RightBlockPos + BlockStep)
            {
                i = RightBlockPos;
                continue;
            }
            else
            {
                break;
            }
        }

        // Block loop controlling - vertical
        j += BlockStep;

        if (j <= BottomBlockPos)
        {
            continue;
        }
        else if (j < BottomBlockPos + BlockStep)
        {
            j = BottomBlockPos;
            continue;
        }
        else
        {
            break;
        }
    }

    // The filtered blocks are sumed and averaged to form the final filtered image
    dst0.transform(ResNum0, ResDen, [](Plane_FL::value_type num, Plane_FL::value_type den)
    {
        return static_cast<Plane::value_type>(num / den + Plane_FL::value_type(0.5));
    });

    dst1.transform(ResNum1, ResDen, [](Plane_FL::value_type num, Plane_FL::value_type den)
    {
        return static_cast<Plane::value_type>(num / den + Plane_FL::value_type(0.5));
    });

    dst2.transform(ResNum2, ResDen, [](Plane_FL::value_type num, Plane_FL::value_type den)
    {
        return static_cast<Plane::value_type>(num / den + Plane_FL::value_type(0.5));
    });

    return dst;
}
