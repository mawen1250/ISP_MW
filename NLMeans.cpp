#include "NLMeans.h"


template < typename _St1, typename _Ty, typename _FTy >
void NLMeans_WeightedAverage(Block<_Ty, _FTy> &dstBlock, const Block<_Ty, _FTy> &refBlock, const _St1 &src,
    const typename Block<_Ty, _FTy>::PosPairCode &posPairCode, PCType GroupSizeMax, double strength)
{
    PCType GroupSize = static_cast<PCType>(posPairCode.size());
    // When GroupSizeMax > 0, limit GroupSize up to GroupSizeMax
    if (GroupSizeMax > 0 && GroupSize > GroupSizeMax)
    {
        GroupSize = GroupSizeMax;
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

    double exponentMul = -1 / (strength * strength);
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


template < typename _St1, typename _Ty, typename _FTy >
void NLMeans_WeightedAverage_Correction(Block<_Ty, _FTy> &dstBlock, const Block<_Ty, _FTy> &refBlock, const _St1 &src,
    const typename Block<_Ty, _FTy>::PosPairCode &posPairCode, PCType GroupSizeMax, double strength, double sigma)
{
    PCType GroupSize = static_cast<PCType>(posPairCode.size());
    // When GroupSizeMax > 0, limit GroupSize up to GroupSizeMax
    if (GroupSizeMax > 0 && GroupSize > GroupSizeMax)
    {
        GroupSize = GroupSizeMax;
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

    double exponentMul = -1 / (strength * strength);
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
Plane &NLMeans(Plane &dst, const Plane &src, const Plane &ref, double sigma, double strength,
    PCType GroupSizeMax, PCType BlockSize, PCType Overlap, PCType BMrange, PCType BMstep, double thMSE)
{
    if (strength <= 0 || GroupSizeMax == 1 || BlockSize <= 0 || BMrange <= 0 || BMrange < BMstep || thMSE <= 0)
    {
        dst = src;
        return dst;
    }

    PCType height = src.Height();
    PCType width = src.Width();

    typedef Block<double, double> BlockT;

    sigma = sigma * src.ValueRange() / 255.; // sigma is converted from 8bit-scale to fit the src range
    PCType BlockStep = BlockSize - Overlap;
    PCType RightBlockPos = width - BlockSize;
    PCType BottomBlockPos = height - BlockSize;
    /*PCType BlockNumV = (height - Overlap) / BlockStep;
    PCType BlockRsdV = (height - Overlap) % BlockStep;
    if (BlockRsdV > 0) ++BlockNumV;
    PCType BlockNumH = (width - Overlap) / BlockStep;
    PCType BlockRsdH = (width - Overlap) % BlockStep;
    if (BlockRsdH > 0) ++BlockNumH;
    PCType BlockNum = BlockNumV * BlockNumH;*/

    BlockT refSlidingWindow(BlockSize, BlockSize, Pos(0, 0), false);
    BlockT dstSlidingWindow(BlockSize, BlockSize, Pos(0, 0), false);

    Plane_FL ResNum(dst, true, 0);
    Plane_FL ResDen(dst, true, 0);

    for (PCType j = 0;;)
    {
        for (PCType i = 0;;)
        {
            // Get reference block from Plane ref
            refSlidingWindow.From(ref, Pos(j, i));

            // Form a group by block matching between reference block and its neighborhood in Plane ref
            auto matchedPosPairGroup = refSlidingWindow.BlockMatchingMulti(ref, BMrange, BMstep, thMSE);

            // Get the filtered block through weighted averaging of matched blocks in Plane src
            // A soft threshold optimal correction is applied by testing staionarity, to improve the NL-means algorithm
            NLMeans_WeightedAverage_Correction(dstSlidingWindow, refSlidingWindow, src, matchedPosPairGroup, GroupSizeMax, strength, sigma);

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
