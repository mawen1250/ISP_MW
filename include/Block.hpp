#ifndef BLOCK_HPP_
#define BLOCK_HPP_


template < typename _St1, typename _Fn1 > inline
void Block_For_each(_St1 &data, _Fn1 &_Func)
{
    auto datap = data.Data();

    for (auto upper = datap + data.PixelCount(); datap != upper; ++datap)
    {
        _Func(*datap);
    }
}

template < typename _St1, typename _St2, typename _Fn1 > inline
void Block_For_each(_St1 &left, _St2 &right, _Fn1 &_Func)
{
    auto leftp = left.Data();
    auto rightp = right.Data();

    for (auto upper = leftp + left.PixelCount(); leftp != upper; ++leftp, ++rightp)
    {
        _Func(*leftp, *rightp);
    }
}

template < typename _St1, typename _Fn1 > inline
void Block_Transform(_St1 &data, _Fn1 &_Func)
{
    auto datap = data.Data();

    for (auto upper = datap + data.PixelCount(); datap != upper; ++datap)
    {
        *datap = _Func(*datap);
    }
}

template < typename _Dt1, typename _St1, typename _Fn1 > inline
void Block_Transform(_Dt1 &dst, const _St1 &src, _Fn1 &_Func)
{
    const char *FunctionName = "Block_Transform";
    if (dst.Width() != src.Width() || dst.Height() != src.Height() || dst.PixelCount() != src.PixelCount())
    {
        std::cerr << FunctionName << ": Width(), Height() and PixelCount() of dst and src must be the same.\n";
        exit(EXIT_FAILURE);
    }

    auto dstp = dst.Data();
    auto srcp = src.Data();

    for (auto upper = dstp + dst.PixelCount(); dstp != upper; ++dstp, ++srcp)
    {
        *datap = _Func(*datap);
    }
}


template < typename _St1, typename _Ty, typename _FTy >
void ExpectationVarianceFromBlocks(Block<_Ty, _FTy> E, Block<_Ty, _FTy> V, const _St1 &src,
    const typename Block<_Ty, _FTy>::PosPairCode &posPairCode, PCType GroupSizeMax = 0)
{
    PCType GroupSize = static_cast<PCType>(posPairCode.size());
    // When GroupSizeMax > 0, limit GroupSize up to GroupSizeMax
    if (GroupSizeMax > 0 && GroupSize > GroupSizeMax)
    {
        GroupSize = GroupSizeMax;
    }

    E.InitValue(true, 0);
    V.InitValue(true, 0);

    _Ty temp;

    for (PCType k = 0; k < GroupSize; ++k)
    {
        Pos pos = posPairCode[k].second;
        auto Ep = E.Data();
        auto Vp = V.Data();
        auto srcp = src.Data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < E.Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + E.Width(); x < upper; ++x, ++Ep, ++Vp)
            {
                temp = static_cast<_Ty>(srcp[x]);
                *Ep += temp;
                *Vp += temp * temp;
            }
        }
    }

    auto Ep = E.Data();
    auto Vp = V.Data();

    for (auto upper = Ep + E.PixelCount(); Ep != upper; ++Ep, ++Vp)
    {
        *Ep /= GroupSize;
        *Vp = *Vp / GroupSize - *Ep * *Ep;
    }
}


#endif
