#ifndef BLOCK_HPP_
#define BLOCK_HPP_


template < typename _St1, typename _Fn1 >
void Block_For_each(_St1 &data, _Fn1 &&_Func)
{
    auto datap = data.data();

    for (auto upper = datap + data.size(); datap != upper; ++datap)
    {
        _Func(*datap);
    }
}

template < typename _St1, typename _St2, typename _Fn1 >
void Block_For_each(_St1 &data1, _St2 &data2, _Fn1 &&_Func)
{
    if (data1.size() != data2.size())
    {
        DEBUG_FAIL("Block_For_each: size() of data1 and data2 must be the same.");
    }

    auto data1p = data1.data();
    auto data2p = data2.data();

    for (auto upper = data1p + data1.size(); data1p != upper; ++data1p, ++data2p)
    {
        _Func(*data1p, *data2p);
    }
}

template < typename _St1, typename _Fn1 >
void Block_Transform(_St1 &data, _Fn1 &&_Func)
{
    auto datap = data.data();

    for (auto upper = datap + data.size(); datap != upper; ++datap)
    {
        *datap = _Func(*datap);
    }
}

template < typename _Dt1, typename _St1, typename _Fn1 >
void Block_Transform(_Dt1 &dst, const _St1 &src, _Fn1 &&_Func)
{
    if (dst.size() != src.size())
    {
        DEBUG_FAIL("Block_Transform: size() of dst and src must be the same.");
    }

    auto dstp = dst.data();
    auto srcp = src.data();

    for (auto upper = dstp + dst.size(); dstp != upper; ++dstp, ++srcp)
    {
        *dstp = _Func(*srcp);
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
        auto Ep = E.data();
        auto Vp = V.data();
        auto srcp = src.data() + pos.y * src.Stride() + pos.x;

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

    auto Ep = E.data();
    auto Vp = V.data();

    for (auto upper = Ep + E.size(); Ep != upper; ++Ep, ++Vp)
    {
        *Ep /= GroupSize;
        *Vp = *Vp / GroupSize - *Ep * *Ep;
    }
}


#endif
