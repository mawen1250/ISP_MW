#ifndef BLOCK_H_
#define BLOCK_H_


#include "Image_Type.h"


template < typename _Ty = double,
    typename _FTy = double >
class Block
{
public:
    typedef Block<_Ty, _FTy> _Myt;
    typedef _Ty value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Ty *pointer;
    typedef const _Ty *const_pointer;
    typedef _Ty &reference;
    typedef const _Ty &const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

    typedef _FTy float_type;

    typedef float_type PosKeyType;
    typedef std::pair<PosKeyType, Pos> PosPair;
    typedef std::vector<PosPair> PosPairCode;
    typedef std::vector<PosKeyType> KeyCode;
    typedef std::vector<Pos> PosCode;

private:
    PCType Height_ = 0;
    PCType Width_ = 0;
    PCType PixelCount_ = 0;
    Pos pos_ = { 0, 0 };
    pointer Data_ = nullptr;

public:
    template < typename _Fn1 >
    void for_each(_Fn1 _Func)
    {
        Block_For_each(*this, _Func);
    }

    template < typename _Fn1 >
    void for_each(_Fn1 _Func) const
    {
        Block_For_each(*this, _Func);
    }

    template < typename _St2, typename _Fn1 >
    void for_each(_St2 &right, _Fn1 _Func)
    {
        Block_For_each(*this, right, _Func);
    }

    template < typename _St2, typename _Fn1 >
    void for_each(_St2 &right, _Fn1 _Func) const
    {
        Block_For_each(*this, right, _Func);
    }

    template < typename _Fn1 >
    void transform(_Fn1 _Func)
    {
        Block_Transform(*this, _Func);
    }

    template < typename _St1, typename _Fn1 >
    void transform(const _St1 &src, _Fn1 _Func)
    {
        Block_Transform(*this, src, _Func);
    }

public:
    // Default constructor
    Block() {}

    Block(PCType _Height, PCType _Width, Pos pos, bool Init = true, value_type Value = 0)
        : Height_(_Height), Width_(_Width), PixelCount_(Height_ * Width_), pos_(pos)
    {
        Data_ = new value_type[PixelCount_];

        InitValue(Init, Value);
    }

    // Constructor from Plane-like classes and Pos
    template < typename _St1 >
    explicit Block(const _St1 &src, PCType _Height = 16, PCType _Width = 16, Pos pos = Pos(0, 0))
        : Block(_Height, _Width, pos, false)
    {
        From(src);
    }

    // Constructor from src
    Block(const _Myt &src, bool Init, value_type Value = 0)
        : Block(src.Height_, src.Width_, src.pos_, Init, Value)
    {}

    // Copy constructor
    Block(const _Myt &src)
        : Block(src.Height_, src.Width_, src.pos_, false)
    {
        memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);
    }

    // Move constructor
    Block(_Myt &&src)
        : Height_(src.Height_), Width_(src.Width_), PixelCount_(src.PixelCount_), pos_(src.pos_)
    {
        Data_ = src.Data_;

        src.Height_ = 0;
        src.Width_ = 0;
        src.PixelCount_ = 0;
        src.Data_ = nullptr;
    }

    // Destructor
    ~Block()
    {
        delete[] Data_;
    }

    // Copy assignment operator
    _Myt &operator=(const _Myt &src)
    {
        if (this == &src)
        {
            return *this;
        }

        Height_ = src.Height_;
        Width_ = src.Width_;
        PixelCount_ = src.PixelCount_;
        pos_ = src.pos_;

        memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);

        return *this;
    }

    // Move assignment operator
    _Myt &operator=(_Myt &&src)
    {
        if (this == &src)
        {
            return *this;
        }

        Height_ = src.Height_;
        Width_ = src.Width_;
        PixelCount_ = src.PixelCount_;
        pos_ = src.pos_;

        delete[] Data_;
        Data_ = src.Data_;

        src.Height_ = 0;
        src.Width_ = 0;
        src.PixelCount_ = 0;
        src.Data_ = nullptr;

        return *this;
    }

    reference operator[](PCType i) { return Data_[i]; }
    const_reference operator[](PCType i) const { return Data_[i]; }
    reference operator()(PCType j, PCType i) { return Data_[j * Stride() + i]; }
    const_reference operator()(PCType j, PCType i) const { return Data_[j * Stride() + i]; }

    iterator begin() { return Data_; }
    const_iterator begin() const { return Data_; }
    iterator end() { return Data_ + PixelCount_; }
    const_iterator end() const { return Data_ + PixelCount_; }
    size_type size() const { return PixelCount_; }
    pointer data() { return Data_; }
    const_pointer data() const { return Data_; }
    value_type value(PCType i) { return Data_[i]; }
    const value_type value(PCType i) const { return Data_[i]; }

    PCType Height() const { return Height_; }
    PCType Width() const { return Width_; }
    PCType Stride() const { return Width_; }
    PCType PixelCount() const { return PixelCount_; }
    Pos GetPos() const { return pos_; }
    PCType PosY() const { return pos_.y; }
    PCType PosX() const { return pos_.x; }

    pointer Data() { return Data_; }
    const_pointer Data() const { return Data_; }

    void SetPos(Pos _pos) { pos_ = _pos; }

    friend std::ostream &operator<<(std::ostream &out, const _Myt &src)
    {
        out << "Block Info\n"
            << "    PixelCount = " << src.PixelCount() << std::endl
            << "    Pos(y, x) = " << src.GetPos() << std::endl
            << "    Data[" << src.Height() << "][" << src.Width() << "] = ";

        auto srcp = src.Data();

        for (PCType j = 0; j < src.Height(); ++j)
        {
            out << "\n        ";

            for (PCType i = 0; i < src.Width(); ++i, ++srcp)
            {
                out << *srcp << " ";
            }
        }

        out << std::endl;

        return out;
    }

    void InitValue(bool Init = true, value_type Value = 0)
    {
        if (Init)
        {
            for_each([&](value_type &x)
            {
                x = Value;
            });
        }
    }

    template < typename _St1 >
    void From(const _St1 &src)
    {
        auto dstp = Data();
        auto srcp = src.Data() + pos_.y * src.Stride() + pos_.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++dstp)
            {
                *dstp = static_cast<value_type>(srcp[x]);
            }
        }
    }

    template < typename _St1 >
    void From(const _St1 &src, Pos pos)
    {
        pos_ = pos;

        From(src);
    }

    template < typename _St1 >
    void AddFrom(const _St1 &src, Pos pos)
    {
        auto dstp = Data();
        auto srcp = src.Data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++dstp)
            {
                *dstp += static_cast<value_type>(srcp[x]);
            }
        }
    }

    template < typename _St1 >
    void AddMulFrom(const _St1 &src, Pos pos, float_type mul)
    {
        auto dstp = Data();
        auto srcp = src.Data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++dstp)
            {
                *dstp += static_cast<value_type>(srcp[x] * mul);
            }
        }
    }

    template < typename _Dt1 >
    void To(_Dt1 &dst) const
    {
        auto srcp = Data();
        auto dstp = dst.Data() + pos_.y * dst.Stride() + pos_.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * dst.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++srcp)
            {
                dstp[x] = static_cast<typename _Dt1::value_type>(*srcp);
            }
        }
    }

    template < typename _Dt1 >
    void AddTo(_Dt1 &dst) const
    {
        auto srcp = Data();
        auto dstp = dst.Data() + pos_.y * dst.Stride() + pos_.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * dst.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++srcp)
            {
                dstp[x] += static_cast<typename _Dt1::value_type>(*srcp);
            }
        }
    }

    template < typename _Dt1 >
    void CountTo(_Dt1 &dst) const
    {
        auto dstp = dst.Data() + pos_.y * dst.Stride() + pos_.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * dst.Stride();

            for (PCType upper = x + Width(); x < upper; ++x)
            {
                ++dstp[x];
            }
        }
    }

    template < typename _Dt1 >
    void CountTo(_Dt1 &dst, typename _Dt1::value_type value) const
    {
        auto dstp = dst.Data() + pos_.y * dst.Stride() + pos_.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * dst.Stride();

            for (PCType upper = x + Width(); x < upper; ++x)
            {
                dstp[x] += value;
            }
        }
    }

    value_type L1Distance(const _Myt &right) const
    {
        value_type dist = 0;

        for_each(right, [&](value_type x, value_type y)
        {
            dist += AbsSub(x, y);
        });

        return dist;
    }

    float_type L2DistanceSquare(const _Myt &right) const
    {
        float_type dist = 0;
        float_type temp;

        for_each(right, [&](value_type x, value_type y)
        {
            temp = x - y;
            dist += temp * temp;
        });

        return dist;
    }

    float_type L2Distance(const _Myt &right) const
    {
        return sqrt(L2DistanceSquare(right));
    }

    value_type LinfDistance(const _Myt &right) const
    {
        value_type dist = 0;
        value_type temp;

        for_each(right, [&](value_type x, value_type y)
        {
            temp = AbsSub(x, y);
            if (temp > dist)
            {
                dist = temp;
            }
        });

        return dist;
    }

    template < typename _St1 >
    Pos BlockMatching(const _St1 &src, bool excludeCurPos = false, PCType range = 48, PCType step = 2, float_type thMSE = 10.) const
    {
        bool end = false;
        Pos pos;
        float_type temp;
        float_type distMin = FLT_MAX;

        range = range / step * step;
        const PCType l = Max(PCType(0), pos_.x - range);
        const PCType r = Min(src.Width() - Width(), pos_.x + range);
        const PCType t = Max(PCType(0), pos_.y - range);
        const PCType b = Min(src.Height() - Height(), pos_.y + range);

        float_type MSE2SSE = static_cast<float_type>(PixelCount()) * src.ValueRange() * src.ValueRange() / float_type(255 * 255);
        float_type thSSE = thMSE * MSE2SSE;

        for (PCType j = t; j <= b; j += step)
        {
            for (PCType i = l; i <= r; i += step)
            {
                float_type dist = 0;

                if (excludeCurPos && j == PosY() && i == PosX())
                {
                    continue;
                }
                else
                {
                    auto refp = Data();
                    auto srcp = src.Data() + j * src.Stride() + i;

                    for (PCType y = 0; y < Height(); ++y)
                    {
                        PCType x = y * src.Stride();

                        for (PCType upper = x + Width(); x < upper; ++x, ++refp)
                        {
                            temp = static_cast<float_type>(*refp) - static_cast<float_type>(srcp[x]);
                            dist += temp * temp;
                        }
                    }
                }

                if (dist < distMin)
                {
                    distMin = dist;
                    pos.y = j;
                    pos.x = i;
                    
                    if (distMin <= thSSE)
                    {
                        end = true;
                        break;
                    }
                }
            }

            if (end)
            {
                break;
            }
        }
        //std::cout << "thSSE = " << thSSE << ", distMin = " << distMin << std::endl;
        return pos;
    }

    template < typename _St1 >
    PosPairCode BlockMatchingMulti(const _St1 &src, PCType range = 48, PCType step = 2, float_type thMSE = 400.) const
    {
        PosPairCode codes;
        float_type temp;
        
        range = range / step * step;
        const PCType l = Max(PCType(0), pos_.x - range);
        const PCType r = Min(src.Width() - Width(), pos_.x + range);
        const PCType t = Max(PCType(0), pos_.y - range);
        const PCType b = Min(src.Height() - Height(), pos_.y + range);

        float_type MSE2SSE = static_cast<float_type>(PixelCount()) * src.ValueRange() * src.ValueRange() / float_type(255 * 255);
        float_type distMul = 1. / MSE2SSE;
        float_type thSSE = thMSE * MSE2SSE;
        //std::cout << "thSSE = " << thSSE << std::endl;
        for (PCType j = t; j <= b; j += step)
        {
            for (PCType i = l; i <= r; i += step)
            {
                float_type dist = 0;

                auto refp = Data();
                auto srcp = src.Data() + j * src.Stride() + i;

                for (PCType y = 0; y < Height(); ++y)
                {
                    PCType x = y * src.Stride();

                    for (PCType upper = x + Width(); x < upper; ++x, ++refp)
                    {
                        temp = static_cast<float_type>(*refp) - static_cast<float_type>(srcp[x]);
                        dist += temp * temp;
                    }
                }

                if (dist <= thSSE)
                {
                    //std::cout << Pos(j, i) << " dist = " << dist << std::endl;
                    codes.push_back(PosPair(dist * distMul, Pos(j, i)));
                }
            }
        }

        std::sort(codes.begin(), codes.end());

        return codes;
    }
};


template < typename _Ty = double >
class BlockGroup
{
public:
    typedef BlockGroup<_Ty> _Myt;
    typedef _Ty value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Ty *pointer;
    typedef const _Ty *const_pointer;
    typedef _Ty &reference;
    typedef const _Ty &const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

    typedef Block<value_type, value_type> block_type;
    typedef typename block_type::PosKeyType PosKeyType;
    typedef typename block_type::PosPair PosPair;
    typedef typename block_type::PosPairCode PosPairCode;
    typedef typename block_type::KeyCode KeyCode;
    typedef typename block_type::PosCode PosCode;

private:
    PCType BlockCount_ = 0;
    PCType Height_ = 0;
    PCType Width_ = 0;
    PCType PixelCount_ = 0;
    KeyCode keyCode_;
    PosCode posCode_;
    pointer Data_ = nullptr;

public:
    template < typename _Fn1 >
    void for_each(_Fn1 _Func)
    {
        Block_For_each(*this, _Func);
    }

    template < typename _Fn1 >
    void for_each(_Fn1 _Func) const
    {
        Block_For_each(*this, _Func);
    }

    template < typename _St2, typename _Fn1 >
    void for_each(_St2 &right, _Fn1 _Func)
    {
        Block_For_each(*this, right, _Func);
    }

    template < typename _St2, typename _Fn1 >
    void for_each(_St2 &right, _Fn1 _Func) const
    {
        Block_For_each(*this, right, _Func);
    }

    template < typename _Fn1 >
    void transform(_Fn1 _Func)
    {
        Block_Transform(*this, _Func);
    }

    template < typename _St1, typename _Fn1 >
    void transform(const _St1 &src, _Fn1 _Func)
    {
        Block_Transform(*this, src, _Func);
    }

public:
    // Default constructor
    BlockGroup() {}

    explicit BlockGroup(PCType _BlockCount, PCType _Height, PCType _Width, bool Init = true, value_type Value = 0)
        : BlockCount_(_BlockCount), Height_(_Height), Width_(_Width), PixelCount_(BlockCount_ * Height_ * Width_)
    {
        Data_ = new value_type[PixelCount_];

        InitValue(Init, Value);
    }

    // Constructor from Plane-like classes and PosPairCode
    template < typename _St1 >
    BlockGroup(const _St1 &src, const PosPairCode &posPairCode, PCType _BlockCount = -1, PCType _Height = 16, PCType _Width = 16)
        : Height_(_Height), Width_(_Width)
    {
        FromPosPairCode(posPairCode, _BlockCount);

        PixelCount_ = BlockCount_ * Height_ * Width_;

        Data_ = new value_type[PixelCount_];

        From(src);
    }

    // Copy constructor
    BlockGroup(const _Myt &src)
        : BlockCount_(src.BlockCount_), Height_(src.Height_), Width_(src.Width_), PixelCount_(src.PixelCount_), 
        keyCode_(src.keyCode_), posCode_(src.posCode_)
    {
        Data_ = new value_type[PixelCount_];

        memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);
    }

    // Move constructor
    BlockGroup(_Myt &&src)
        : BlockCount_(src.BlockCount_), Height_(src.Height_), Width_(src.Width_), PixelCount_(src.PixelCount_), 
        keyCode_(std::move(src.keyCode_)), posCode_(std::move(src.posCode_))
    {
        Data_ = src.Data_;

        src.BlockCount_ = 0;
        src.Height_ = 0;
        src.Width_ = 0;
        src.PixelCount_ = 0;
        src.Data_ = nullptr;
    }

    // Destructor
    ~BlockGroup()
    {
        delete[] Data_;
    }

    // Copy assignment operator
    _Myt &operator=(const _Myt &src)
    {
        if (this == &src)
        {
            return *this;
        }

        BlockCount_ = src.BlockCount_;
        Height_ = src.Height_;
        Width_ = src.Width_;
        PixelCount_ = src.PixelCount_;
        keyCode_ = src.keyCode_;
        posCode_ = src.posCode_;

        memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);

        return *this;
    }

    // Move assignment operator
    _Myt &operator=(_Myt &&src)
    {
        if (this == &src)
        {
            return *this;
        }

        BlockCount_ = src.BlockCount_;
        Height_ = src.Height_;
        Width_ = src.Width_;
        PixelCount_ = src.PixelCount_;
        keyCode_ = std::move(src.keyCode_);
        posCode_ = std::move(src.posCode_);

        delete[] Data_;
        Data_ = src.Data_;

        src.BlockCount_ = 0;
        src.Height_ = 0;
        src.Width_ = 0;
        src.PixelCount_ = 0;
        src.Data_ = nullptr;

        return *this;
    }

    reference operator[](PCType i) { return Data_[i]; }
    const_reference operator[](PCType i) const { return Data_[i]; }
    reference operator()(PCType k, PCType j, PCType i) { return Data_[(k * Height() + j) * Stride() + i]; }
    const_reference operator()(PCType k, PCType j, PCType i) const { return Data_[(k * Height() + j) * Stride() + i]; }

    iterator begin() { return Data_; }
    const_iterator begin() const { return Data_; }
    iterator end() { return Data_ + PixelCount_; }
    const_iterator end() const { return Data_ + PixelCount_; }
    size_type size() const { return PixelCount_; }
    pointer data() { return Data_; }
    const_pointer data() const { return Data_; }
    value_type value(PCType i) { return Data_[i]; }
    const value_type value(PCType i) const { return Data_[i]; }

    PCType BlockCount() const { return BlockCount_; }
    PCType Height() const { return Height_; }
    PCType Width() const { return Width_; }
    PCType Stride() const { return Width_; }
    PCType PixelCount() const { return PixelCount_; }
    const KeyCode &GetKeyCode() const { return keyCode_; }
    const PosCode &GetPosCode() const { return posCode_; }

    pointer Data() { return Data_; }
    const_pointer Data() const { return Data_; }

    void InitValue(bool Init = true, value_type Value = 0)
    {
        if (Init)
        {
            keyCode_.resize(BlockCount_, 0);
            posCode_.resize(BlockCount_, Pos(0, 0));

            for_each([&](value_type &x)
            {
                x = Value;
            });
        }
        else
        {
            keyCode_.resize(BlockCount_);
            posCode_.resize(BlockCount_);
        }
    }

    void FromPosPairCode(const PosPairCode &src)
    {
        keyCode_.resize(BlockCount_);
        posCode_.resize(BlockCount_);

        for (int i = 0; i < BlockCount_; ++i)
        {
            keyCode_[i] = src[i].first;
            posCode_[i] = src[i].second;
        }
    }

    void FromPosPairCode(const PosPairCode &src, PCType _BlockCount = -1)
    {
        if (_BlockCount < 0)
        {
            BlockCount_ = src.size();
        }
        else
        {
            BlockCount_ = Min(src.size(), static_cast<size_type>(_BlockCount));
        }

        FromPosPairCode(src);
    }

    template < typename _St1 >
    _Myt &From(const _St1 &src)
    {
        for (PCType k = 0; k < BlockCount(); ++k)
        {
            auto dstp = Data() + k * Height() * Stride();
            auto srcp = src.Data() + posCode_[k].y * src.Stride() + posCode_[k].x;

            for (PCType j = 0; j < Height(); ++j)
            {
                PCType i = j * src.Stride();

                for (PCType upper = i + Width(); i < upper; ++i, ++dstp)
                {
                    *dstp = static_cast<value_type>(srcp[i]);
                }
            }
        }

        return *this;
    }
};


#include "Block.hpp"


#endif
