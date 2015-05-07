#ifndef BLOCK_H_
#define BLOCK_H_


#include "Image_Type.h"


template < typename _Ty = double,
    typename _DTy = double >
class Block
{
public:
    typedef Block<_Ty, _DTy> _Myt;
    typedef _Ty value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Ty *pointer;
    typedef const _Ty *const_pointer;
    typedef _Ty &reference;
    typedef const _Ty &const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

    typedef _DTy dist_type;

    typedef dist_type KeyType;
    typedef Pos PosType;
    typedef std::pair<KeyType, PosType> PosPair;
    typedef std::vector<PosPair> PosPairCode;
    typedef std::vector<KeyType> KeyCode;
    typedef std::vector<PosType> PosCode;

private:
    PCType Height_ = 0;
    PCType Width_ = 0;
    PCType PixelCount_ = 0;
    PosType pos_ = { 0, 0 };
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

    Block(PCType _Height, PCType _Width, PosType pos, bool Init = true, value_type Value = 0)
        : Height_(_Height), Width_(_Width), PixelCount_(Height_ * Width_), pos_(pos)
    {
        AlignedMalloc(Data_, PixelCount_);

        InitValue(Init, Value);
    }

    // Constructor from Plane-like classes and PosType
    template < typename _St1 >
    explicit Block(const _St1 &src, PCType _Height, PCType _Width, PosType pos)
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
        AlignedFree(Data_);
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

        AlignedFree(Data_);
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
    PosType GetPos() const { return pos_; }
    PCType PosY() const { return pos_.y; }
    PCType PosX() const { return pos_.x; }

    pointer Data() { return Data_; }
    const_pointer Data() const { return Data_; }

    void SetPos(PosType _pos) { pos_ = _pos; }

    friend std::ostream &operator<<(std::ostream &out, const _Myt &src)
    {
        out << "Block Info\n"
            << "    PixelCount = " << src.PixelCount() << std::endl
            << "    Pos(y, x) = " << src.GetPos() << std::endl
            << "    Data[" << src.Height() << "][" << src.Width() << "] = ";

        auto srcp = src.data();

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
        auto dstp = data();
        auto srcp = src.data() + PosY() * src.Stride() + PosX();

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
    void From(const _St1 &src, PosType pos)
    {
        pos_ = pos;

        From(src);
    }

    template < typename _St1 >
    void AddFrom(const _St1 &src, PosType pos)
    {
        auto dstp = data();
        auto srcp = src.data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++dstp)
            {
                *dstp += static_cast<value_type>(srcp[x]);
            }
        }
    }

    template < typename _St1, typename _Gt1 >
    void AddFrom(const _St1 &src, PosType pos, _Gt1 gain)
    {
        auto dstp = data();
        auto srcp = src.data() + pos.y * src.Stride() + pos.x;

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * src.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++dstp)
            {
                *dstp += static_cast<value_type>(srcp[x] * gain);
            }
        }
    }

    template < typename _Dt1 >
    void To(_Dt1 &dst) const
    {
        auto srcp = data();
        auto dstp = dst.data() + PosY() * dst.Stride() + PosX();

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
        auto srcp = data();
        auto dstp = dst.data() + PosY() * dst.Stride() + PosX();

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * dst.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++srcp)
            {
                dstp[x] += static_cast<typename _Dt1::value_type>(*srcp);
            }
        }
    }

    template < typename _Dt1, typename _Gt1 >
    void AddTo(_Dt1 &dst, _Gt1 gain) const
    {
        auto srcp = data();
        auto dstp = dst.data() + PosY() * dst.Stride() + PosX();

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * dst.Stride();

            for (PCType upper = x + Width(); x < upper; ++x, ++srcp)
            {
                dstp[x] += static_cast<typename _Dt1::value_type>(*srcp * gain);
            }
        }
    }

    template < typename _Dt1 >
    void CountTo(_Dt1 &dst) const
    {
        auto dstp = dst.data() + PosY() * dst.Stride() + PosX();

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
        auto dstp = dst.data() + PosY() * dst.Stride() + PosX();

        for (PCType y = 0; y < Height(); ++y)
        {
            PCType x = y * dst.Stride();

            for (PCType upper = x + Width(); x < upper; ++x)
            {
                dstp[x] += value;
            }
        }
    }

    dist_type L1Distance(const _Myt &right) const
    {
        dist_type dist = 0;

        for_each(right, [&](value_type x, value_type y)
        {
            dist += static_cast<dist_type>(AbsSub(x, y));
        });

        return dist;
    }

    dist_type L2DistanceSquare(const _Myt &right) const
    {
        dist_type dist = 0;
        dist_type temp;

        for_each(right, [&](value_type x, value_type y)
        {
            temp = static_cast<dist_type>(x) - static_cast<dist_type>(y);
            dist += temp * temp;
        });

        return dist;
    }

    dist_type L2Distance(const _Myt &right) const
    {
        return sqrt(L2DistanceSquare(right));
    }

    dist_type LinfDistance(const _Myt &right) const
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

        return static_cast<dist_type>(dist);
    }

    template < typename _St1 >
    PosType BlockMatching(const _St1 &src, bool excludeCurPos = false, PCType range = 48, PCType step = 2, double thMSE = 10) const
    {
        bool end = false;
        PosType pos;
        dist_type temp;
        dist_type distMin = TypeMax<dist_type>();

        range = range / step * step;
        const PCType l = SearchBoundary(PCType(0), range, step, false);
        const PCType r = SearchBoundary(src.Width() - Width(), range, step, false);
        const PCType t = SearchBoundary(PCType(0), range, step, true);
        const PCType b = SearchBoundary(src.Height() - Height(), range, step, true);

        double MSE2SSE = static_cast<double>(PixelCount()) * src.ValueRange() * src.ValueRange() / double(255 * 255);
        dist_type thSSE = static_cast<dist_type>(thMSE * MSE2SSE);

        for (PCType j = t; j <= b; j += step)
        {
            for (PCType i = l; i <= r; i += step)
            {
                dist_type dist = 0;

                if (excludeCurPos && j == PosY() && i == PosX())
                {
                    continue;
                }
                else
                {
                    auto refp = data();
                    auto srcp = src.data() + j * src.Stride() + i;

                    for (PCType y = 0; y < Height(); ++y)
                    {
                        PCType x = y * src.Stride();

                        for (PCType upper = x + Width(); x < upper; ++x, ++refp)
                        {
                            temp = static_cast<dist_type>(*refp) - static_cast<dist_type>(srcp[x]);
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
        
        return pos;
    }

    template < typename _St1 >
    PosPairCode BlockMatchingMulti(const _St1 &src, PCType range = 48, PCType step = 2, double thMSE = 400) const
    {
        range = range / step * step;
        const PCType l = SearchBoundary(PCType(0), range, step, false);
        const PCType r = SearchBoundary(src.Width() - Width(), range, step, false);
        const PCType t = SearchBoundary(PCType(0), range, step, true);
        const PCType b = SearchBoundary(src.Height() - Height(), range, step, true);

        double MSE2SSE = static_cast<double>(PixelCount()) * src.ValueRange() * src.ValueRange() / double(255 * 255);
        double distMul = double(1) / MSE2SSE;
        dist_type thSSE = static_cast<dist_type>(thMSE * MSE2SSE);
        
        PosPairCode codes(((r - l) / step + 1) * ((b - t) / step + 1));
        PCType index = 0;
        codes[index++] = PosPair(static_cast<KeyType>(0), PosType(PosY(), PosX()));

        for (PCType j = t; j <= b; j += step)
        {
            for (PCType i = l; i <= r; i += step)
            {
                if (j == PosY() && i == PosX())
                {
                    continue;
                }

                dist_type dist = 0;

                auto refp = data();
                auto srcp = src.data() + j * src.Stride() + i;

                for (PCType y = 0; y < Height(); ++y)
                {
                    PCType x = y * src.Stride();

                    for (PCType upper = x + Width(); x < upper; ++x, ++refp)
                    {
                        dist_type temp = static_cast<dist_type>(*refp) - static_cast<dist_type>(srcp[x]);
                        dist += temp * temp;
                    }
                }

                if (dist <= thSSE)
                {
                    codes[index++] = PosPair(static_cast<KeyType>(dist * distMul), PosType(j, i));
                }
            }
        }

        codes.resize(index);
        std::sort(codes.begin(), codes.end());

        return codes;
    }

protected:
    PCType SearchBoundary(PCType plane_boundary, PCType search_range, PCType search_step, bool vertical = false) const
    {
        const PCType pos = vertical ? PosY() : PosX();
        PCType search_boundary;

        search_range = search_range / search_step * search_step;

        if (pos == plane_boundary)
        {
            search_boundary = plane_boundary;
        }
        else if (pos > plane_boundary)
        {
            search_boundary = pos - search_range;

            while (search_boundary < plane_boundary)
            {
                search_boundary += search_step;
            }
        }
        else
        {
            search_boundary = pos + search_range;

            while (search_boundary > plane_boundary)
            {
                search_boundary -= search_step;
            }
        }

        return search_boundary;
    }
};


template < typename _Ty = double,
    typename _DTy = double >
class BlockGroup
{
public:
    typedef BlockGroup<_Ty, _DTy> _Myt;
    typedef _Ty value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Ty *pointer;
    typedef const _Ty *const_pointer;
    typedef _Ty &reference;
    typedef const _Ty &const_reference;

    typedef pointer iterator;
    typedef const_pointer const_iterator;

    typedef _DTy dist_type;

    typedef Block<value_type, dist_type> block_type;
    typedef typename block_type::KeyType KeyType;
    typedef typename block_type::PosType PosType;
    typedef typename block_type::PosPair PosPair;
    typedef typename block_type::PosPairCode PosPairCode;
    typedef typename block_type::KeyCode KeyCode;
    typedef typename block_type::PosCode PosCode;

private:
    PCType GroupSize_ = 0;
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

    explicit BlockGroup(PCType _GroupSize, PCType _Height, PCType _Width, bool Init = true, value_type Value = 0)
        : GroupSize_(_GroupSize), Height_(_Height), Width_(_Width), PixelCount_(GroupSize_ * Height_ * Width_)
    {
        AlignedMalloc(Data_, PixelCount_);

        InitValue(Init, Value);
    }

    // Constructor from Plane-like classes and PosPairCode
    template < typename _St1 >
    BlockGroup(const _St1 &src, const PosPairCode &posPairCode, PCType _GroupSize = -1, PCType _Height = 16, PCType _Width = 16)
        : Height_(_Height), Width_(_Width)
    {
        FromPosPairCode(posPairCode, _GroupSize);

        PixelCount_ = GroupSize_ * Height_ * Width_;

        AlignedMalloc(Data_, PixelCount_);

        From(src);
    }

    // Copy constructor
    BlockGroup(const _Myt &src)
        : GroupSize_(src.GroupSize_), Height_(src.Height_), Width_(src.Width_), PixelCount_(src.PixelCount_), 
        keyCode_(src.keyCode_), posCode_(src.posCode_)
    {
        AlignedMalloc(Data_, PixelCount_);

        memcpy(Data_, src.Data_, sizeof(value_type) * PixelCount_);
    }

    // Move constructor
    BlockGroup(_Myt &&src)
        : GroupSize_(src.GroupSize_), Height_(src.Height_), Width_(src.Width_), PixelCount_(src.PixelCount_), 
        keyCode_(std::move(src.keyCode_)), posCode_(std::move(src.posCode_))
    {
        Data_ = src.Data_;

        src.GroupSize_ = 0;
        src.Height_ = 0;
        src.Width_ = 0;
        src.PixelCount_ = 0;
        src.Data_ = nullptr;
    }

    // Destructor
    ~BlockGroup()
    {
        AlignedFree(Data_);
    }

    // Copy assignment operator
    _Myt &operator=(const _Myt &src)
    {
        if (this == &src)
        {
            return *this;
        }

        GroupSize_ = src.GroupSize_;
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

        GroupSize_ = src.GroupSize_;
        Height_ = src.Height_;
        Width_ = src.Width_;
        PixelCount_ = src.PixelCount_;
        keyCode_ = std::move(src.keyCode_);
        posCode_ = std::move(src.posCode_);

        AlignedFree(Data_);
        Data_ = src.Data_;

        src.GroupSize_ = 0;
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

    PCType GroupSize() const { return GroupSize_; }
    PCType Height() const { return Height_; }
    PCType Width() const { return Width_; }
    PCType Stride() const { return Width_; }
    PCType PixelCount() const { return PixelCount_; }
    const KeyCode &GetKeyCode() const { return keyCode_; }
    const PosCode &GetPosCode() const { return posCode_; }
    KeyType GetKey(PCType i) const { return keyCode_[i]; }
    PosType GetPos(PCType i) const { return posCode_[i]; }

    pointer Data() { return Data_; }
    const_pointer Data() const { return Data_; }

    void InitValue(bool Init = true, value_type Value = 0)
    {
        if (Init)
        {
            keyCode_.resize(GroupSize(), 0);
            posCode_.resize(GroupSize(), PosType(0, 0));

            for_each([&](value_type &x)
            {
                x = Value;
            });
        }
        else
        {
            keyCode_.resize(GroupSize());
            posCode_.resize(GroupSize());
        }
    }

    void FromPosPairCode(const PosPairCode &src, PCType _GroupSize = -1)
    {
        if (_GroupSize < 0)
        {
            GroupSize_ = static_cast<PCType>(src.size());
        }
        else
        {
            GroupSize_ = static_cast<PCType>(Min(src.size(), static_cast<size_type>(_GroupSize)));
        }

        keyCode_.resize(GroupSize());
        posCode_.resize(GroupSize());

        for (int i = 0; i < GroupSize(); ++i)
        {
            keyCode_[i] = src[i].first;
            posCode_[i] = src[i].second;
        }
    }

    template < typename _St1 >
    _Myt &From(const _St1 &src)
    {
        auto dstp = data();

        for (PCType z = 0; z < GroupSize(); ++z)
        {
            auto srcp = src.data() + GetPos(z).y * src.Stride() + GetPos(z).x;

            for (PCType y = 0; y < Height(); ++y)
            {
                PCType x = y * src.Stride();

                for (PCType upper = x + Width(); x < upper; ++x, ++dstp)
                {
                    *dstp = static_cast<value_type>(srcp[x]);
                }
            }
        }

        return *this;
    }

    template < typename _Dt1 >
    void To(_Dt1 &dst) const
    {
        auto srcp = data();

        for (PCType z = 0; z < GroupSize(); ++z)
        {
            auto dstp = dst.data() + GetPos(z).y * dst.Stride() + GetPos(z).x;

            for (PCType y = 0; y < Height(); ++y)
            {
                PCType x = y * dst.Stride();

                for (PCType upper = x + Width(); x < upper; ++x, ++srcp)
                {
                    dstp[x] = static_cast<typename _Dt1::value_type>(*srcp);
                }
            }
        }
    }

    template < typename _Dt1 >
    void AddTo(_Dt1 &dst) const
    {
        auto srcp = data();

        for (PCType z = 0; z < GroupSize(); ++z)
        {
            auto dstp = dst.data() + GetPos(z).y * dst.Stride() + GetPos(z).x;

            for (PCType y = 0; y < Height(); ++y)
            {
                PCType x = y * dst.Stride();

                for (PCType upper = x + Width(); x < upper; ++x, ++srcp)
                {
                    dstp[x] += static_cast<typename _Dt1::value_type>(*srcp);
                }
            }
        }
    }

    template < typename _Dt1, typename _Gt1 >
    void AddTo(_Dt1 &dst, _Gt1 gain) const
    {
        auto srcp = data();

        for (PCType z = 0; z < GroupSize(); ++z)
        {
            auto dstp = dst.data() + GetPos(z).y * dst.Stride() + GetPos(z).x;

            for (PCType y = 0; y < Height(); ++y)
            {
                PCType x = y * dst.Stride();

                for (PCType upper = x + Width(); x < upper; ++x, ++srcp)
                {
                    dstp[x] += static_cast<typename _Dt1::value_type>(*srcp * gain);
                }
            }
        }
    }

    template < typename _Dt1 >
    void CountTo(_Dt1 &dst) const
    {
        for (PCType z = 0; z < GroupSize(); ++z)
        {
            auto dstp = dst.data() + GetPos(z).y * dst.Stride() + GetPos(z).x;

            for (PCType y = 0; y < Height(); ++y)
            {
                PCType x = y * dst.Stride();

                for (PCType upper = x + Width(); x < upper; ++x)
                {
                    ++dstp[x];
                }
            }
        }
    }

    template < typename _Dt1 >
    void CountTo(_Dt1 &dst, typename _Dt1::value_type value) const
    {
        for (PCType z = 0; z < GroupSize(); ++z)
        {
            auto dstp = dst.data() + GetPos(z).y * dst.Stride() + GetPos(z).x;

            for (PCType y = 0; y < Height(); ++y)
            {
                PCType x = y * dst.Stride();

                for (PCType upper = x + Width(); x < upper; ++x)
                {
                    dstp[x] += value;
                }
            }
        }
    }
};


#include "Block.hpp"


#endif
