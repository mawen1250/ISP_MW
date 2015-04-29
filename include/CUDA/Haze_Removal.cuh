#ifndef HAZE_REMOVAL_CUH_
#define HAZE_REMOVAL_CUH_


#include "Helper.cuh"
#include "Haze_Removal.h"


struct CUDA_Haze_Removal_Para
    : public Haze_Removal_Para
{
    typedef CUDA_Haze_Removal_Para _Myt;
    typedef Haze_Removal_Para _Mybase;

    _Myt() : _Mybase() { ppmode = 2; }
};

extern const CUDA_Haze_Removal_Para CUDA_Haze_Removal_Default;


class CUDA_Haze_Removal_Retinex
    : public Haze_Removal
{
public:
    typedef CUDA_Haze_Removal_Retinex _Myt;
    typedef Haze_Removal _Mybase;
    typedef CUDA_FilterData<FLType> _Dt;

    static const cuIdx BLOCK_DIM = 256;

private:
    _Dt d;

    PCType pcount;
    DType sFloor;
    DType sCeil;

    DType *RdevInt = nullptr;
    DType *GdevInt = nullptr;
    DType *BdevInt = nullptr;
    FLType *Rdev = nullptr;
    FLType *Gdev = nullptr;
    FLType *Bdev = nullptr;
    FLType *tMapInv_dev = nullptr;

    cuIdx GRID_DIM;

public:
    _Myt(const Haze_Removal_Para &_para = CUDA_Haze_Removal_Default, CudaMemMode _mem_mode = CudaMemMode::Host2Device)
        : _Mybase(_para), d(_mem_mode, BLOCK_DIM)
    {}

protected:
    void Init(const Frame &src);
    void End();
    void IntToLinear();
    void LinearToInt();
    //FLType HistMax() const;

    // Main process flow
    virtual Frame &process_Frame(Frame &dst, const Frame &src);

    // Generate the Inverted Transmission Map from intensity image
    virtual void GetTMapInv();

    // Get the Global Atmospheric Light from Inverted Transmission Map and src
    virtual void GetAtmosLight();

    // Generate the haze-free image
    virtual void RemoveHaze();

    // Store the filtered result to a Frame with range scaling
    virtual void StoreResult(Frame &dst);
};


class CUDA_Haze_Removal_Retinex_IO
    : public Haze_Removal_Retinex_IO
{
public:
    typedef CUDA_Haze_Removal_Retinex_IO _Myt;
    typedef Haze_Removal_Retinex_IO _Mybase;

protected:
    virtual void arguments_process()
    {
        para = CUDA_Haze_Removal_Default;
        _Mybase::arguments_process();
    }

    virtual Frame process(const Frame &src)
    {
        CUDA_Haze_Removal_Retinex filter(para);
        return filter(src);
    }

public:
    _Myt(std::string _Tag = ".Haze_Removal_Retinex")
        : _Mybase(std::move(_Tag))
    {}
};


#endif
