#ifndef GAUSSIAN_CUH_
#define GAUSSIAN_CUH_


#include "Helper.cuh"
#include "Gaussian.h"


class CUDA_Gaussian2D
    : public Gaussian2D
{
public:
    typedef CUDA_Gaussian2D _Myt;
    typedef Gaussian2D _Mybase;

private:
    CudaMemMode mem_mode;

public:
    CUDA_Gaussian2D(const Gaussian2D_Para &_para = Gaussian2D_Default, CudaMemMode _mem_mode = CudaMemMode::Host2Device)
        : _Mybase(_para), mem_mode(_mem_mode)
    {}

protected:
    virtual Plane &process_Plane(Plane &dst, const Plane &src);
};


class CUDA_Gaussian2D_IO
    : public Gaussian2D_IO
{
public:
    typedef CUDA_Gaussian2D_IO _Myt;
    typedef Gaussian2D_IO _Mybase;

protected:
    virtual Frame process(const Frame &src)
    {
        CUDA_Gaussian2D filter(para);
        return filter(src);
    }

public:
    CUDA_Gaussian2D_IO(std::string _Tag = ".Gaussian")
        : _Mybase(std::move(_Tag)) {}
};


// CUDA Implementation of recursive Gaussian algorithm from "Ian T. Young, Lucas J. van Vliet - Recursive implementation of the Gaussian filter"
// For 32bit float type, the maximum sigma with valid result is about 80.
class CUDA_RecursiveGaussian
    : public RecursiveGaussian
{
public:
    typedef CUDA_RecursiveGaussian _Myt;
    typedef RecursiveGaussian _Mybase;
    typedef CUDA_FilterData<FLType> _Dt;

    static const cuIdx BLOCK_DIM = 64;

private:
    _Dt d;

public:
    CUDA_RecursiveGaussian(long double sigma, bool _allow_negative = true, CudaMemMode _mem_mode = CudaMemMode::Host2Device)
        : _Mybase(sigma, _allow_negative), d(_mem_mode, BLOCK_DIM)
    {}

    virtual void FilterV(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride);
    virtual void FilterH(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride);
    virtual void Filter(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride);

    virtual void FilterV(Plane_FL &dst, const Plane_FL &src);
    virtual void FilterH(Plane_FL &dst, const Plane_FL &src);
    virtual void Filter(Plane_FL &dst, const Plane_FL &src);

    virtual void FilterV(Plane_FL &data) { FilterV(data, data); }
    virtual void FilterH(Plane_FL &data) { FilterH(data, data); }
    virtual void Filter(Plane_FL &data) { Filter(data, data); }
};


#endif
