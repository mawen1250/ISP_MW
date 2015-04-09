#ifndef GAUSSIAN_CUH_
#define GAUSSIAN_CUH_


#include "Helper.cuh"
#include "Gaussian.h"


extern const CUDA_FilterMode CUDA_Gaussian2D_Default;
extern const CUDA_FilterMode CUDA_RecursiveGaussian_Default;


class CUDA_Gaussian2D
    : public Gaussian2D
{
public:
    typedef CUDA_Gaussian2D _Myt;
    typedef Gaussian2D _Mybase;

private:
    CUDA_FilterMode cuFM;

public:
    _Myt(const Gaussian2D_Para &_para = Gaussian2D_Default, const CUDA_FilterMode &_cuFM = CUDA_Gaussian2D_Default)
        : _Mybase(_para), cuFM(_cuFM)
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
    _Myt(std::string _Tag = ".Gaussian")
        : _Mybase(std::move(_Tag)) {}
};


// Implementation of recursive Gaussian algorithm from "Ian T. Young, Lucas J. van Vliet - Recursive implementation of the Gaussian filter"
// For 32bit float type, the maximum sigma with valid result is about 120.
// CUDA version
class CUDA_RecursiveGaussian
    : public RecursiveGaussian
{
public:
    typedef CUDA_RecursiveGaussian _Myt;
    typedef RecursiveGaussian _Mybase;
    typedef CUDA_FilterData<FLType> _Dt;

private:
    _Dt d;

public:
    _Myt(long double sigma, const CUDA_FilterMode &_cuFM = CUDA_RecursiveGaussian_Default)
        : _Mybase(sigma), d(_cuFM)
    {}

    __device__ const _Dt &D() const { return d; }

    virtual void FilterV(Plane_FL &dst, const Plane_FL &src);
    void FilterV(Plane_FL &data) { FilterV(data, data); }

    virtual void FilterH(Plane_FL &dst, const Plane_FL &src);
    void FilterH(Plane_FL &data) { FilterH(data, data); }

    virtual void Filter(Plane_FL &dst, const Plane_FL &src);
    void Filter(Plane_FL &data) { Filter(data, data); }

    virtual void Filter(FLType *dst, const FLType *src, PCType height, PCType width, PCType stride);
};


#endif
