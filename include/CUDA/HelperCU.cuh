#ifndef HELPERCU_CUH_
#define HELPERCU_CUH_


#include "Helper.cuh"
#include <device_functions.h>
#include <math_functions.h>


#ifdef __CUDACC__


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Max Min
template < typename T >
static __inline__ __device__ T cuMax(T a, T b)
{
    return a < b ? b : a;
}

template < typename T >
static __inline__ __device__ T cuMin(T a, T b)
{
    return a > b ? b : a;
}

template < typename T >
static __inline__ __device__ T cuClip(T input, T lower, T upper)
{
    return input >= upper ? upper : input <= lower ? lower : input;
}


// Abs
template < typename T >
static __inline__ __device__ T cuAbs(T input)
{
    return input < 0 ? -input : input;
}

template < typename T >
static __inline__ __device__ T cuAbsSub(T a, T b)
{
    return a >= b ? a - b : b - a;
}


// Initialization
template < typename _Ty >
__global__ void CUDA_Set_Kernel(_Ty *dst, const cuIdx count, _Ty value)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count)
    {
        dst[idx] = value;
    }
}

template < typename _Ty >
inline void CUDA_Set(_Ty *dst, const cuIdx count, _Ty value = 0, cuIdx _block_dim = 256)
{
    if (count < _block_dim) _block_dim = count;
    CudaGlobalCall(CUDA_Set_Kernel, CudaGridDim(count, _block_dim), _block_dim)(dst, count, value);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static __inline__ __device__ float atomicMinFloat(float *address, float val)
{
    unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_ul, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ul, assumed,
            __float_as_int(cuMin(val, __int_as_float(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}

static __inline__ __device__ float atomicMaxFloat(float *address, float val)
{
    unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_ul, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ul, assumed,
            __float_as_int(cuMax(val, __int_as_float(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}

#ifdef _CUDA_1_2_
static __inline__ __device__ float atomicAddFloat(float *address, float val)
{
    unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_ul, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ul, assumed,
            __float_as_int(val + __int_as_float(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}
#else
static __inline__ __device__ float atomicAddFloat(float *address, float val)
{
    return atomicAdd(address, val);
}

static __inline__ __device__ double atomicAddFloat(double *address, double val)
{
    unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

static __inline__ __device__ double atomicMinFloat(double *address, double val)
{
    unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(cuMin(val, __longlong_as_double(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

static __inline__ __device__ double atomicMaxFloat(double *address, double val)
{
    unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(cuMax(val, __longlong_as_double(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template < typename _Ty1, typename _Ty2 >
struct CUDA_KeyPair
    : public std::pair<_Ty1, _Ty2>
{
    typedef CUDA_KeyPair<_Ty1, _Ty2> _Myt;
    typedef std::pair<_Ty1, _Ty2> _Mybase;

    typedef _Ty1 KeyType;
    typedef _Ty2 ValType;

    CUDA_KeyPair()
        : _Mybase()
    {}

    CUDA_KeyPair(const _Ty1& _Val1, const _Ty2& _Val2)
        : _Mybase(_Val1, _Val2)
    {}

    CUDA_KeyPair(const _Myt &_Right)
        : _Mybase(_Right)
    {}

    CUDA_KeyPair(_Myt &&_Right)
        : _Mybase(_Right)
    {}

    __device__ _Myt &operator=(const _Myt &_Right)
    {
        _Mybase::operator=(_Right);
        return *this;
    }

    __device__ _Myt &operator=(_Myt &&_Right)
    {
        _Mybase::operator=(_Right);
        return *this;
    }

    __device__ bool operator==(const _Myt &_Right)
    {
        return this->first == _Right.first;
    }

    __device__ bool operator!=(const _Myt &_Right)
    {
        return this->first != _Right.first;
    }

    __device__ bool operator<(const _Myt &_Right)
    {
        return this->first < _Right.first;
    }

    __device__ bool operator>(const _Myt &_Right)
    {
        return this->first > _Right.first;
    }

    __device__ bool operator<=(const _Myt &_Right)
    {
        return this->first <= _Right.first;
    }

    __device__ bool operator>=(const _Myt &_Right)
    {
        return this->first >= _Right.first;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif // __CUDACC__


#endif
