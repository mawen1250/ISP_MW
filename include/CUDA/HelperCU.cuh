#ifdef __CUDACC__

#ifndef HELPERCU_CUH_
#define HELPERCU_CUH_


#include "Helper.cuh"
#include <device_functions.h>
#include <math_functions.h>


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
    atomicAdd(address, val);
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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif

#endif // __CUDACC__
