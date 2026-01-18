// index_macro.hpp
#ifndef __header__  // Include guard to prevent multiple inclusions
#define __header__

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <complex>
#include <stdlib.h>
#include <iostream>  // Required for std::cerr
#include <cmath>
#include <algorithm>  // for std::any_of
#include <chrono>



// Macro to calculate the 1D index from 3D coordinates (x, y, z)
#define ID(x, y, z, NX, NY) ((z) * (NX) * (NY) + (y) * (NX) + (x))

#define NTHREADS 256
#define NO_HAMIL_PARAMS 32

// 2bec: undefine to enforce each particle number conservation separately
//#define PROPER_2BEC_NORMALIZATION 1 

#define pi 3.141592653589793
#define hbar 63507.799
#define kB (0.831466*1.E7)
#define abohr 5.2917721E-5

typedef std::complex<double> complex;
typedef cufftDoubleComplex cufftDoubleComplex;

template <typename Scalar>
__host__ __device__
inline cufftDoubleComplex operator*(cufftDoubleComplex a, Scalar s)
{
    return {a.x * s, a.y * s};
}

template <typename Scalar>
__host__ __device__
inline cufftDoubleComplex operator*(Scalar s, cufftDoubleComplex a)
{
    return {a.x * s, a.y * s};
}

__host__ __device__
inline cufftDoubleComplex operator*(cufftDoubleComplex a,
                                    cufftDoubleComplex b)
{
    return {
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    };
}

__host__ __device__
inline cufftDoubleComplex operator+(cufftDoubleComplex a, 
									cufftDoubleComplex b)
{
    return {
        a.x + b.x,
        a.y + b.y
    };
}


template <typename Scalar>
__host__ __device__
inline cufftDoubleComplex& operator*=(cufftDoubleComplex& a, Scalar s)
{
    a.x *= s;
    a.y *= s;
    return a;
}


__host__ __device__
inline cufftDoubleComplex& operator+=(cufftDoubleComplex& a, cufftDoubleComplex s)
{
    a.x += s.x;
    a.y += s.y;
    return a;
}






#endif // INDEX_MACRO_HPP


