#ifndef __parallel__
#define __parallel__

#include "header.hpp"

// Error handling function for CUDA functions
void CCE(cudaError_t error, const char* msg) ;



class Cuda {

public:
    cufftHandle planForward;
    cufftHandle planBackward;
    const int isTwoComponent;

    int size;
    int noThreads;
    int gridSize;

    cufftDoubleComplex *d_psi_old;  
    cufftDoubleComplex *d_psi;  
    cufftDoubleComplex *d_psi_new;  
    cufftDoubleComplex *d_hpsi;  
    cufftDoubleComplex *d_hpsi_en; 
    cufftDoubleComplex *d_vext;  
    cufftDoubleComplex *d_aux;  
 

    cufftDoubleComplex *d_psi2_old;  
    cufftDoubleComplex *d_psi2;  
    cufftDoubleComplex *d_psi2_new;  
    cufftDoubleComplex *d_hpsi2;  
    cufftDoubleComplex *d_hpsi2_en;  
    cufftDoubleComplex *d_vext2;  
    cufftDoubleComplex *d_aux2;  

    double *d_h_params;


    double *d_partial;
    double *d_partial2;

    double *d_final;
    double *d_final2;
    double *h_final;
    double *h_final2;


    double *h_partialSums;
    double *h_partialSums2;



    double *d_kx;
    double *d_ky;
    double *d_kz;

    double *d_x;
    double *d_y;
    double *d_z;


    // Constructor: Allocate memory on the device
    Cuda(): isTwoComponent(false), size(0), noThreads(NTHREADS) {};
    Cuda(const int *NX, int nth,  bool _is2comp);
    Cuda(const Cuda&) = delete;
    Cuda& operator=(const Cuda&) = delete;

    void PrintGPUInfo();
    
    ~Cuda() {
            
        cudaFree(d_h_params);
        cudaFree(d_partial);
        cudaFree(d_partial2);
        cudaFree(d_final);
        cudaFree(d_final2);
        free(h_final);
        free(h_final2);

        free(h_partialSums);
        free(h_partialSums2);


        cufftDestroy(planForward);
        cufftDestroy(planBackward);
        cudaFree(d_psi_old);
        cudaFree(d_psi);
        cudaFree(d_psi_new);
        cudaFree(d_hpsi);
        cudaFree(d_hpsi_en);
        cudaFree(d_vext);
        cudaFree(d_aux);

        if(this->isTwoComponent){
            cudaFree(d_psi2_old);
            cudaFree(d_psi2);
            cudaFree(d_psi2_new);
            cudaFree(d_hpsi2);
            cudaFree(d_hpsi2_en);
            cudaFree(d_vext2);
            cudaFree(d_aux2);
        }

        cudaFree(d_kx);
        cudaFree(d_ky);
        cudaFree(d_kz);

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);

    }

};

__host__ __device__ cufftDoubleComplex complexSqrt(cufftDoubleComplex z);

// 1-COMPONEN KERNELS
__global__ void Laplace( cufftDoubleComplex* v1, double* kx,  double* ky, double* kz, cufftDoubleComplex* result, int N, int NX, int NY, int NZ);
__global__ void NormalizePsi(cufftDoubleComplex *input, double *output, int N);
__global__ void CalculateObservable(cufftDoubleComplex *psi, cufftDoubleComplex *Opsi, double *output, int N);
__global__ void SumArrays( cufftDoubleComplex* vout, cufftDoubleComplex* v2, cufftDoubleComplex a, cufftDoubleComplex* v1, int N);
// GENERAL BUT USER-SPECIFIC
__global__ void CalcAverageCos(cufftDoubleComplex *psi, double* x, double *output, double L, int NX, int N) ;
__global__ void BECVPx( cufftDoubleComplex* v1, double* kx, cufftDoubleComplex* result, int N, int NX);


// RK4 algorithm-specific kernels
__global__ void SumRK4( cufftDoubleComplex* psi_out, cufftDoubleComplex* psi_in, cufftDoubleComplex* k1, cufftDoubleComplex* k2, cufftDoubleComplex* k3, cufftDoubleComplex* k4, int N);
__global__ void CalcK( cufftDoubleComplex* k, cufftDoubleComplex* hpsi, double dt, int N);
__global__ void UpdateRKStep( cufftDoubleComplex* psi, cufftDoubleComplex* psi_old, cufftDoubleComplex* k, double sc, int N);
__global__ void UpdateRKStep( cufftDoubleComplex* psi, cufftDoubleComplex* psi_old, cufftDoubleComplex* hpsi, cufftDoubleComplex nIdt, int N);
__global__ void FinalRKStep( cufftDoubleComplex* psi, cufftDoubleComplex* psi_old, cufftDoubleComplex* _k, cufftDoubleComplex* hpsi, cufftDoubleComplex nIdt, int N);
__global__ void BECUpdatePsi( cufftDoubleComplex* psi, cufftDoubleComplex* newpsi,  cufftDoubleComplex* oldpsi, cufftDoubleComplex* hpsi, double mu, double dt, double beta, int N );


// DIPOLAR
__global__ void DipoleDipoleInteraction( double* kx,  double* ky, double* kz, cufftDoubleComplex* vdd, double a_dd, double d_x, double d_y, double d_z, int N, int NX, int NY, int NZ) ;
// SOFT CORE
__global__ void SoftCoreInteraction_1D( double* kx,  double* ky, double* kz, cufftDoubleComplex* vtilde, int N, int NX, int NY, int NZ, double* params);
__global__ void SoftCoreInteraction_2D( double* kx,  double* ky, double* kz, cufftDoubleComplex* vtilde, int N, int NX, int NY, int NZ, double* params);
__global__ void SoftCoreInteraction_3D( double* kx,  double* ky, double* kz, cufftDoubleComplex* vtilde, int N, int NX, int NY, int NZ, double* params);





// 2-COMPONENT KERNELS
__global__ void Laplace_2( cufftDoubleComplex* v1, cufftDoubleComplex* v2, double* kx,  double* ky, double* kz, cufftDoubleComplex* result1, cufftDoubleComplex* result2, int N, int NX, int NY, int NZ); 
__global__ void Calculate2Observables(cufftDoubleComplex *psi, cufftDoubleComplex *O1psi, double *output1, cufftDoubleComplex *O2psi, double *output2, int N);
__global__ void CalcAverageCos(cufftDoubleComplex *psi1, cufftDoubleComplex *psi2, double* x, double *output, double L, int NX, int N, int sign=1) ;

__global__ void CalculateRelativePhase(cufftDoubleComplex *psi1, cufftDoubleComplex *psi2, double *output, int N) ;
__global__ void CalculateDszDt(cufftDoubleComplex *psi1, cufftDoubleComplex *psi2, double *output, int N);



/*
    Generic functions for parallel operations
*/
__global__ void sumReductionKernel(double *input, double *output, int size);


/* @brief parallel scalar multiplication
 */
template <typename T, typename S>
__global__ void ScalarMultiply(T* vout, const T* vin, S a, int N)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix] = static_cast<T>(a * vin[ix]);
    }
}

/* @brief parallel scalar multiplication inplace
 */
template <typename T, typename S>
__global__ void ScalarMultiply(T* vout, S a, int N)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix] *= a;
    }
}


/* @brief parallel multiplication of an array and a scalar

    vout[] += a * vin[]
 */
template <typename T, typename S>
__global__ void AppendArray(T* vout,  S a, const T* vin, int N)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix] += static_cast<T>(a * vin[ix]);
    }
}

/* @brief parallel dot product
 */
template<typename T>
__global__ void MultiplyArrays(const T* a, const T* b, T* result, int N)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        result[ix] = a[ix] * b[ix];
    }
}

/* @brief parallel square
 */
template<typename T, typename S>
__global__ void SquareArray( const T* in, S* out, int N)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        out[ix] =  in[ix]*in[ix] ;
    }
}



#endif
