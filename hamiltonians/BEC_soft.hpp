#ifndef __BEC_soft__
#define __BEC_soft__

#include "oneComponentGPSolver.hpp"

class BECsoft : public OneComponentGPSolver {
public:
    Params p;


     // FFT handles
    cufftHandle planForwardD2Z;
    cufftHandle planBackwardZ2D;
    // additional device arrays for DDI
    double *d_phidd;
    double *d_rho;
    cufftDoubleComplex *d_vtilde;
    cufftDoubleComplex *d_rhotilde;

    cufftDoubleComplex* h_temp_c;
    double* h_temp;

    // Constructor to initialize the two pointers
    BECsoft(Params &p);
    ~BECsoft()
    {
        cudaFree(d_phidd);
        cudaFree(d_vtilde);
        cudaFree(d_rhotilde);
        cudaFree(d_rho);

        free(h_temp);
        free(h_temp_c);

        cufftDestroy(planForwardD2Z);
        cufftDestroy(planBackwardZ2D);

    }
   
    // functions, which can be overriden 
    Psi InitPsi();
    Vext InitVext();

    

    void alg_calcHpsi() override;
    void alg_calcHpsiMU() override;
    void alg_Soft(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_vtilde, double* _d_rho, cufftDoubleComplex* _d_rhotilde, double* _d_phidd);
    void initialize_DeviceArrays();
    void initialize_softcoreFFT();

    
    // functions, which can be overriden (or left empty {}) 
    void call_IM_init() override;
    void call_IM_loop_before_step() override;
    void call_IM_loop_after_step() override;
    void call_IM_loop_convergence() override;
    void call_IM_end() override;

    void call_RE_init() override;
    void call_RE_loop_before_step() override;
    void call_RE_loop_after_step() override;
    void call_RE_end() override;

    // Define your functions here
    void alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi );

        
};

__global__ void BECSoftHamiltonian  ( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, double* phidd, cufftDoubleComplex* h_en, cufftDoubleComplex* h_mu, int N, double* a);
__global__ void BECSoftHamiltonianMU( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, double* phidd, cufftDoubleComplex* h_mu, int N, double* a);


#endif
