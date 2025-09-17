#ifndef __BEC_one_component__
#define __BEC_one_component__


#include "oneComponentGPSolver.hpp"

class BEConeComponent : public OneComponentGPSolver {
public:
    Params p;
    
    // Constructor to initialize the two pointers
    BEConeComponent(Params &p);

    Psi InitPsi();
    Vext InitVext();

    // functions, which have be overriden 
    void alg_calcHpsi() override;
    void alg_calcHpsiMU() override;

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

    // Define your functions
    void alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi );

        
};

__global__ void BECHamiltonian  ( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_en, cufftDoubleComplex* h_mu, int N, double* a);
__global__ void BECHamiltonianMU( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_mu, int N, double* a);


#endif