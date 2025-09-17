#ifndef __single_template__
#define __single_template__


#include "oneComponentGPSolver.hpp"

class SingleTemplate : public OneComponentGPSolver {
public:
    Params p;
    
    // Constructor to initialize the two pointers
    SingleTemplate(Params &p);

   
    // functions, which can be overrivden 
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

    // Define your functions...


        
};


// Define your Hamiltonian kernels.
// Adjust arguments at will
__global__ void SingleTemplate_template  ( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_en, cufftDoubleComplex* h_mu, int N, double* prms);
__global__ void SingleTemplateMU_template( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_mu, int N, double* prms);


#endif