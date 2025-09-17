#ifndef __double_template__
#define __double_template__


#include "twoComponentGPSolver.hpp"

class DoubleTemplate : public TwoComponentGPSolver {
public:
    Params2 p;
    
    // Constructor to initialize the two pointers
    DoubleTemplate(Params2 &p);

   
    // functions, which can be overrivden 
    void InitPsi();
    void InitVext();

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
__global__ void DoubleTemplate_template( cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
                                  cufftDoubleComplex* hpsi1, cufftDoubleComplex* hpsi2,  
                                  cufftDoubleComplex* vext1, cufftDoubleComplex* vext2, 
                                  cufftDoubleComplex* h1_en, cufftDoubleComplex* h2_en, 
                                  int N, double* prms);
__global__ void DoubleTemplateMU_template( cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
                                  cufftDoubleComplex* hpsi1, cufftDoubleComplex* hpsi2,  
                                  cufftDoubleComplex* vext1, cufftDoubleComplex* vext2, 
                                  int N, double* prms);
#endif