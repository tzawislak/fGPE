#ifndef __BEC_rabi__
#define __BEC_rabi__


#include "twoComponentGPSolver.hpp"

class BECRabi : public TwoComponentGPSolver {
public:
    Params2 p;
    
    // Constructor to initialize the two pointers
    BECRabi(Params2 &p);

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

    // Define your functions
    void alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi );
    void alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi1, cufftDoubleComplex* _psi2, int sign );
    void alg_calcRelativePhase(double *_phi, cufftDoubleComplex* _d_psi1, cufftDoubleComplex* _d_psi2 );
    void alg_calcDszDt(double *_double, cufftDoubleComplex* _d_psi1, cufftDoubleComplex* _d_psi2 );


        
};

__global__ void BECRabi_2(cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
                          cufftDoubleComplex* hpsi1, cufftDoubleComplex* hpsi2,  
                          cufftDoubleComplex* vext1, cufftDoubleComplex* vext2, 
                          cufftDoubleComplex* h1_en, cufftDoubleComplex* h2_en, 
                          int N, double* prms);
__global__ void BECRabiMU_2(  cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
                              cufftDoubleComplex* hpsi1, cufftDoubleComplex* hpsi2,  
                              cufftDoubleComplex* vext1, cufftDoubleComplex* vext2, 
                              int N, double* prms);

#endif