#ifndef __base_hamiltonian__
#define __base_hamiltonian__

#include "params.hpp"
#include "output.hpp"
#include "parallel.hpp"
#include "header.hpp"



class BaseHamiltonian: public Cuda {
protected:


     


public:
    ParamsBase* pars;
    Output output;

    double *x, *y, *z; 

    double *kx, *ky, *kz;
    explicit BaseHamiltonian(ParamsBase* par);
    Output getOutput() { return output;  }

    inline int iX(int index ) { return index % pars->NX[0]; };
    inline int iY(int index ) { return (index / pars->NX[0]) % pars->NX[1]; };
    inline int iZ(int index ) { return index / (pars->NX[0]*pars->NX[1]); };

    void InitializeKspace();
    void InitializeSpace();

   
    typedef void (BaseHamiltonian::*ReductionFuncPtr)(double*);
    typedef void (BaseHamiltonian::*Reduction2FuncPtr)(double*, double*, double);

    ReductionFuncPtr reduction;
    Reduction2FuncPtr reduction2;
    


    void _simple_reduction(double* norm);
    void _simple_reduction2(double* o1, double* o2, double norm);
    void _parallel_reduction(double* norm);
    void _parallel_reduction2(double* o1, double* o2, double norm);


    void alg_Laplace(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_hpsi);
    void alg_updatePsi(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_psi_new, cufftDoubleComplex* _d_psi_old, cufftDoubleComplex* _d_hpsi, const double &mu, double &dt, double &beta);
    void alg_addVPx(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_aux, cufftDoubleComplex* _d_hpsi, cufftDoubleComplex* _d_hpsi_en, const double &_lm);
    void alg_addOmegaLz(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_aux, cufftDoubleComplex* _d_hpsi, cufftDoubleComplex* _d_hpsi_en, const double &_lm);


    void alg_calcNorm(double *norm, cufftDoubleComplex* _d_psi);
    void alg_calcPx(double *px, cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_Pxpsi);


    void alg_updateWavefunctions(double norm, cufftDoubleComplex* _d_psi_old, cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_psi_new);
    void alg_calc2Observables(double norm, double *_o1, double *_o2, cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_o1psi, cufftDoubleComplex* _d_o2psi );

    void runImagTimeEvol();
    void runRealTimeEvol();
    void runRealTimeEvolCPU();

    
    // Virtual destructor to allow proper cleanup when deleting derived objects
    virtual ~BaseHamiltonian() {
      
    }
};



#endif
