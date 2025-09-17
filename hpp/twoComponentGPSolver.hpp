#ifndef __two_component_GP_solver__
#define __two_component_GP_solver__

#include "psi.hpp"
#include "vext.hpp"
#include "output.hpp"
#include "baseHamiltonian.hpp"

#include <cstring>


// Derived class
class TwoComponentGPSolver : public BaseHamiltonian {
protected:
    complex* psi1;
    complex* psi2;
    complex* vext1;
    complex* vext2;
    
public:
    Params2 p;
    // variables used during  time evolutions
    double PARS[64] = {0.0};
    enum Variables { EN1, EN2, MU1, MU2, NORM1, NORM2, PX1, PX2, EN1_PREV, EN2_PREV, AVG_TIME, ITER, ITER_TIME, EN1_0, EN2_0, MU1_0, MU2_0, NCYCLES, AUX1_1, AUX2_1, AUX3_1, AUX4_1, AUX5_1 ,AUX1_2, AUX2_2, AUX3_2, AUX4_2, AUX5_2} ;


    TwoComponentGPSolver(Params2 &par);
    complex* getPsi(int i) { return (i==1) ? psi1 : psi2;  }
    complex* getVext(int i) { return (i==1) ? vext1 : vext2;  }

    void runImagTimeEvol();
    void runRealTimeEvol();

    
    virtual void alg_calcHpsi() = 0;
    virtual void alg_calcHpsiMU() = 0;

    virtual void call_IM_init() = 0;
    virtual void call_IM_loop_before_step() = 0;
    virtual void call_IM_loop_after_step() = 0;
    virtual void call_IM_loop_convergence() = 0;
    virtual void call_IM_end() = 0;

    virtual void call_RE_init() = 0;
    virtual void call_RE_loop_before_step() = 0;
    virtual void call_RE_loop_after_step() = 0;
    virtual void call_RE_end() = 0;


    virtual ~TwoComponentGPSolver() {

        free(psi1);
        free(psi2);
        free(vext1); 
        free(vext2); 
        free(kx);
        free(ky);
        free(kz);
        free(x);
        free(y);
        free(z);
    }
};

#endif