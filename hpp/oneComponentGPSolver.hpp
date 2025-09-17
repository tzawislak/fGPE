#ifndef __one_component_GP_solver__
#define __one_component_GP_solver__

#include "psi.hpp"
#include "vext.hpp"
#include "output.hpp"
#include "baseHamiltonian.hpp"
#include <cstring>

// Derived class
class OneComponentGPSolver : public BaseHamiltonian {
protected:
    complex* psi;
    complex* vext;

public:
    Params p;

    // variables used during  time evolutions
    double PARS[32] = {0.0};
    enum Variables { EN, MU, NORM, PX, EN_PREV, AVG_TIME, ITER, ITER_TIME, EN_0, MU_0, NCYCLES, AUX1, AUX2, AUX3, AUX4, AUX5} ;

    
    // Constructor to initialize the two pointers
    OneComponentGPSolver(Params &p);

    complex* getPsi() { return psi;  }
    complex* getVext() { return vext;  }
    


    void runImagTimeEvol();
    void runRealTimeEvol();

    // 
    // functions, which have to be overriden 
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



        

    // Destructor
    ~OneComponentGPSolver() override {
        free(psi);
        free(vext); 
        free(kx);
        free(ky);
        free(kz);
        free(x);
        free(y);
        free(z);
    }
};

#endif