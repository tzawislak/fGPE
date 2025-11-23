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
    enum Variables { T,         // current time
                     EN,        // current energy
                     MU,        // current chemical potential
                     NORM,      // current norm
                     PX,        // current momentum along the x axis
                     LZ,        // current angular momentum along the z axis
                     EN_PREV,   // energy in the previous iteration step
                     AVG_TIME,  // stores the total (real) time of the simulation
                     ITER,      // number of completed iterations
                     ITER_TIME, // time needed for the full iteration step
                     EN_0,      // initial energy
                     MU_0,      // initial chemical potential
                     NCYCLES,   // number of intermediate wavefunctions written to the .wdat file
                     AUX1,      // auxilliary variables, use it to store any double you wish
                     AUX2,
                     AUX3,
                     AUX4,
                     AUX5
                    } ;

    
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