#include "oneComponentGPSolver.hpp"

OneComponentGPSolver::OneComponentGPSolver(Params &par): BaseHamiltonian(&par) {
    this->p = par;
    
    //
    // initialize the wavefunction array, the user will define it in derived class
    // 
    this->psi = (complex*)malloc( p.Npoints * sizeof(complex) );
    

    //
    // initialize the external potential
    // 
    this->vext = (complex*)malloc( p.Npoints * sizeof(complex) );
   
}





void OneComponentGPSolver::runImagTimeEvol(){
    
    // Execute code before the IMTE loop
    this->call_IM_init();

    // Copy the data to the Device
    CCE(cudaMemcpy(this->d_vext, this->vext, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: vext");
    CCE(cudaMemcpy(this->d_psi_old, this->psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi_old");
    CCE(cudaMemcpy(this->d_psi, this->psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi");
    


    // run params
    double dt = p.dt*p.omho/1000;
    double beta = p.getDouble("beta");

    //
    //  Imaginary time evolution loop
    //
    for (int iteration = 0; iteration < p.niter; ++iteration)
    {
        // begin time measurement
        auto start = std::chrono::high_resolution_clock::now();

        PARS[EN_PREV] = PARS[EN];

        // Execute code before the step
        this->call_IM_loop_before_step();
        /*

            THE CORE - DO NOT MODIFY    
        
        */

        // APPLY THE HAMILTONIAN
        this->alg_calcHpsi();

        // UPADATE THE WAVEFUNCTION                                                      this mu has no effect, but slows down
        this->alg_updatePsi(this->d_psi, this->d_psi_new, this->d_psi_old, this->d_hpsi, 0*PARS[MU], dt, beta);

        // UPDATE WAVEFUNCTIONS
        this->alg_calcNorm(&PARS[NORM], this->d_psi_new);
        this->alg_updateWavefunctions(std::pow( p.npart/PARS[NORM], 0.5), this->d_psi_old, this->d_psi, this->d_psi_new);

        // CALCULATE OBSERVABLES
        PARS[NORM]=p.npart;
        this->alg_calc2Observables(PARS[NORM], &PARS[MU], &PARS[EN], this->d_psi, this->d_hpsi, this->d_hpsi_en );
             
        // END OF THIS STEP

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_time = end - start;
        PARS[AVG_TIME] += iter_time.count();
        PARS[ITER_TIME] = iter_time.count();
        

        // CHECK CONVERGENCE
        if ( abs(PARS[EN]-PARS[EN_PREV])/PARS[EN_PREV] < p.epsilon_e )
        {
            this->call_IM_loop_convergence();
            break;
        }


        // Execute code after the step
        this->call_IM_loop_after_step();
        
        PARS[ITER]++;
    }

    // copy the wavefunction back to the host
    CCE(cudaMemcpy(this->psi, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
    // TODO: copy also the vext

    // Execute code after the time evolution has finished
    this->call_IM_end();

}

















void OneComponentGPSolver::runRealTimeEvol(){


       
    // copy the data to the Device
    CCE(cudaMemcpy(this->d_vext,    this->vext, p.Npoints *sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: vext");
    CCE(cudaMemcpy(this->d_psi_old, this->psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi_old");
    CCE(cudaMemcpy(this->d_psi,     this->psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi");

    // rename an array pointer
    cufftDoubleComplex *d_k = this->d_psi_new;  


    double dt=p.dt*p.omho/1000;

    // auxilliary numbers
    cufftDoubleComplex m05Idt = {0.0, -0.5*dt}; 
    cufftDoubleComplex mIdt =   {0.0, -1.0*dt}; 
    cufftDoubleComplex m16Idt =   {0.0, (-1.0*dt)/6.0}; 

    // Execute code before the RETE loop
    this->call_RE_init();

    //
    //  Run real time evolution
    //
    for ( int iteration = 1; iteration < p.niter; ++iteration)
    {

        auto start = std::chrono::high_resolution_clock::now();

        // Execute code before the step
        this->call_RE_loop_before_step();
        /*

            THE CORE - DO NOT MODIFY    
        
        */

        CCE(cudaMemcpy(this->d_psi_old, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at memcpy: psi_old");
        
        //
        // RK4 steps (I think the current implementation is only for time-independent H (see RK4 ))
        // 
        this->alg_calcHpsiMU();
        CCE(cudaMemcpy(d_k, this->d_hpsi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at memcpy: psi_old");
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi, this->d_psi_old, this->d_hpsi, m05Idt, p.Npoints);

        this->alg_calcHpsiMU();
        UpdateRKStep<<<gridSize, noThreads>>>( d_k, d_k, this->d_hpsi, 2.0 , p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi, this->d_psi_old, this->d_hpsi, m05Idt, p.Npoints);
    
        this->alg_calcHpsiMU();
        UpdateRKStep<<<gridSize, noThreads>>>( d_k, d_k, this->d_hpsi, 2.0 , p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi, this->d_psi_old, this->d_hpsi, mIdt, p.Npoints);
     
        this->alg_calcHpsi();
        FinalRKStep<<<gridSize, noThreads>>>( this->d_psi, this->d_psi_old, d_k, this->d_hpsi, mIdt, p.Npoints);
    
        // Do not normalize the state after the real time evolution
        // TE should be unitary, so any divergence from the norm will point out 
        // the code's imprecision. 
        // 
        //this->alg_calcNorm(&PARS[NORM], this->d_psi);
        //ScalarMultiply<<<gridSize, noThreads>>>( this->d_psi, std::pow( p.npart/PARS[NORM], 0.5), p.Npoints);
        //CCE(cudaGetLastError(), "Scalar Multiply Kernel launch failed");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_time = end - start;
        PARS[AVG_TIME] += iter_time.count();
        PARS[ITER_TIME] = iter_time.count();

        // Execute code after the step
        this->call_RE_loop_after_step();

        PARS[ITER]++;


    } // END of the real time evolution


    CCE(cudaMemcpy(this->psi, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
    // TODO: copy also the vext

    // Execute code after the time evolution has finished
    this->call_RE_end();

}



