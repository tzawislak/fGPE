#include "twoComponentGPSolver.hpp"

TwoComponentGPSolver::TwoComponentGPSolver(Params2 &par): BaseHamiltonian(&par)  {
    
    this->p = par;
    // initialize device arrays
    this->psi1 = (complex*)malloc( p.Npoints * sizeof(complex) );
    this->psi2 = (complex*)malloc( p.Npoints * sizeof(complex) );
    this->vext1 = (complex*)malloc( p.Npoints * sizeof(complex) );
    this->vext2 = (complex*)malloc( p.Npoints * sizeof(complex) );
}



void TwoComponentGPSolver::runImagTimeEvol()
{


    // Execute code before the IMTE loop
    this->call_IM_init();
    

    CCE(cudaMemcpy(this->d_vext , this->vext1, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: vext");
    CCE(cudaMemcpy(this->d_vext2, this->vext2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: vext");
    CCE(cudaMemcpy(this->d_psi_old, this->psi1, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi_old");
    CCE(cudaMemcpy(this->d_psi2_old, this->psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi_old");
    CCE(cudaMemcpy(this->d_psi, this->psi1, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi");
    CCE(cudaMemcpy(this->d_psi2, this->psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi");

    

    double dt = p.dt*p.omho1/1000; // omho1 --> mind the scale
    double beta = p.getDouble("beta");

    for (int iteration = 0; iteration < p.niter; ++iteration)
    {

        auto start = std::chrono::high_resolution_clock::now();
        PARS[EN1_PREV] = PARS[EN1];
        PARS[EN2_PREV] = PARS[EN2];

        // Execute code before the step
        this->call_IM_loop_before_step();
        /*

            THE CORE - DO NOT MODIFY    
        
        */       
        // APPLY THE HAMILTONIAN
        this->alg_calcHpsi();

        // UPADATE THE WAVEFUNCTION
#ifndef PROPER_2BEC_NORMALIZATION
        this->alg_updatePsi(this->d_psi,  this->d_psi_new,  this->d_psi_old,  this->d_hpsi, PARS[MU1], dt, beta);
        this->alg_updatePsi(this->d_psi2, this->d_psi2_new, this->d_psi2_old, this->d_hpsi2, PARS[MU2], dt, beta);
#endif

#ifdef  PROPER_2BEC_NORMALIZATION
        this->alg_updatePsi(this->d_psi,  this->d_psi_new,  this->d_psi_old,  this->d_hpsi, PARS[MU1]+PARS[MU2], dt, beta);
        this->alg_updatePsi(this->d_psi2, this->d_psi2_new, this->d_psi2_old, this->d_hpsi2, PARS[MU1]+PARS[MU2], dt, beta);
#endif

        // CALCULATE NORMS OF THE TWO COMPONENTS
        this->alg_calcNorm(&PARS[NORM1], this->d_psi_new);
        this->alg_calcNorm(&PARS[NORM2], this->d_psi2_new);
        

        // NORMALIZE AND UPDATE WAVEFUNCTIONS
#ifndef PROPER_2BEC_NORMALIZATION
        this->alg_updateWavefunctions(std::pow( p.npart1/PARS[NORM1], 0.5), this->d_psi_old, this->d_psi, this->d_psi_new);
        this->alg_updateWavefunctions(std::pow( p.npart2/PARS[NORM2], 0.5), this->d_psi2_old, this->d_psi2, this->d_psi2_new);
#endif
        // CALCULATE OBSERVABLES
        //PARS[NORM1]=p.npart1;
        //PARS[NORM2]=p.npart2;
        //
#ifdef  PROPER_2BEC_NORMALIZATION
        this->alg_updateWavefunctions(std::pow( (p.getDouble("npart1")+p.getDouble("npart2"))/(PARS[NORM1]+PARS[NORM2]), 0.5), this->d_psi_old, this->d_psi, this->d_psi_new);
        this->alg_updateWavefunctions(std::pow( (p.getDouble("npart1")+p.getDouble("npart2"))/(PARS[NORM1]+PARS[NORM2]), 0.5), this->d_psi2_old, this->d_psi2, this->d_psi2_new);
#endif

        this->alg_calc2Observables(PARS[NORM1], &PARS[MU1], &PARS[EN1], this->d_psi, this->d_hpsi, this->d_hpsi_en );
        this->alg_calc2Observables(PARS[NORM2], &PARS[MU2], &PARS[EN2], this->d_psi2, this->d_hpsi2, this->d_hpsi2_en );

             
        // END OF THIS STEP

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_time = end - start;
        PARS[AVG_TIME] += iter_time.count();
        PARS[ITER_TIME] = iter_time.count();

        // CHECK CONVERGENCE
        if ( (abs(PARS[EN1]-PARS[EN1_PREV]) < p.epsilon_e) && (abs(PARS[EN2]-PARS[EN2_PREV]) < p.epsilon_e)  )
        {
            this->call_IM_loop_convergence();
            break;
        }
        
         // Execute code after the step
        this->call_IM_loop_after_step();
        
        PARS[ITER]++;

    }

    CCE(cudaMemcpy(this->psi1, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
    CCE(cudaMemcpy(this->psi2, this->d_psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
   
    // Execute code after the time evolution has finished
    this->call_IM_end();
}







void TwoComponentGPSolver::runRealTimeEvol(){

    CCE(cudaMemcpy(this->d_vext,    this->vext1, p.Npoints *sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: vext");
    CCE(cudaMemcpy(this->d_psi_old, this->psi1, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi_old");
    CCE(cudaMemcpy(this->d_psi,     this->psi1, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi");

    CCE(cudaMemcpy(this->d_vext2,    this->vext2, p.Npoints *sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: vext");
    CCE(cudaMemcpy(this->d_psi2_old, this->psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi_old");
    CCE(cudaMemcpy(this->d_psi2,     this->psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi");

    cufftDoubleComplex *d_k1 = this->d_psi_new;  
    cufftDoubleComplex *d_k2 = this->d_psi2_new;  

    double dt=p.dt*p.omho1/1000;

    cufftDoubleComplex m05Idt = {0.0, -0.5*dt}; 
    cufftDoubleComplex mIdt =   {0.0, -1.0*dt}; 
    cufftDoubleComplex m16Idt = {0.0, (-1.0*dt)/6.0}; 

    // Execute code before the RETE loop
    this->call_RE_init();
    
    for ( int iteration = 1; iteration < p.niter; ++iteration)
    {

        auto start = std::chrono::high_resolution_clock::now();
        // Execute code before the step
        this->call_RE_loop_before_step();
        /*

            THE CORE - DO NOT MODIFY    
        
        */

        CCE(cudaMemcpy(this->d_psi_old, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at memcpy: psi_old");
        CCE(cudaMemcpy(this->d_psi2_old, this->d_psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at memcpy: psi_old");
         
        this->alg_calcHpsiMU();
        CCE(cudaMemcpy(d_k1, this->d_hpsi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at memcpy: k_1");
        CCE(cudaMemcpy(d_k2, this->d_hpsi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at memcpy: k_2");
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi,  this->d_psi_old,  this->d_hpsi,  m05Idt, p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi2, this->d_psi2_old, this->d_hpsi2, m05Idt, p.Npoints);

        this->alg_calcHpsiMU();
        UpdateRKStep<<<gridSize, noThreads>>>( d_k1, d_k1, this->d_hpsi,  2.0 , p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( d_k2, d_k2, this->d_hpsi2, 2.0 , p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi,  this->d_psi_old,  this->d_hpsi,  m05Idt, p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi2, this->d_psi2_old, this->d_hpsi2, m05Idt, p.Npoints);
    
        this->alg_calcHpsiMU();
        UpdateRKStep<<<gridSize, noThreads>>>( d_k1, d_k1, this->d_hpsi,  2.0 , p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( d_k2, d_k2, this->d_hpsi2, 2.0 , p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi,  this->d_psi_old,  this->d_hpsi,  mIdt, p.Npoints);
        UpdateRKStep<<<gridSize, noThreads>>>( this->d_psi2, this->d_psi2_old, this->d_hpsi2, mIdt, p.Npoints);
     
        this->alg_calcHpsiMU();
        FinalRKStep<<<gridSize, noThreads>>>( this->d_psi,  this->d_psi_old,  d_k1, this->d_hpsi,  mIdt, p.Npoints);
        FinalRKStep<<<gridSize, noThreads>>>( this->d_psi2, this->d_psi2_old, d_k2, this->d_hpsi2, mIdt, p.Npoints);

        
        // Do not normalize the state after the real time evolution
        // TE should be unitary, so any divergence from the norm will point out 
        // the code's imprecision. 
        // 
        //this->alg_calcNorm(&norm1, this->d_psi);
        //this->alg_calcNorm(&norm2, this->d_psi2);
        //ScalarMultiply<<<gridSize, noThreads>>>( this->d_psi, std::pow( p.npart/norm, 0.5), p.Npoints);
        //CCE(cudaGetLastError(), "Scalar Multiply Kernel launch failed");
    
        
        // END OF THIS STEP

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_time = end - start;
        PARS[AVG_TIME] += iter_time.count();
        PARS[ITER_TIME] = iter_time.count();
    
        // Execute code after the step
        this->call_RE_loop_after_step();

        PARS[ITER]++;
        
    }


    CCE(cudaMemcpy(this->psi1, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi1");
    CCE(cudaMemcpy(this->psi2, this->d_psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi2");
    // Execute code after the time evolution has finished
    this->call_RE_end();
}
