#include "BEC_oneComponent.hpp"


BEConeComponent::BEConeComponent(Params &par): OneComponentGPSolver(par)
{
    this->p = par;
    InitPsi();      // set the initial wavefunction
    InitVext();     // set the potential

    double h_params[NO_HAMIL_PARAMS] = {0};
    /*
        Define parameters entering your Hamiltonian
    */
    h_params[0] = p.a; // init a
    CCE(cudaMemcpy(this->d_h_params, h_params, NO_HAMIL_PARAMS * sizeof(double), cudaMemcpyHostToDevice), "CUDA error at memcpy: d_h_params");
    



    // choose the type of time evolution
    if( p.getBool("treal"))
    {
        std::cout << "# Running real time evolution" << std::endl;
        runRealTimeEvol();
    }
    else
    {
        std::cout << "# Running imaginary time evolution" << std::endl;
        runImagTimeEvol();
    }
}

   

// Initialize your wavefunction
Psi BEConeComponent::InitPsi()
{
    // You can access init params from this->p object
    Psi psi_(p, "");
    
    // BEGIN:: Manipulate your wavefunction

    // imprint vortex 
    if( p.getBool("tpi_vortex") )
    {
        psi_.imprintVortexTube(p, p.getDouble("api_prc")); 
    }
    
    // write initial wavefunction
    output.WritePsiInit( psi_.getPsi(), p.Npoints);



    // END:: Manipulate your wavefunction
    // CRUCIAL:: copy the wavefunction to psi pointer
    std::memcpy(this->psi, psi_.getPsi(), p.Npoints * sizeof(complex));
    return psi_;
} 



// Initialize your external potential
Vext BEConeComponent::InitVext()
{
    Vext vext_(p, "");
    // BEGIN:: Manipulate your external potential
    

    // the protocol
    if( p.getBool("tvseed") )
    { 
        vext_.addProtocolPotential(p, 0); 
    }

    

    output.WriteVext( vext_.getVext(), p.Npoints);
    // END:: Manipulate your external potential


    std::memcpy(this->vext, vext_.getVext(), p.Npoints * sizeof(complex));
    return vext_;
}





// ------------------------------------ 
//        Hamiltonian functions
// ------------------------------------
void BEConeComponent::alg_calcHpsi(){

    this->alg_Laplace(this->d_psi, this->d_hpsi);

    BECHamiltonian<<<gridSize, noThreads>>>( PARS[T],this->d_psi, this->d_hpsi, this->d_vext, this->d_hpsi_en, this->d_hpsi, p.Npoints, this->d_h_params);

    if( p.getBool("trotate") )
    {
        this->alg_addVPx( this->d_psi, this->d_aux, this->d_hpsi, this->d_hpsi_en, 2*pi*p.getDouble("omega")/(p.omho*p.aho) ); // this aho may be only due to compatibility with previous analysis
    }
    CCE(cudaGetLastError(), "BEC Hamiltonian Kernel launch failed");
}

void BEConeComponent::alg_calcHpsiMU(){

    this->alg_Laplace(this->d_psi, this->d_hpsi);

    BECHamiltonianMU<<<gridSize, noThreads>>>( PARS[T], this->d_psi, this->d_hpsi, this->d_vext, this->d_hpsi, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "BEC Hamiltonian Kernel launch failed");
}



void BEConeComponent::alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi ){
    CalcAverageCos<<<gridSize, noThreads>>>( _psi, this->d_x, this->d_partial, 2*p.XMAX[0], p.NX[0], p.Npoints);
    CCE(cudaGetLastError(), "Calculate Average Cos Kernel launch failed");

    (this->*reduction)(_avgcos);

}



// ------------------------------------ 
//            CUDA kernels
// ------------------------------------
__global__ void BECHamiltonian( double time, cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_en, cufftDoubleComplex* h_mu, int N, double* prms){
    /* AUXILLIARY VARIABLES */
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double rX=0.0;
    double rY=0.0;
    
    /* PARAMETERS */
    //double a = prms[0];
    double a = d_quench_as( time, prms );

    if (ix < N) {
        /* CALC H_EN */
        rX = (vext[ix].x + 2*pi*a*( psi[ix].x*psi[ix].x + psi[ix].y*psi[ix].y ));
        rY = (vext[ix].y);
        h_en[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_en[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;
        /* CALC H_MU */
        rX = (vext[ix].x + 4*pi*a*( psi[ix].x*psi[ix].x + psi[ix].y*psi[ix].y ));
        rY = (vext[ix].y);
        h_mu[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_mu[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;
    }
}

__global__ void BECHamiltonianMU( double time, cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_mu, int N, double* prms){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double rX=0.0;
    double rY=0.0;
    //double a = prms[0];
    double a = d_quench_as(time, prms);
    if (ix < N) {
        rX = (vext[ix].x + 4*pi*a*( psi[ix].x*psi[ix].x + psi[ix].y*psi[ix].y ));
        rY = (vext[ix].y);
        h_mu[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_mu[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;
    }      
}

__device__ double d_quench_as(double time, double* prms){
    double a0 = prms[0]; // init a
    double a1 = prms[1]; // 1 final a
    double a2 = prms[2]; // 2 final a
    double t1 = prms[3]; // 1 time 
    double t2 = prms[4]; // 2 time
    double t3 = prms[5]; // 3 time 
    if ( 0  < time && time <= t1 ) return a0 + (a1 - a0)/(t1) *time;
    if ( t1 < time && time <= t2 ) return a1;
    if ( t2 < time && time <= t3 ) return a1 + (a2 - a1)/(t3-t2)* (time - t2);
    return a2;
}



/*

    Imaginary time evoltuion calls

*/
/**
 * @brief Calls user-defined code before the imaginary time evolution loop
 * @note The user may access:
 *  -   this-p (input params),
 *  -   this->PARS (iteration parameters e.g. energy, chemical potential)
 *  -   this->output (output haandler),
 *  -   this->psi (current wavefunction),
 *  -   this->vext (current external potential),
 *  -   this->alg* (algorithm functions)
 */
void BEConeComponent::call_IM_init()
{
    static char status_buffer[256];
    sprintf(status_buffer, "%-7s %-10s %-10s %-15s %-15s %-10s %-15s\n", "Iter.", "Time", "Norm", "mu", "Energy", "Ediff", "<Px>");
    output.CreateStatusFile(status_buffer);
    output.WritePsiInit( this->psi, p.Npoints);
}


/**
 * @brief Calls user-defined code before every IMTE step
 * @note The user may access:
 */
void BEConeComponent::call_IM_loop_before_step() 
{

}


/**
 * @brief Calls user-defined code after every single IMTE step
 * @note The user may access:
 */
void BEConeComponent::call_IM_loop_after_step() 
{
    static char status_buffer[256];

    int iteration = (int)PARS[ITER];
    // write status every 10th iteration
    if(iteration%p.getInt("itStatMod")==0){

        this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
        PARS[PX] /= PARS[NORM]/p.aho;

        sprintf(status_buffer, "%-7d %-10.4f %-10.1f %-15.12f %-15.12f %-10.3E %-15.12f\n", (int)PARS[ITER], PARS[ITER_TIME], PARS[NORM], PARS[MU], PARS[EN], PARS[EN_PREV]-PARS[EN], PARS[PX]/hbar);
        output.WriteStatus(status_buffer);
    }


    if( (iteration+1)%p.itmod == 0 ){
        // TODO: function to execute when....
        CCE(cudaMemcpy(this->psi, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        output.WritePsi(this->psi, p.Npoints);
    }
}


/**
 * @brief Calls user-defined code when convergence is met
 * @note The user may access:
 */
void BEConeComponent::call_IM_loop_convergence() 
{
    static char status_buffer[256];

    this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
    PARS[PX] /= PARS[NORM]/p.aho*hbar;

    sprintf(status_buffer, "%-7d %-10.4f %-10.1f %-15.12f %-15.12f %-10.3E %-15.12f\n", (int)PARS[ITER], PARS[ITER_TIME], PARS[NORM], PARS[MU], PARS[EN], PARS[EN_PREV]-PARS[EN], PARS[PX]);
    output.WriteStatus(status_buffer);
    std::cout << "Convergence reached: dE = " << PARS[EN_PREV]-PARS[EN] << std::endl;
}


/**
 * @brief Calls user-defined code aftger the imaginary time evolution loop
 * @note The user may access:
 */
void BEConeComponent::call_IM_end() 
{
    output.WritePsiFinal( this->psi, p.Npoints);
    output.CloseStatusFile();

    output.WriteWtxt( p, 1);
    // calculate observables at the end:
    
    output.WriteVariable2WTXT( "en", PARS[EN], "code");
    output.WriteVariable2WTXT( "mu", PARS[MU], "code");
    output.WriteVariable2WTXT( "px", PARS[PX], "hbar");

    output.WriteVariable2WTXT( "convergence", PARS[EN_PREV]-PARS[EN], "code");

    std::cout << "Average iteration time: " << PARS[AVG_TIME]/PARS[ITER] << std::endl;

    std::cout << "Files saved under prefix: " << p.outpath << std::endl;
}

















/*

    Real time evoltuion calls

*/
void BEConeComponent::call_RE_init()
{
    static char status_buffer[256];

    // Prepare output file header
    sprintf(status_buffer, "%-15s %-10s %-10s %-15s %-15s %-15s %-15s %-10s\n", "Time", "ItTime", "Norm", "mu", "Energy", "Px", "<cos>", "Ediff");
    output.CreateRealStatusFile(status_buffer);

    output.WritePsiInit( this->psi, p.Npoints);

    PARS[NORM]=p.npart;

    // calculate initial observable values     
    this->alg_calcHpsi();
    this->alg_calc2Observables(PARS[NORM], &PARS[MU_0], &PARS[EN_0], this->d_psi, this->d_hpsi, this->d_hpsi_en );
    this->alg_calcNorm(&PARS[NORM], this->d_psi);
    this->alg_calcCos( &PARS[AUX1], this->d_psi);
    PARS[AUX1] /= PARS[NORM];
    this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
    PARS[PX] /= (PARS[NORM]*hbar)/p.aho;
    
    

    // write pre-run variables to the input file
    sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E\n", 0.0, 0.0, PARS[NORM], PARS[MU_0], PARS[EN_0], PARS[PX], PARS[AUX1], PARS[EN_0]-PARS[EN]);
    output.WriteStatusReal(status_buffer);
}


void BEConeComponent::call_RE_loop_before_step()
{
    //static char status_buffer[256];

}


void BEConeComponent::call_RE_loop_after_step()
{
    static char status_buffer[256];
    // write status every (itStatMod)th iteration
    if(((int)PARS[ITER])%p.getInt("itStatMod")==0){

        this->alg_calcNorm(&PARS[NORM], this->d_psi);
        this->alg_calc2Observables(PARS[NORM], &PARS[MU], &PARS[EN], this->d_psi, this->d_hpsi, this->d_hpsi_en );
        this->alg_calcCos( &PARS[AUX1], this->d_psi);
        PARS[AUX1]/=PARS[NORM];

        this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
        PARS[PX] /= (PARS[NORM]*hbar)/p.aho;
        double time = p.time0 + PARS[ITER] * p.dt;   // physical time in ms

        sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E\n", time, PARS[ITER_TIME], PARS[NORM], PARS[MU], PARS[EN], PARS[PX], PARS[AUX1], PARS[EN]-PARS[EN_0]);
        output.WriteStatusReal(status_buffer);

        double a0 = p.a; // init a
        double a1 = p.a*1.02; // 1 final a
        double a2 = p.a; // 2 final a
        double t1 = 1; // 1 time 
        double t2 = 1; // 2 time
        double t3 = 2; // 3 time 
        double aaa=a2;
        if ( 0  < PARS[T] && PARS[T] <= t1 ) aaa= a0 + (a1 - a0)/(t1) *PARS[T];
        if ( t1 < PARS[T] && PARS[T] <= t2 ) aaa= a1;
        if ( t2 < PARS[T] && PARS[T] <= t3 ) aaa= a1 + (a2 - a1)/(t3-t2)* (PARS[T] - t2);

        std::cout << "a: " << aaa/abohr*p.aho << "  Ediff: " << PARS[MU]-PARS[MU_0] << std::endl;
        // TODO:Add energy conservation termination
    }

    //  Save the wavefunction every p.itmod iteration
    if( ((int)PARS[ITER]+1)%p.itmod == 0 ){
        CCE(cudaMemcpy(this->psi, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        output.WritePsi(this->psi, p.Npoints);
        PARS[NCYCLES]++;
    }
}


void BEConeComponent::call_RE_end()
{
    //static char status_buffer[256];
    output.WritePsiFinal( this->psi, p.Npoints);
    output.CloseRealStatusFile();

    output.WriteWtxt( p, PARS[NCYCLES]);
    output.WriteVariable2WTXT( "en0", PARS[EN_0], "code");
    output.WriteVariable2WTXT( "mu0", PARS[MU_0], "code");
    output.WriteVariable2WTXT( "en", PARS[EN], "code");
    output.WriteVariable2WTXT( "mu", PARS[MU_0], "code");
    output.WriteVariable2WTXT( "px", PARS[PX], "hbar");

    std::cout << "Average iteration time: " << PARS[AVG_TIME]/PARS[ITER] << std::endl;
    std::cout << "Files saved under prefix: " << p.outpath << std::endl;
   
}

