#include "single_template.hpp"


SingleTemplate::SingleTemplate(Params &par): OneComponentGPSolver(par)
{
    this->p = par;
    InitPsi();      // set the initial wavefunction
    InitVext();     // set the potential

    double h_params[NO_HAMIL_PARAMS] = {0};
    /*
        MODIFY:
        Define parameters entering your Hamiltonian
    */
    



    /*
        END MODIFY
    */

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
Psi SingleTemplate::InitPsi()
{
    // You can access init params from this->p object
    Psi psi_(p, "");
    
    /*
        BEGIN:: Manipulate your wavefunction
        HINT:   You may want to define your own initialization procedure
                as a new member function of hpp/psi.hpp
    */

    
    
    // write initial wavefunction
    // output.WritePsiInit( psi_.getPsi(), p.Npoints);
    /*
        END:: Manipulate your wavefunction
    */

    // CRUCIAL:: copy the wavefunction to psi pointer
    std::memcpy(this->psi, psi_.getPsi(), p.Npoints * sizeof(complex));
    return psi_;
} 



// Initialize your external potential
Vext SingleTemplate::InitVext()
{
    Vext vext_(p, "");
    /*
        BEGIN:: Manipulate your external potential
        HINT:   You may want to define your own initialization procedure
                as a new member function of hpp/vext.hpp
    */





    //output.WriteVext( vext_.getVext(), p.Npoints);
    /*
        END:: Manipulate your external potential
    */
    std::memcpy(this->vext, vext_.getVext(), p.Npoints * sizeof(complex));
    return vext_;
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
void SingleTemplate::call_IM_init()
{
    // initialize status file 
    static char status_buffer[256];
    sprintf(status_buffer, "%-7s %-10s %-10s %-15s %-15s %-10s %-15s\n", "Iter.", "Time", "Norm", "mu", "Energy", "Ediff", "<Px>");
    output.CreateStatusFile(status_buffer);

    // write initial wavefunction
    output.WritePsiInit( this->psi, p.Npoints);
}


/**
 * @brief Calls user-defined code before every IMTE step
 * @note The user may access:
 */
void SingleTemplate::call_IM_loop_before_step() 
{

}


/**
 * @brief Calls user-defined code after every single IMTE step
 * @note The user may access:
 */
void SingleTemplate::call_IM_loop_after_step() 
{
    static char status_buffer[256];
    int iteration = (int)PARS[ITER];

    // write status every 10th iteration
    if(iteration%p.getInt("itStatMod")==0){

        // calculate <Px>
        this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
        PARS[PX] /= PARS[NORM]/p.aho;

        // print status
        sprintf(status_buffer, "%-7d %-10.4f %-10.1f %-15.12f %-15.12f %-10.3E %-15.12f\n", (int)PARS[ITER], PARS[ITER_TIME], PARS[NORM], PARS[MU], PARS[EN], PARS[EN_PREV]-PARS[EN], PARS[PX]/hbar);
        output.WriteStatus(status_buffer);
    }


    // write current wavefunction
    if( (iteration+1)%p.itmod == 0 ){
        CCE(cudaMemcpy(this->psi, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        output.WritePsi(this->psi, p.Npoints);
    }
}


/**
 * @brief Calls user-defined code when convergence is met
 * @note The user may access:
 */
void SingleTemplate::call_IM_loop_convergence() 
{
    static char status_buffer[256];

    // calculate <Px>
    this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
    PARS[PX] /= PARS[NORM]/p.aho;

    // print status
    sprintf(status_buffer, "%-7d %-10.4f %-10.1f %-15.12f %-15.12f %-10.3E %-15.12f\n", (int)PARS[ITER], PARS[ITER_TIME], PARS[NORM], PARS[MU], PARS[EN], PARS[EN_PREV]-PARS[EN], PARS[PX]/hbar);
    output.WriteStatus(status_buffer);
    std::cout << "Convergence reached: dE = " << PARS[EN_PREV]-PARS[EN] << std::endl;
}


/**
 * @brief Calls user-defined code aftger the imaginary time evolution loop
 * @note The user may access:
 */
void SingleTemplate::call_IM_end() 
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
void SingleTemplate::call_RE_init()
{
    static char status_buffer[256];

    // Prepare output file header
    sprintf(status_buffer, "%-15s %-10s %-10s %-15s %-15s %-15s %-10s\n", "Time", "ItTime", "Norm", "mu", "Energy", "Px", "<cos>", "Ediff");
    output.CreateRealStatusFile(status_buffer);

    output.WritePsiInit( this->psi, p.Npoints);

    PARS[NORM]=p.npart;

    // calculate initial observable values     
    this->alg_calcHpsi();
    this->alg_calc2Observables(PARS[NORM], &PARS[MU_0], &PARS[EN_0], this->d_psi, this->d_hpsi, this->d_hpsi_en );
    this->alg_calcNorm(&PARS[NORM], this->d_psi);
    this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
    PARS[PX] /= (PARS[NORM]*hbar)/p.aho;
    
    

    // write pre-run variables to the input file
    sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E\n", 0.0, 0.0, PARS[NORM], PARS[MU_0], PARS[EN_0], PARS[PX], PARS[AUX1], PARS[EN_0]-PARS[EN]);
    output.WriteStatusReal(status_buffer);
}


void SingleTemplate::call_RE_loop_before_step()
{
    static char status_buffer[256];

}


void SingleTemplate::call_RE_loop_after_step()
{
    static char status_buffer[256];
    // write status every (itStatMod)th iteration
    if(((int)PARS[ITER])%p.getInt("itStatMod")==0){

        this->alg_calcNorm(&PARS[NORM], this->d_psi);
        this->alg_calc2Observables(PARS[NORM], &PARS[MU], &PARS[EN], this->d_psi, this->d_hpsi, this->d_hpsi_en );

        this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
        PARS[PX] /= (PARS[NORM]*hbar)/p.aho;
        double time = p.time0 + PARS[ITER] * p.dt;   // physical time in ms

        sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E\n", time, PARS[ITER_TIME], PARS[NORM], PARS[MU], PARS[EN], PARS[PX], PARS[AUX1], PARS[EN]-PARS[EN_0]);
        output.WriteStatusReal(status_buffer);

        // TODO:Add energy conservation termination
    }

    //  Save the wavefunction every p.itmod iteration
    if( ((int)PARS[ITER]+1)%p.itmod == 0 ){
        CCE(cudaMemcpy(this->psi, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        output.WritePsi(this->psi, p.Npoints);
        PARS[NCYCLES]++;
    }
}


void SingleTemplate::call_RE_end()
{
    static char status_buffer[256];
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







// ------------------------------------ 
//        Hamiltonian functions
// ------------------------------------
void SingleTemplate::alg_calcHpsi(){
    // Calculate the kinetic energy
    this->alg_Laplace(this->d_psi, this->d_hpsi);

    SingleTemplate_template<<<gridSize, noThreads>>>( this->d_psi, this->d_hpsi, this->d_vext, this->d_hpsi_en, this->d_hpsi, p.Npoints, this->d_h_params);
    
    CCE(cudaGetLastError(), "Hamiltonian Kernel launch failed");
}

void SingleTemplate::alg_calcHpsiMU(){
    // Calculate kinetic energy term
    this->alg_Laplace(this->d_psi, this->d_hpsi);

    SingleTemplateMU_template<<<gridSize, noThreads>>>( this->d_psi, this->d_hpsi, this->d_vext, this->d_hpsi, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "BEC Hamiltonian Kernel launch failed");
}

// ------------------------------------ 
//            CUDA kernels
// ------------------------------------
__global__ void SingleTemplate_template( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_en, cufftDoubleComplex* h_mu, int N, double* prms){
    /* AUXILLIARY VARIABLES */
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
   
    /*  
        PARAMETERS:
        PRE: must have been defined in d_h_params array
     */
    double example = prms[0];


    if (ix < N) {
        /* CALC H_EN */
                   //kinetic energy
        h_en[ix].x = KEpsi[ix].x; /* + real part of your terms */
        h_en[ix].y = KEpsi[ix].y; /* + imaginary part of your terms */

        /* CALC H_MU */
        h_mu[ix].x = KEpsi[ix].x;  /* + real part of your terms */
        h_mu[ix].y = KEpsi[ix].y;  /* + imaginary part of your terms */
    }
}



__global__ void SingleTemplateMU_template( cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_mu, int N, double* prms){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    /*  
        PARAMETERS:
        PRE: must have been defined in d_h_params array
     */
    double example = prms[0];


    if (ix < N) {
        
        h_mu[ix].x = KEpsi[ix].x;  /* + real part of your terms */
        h_mu[ix].y = KEpsi[ix].y;  /* + imaginary part of your terms */
    }      
}


