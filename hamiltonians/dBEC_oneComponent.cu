
#include "dBEC_oneComponent.hpp"

dBEConeComponent::dBEConeComponent(Params &par): OneComponentGPSolver(par)
{
    this->p = par;

    initialize_DeviceArrays();
     // Allocate memory for device arrays
    CCE(cudaMalloc((void**)&d_vtilde,   (p.NX[0]/2 +1)*p.NX[1]*p.NX[2] * sizeof(cufftDoubleComplex)), "CUDA malloc: d_vtilde");
    CCE(cudaMalloc((void**)&d_rhotilde, (p.NX[0]/2 +1)*p.NX[1]*p.NX[2] * sizeof(cufftDoubleComplex)), "CUDA malloc: d_rhotilde");
    CCE(cudaMalloc((void**)&d_rho,         p.NX[0]    *p.NX[1]*p.NX[2] * sizeof(double)), "CUDA malloc: d_rho");
    CCE(cudaMalloc((void**)&d_phidd,       p.NX[0]    *p.NX[1]*p.NX[2] * sizeof(double)), "CUDA malloc: d_phidd");

    // Create FFT plans
    if (cufftPlan3d(&planForwardD2Z, p.NX[2], p.NX[1], p.NX[0], CUFFT_D2Z) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Plan ForwardZ2D creation failed" << std::endl;
    }

    if (cufftPlan3d(&planBackwardZ2D, p.NX[2], p.NX[1], p.NX[0], CUFFT_Z2D) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Plan BackwordD2Z creation failed" << std::endl;
    }

    InitPsi();      // set the initial wavefunction
    InitVext();     // set the potential
    initialize_dipolarFFT();

    double h_params[NO_HAMIL_PARAMS] = {0};
    /*
        Define parameters entering your Hamiltonian
    */
    h_params[0] = p.a;
    h_params[1] = calcluate_LHY();
    std::cout << "# LHY: " <<  h_params[1] << std::endl;
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

   
//
// Initialize your wavefunction
// 
Psi dBEConeComponent::InitPsi()
{
    // You can access init params from this->p object
    Psi psi_(p, "");
    
    /*
        BEGIN:: Manipulate your wavefunction
    */

    // imprint vortex 
    if( p.getBool("tpi_vortex") )
    {
        psi_.imprintVortexTube(p, p.getDouble("api_prc")); 
    }
    
    // write initial wavefunction
    output.WritePsiInit( psi_.getPsi(), p.Npoints);


    if( p.getBool("tforce") )
    { 
        double offset = p.getDouble("Fdelta");
        double Ndrpl = p.getDouble("FNdrpl");
        double amp = p.getDouble("Famp"); 
        double L = 2 * p.XMAX[0];
        double k = 2*pi / L * Ndrpl;

        psi_.force1DLattice(p, amp, k, offset); 
    }
    /*
        END:: Manipulate your wavefunction
    */

    // CRUCIAL:: copy the wavefunction to psi pointer
    std::memcpy(this->psi, psi_.getPsi(), p.Npoints * sizeof(complex));
    return psi_;
} 




//
// Initialize your external potential
//
Vext dBEConeComponent::InitVext()
{
    Vext vext_(p, "");
    /*
        BEGIN:: Manipulate your external potential
    */

    // the protocol
    if( p.getBool("tvseed") )
    { 
        vext_.addProtocolPotential(p, 0); 
    }

    // optical lattice
    if( p.getBool("topt") )
    { 
        vext_.addOpticalLattice(p, 0); 
    }

    output.WriteVext( vext_.getVext(), p.Npoints);
    /*
        END:: Manipulate your external potential
    */

    std::memcpy(this->vext, vext_.getVext(), p.Npoints * sizeof(complex));
    return vext_;
}

void dBEConeComponent::alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi ){
    CalcAverageCos<<<gridSize, noThreads>>>( _psi, this->d_x, this->d_partial, 2*p.XMAX[0], p.NX[0], p.Npoints);
    CCE(cudaGetLastError(), "Calculate Average Cos Kernel launch failed");

    (this->*reduction)(_avgcos);

}

/*
 *
 *  Imaginary time evolution calls
 *
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
void dBEConeComponent::call_IM_init()
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
void dBEConeComponent::call_IM_loop_before_step() 
{

}

/**
 * @brief Calls user-defined code after every single IMTE step
 * @note The user may access:
 */
void dBEConeComponent::call_IM_loop_after_step() 
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
void dBEConeComponent::call_IM_loop_convergence() 
{
    static char status_buffer[256];

    this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
    PARS[PX] /= PARS[NORM]/p.aho;

    sprintf(status_buffer, "%-7d %-10.4f %-10.1f %-15.12f %-15.12f %-10.3E %-15.12f\n", (int)PARS[ITER], PARS[ITER_TIME], PARS[NORM], PARS[MU], PARS[EN], PARS[EN_PREV]-PARS[EN], PARS[PX]/hbar);
    output.WriteStatus(status_buffer);
    std::cout << "Convergence reached: dE = " << PARS[EN_PREV]-PARS[EN] << std::endl;
}

/**
 * @brief Calls user-defined code aftger the imaginary time evolution loop
 * @note The user may access:
 */
void dBEConeComponent::call_IM_end() 
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
 *
 *  Real time evolution calls
 *
*/
void dBEConeComponent::call_RE_init()
{
    static char status_buffer[256];

    // Prepare output file header
    sprintf(status_buffer, "%-15s %-10s %-10s %-15s %-15s %-15s %-15s %-15s\n", "Time", "ItTime", "Norm", "mu", "Energy", "Px", "<cos>", "Ediff");
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
    output.WriteWtxt( p, PARS[NCYCLES]);

}


void dBEConeComponent::call_RE_loop_before_step()
{
    //static char status_buffer[256];

}


void dBEConeComponent::call_RE_loop_after_step()
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

        // TODO:Add energy conservation termination
    }

    //  Save the wavefunction every p.itmod iteration
    if( ((int)PARS[ITER]+1)%p.itmod == 0 ){
        CCE(cudaMemcpy(this->psi, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        output.WritePsi(this->psi, p.Npoints);
        PARS[NCYCLES]++;
    }
}


void dBEConeComponent::call_RE_end()
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







// ------------------------------------ 
//            CUDA kernels
// ------------------------------------
__global__ void dBECHamiltonian( cufftDoubleComplex* psi, 
                                 cufftDoubleComplex* KEpsi,  
                                 cufftDoubleComplex* vext,
                                             double* phidd,
                                 cufftDoubleComplex* h_en, 
                                 cufftDoubleComplex* h_mu, 
                                 int N, double* prms){
    /* AUXILLIARY VARIABLES */
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double rX=0.0;
    double rY=0.0;

    double rho=0.0;
    /* PARAMETERS */
    double a        = prms[0];
    double gammaeps = prms[1];

    if (ix < N) {
        rho = psi[ix].x*psi[ix].x + psi[ix].y*psi[ix].y ;

        /* CALC H_EN */
        rX = vext[ix].x + 2*pi*a*rho + 0.5*phidd[ix] + 0.4*gammaeps*rho*sqrt(rho);
        rY = vext[ix].y;

        h_en[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_en[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;

        /* CALC H_MU */
        rX = vext[ix].x + 4*pi*a*rho + phidd[ix] + gammaeps*rho*sqrt(rho);
        rY = vext[ix].y;
        h_mu[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_mu[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;
    }
}





__global__ void dBECHamiltonianMU( cufftDoubleComplex* psi, 
                                   cufftDoubleComplex* KEpsi,
                                   cufftDoubleComplex* vext, 
                                               double* phidd,
                                   cufftDoubleComplex* h_mu, 
                                   int N, double* prms){

    /* AUXILLIARY VARIABLES */
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double rX=0.0;
    double rY=0.0;

    double rho=0.0;
    /* PARAMETERS */
    double a        = prms[0];
    double gammaeps = prms[1];

    if (ix < N) {
        rho = psi[ix].x*psi[ix].x + psi[ix].y*psi[ix].y ;

        /* CALC H_MU */
        rX = vext[ix].x + 4*pi*a*rho + phidd[ix] + gammaeps*rho*sqrt(rho);
        rY = vext[ix].y;
        h_mu[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_mu[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;
    }      
}


// ------------------------------------ 
//        Hamiltonian functions
// ------------------------------------
void dBEConeComponent::alg_calcHpsi(){

    this->alg_Laplace(this->d_psi, this->d_hpsi);

    alg_Dipolar(this->d_psi, this->d_vtilde, this->d_rho, this->d_rhotilde, this->d_phidd);

    dBECHamiltonian<<<gridSize, noThreads>>>( this->d_psi, this->d_hpsi, this->d_vext, this->d_phidd, this->d_hpsi_en, this->d_hpsi, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "dBEC Hamiltonian Kernel launch failed");

    if( p.getBool("trotate") )
    {
        this->alg_addVPx( this->d_psi, this->d_aux, this->d_hpsi, this->d_hpsi_en, 2*pi*p.getDouble("omega")/(p.omho*p.aho) );
    }
    CCE(cudaGetLastError(), "dBEC Hamiltonian Kernel launch failed");
}


void dBEConeComponent::alg_calcHpsiMU(){

    this->alg_Laplace(this->d_psi, this->d_hpsi);

    alg_Dipolar(this->d_psi, this->d_vtilde, this->d_rho, this->d_rhotilde, this->d_phidd);

    dBECHamiltonianMU<<<gridSize, noThreads>>>( this->d_psi, this->d_hpsi, this->d_vext, this->d_phidd, this->d_hpsi, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "dBEC Hamiltonian Kernel launch failed");
}





double dBEConeComponent::calcluate_LHY()
{
    std::cout << "# Initialize the LHY correction..." << std::endl;
    int ntheta = 10000;
    double dtheta = pi/(ntheta-1);

    cufftDoubleComplex sum = {0.0, 0.0}; 
    cufftDoubleComplex caux = {0.0, 0.0}; 

    for (int i = 0; i < ntheta; ++i)
    {
        double theta = i * dtheta;
        caux.x = pow(1.0 + (p.edd) * (3.0 * cos(theta)*cos(theta) - 1.0), 5);
        caux.y = 0.0;
        caux = complexSqrt(caux);
        sum.x += sin(theta)*dtheta*caux.x;
        sum.y += sin(theta)*dtheta*caux.y;
    }
    
    // discard the imaginary part - one should not, in principle, but this is what everyone does
    double feps = cuCreal(sum);
    double gammaeps = (64*sqrt(pi)/(3.0)) * sqrt(pow(p.a,5)) * feps;
        
    return gammaeps;
}

        

void dBEConeComponent::initialize_dipolarFFT()
{
    std::cout << "# Initialize the dipole-dipole interaction in kspace..." << std::endl;

    // FT of the dipole-dipole interaction.
    double dd_x = p.getDouble("dipole_x");
    double dd_y = p.getDouble("dipole_y");
    double dd_z = p.getDouble("dipole_z");
    double norm = sqrt(dd_x*dd_x + dd_y*dd_y + dd_z*dd_z);

    dd_x = dd_x / norm;
    dd_y = dd_y / norm;
    dd_z = dd_z / norm;
    
    DipoleDipoleInteraction<<<gridSize, noThreads>>>( this->d_kx, this->d_ky, this->d_kz, this->d_vtilde, p.add, dd_x, dd_y, dd_z, p.Npoints, p.NX[0]/2 +1, p.NX[1], p.NX[2]); 
    CCE(cudaGetLastError(), "DipoleDipoleInteraction kernel launch failed");
}




void dBEConeComponent::alg_Dipolar(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_vtilde, double* _d_rho, cufftDoubleComplex* _d_rhotilde, double* _d_phidd){
    // calculate rho   
    SquareArray<<<gridSize, noThreads>>>( _d_psi, _d_rho, p.NX[0]*p.NX[1]*p.NX[2]);
    CCE(cudaGetLastError(), "Square Array kernel launch failed");
    // FT of rho
    if (cufftExecD2Z(planForwardD2Z, _d_rho, _d_rhotilde) != CUFFT_SUCCESS) {   std::cerr << "CUFFT error: Forward FFT failed at DDI" << std::endl; }
    // multiply with this->d_vdd
    MultiplyArrays<<<gridSize, noThreads>>>(_d_rhotilde, _d_vtilde, _d_rhotilde, (p.NX[0]/2 +1)*p.NX[1]*p.NX[2]);
    CCE(cudaGetLastError(), "Multiply Array kernel launch failed");
 
    if( cufftExecZ2D(planBackwardZ2D, _d_rhotilde, _d_phidd) != CUFFT_SUCCESS) { std::cerr << "CUFFT error: Inverse FFT failed at DDI" << std::endl; }

    ScalarMultiply<<<gridSize, noThreads>>>(  _d_phidd, 1.0/(p.NX[0]*p.NX[1]*p.NX[2]), p.NX[0]*p.NX[1]*p.NX[2]);
    CCE(cudaGetLastError(), "Scalar Multiply Kernel launch failed");
}



void dBEConeComponent::initialize_DeviceArrays()
{
   
}