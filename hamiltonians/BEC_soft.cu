
#include "BEC_soft.hpp"

BECsoft::BECsoft(Params &par): OneComponentGPSolver(par)
{
    this->p = par;

    h_temp =               (double*) malloc( p.NX[0]*p.NX[1]*p.NX[2] * sizeof(double));
    h_temp_c = (cufftDoubleComplex*) malloc( (p.NX[0]/2 +1)*p.NX[1]*p.NX[2] * sizeof(cufftDoubleComplex));


    initialize_DeviceArrays();
     // Allocate memory for device arrays
    CCE(cudaMalloc((void**)&d_vtilde,   (p.NX[0]/2 +1)*p.NX[1]*p.NX[2] * sizeof(cufftDoubleComplex)), "CUDA malloc: d_vtilde");
    CCE(cudaMalloc((void**)&d_rhotilde, (p.NX[0]/2 +1)*p.NX[1]*p.NX[2] * sizeof(cufftDoubleComplex)), "CUDA malloc: d_rhotilde");
    CCE(cudaMalloc((void**)&d_rho,         p.NX[0]    *p.NX[1]*p.NX[2] * sizeof(double)), "CUDA malloc: d_rho");
    CCE(cudaMalloc((void**)&d_phidd,       p.NX[0]    *p.NX[1]*p.NX[2] * sizeof(double)), "CUDA malloc: d_phidd");
    std::cout << "# Simulation dimension: " << p.DIM << std::endl;
    switch (p.DIM) {
        case 1:
            // Create FFT plans
            if (cufftPlan1d(&planForwardD2Z, p.NX[0], CUFFT_D2Z, 1) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan ForwardZ2D creation failed" << std::endl;
            }

            if (cufftPlan1d(&planBackwardZ2D, p.NX[0], CUFFT_Z2D, 1) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan BackwordD2Z creation failed" << std::endl;
            }
            break;

        case 2:
            // Create FFT plans
            if (cufftPlan2d(&planForwardD2Z, p.NX[1], p.NX[0], CUFFT_D2Z) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan ForwardZ2D creation failed" << std::endl;
            }

            if (cufftPlan2d(&planBackwardZ2D, p.NX[1], p.NX[0], CUFFT_Z2D) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan BackwordD2Z creation failed" << std::endl;
            }
            break;

        case 3:
            // Create FFT plans
            if (cufftPlan3d(&planForwardD2Z, p.NX[2], p.NX[1], p.NX[0], CUFFT_D2Z) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan ForwardZ2D creation failed" << std::endl;
            }

            if (cufftPlan3d(&planBackwardZ2D, p.NX[2], p.NX[1], p.NX[0], CUFFT_Z2D) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan BackwordD2Z creation failed" << std::endl;
            }
            break;

        default:
            std::cerr << "Dimension has to be 1, 2 or 3." << std::endl;
            break;
    }
    

    

    double h_params[NO_HAMIL_PARAMS] = {0};
     // Calculate Lambda (units!)
    double rho;
    double U;
    // Regardless the dimension, the wavefunction will be 3D. To scale properly, introduce transverse volume so that transverse
    // lengths do not influence the result
    double tV = 1.0;
    double asc = p.getDouble("a_soft") / p.aho;
    switch (p.DIM) {
        case 1:
            std::cout << "# Initialize 1D soft core" << std::endl;
            rho = p.npart / (p.NX[0]*p.DX[0]);
            tV  = (p.NX[1]*p.DX[1]) * (p.NX[2]*p.DX[2]);
            U   = p.getDouble("Lambda") / (2* (std::pow(asc,3)) * rho)   *tV;
            break;
        case 2:
            std::cout << "# Initialize 2D soft core" << std::endl;
            rho = p.npart / (p.NX[0]*p.DX[0] * p.NX[1]*p.DX[1]);
            tV  = p.NX[2]*p.DX[2];
            U   = p.getDouble("Lambda") / ( pi* (std::pow(asc,4)) * rho)  *tV;
            break;
        case 3:
            std::cout << "# Initialize 3D soft core" << std::endl;
            rho = p.npart / (p.NX[0]*p.DX[0] * p.NX[1]*p.DX[1] * p.NX[2]*p.DX[2]);
            tV  = 1.0;
            U   = p.getDouble("Lambda") / ( 4./3. * pi * (std::pow(asc,5)) * rho)  *tV;
            break;

        default:
            std::cerr << "Dimension has to be 1, 2 or 3." << std::endl;
            break;
    }
    
    std::cout << rho << " " << U << " " << p.aho << std::endl;
    /*
        Define parameters entering your Hamiltonian
    */
    h_params[0] = asc;
    h_params[1] = U;
    CCE(cudaMemcpy(this->d_h_params, h_params, NO_HAMIL_PARAMS * sizeof(double), cudaMemcpyHostToDevice), "CUDA error at memcpy: d_h_params");
    
    InitPsi();      // set the initial wavefunction
    InitVext();     // set the potential
    initialize_softcoreFFT();

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
Psi BECsoft::InitPsi()
{
    // You can access init params from this->p object
    Psi psi_(p, "");
    
    /*
        BEGIN:: Manipulate your wavefunction
    */
    if( p.getBool("tfree") )
    { 
        psi_.initVoid(p,0); 
        psi_.normalize(p);
    }



    if( p.getBool("tforce") )
    { 

        double akrot = p.getDouble("akrot") / (2*pi);
        double a     = p.getDouble("a_soft") / p._aho(1);
        double idd   = a/akrot; // inter-droplet distance
        double amp = p.getDouble("Famp"); 
        

        switch(p.DIM){
            case 1:{
                double k = 2*pi / idd;
                double offset = p.getDouble("Fdelta") * idd;

                psi_.force1DLattice(p, amp, k, offset); 
                break;
            } 
            case 2:{
                std::string latticetype = p.getString("Latticetype");
                if( latticetype == "triangular"){
                    double k = 2*pi / idd;

                    double offsetx = p.getDouble("Fdeltax") * idd;
                    double offsety = p.getDouble("Fdeltay") * idd;
                    psi_.force2DTriangularLattice(p, amp, k, offsetx, offsety);
                    break;
                }

                if( latticetype == "square"){
                    double k = 2*pi / idd;
                    double offsetx = p.getDouble("Fdeltax") * idd;
                    double offsety = p.getDouble("Fdeltay") * idd;
                    psi_.force2DSquareLattice(p, amp, k, offsetx, offsety);
                }
                break;
            }
            case 3:{
                std::string latticetype = p.getString("Latticetype");
                if( latticetype == "FCC"){
                    double k =  2*pi / idd;
                    double offsetx = p.getDouble("Fdeltax") * idd;
                    double offsety = p.getDouble("Fdeltay") * idd;
                    double offsetz = p.getDouble("Fdeltaz") * idd;
                    psi_.force3DFCCLattice(p, amp, k, offsetx, offsety, offsetz);
                    break;
                }

                if( latticetype == "BCC"){
                    double k =  2*pi / idd;
                    double offsetx = p.getDouble("Fdeltax") * idd;
                    double offsety = p.getDouble("Fdeltay") * idd;
                    double offsetz = p.getDouble("Fdeltaz") * idd;
                    psi_.force3DBCCLattice(p, amp, k, offsetx, offsety, offsetz);
                    break;
                }

                if( latticetype == "HEX"){
                    double k =  2*pi / idd;
                    double offsetx = p.getDouble("Fdeltax") * idd;
                    double offsety = p.getDouble("Fdeltay") * idd;
                    double offsetz = p.getDouble("Fdeltaz") * idd;
                    psi_.force3DHEXLattice(p, amp, k, offsetx, offsety, offsetz);
                    break;
                }

                if( latticetype == "SC"){
                    double k = 2*pi / idd;
                    double offsetx = p.getDouble("Fdeltax") * idd;
                    double offsety = p.getDouble("Fdeltay") * idd;
                    double offsetz = p.getDouble("Fdeltaz") * idd;
                    psi_.force3DSCLattice(p, amp, k, offsetx, offsety, offsetz);
                }
                break;
                break;
            }
        }
        
    }
    

    // imprint vortex 
    if( p.getBool("tpi_vortex") )
    {
        psi_.imprintVortexTube(p, p.getDouble("api_prc")); 
    }
    
    // write initial wavefunction
    output.WritePsiInit( psi_.getPsi(), p.Npoints);



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
Vext BECsoft::InitVext()
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

    if( p.getBool("tbox") )
    { 
        vext_.initBox(p,0); 
    }

    if( p.getBool("tfree") )
    { 
        vext_.initVoid(p,0); 
        std::cout << "Aho: " << p.aho << std::endl;
    }



    output.WriteVext( vext_.getVext(), p.Npoints);
    /*
        END:: Manipulate your external potential
    */

    std::memcpy(this->vext, vext_.getVext(), p.Npoints * sizeof(complex));
    return vext_;
}

void BECsoft::alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi ){
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
void BECsoft::call_IM_init()
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
void BECsoft::call_IM_loop_before_step() 
{

}

/**
 * @brief Calls user-defined code after every single IMTE step
 * @note The user may access:
 */
void BECsoft::call_IM_loop_after_step() 
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
void BECsoft::call_IM_loop_convergence() 
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
void BECsoft::call_IM_end() 
{
    output.WritePsiFinal( this->psi, p.Npoints);
    output.CloseStatusFile();

    output.WriteWtxt( p, 1);
    // calculate observables at the end:
    
    output.WriteVariable2WTXT( "en", PARS[EN], "code");
    output.WriteVariable2WTXT( "mu", PARS[MU], "code");
    output.WriteVariable2WTXT( "px", PARS[PX], "hbar");

    double h_params[NO_HAMIL_PARAMS] = {0};
    CCE(cudaMemcpy(h_params, this->d_h_params, NO_HAMIL_PARAMS * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error at memcpy: d_h_params");
    output.WriteVariable2WTXT( "a_soft", p.getDouble("a_soft"), "mum");
    output.WriteVariable2WTXT( "U_soft", h_params[1], "code");
    output.WriteVariable2WTXT( "Lambda", p.getDouble("Lambda"), "1");


    output.WriteVariable2WTXT( "convergence", PARS[EN_PREV]-PARS[EN], "code");

    std::cout << "Average iteration time: " << PARS[AVG_TIME]/PARS[ITER] << std::endl;

    std::cout << "Files saved under prefix: " << p.outpath << std::endl;
}

/*
 *
 *  Real time evolution calls
 *
*/
void BECsoft::call_RE_init()
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
    this->alg_calcCos( &PARS[AUX1], this->d_psi);
    PARS[AUX1] /= PARS[NORM];
    this->alg_calcPx( &PARS[PX], this->d_psi, this->d_aux);
    PARS[PX] /= (PARS[NORM]*hbar)/p.aho;
    
    

    // write pre-run variables to the input file
    sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E\n", 0.0, 0.0, PARS[NORM], PARS[MU_0], PARS[EN_0], PARS[PX], PARS[AUX1], PARS[EN_0]-PARS[EN]);
    output.WriteStatusReal(status_buffer);
}


void BECsoft::call_RE_loop_before_step()
{
    //static char status_buffer[256];

}


void BECsoft::call_RE_loop_after_step()
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


void BECsoft::call_RE_end()
{
    //static char status_buffer[256];
    output.WritePsiFinal( this->psi, p.Npoints);
    output.CloseRealStatusFile();

    output.WriteWtxt( p, PARS[NCYCLES]);
    double h_params[NO_HAMIL_PARAMS] = {0};
    CCE(cudaMemcpy(h_params, this->d_h_params, NO_HAMIL_PARAMS * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error at memcpy: d_h_params");
    output.WriteVariable2WTXT( "a_soft", p.getDouble("a_soft"), "mum");
    output.WriteVariable2WTXT( "U_soft", h_params[1], "code");
    output.WriteVariable2WTXT( "Lambda", p.getDouble("Lambda"), "1");

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
__global__ void BECSoftHamiltonian( cufftDoubleComplex* psi, 
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

    /* PARAMETERS */
    

    if (ix < N) {
        /* CALC H_EN */
        rX = vext[ix].x + 0.5*phidd[ix];
        rY = vext[ix].y;

        h_en[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_en[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;

        /* CALC H_MU */
        rX = vext[ix].x + phidd[ix];
        rY = vext[ix].y;
        h_mu[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_mu[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;
    }
}





__global__ void BECSoftHamiltonianMU( cufftDoubleComplex* psi, 
                                   cufftDoubleComplex* KEpsi,
                                   cufftDoubleComplex* vext, 
                                               double* phidd,
                                   cufftDoubleComplex* h_mu, 
                                   int N, double* prms){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double rX=0.0;
    double rY=0.0;

    if (ix < N) {
        rX = vext[ix].x + phidd[ix];
        rY = vext[ix].y;
        h_mu[ix].x = KEpsi[ix].x  +  rX*psi[ix].x - rY*psi[ix].y;
        h_mu[ix].y = KEpsi[ix].y  +  rX*psi[ix].y + rY*psi[ix].x;
    }      
}


// ------------------------------------ 
//        Hamiltonian functions
// ------------------------------------
void BECsoft::alg_calcHpsi(){

    this->alg_Laplace(this->d_psi, this->d_hpsi);

    alg_Soft(this->d_psi, this->d_vtilde, this->d_rho, this->d_rhotilde, this->d_phidd);

    BECSoftHamiltonian<<<gridSize, noThreads>>>( this->d_psi, this->d_hpsi, this->d_vext, this->d_phidd, this->d_hpsi_en, this->d_hpsi, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "dBEC Hamiltonian Kernel launch failed");

    if( p.getBool("trotate") )
    {
        this->alg_addVPx( this->d_psi, this->d_aux, this->d_hpsi, this->d_hpsi_en, 2*pi*p.getDouble("omega")/(p.omho*p.aho) );
    }
    CCE(cudaGetLastError(), "dBEC Hamiltonian Kernel launch failed");
}


void BECsoft::alg_calcHpsiMU(){

    this->alg_Laplace(this->d_psi, this->d_hpsi);

    alg_Soft(this->d_psi, this->d_vtilde, this->d_rho, this->d_rhotilde, this->d_phidd);

    BECSoftHamiltonianMU<<<gridSize, noThreads>>>( this->d_psi, this->d_hpsi, this->d_vext, this->d_phidd, this->d_hpsi, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "dBEC Hamiltonian Kernel launch failed");
}




        

void BECsoft::initialize_softcoreFFT()
{

     switch (p.DIM) {
        case 1:
            std::cout << "# Initialize the soft-core interaction in 1D kspace..." << std::endl;
            SoftCoreInteraction_1D<<<gridSize, noThreads>>>( this->d_kx, this->d_ky, this->d_kz, this->d_vtilde, p.Npoints, p.NX[0]/2 +1, p.NX[1], p.NX[2], this->d_h_params);
            CCE(cudaGetLastError(), "Soft core kernel launch failed");
            break;

        case 2:
            std::cout << "# Initialize the soft-core interaction in 2D kspace..." << std::endl;
            SoftCoreInteraction_2D<<<gridSize, noThreads>>>( this->d_kx, this->d_ky, this->d_kz, this->d_vtilde, p.Npoints, p.NX[0]/2 +1, p.NX[1], p.NX[2], this->d_h_params);
            CCE(cudaGetLastError(), "Soft core kernel launch failed");
            break;

        case 3:
            std::cout << "# Initialize the soft-core interaction in 3D kspace..." << std::endl;
            SoftCoreInteraction_3D<<<gridSize, noThreads>>>( this->d_kx, this->d_ky, this->d_kz, this->d_vtilde, p.Npoints, p.NX[0]/2 +1, p.NX[1], p.NX[2], this->d_h_params);
            CCE(cudaGetLastError(), "Soft core kernel launch failed");
            break;

        default:
            std::cerr << "Dimension has to be 1, 2 or 3." << std::endl;
            exit(3);
            break;
    }

    CCE(cudaMemcpy(this->h_temp_c, this->d_vtilde, (p.NX[0]/2 +1)*p.NX[1]*p.NX[2] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at memcpy: vtilde");
    output.Write3DMatrix( (complex*) this->h_temp_c, (p.NX[0]/2 +1)*p.NX[1]*p.NX[2], "vtilde.wdat"  );
}




void BECsoft::alg_Soft(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_vtilde, double* _d_rho, cufftDoubleComplex* _d_rhotilde, double* _d_phidd){
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



void BECsoft::initialize_DeviceArrays()
{
   
}