#include "BEC_Rabi.hpp"


BECRabi::BECRabi(Params2 &par): TwoComponentGPSolver(par)
{
    this->p = par;
    InitPsi();      // set the initial wavefunction
    InitVext();     // set the potential

    double h_params[NO_HAMIL_PARAMS] = {0};
    /*
        Define parameters entering your Hamiltonian
    */
    double hbarOmega = 4*pi*(p.npart1 + p.npart2)/(p.getDouble("Lambda")*p.V) * ((p.a11+p.a22)/2.0 - p.a12); 


    h_params[0] = p.a11;
    h_params[1] = p.a22;
    h_params[2] = p.a12;
    h_params[3] = -1*p.getDouble("d_rabi")/(p.omho1*2.);
    h_params[4] = -1*hbarOmega/2.;
    CCE(cudaMemcpy(this->d_h_params, h_params, NO_HAMIL_PARAMS * sizeof(double), cudaMemcpyHostToDevice), "CUDA error at memcpy: d_h_params");
    



    //
    // choose the type of time evolution
    //
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
void BECRabi::InitPsi()
{
    // You can access init params from this->p object
    Psi psi_1(p, 1);
    Psi psi_2(p, 2);
    
    /*
        BEGIN:: Manipulate your wavefunction
    */

    // imprint vortex 
    if( p.getBool("tpi_vortex") )
    {
        psi_1.imprintVortexTube(p, p.getDouble("api_prc1")); 
        psi_2.imprintVortexTube(p, p.getDouble("api_prc2")); 
    }
    
    // write initial wavefunction
    //output.WritePsiInit( psi_1.getPsi(), p.Npoints);



    /*
        END:: Manipulate your wavefunction
    */

    // CRUCIAL:: copy the wavefunction to psi pointer
    std::memcpy(this->psi1, psi_1.getPsi(), p.Npoints * sizeof(complex));
    std::memcpy(this->psi2, psi_2.getPsi(), p.Npoints * sizeof(complex)); 
} 




//
// Initialize your external potential
//
void BECRabi::InitVext()
{
    Vext vext_1(p, 1);
    Vext vext_2(p, 2);
    /*
        BEGIN:: Manipulate your external potential
    */
    // the protocol
    if( p.getBool("tvseed") )
    { 
        vext_1.addProtocolPotential(p, 1, 1); 
        if( p.getBool("tdensity") )
        {
            vext_2.addProtocolPotential(p, 2, 1); 
        }else{
            vext_2.addProtocolPotential(p, 2, -1); 
        }
        
    }

    // optical lattice
    //if( p.getBool("topt") )
    //{ 
        // CHECK 
        //vext_1.addOpticalLattice(p, 0); 
        //vext_2.addOpticalLattice(p, 0); 
    //}

    //output.WriteVext( vext_1.getVext(), p.Npoints, "1");
    //output.WriteVext( vext_2.getVext(), p.Npoints, "2");
    /*
        END:: Manipulate your external potential
    */
    std::memcpy(this->vext1, vext_1.getVext(), p.Npoints * sizeof(complex));
    std::memcpy(this->vext2, vext_2.getVext(), p.Npoints * sizeof(complex));
}







// ------------------------------------ 
//        Hamiltonian functions
// ------------------------------------
void BECRabi::alg_calcHpsi(){

    this->alg_Laplace(this->d_psi , this->d_hpsi );
    this->alg_Laplace(this->d_psi2, this->d_hpsi2);

    if( p.getBool("trotate") )
    {
        this->alg_addVPx( this->d_psi,  this->d_psi_new,  this->d_hpsi,  this->d_hpsi_en,  2*pi*p.getDouble("omega")/(p.omho1*p.aho1) );
        this->alg_addVPx( this->d_psi2, this->d_psi2_new, this->d_hpsi2, this->d_hpsi2_en, 2*pi*p.getDouble("omega")/(p.omho2*p.aho2) );
    }

    BECRabi_2<<<gridSize, noThreads>>>( this->d_psi, this->d_psi2, this->d_hpsi, this->d_hpsi2, this->d_vext, this->d_vext2, this->d_hpsi_en, this->d_hpsi2_en, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "BEC 2 Hamiltonian Kernel launch failed");
}

void BECRabi::alg_calcHpsiMU(){

    this->alg_Laplace(this->d_psi, this->d_hpsi);
    this->alg_Laplace(this->d_psi2, this->d_hpsi2);

    BECRabiMU_2<<<gridSize, noThreads>>>( this->d_psi, this->d_psi2, this->d_hpsi, this->d_hpsi2, this->d_vext, this->d_vext2, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "BEC 2 Hamiltonian Kernel launch failed");
}




void BECRabi::alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi ){
    CalcAverageCos<<<gridSize, noThreads>>>( _psi, this->d_x, this->d_partial, 2*p.XMAX[0], p.NX[0], p.Npoints);
    CCE(cudaGetLastError(), "Calculate Average Cos Kernel launch failed");

    (this->*reduction)(_avgcos);
}

void BECRabi::alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi1, cufftDoubleComplex* _psi2, int sign )
{
    CalcAverageCos<<<gridSize, noThreads>>>( _psi1, _psi2, this->d_x, this->d_partial, 2*p.XMAX[0], p.NX[0], p.Npoints);
    CCE(cudaGetLastError(), "Calculate Average Cos Kernel launch failed");

    (this->*reduction)(_avgcos);

}



// ------------------------------------ 
//            CUDA kernels
// ------------------------------------
__global__ void BECRabi_2( cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
                                  cufftDoubleComplex* hpsi1, cufftDoubleComplex* hpsi2,  
                                  cufftDoubleComplex* vext1, cufftDoubleComplex* vext2, 
                                  cufftDoubleComplex* h1_en, cufftDoubleComplex* h2_en, 
                                  int N, double* prms){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double aX=0.0;  // elements of single-particle hamiltonian (execpt for the kinetic term)
    double aY=0.0;
    /* PARAMETERS */
    double a11 = prms[0];
    double a22 = prms[1];
    double a12 = prms[2];
    double delta = prms[3];
    double Omega = prms[4];
    double rho1=0.0;
    double rho2=0.0;
    if (ix < N) {
        rho1 = psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y;
        rho2 = psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y;
        //
        //  | H11 H12 | |psi1|  = |H11psi1 + H12psi2|
        //  | H21 H22 | |psi2|  = |H21psi1 + H22psi2|
        //
        //  where H12 = H21 = real number

        // Energy
        aX = vext1[ix].x + 2*pi*a11*rho1 + 2*pi*a12*rho2 + delta ;
        aY = vext1[ix].y ;
        h1_en[ix].x = hpsi1[ix].x  +  aX*psi1[ix].x-aY*psi1[ix].y  +  Omega*psi2[ix].x;
        h1_en[ix].y = hpsi1[ix].y  +  aX*psi1[ix].y+aY*psi1[ix].x  +  Omega*psi2[ix].y;

        aX = vext2[ix].x + 2*pi*a22*rho2 + 2*pi*a12*rho1 - delta ;
        aY = vext2[ix].y;
        h2_en[ix].x = hpsi2[ix].x  +  aX*psi2[ix].x-aY*psi2[ix].y  +  Omega*psi1[ix].x;
        h2_en[ix].y = hpsi2[ix].y  +  aX*psi2[ix].y+aY*psi2[ix].x  +  Omega*psi1[ix].y;



        // chemical potential:
        aX = vext1[ix].x + 4*pi*a11*rho1 + 4*pi*a12*rho2 + delta;
        aY = vext1[ix].y;
        hpsi1[ix].x += aX*psi1[ix].x-aY*psi1[ix].y  +  Omega*psi2[ix].x;
        hpsi1[ix].y += aX*psi1[ix].y+aY*psi1[ix].x  +  Omega*psi2[ix].y;

        aX = vext2[ix].x + 4*pi*a22*rho2 + 4*pi*a12*rho1 - delta;
        aY = vext2[ix].y;
        hpsi2[ix].x += aX*psi2[ix].x-aY*psi2[ix].y  +  Omega*psi1[ix].x;
        hpsi2[ix].y += aX*psi2[ix].y+aY*psi2[ix].x  +  Omega*psi1[ix].y;
    }      
}


__global__ void BECRabiMU_2( cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
                                  cufftDoubleComplex* hpsi1, cufftDoubleComplex* hpsi2,  
                                  cufftDoubleComplex* vext1, cufftDoubleComplex* vext2, 
                                  int N, double* prms){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double aX=0.0;
    double aY=0.0;
    /* PARAMETERS */
    double a11 = prms[0];
    double a22 = prms[1];
    double a12 = prms[2];
    double delta = prms[3];
    double Omega = prms[4];
    double rho1=0.0;
    double rho2=0.0;
    if (ix < N) {
    // FIRST COMPONENT
        rho1 = psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y;
        rho2 = psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y;
        // chemical potential:
        aX = vext1[ix].x + 4*pi*a11*rho1 + 4*pi*a12*rho2 + delta;
        aY = vext1[ix].y;
        hpsi1[ix].x += aX*psi1[ix].x-aY*psi1[ix].y  +  Omega*psi2[ix].x;
        hpsi1[ix].y += aX*psi1[ix].y+aY*psi1[ix].x  +  Omega*psi2[ix].y;

        aX = vext2[ix].x + 4*pi*a22*rho2 + 4*pi*a12*rho1 - delta;
        aY = vext2[ix].y;
        hpsi2[ix].x += aX*psi2[ix].x-aY*psi2[ix].y  +  Omega*psi1[ix].x;
        hpsi2[ix].y += aX*psi2[ix].y+aY*psi2[ix].x  +  Omega*psi1[ix].y;
    }      
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
void BECRabi::call_IM_init()
{
    static char status_buffer[256];

    sprintf(status_buffer, "%-7s %-10s %-10s %-15s %-15s %-15s %-10s %-15s %-15s %-15s %-10s %-10s\n", "Iter.", "Time", "Norm1", "mu1", "Energy1", "Px1", "Norm2", "mu2", "Energy2", "Px2", "Ediff1", "Ediff2");
    output.CreateStatusFile(status_buffer);

    output.WritePsiInit( this->psi1, p.Npoints, "1");
    output.WritePsiInit( this->psi2, p.Npoints, "2");
}


/**
 * @brief Calls user-defined code before every IMTE step
 * @note The user may access:
 */
void BECRabi::call_IM_loop_before_step() 
{

}


/**
 * @brief Calls user-defined code after every single IMTE step
 * @note The user may access:
 */
void BECRabi::call_IM_loop_after_step() 
{
    static char status_buffer[256];

    int iteration = (int)PARS[ITER];
    // write status every 10th iteration
    if(iteration%p.getInt("itStatMod")==0){

        this->alg_calcPx( &PARS[PX1], this->d_psi, this->d_aux);
        this->alg_calcPx( &PARS[PX2], this->d_psi2, this->d_aux2);

        PARS[PX1] /= PARS[NORM1]/p.aho1;
        PARS[PX2] /= PARS[NORM2]/p.aho1;

        sprintf(status_buffer, "%-7d %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-10.1f %-15.12f %-15.12f %-15.12f %-10.3E %-10.3E\n", (int)PARS[ITER], PARS[ITER_TIME], PARS[NORM1], PARS[MU1], PARS[EN1], PARS[PX1]/hbar, PARS[NORM2], PARS[MU2], PARS[EN2], PARS[PX2]/hbar, PARS[EN1_PREV]-PARS[EN1], PARS[EN2_PREV]-PARS[EN2]);
        output.WriteStatus(status_buffer);
    }


    if( (iteration+1)%p.itmod == 0 ){
        // TODO: function to execute when....
        CCE(cudaMemcpy(this->psi1, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        CCE(cudaMemcpy(this->psi2, this->d_psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        output.WritePsi(this->psi1, p.Npoints);
        output.WritePsi(this->psi2, p.Npoints);
        PARS[NCYCLES]++;
    }
}


/**
 * @brief Calls user-defined code when convergence is met
 * @note The user may access:
 */
void BECRabi::call_IM_loop_convergence() 
{
    static char status_buffer[256];

    this->alg_calcPx( &PARS[PX1], this->d_psi, this->d_aux);
    this->alg_calcPx( &PARS[PX2], this->d_psi2, this->d_aux2);
    PARS[PX1] /= PARS[NORM1]/p.aho1;
    PARS[PX2] /= PARS[NORM2]/p.aho1;

    sprintf(status_buffer, "%-7d %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-10.1f %-15.12f %-15.12f %-15.12f %-10.3E %-10.3E\n", (int)PARS[ITER], PARS[ITER_TIME], PARS[NORM1], PARS[MU1], PARS[EN1], PARS[PX1]/hbar, PARS[NORM2], PARS[MU2], PARS[EN2], PARS[PX2]/hbar, PARS[EN1_PREV]-PARS[EN1], PARS[EN2_PREV]-PARS[EN2]);
    output.WriteStatus(status_buffer);
    std::cout << "Convergence reached: dE1 = " << PARS[EN1_PREV]-PARS[EN1] << std::endl;
    std::cout << "                     dE2 = " << PARS[EN2_PREV]-PARS[EN2] << std::endl;
}



/**
 * @brief Calls user-defined code aftger the imaginary time evolution loop
 * @note The user may access:
 */
void BECRabi::call_IM_end() 
{
    output.WritePsiFinal( this->psi1, p.Npoints, "1");
    output.WritePsiFinal( this->psi2, p.Npoints, "2");
    output.CloseStatusFile();

    output.WriteWtxt_2c( p, PARS[NCYCLES]);
    // calculate observables at the end:
    
    output.WriteVariable2WTXT( "en1", PARS[EN1], "code");
    output.WriteVariable2WTXT( "en2", PARS[EN2], "code");
    output.WriteVariable2WTXT( "mu1", PARS[MU1], "code");
    output.WriteVariable2WTXT( "mu2", PARS[MU2], "code");
    output.WriteVariable2WTXT( "px1", PARS[PX1], "hbar");
    output.WriteVariable2WTXT( "px2", PARS[PX2], "hbar");
    output.WriteVariable2WTXT( "convergence1", PARS[EN1_PREV]-PARS[EN1], "code");
    output.WriteVariable2WTXT( "convergence2", PARS[EN2_PREV]-PARS[EN2], "code");

    double h_params[NO_HAMIL_PARAMS] = {0};
    CCE(cudaMemcpy(h_params, this->d_h_params, NO_HAMIL_PARAMS * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error at memcpy: h_params");
    output.WriteVariable2WTXT( "Lambda_rabi", p.getDouble("Lambda"), "1");
    output.WriteVariable2WTXT( "d_rabi", p.getDouble("d_rabi"), "code");
    output.WriteVariable2WTXT( "O_rabi", h_params[4], "code");
    double n =  (PARS[NORM1] + PARS[NORM2])/p.V;
    double n1 = PARS[NORM1]/p.V;
    double n2 = PARS[NORM2]/p.V;
    double sz = (PARS[NORM1] - PARS[NORM2])/p.V ;
    output.WriteVariable2WTXT( "N",  (PARS[NORM1] + PARS[NORM2]) , "1");

    output.WriteVariable2WTXT( "n",  n /(std::pow(p.aho1,3)) , "1/mum^3");
    output.WriteVariable2WTXT( "n1",  n1 /(std::pow(p.aho1,3)) , "1/mum^3");
    output.WriteVariable2WTXT( "n2",  n2 /(std::pow(p.aho1,3)) , "1/mum^3");

    output.WriteVariable2WTXT( "V",  p.V*(std::pow(p.aho1,3)) , "mum^3");

    output.WriteVariable2WTXT( "<sz>", sz/n, "1");
    output.WriteVariable2WTXT( "P", sz/n, "1");
    output.WriteVariable2WTXT( "G", (p.a11 - p.a22)/(p.a11 + p.a22), "1");
    output.WriteVariable2WTXT( "G12",     2*(p.a12)/(p.a11 + p.a22), "1");

    double phi = 0.0;
    alg_calcRelativePhase(&phi, this->d_psi, this->d_psi2);
    std::cout << "REL PHI: " <<  phi/pi << std::endl;
    output.WriteVariable2WTXT( "phir", phi/pi, "pi");

    

    std::cout << "Average iteration time: " << PARS[AVG_TIME]/PARS[ITER] << std::endl;

    std::cout << "Files saved under prefix: " << p.outpath << std::endl;
}



void BECRabi::alg_calcRelativePhase(double *_phi, cufftDoubleComplex* _d_psi1, cufftDoubleComplex* _d_psi2 ){
    CalculateRelativePhase<<<gridSize, noThreads>>>( _d_psi1, _d_psi2, this->d_partial, p.Npoints);
    CCE(cudaGetLastError(), "Relative phase Kernel launch failed");
    // here norm has to be p.Npoints 
    (this->*reduction)(_phi);
    *_phi /= p.Npoints*p.DV;
}


void BECRabi::alg_calcDszDt(double *_double, cufftDoubleComplex* _d_psi1, cufftDoubleComplex* _d_psi2 ){

    CalculateRelativePhase<<<gridSize, noThreads>>>( _d_psi1, _d_psi2, this->d_partial, p.Npoints);
    CCE(cudaGetLastError(), "dsz/dt Kernel launch failed");
    // here norm has to be p.Npoints 
    (this->*reduction)(_double);
}







/*

    Real time evoltuion calls

*/
void BECRabi::call_RE_init()
{
    static char status_buffer[256];

    // Prepare output file header
    //sprintf(status_buffer, "%-15s %-10s %-10s %-15s %-15s %-15s %-10s %-15s %-15s  %-15s %-15s %-15s %-10s %-10s\n", "Time", "ItTime", "Norm1", "mu1", "Energy1", "Px1", "Norm2", "mu2", "Energy2", "Px2", "<cos_1>", "<cos_2>", "Ediff1", "Ediff2");
    sprintf(status_buffer, "%-15s %-10s %-10s %-15s %-15s %-15s %-10s %-15s %-15s  %-15s %-15s %-15s %-15s %-10s %-10s\n", "Time", "ItTime", "Norm1", "mu1", "Energy1", "Px1", "Norm2", "mu2", "Energy2", "Px2", "<cos_1>", "<cos_2>", "dszdt", "Ediff1", "Ediff2");
    output.CreateRealStatusFile(status_buffer);

    output.WriteWtxt_2c( p, 0);
    output.WritePsiInit( this->psi1, p.Npoints, "1");
    output.WritePsiInit( this->psi2, p.Npoints, "2");

    PARS[NORM1]=p.npart1;
    PARS[NORM2]=p.npart2;

    // calculate initial observable values     
    this->alg_calcHpsi();
    this->alg_calc2Observables(PARS[NORM1], &PARS[MU1_0], &PARS[EN1_0], this->d_psi, this->d_hpsi, this->d_hpsi_en );
    this->alg_calc2Observables(PARS[NORM2], &PARS[MU2_0], &PARS[EN2_0], this->d_psi2, this->d_hpsi2, this->d_hpsi2_en );
    this->alg_calcNorm(&PARS[NORM1], this->d_psi);
    this->alg_calcNorm(&PARS[NORM2], this->d_psi2);
    this->alg_calcCos( &PARS[AUX1_1], this->d_psi);
    PARS[AUX1_1] /= PARS[NORM1];
    this->alg_calcCos( &PARS[AUX1_2], this->d_psi2);
    PARS[AUX1_2] /= PARS[NORM2];
    this->alg_calcPx( &PARS[PX1], this->d_psi, this->d_aux);
    this->alg_calcPx( &PARS[PX2], this->d_psi2, this->d_aux2);
    PARS[PX1] /= (PARS[NORM1]*hbar)/p.aho1;
    PARS[PX2] /= (PARS[NORM2]*hbar)/p.aho1;
    //relative phase
    alg_calcDszDt(&PARS[AUX3_1], this->d_psi, this->d_psi2);
    //PARS[AUX3_1] /= pi;
    

    // write pre-run variables to the input file
    sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E %-10.3E\n", 0.00, 0.00, PARS[NORM1], PARS[MU1_0], PARS[EN1_0], PARS[PX1], PARS[NORM2], PARS[MU2_0], PARS[EN2_0], PARS[PX2], PARS[AUX1_1], PARS[AUX1_2], PARS[AUX3_1], PARS[EN1]-PARS[EN1_0], PARS[EN2]-PARS[EN2_0]);
    //sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E %-10.3E\n", 0.00, 0.00, PARS[NORM1], PARS[MU1_0], PARS[EN1_0], PARS[PX1], PARS[NORM2], PARS[MU2_0], PARS[EN2_0], PARS[PX2], PARS[AUX1_1], PARS[AUX1_2], PARS[EN1]-PARS[EN1_0], PARS[EN2]-PARS[EN2_0]);
    output.WriteStatusReal(status_buffer);
}


void BECRabi::call_RE_loop_before_step()
{
    //static char status_buffer[256];

}


void BECRabi::call_RE_loop_after_step()
{
    static char status_buffer[256];
    // write status every (itStatMod)th iteration
    if(((int)PARS[ITER])%p.getInt("itStatMod")==0){

        this->alg_calcNorm(&PARS[NORM1], this->d_psi);
        this->alg_calcNorm(&PARS[NORM2], this->d_psi2);
        this->alg_calc2Observables(PARS[NORM1], &PARS[MU1], &PARS[EN1], this->d_psi, this->d_hpsi, this->d_hpsi_en );
        this->alg_calc2Observables(PARS[NORM2], &PARS[MU2], &PARS[EN2], this->d_psi2, this->d_hpsi2, this->d_hpsi2_en );

        this->alg_calcCos( &PARS[AUX1_1], this->d_psi);
        PARS[AUX1_1]/=PARS[NORM1];
        this->alg_calcCos( &PARS[AUX1_2], this->d_psi2);
        PARS[AUX1_2]/=PARS[NORM2];

        this->alg_calcPx( &PARS[PX1], this->d_psi, this->d_aux);
        this->alg_calcPx( &PARS[PX2], this->d_psi2, this->d_aux);
        //this->alg_calcCos( &avgcos_s, this->d_psi, this->d_psi2, -1);

         //relative phase
        alg_calcDszDt(&PARS[AUX3_1], this->d_psi, this->d_psi2);
        //PARS[AUX3_1] /= pi;

        PARS[PX1] /= (PARS[NORM1]*hbar)/p.aho1;
        PARS[PX2] /= (PARS[NORM2]*hbar)/p.aho1;
        double time = p.time0 + PARS[ITER] * p.dt;   // physical time in ms
        //sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E %-10.3E\n", time, PARS[ITER_TIME], PARS[NORM1], PARS[MU1], PARS[EN1], PARS[PX1], PARS[NORM2], PARS[MU2], PARS[EN2], PARS[PX2], PARS[AUX1_1], PARS[AUX1_2], PARS[EN1]-PARS[EN1_0], PARS[EN2]-PARS[EN2_0]);
        sprintf(status_buffer, "%-15.5f %-10.4f %-10.1f %-15.12f %-15.12f %-15.12f %-10.1f %-15.12f %-15.12f %-15.12f %-15.12f %-15.12f %-15.12f %-10.3E %-10.3E\n", time, PARS[ITER_TIME], PARS[NORM1], PARS[MU1], PARS[EN1], PARS[PX1], PARS[NORM2], PARS[MU2], PARS[EN2], PARS[PX2], PARS[AUX1_1], PARS[AUX1_2], PARS[AUX3_1], PARS[EN1]-PARS[EN1_0], PARS[EN2]-PARS[EN2_0]);

        output.WriteStatusReal(status_buffer);

        // TODO:Add energy conservation termination
    }

    //  Save the wavefunction every p.itmod iteration
    if( ((int)PARS[ITER]+1)%p.itmod == 0 ){
        CCE(cudaMemcpy(this->psi1, this->d_psi, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        CCE(cudaMemcpy(this->psi2, this->d_psi2, p.Npoints * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost), "CUDA error at malloc: psi");
        output.WritePsi( this->psi1, p.Npoints, "1");
        output.WritePsi( this->psi2, p.Npoints, "2");
        PARS[NCYCLES]++;
        output.WriteWtxt_2c( p, PARS[NCYCLES]);
    }
}



void BECRabi::call_RE_end()
{
    //static char status_buffer[256];
    output.WritePsiFinal( this->psi1, p.Npoints, "1");
    output.WritePsiFinal( this->psi2, p.Npoints, "2");
    output.CloseRealStatusFile();

    output.WriteWtxt_2c( p, PARS[NCYCLES]);
    output.WriteVariable2WTXT( "en10", PARS[EN1_0], "code");
    output.WriteVariable2WTXT( "en20", PARS[EN2_0], "code");
    output.WriteVariable2WTXT( "mu10", PARS[MU1_0], "code");
    output.WriteVariable2WTXT( "mu20", PARS[MU2_0], "code");
    output.WriteVariable2WTXT( "en1", PARS[EN1], "code");
    output.WriteVariable2WTXT( "en1", PARS[EN2], "code");
    output.WriteVariable2WTXT( "mu1", PARS[MU1_0], "code");
    output.WriteVariable2WTXT( "mu2", PARS[MU2_0], "code");
    output.WriteVariable2WTXT( "px1", PARS[PX1], "hbar");
    output.WriteVariable2WTXT( "px2", PARS[PX2], "hbar");
   

    std::cout << "Average iteration time: " << PARS[AVG_TIME]/PARS[ITER] << std::endl;
    std::cout << "Files saved under prefix: " << p.outpath << std::endl;
   
}

