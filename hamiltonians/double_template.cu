#include "double_template.hpp"


DoubleTemplate::DoubleTemplate(Params2 &par): TwoComponentGPSolver(par)
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
void DoubleTemplate::InitPsi()
{
    // You can access init params from this->p object
    Psi psi_1(p, 1);
    Psi psi_2(p, 2);
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
    std::memcpy(this->psi1, psi_1.getPsi(), p.Npoints * sizeof(complex));
    std::memcpy(this->psi2, psi_2.getPsi(), p.Npoints * sizeof(complex));
} 



// Initialize your external potential
void DoubleTemplate::InitVext()
{
    Vext vext_1(p, 1);
    Vext vext_2(p, 2);
    /*
        BEGIN:: Manipulate your external potential
        HINT:   You may want to define your own initialization procedure
                as a new member function of hpp/vext.hpp
    */





    //output.WriteVext( vext_1.getVext(), p.Npoints);
    /*
        END:: Manipulate your external potential
    */
    std::memcpy(this->vext1, vext_1.getVext(), p.Npoints * sizeof(complex));
    std::memcpy(this->vext2, vext_2.getVext(), p.Npoints * sizeof(complex));
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
void DoubleTemplate::call_IM_init()
{
 
}


/**
 * @brief Calls user-defined code before every IMTE step
 * @note The user may access:
 */
void DoubleTemplate::call_IM_loop_before_step() 
{

}


/**
 * @brief Calls user-defined code after every single IMTE step
 * @note The user may access:
 */
void DoubleTemplate::call_IM_loop_after_step() 
{
   
}


/**
 * @brief Calls user-defined code when convergence is met
 * @note The user may access:
 */
void DoubleTemplate::call_IM_loop_convergence() 
{
   
}


/**
 * @brief Calls user-defined code aftger the imaginary time evolution loop
 * @note The user may access:
 */
void DoubleTemplate::call_IM_end() 
{
  
}


















/*

    Real time evoltuion calls

*/
void DoubleTemplate::call_RE_init()
{
  
}


void DoubleTemplate::call_RE_loop_before_step()
{
   
}


void DoubleTemplate::call_RE_loop_after_step()
{
  
}


void DoubleTemplate::call_RE_end()
{
   
}







// ------------------------------------ 
//        Hamiltonian functions
// ------------------------------------
void DoubleTemplate::alg_calcHpsi(){
    // Calculate the kinetic energy
    this->alg_Laplace(this->d_psi, this->d_hpsi);

    DoubleTemplate_template<<<gridSize, noThreads>>>( this->d_psi, this->d_psi2, this->d_hpsi, this->d_hpsi2, this->d_vext, this->d_vext2, this->d_hpsi_en, this->d_hpsi2_en, p.Npoints, this->d_h_params);

    CCE(cudaGetLastError(), "Hamiltonian Kernel launch failed");
}

void DoubleTemplate::alg_calcHpsiMU(){
    // Calculate kinetic energy term
    this->alg_Laplace(this->d_psi, this->d_hpsi);

    DoubleTemplateMU_template<<<gridSize, noThreads>>>( this->d_psi, this->d_psi2, this->d_hpsi, this->d_hpsi2, this->d_vext, this->d_vext2, p.Npoints, this->d_h_params);
    CCE(cudaGetLastError(), "BEC Hamiltonian Kernel launch failed");
}

// ------------------------------------ 
//            CUDA kernels
// ------------------------------------
// -------------------------------------  
//
//  :::     BEC_twoComponents        :::
//
// -------------------------------------
__global__ void DoubleTemplate_template( cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
                                  cufftDoubleComplex* hpsi1, cufftDoubleComplex* hpsi2,  
                                  cufftDoubleComplex* vext1, cufftDoubleComplex* vext2, 
                                  cufftDoubleComplex* h1_en, cufftDoubleComplex* h2_en, 
                                  int N, double* prms){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double aX=0.0;
    double aY=0.0;
    /* PARAMETERS */
    double a11 = prms[0];
    double a22 = prms[1];
    double a12 = prms[2];
    if (ix < N) {
    // FIRST COMPONENT
        // energy:
        aX = (vext1[ix].x + 2*pi*a11*( psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y ) 
                          + 2*pi*a12*( psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y ));
        aY = (vext1[ix].y);
        h1_en[ix].x = hpsi1[ix].x  +  aX*psi1[ix].x - aY*psi1[ix].y;
        h1_en[ix].y = hpsi1[ix].y  +  aX*psi1[ix].y + aY*psi1[ix].x;

        // chemical potential:
        aX = (vext1[ix].x + 4*pi*a11*( psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y )
                          + 4*pi*a12*( psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y ));
        aY = (vext1[ix].y);
        hpsi1[ix].x += aX*psi1[ix].x - aY*psi1[ix].y;
        hpsi1[ix].y += aX*psi1[ix].y + aY*psi1[ix].x;

    // SECOND COMPONENT
        // energy:
        aX = (vext2[ix].x + 2*pi*a22*( psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y ) 
                          + 2*pi*a12*( psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y ));
        aY = (vext2[ix].y);
        h2_en[ix].x = hpsi2[ix].x  +  aX*psi2[ix].x - aY*psi2[ix].y;
        h2_en[ix].y = hpsi2[ix].y  +  aX*psi2[ix].y + aY*psi2[ix].x;

        // chemical potential:
        aX = (vext2[ix].x + 4*pi*a22*( psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y )
                          + 4*pi*a12*( psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y ));
        aY = (vext1[ix].y);
        hpsi2[ix].x += aX*psi2[ix].x - aY*psi2[ix].y;
        hpsi2[ix].y += aX*psi2[ix].y + aY*psi2[ix].x;
    }      
}


__global__ void DoubleTemplateMU_template( cufftDoubleComplex* psi1, cufftDoubleComplex* psi2, 
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
    if (ix < N) {
    // FIRST COMPONENT
        // chemical potential:
        aX = (vext1[ix].x + 4*pi*a11*( psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y )
                          + 4*pi*a12*( psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y ));
        aY = (vext1[ix].y);
        hpsi1[ix].x += aX*psi1[ix].x - aY*psi1[ix].y;
        hpsi1[ix].y += aX*psi1[ix].y + aY*psi1[ix].x;

    // SECOND COMPONENT
        // chemical potential:
        aX = (vext2[ix].x + 4*pi*a22*( psi2[ix].x*psi2[ix].x + psi2[ix].y*psi2[ix].y )
                          + 4*pi*a12*( psi1[ix].x*psi1[ix].x + psi1[ix].y*psi1[ix].y ));
        aY = (vext1[ix].y);
        hpsi2[ix].x += aX*psi2[ix].x - aY*psi2[ix].y;
        hpsi2[ix].y += aX*psi2[ix].y + aY*psi2[ix].x;
    }      
}