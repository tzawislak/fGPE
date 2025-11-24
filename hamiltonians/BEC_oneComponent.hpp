#ifndef __BEC_one_component__
#define __BEC_one_component__


#include "oneComponentGPSolver.hpp"


/**
 * @brief single-component, ordinary BEC hamiltonian class
 * 
 * @param p the Params class objects (stores all parameters from the input file)
 * 
 * @section usage Usage
 * 1. In constructor BEConeComponent(), you can define up to NO_HAMIL_PARAMS parameters your CUDA kernel Hamiltonian will have access to at runtime.
 * 2. Initialize your wavefunction in InitPsi(). You may write your own initialization as the class member function of Psi.
 * 3. Initialize your external potential in InitVext(). You may write your own initialization as the class member function of Vext.
 * 4. Write your CUDA kernel BECHamiltonian that will calculate the hamiltonian \f$\mathcal{H}|\psi\rangle\f$ and energ density functional\f$\mathcal{E}[\psi]\f$. The EDF is crucial in Imaginary time evolution and optional in real time evolution.
 * 5. Write your CUDA kernel BECHamiltonianMU that will calculate the hamiltonian \f$\mathcal{H}|\psi\rangle\f$. This is used only in real time evolution and is needed alongside BECHamiltonian kernel for time efficiency
 * @warning Make sure your CUDA kernels BECHamiltonian and BECHamiltonianMU provide the same \f$\mathcal{H}|\psi\rangle\f$.
 * 6. You may use the member functions call_IM* to log the imaginary time evolution (see descriptions of each function).
 * 7. You may use the member functions call_RE* to log the real time evolution (see descriptions of each function).
 */
class BEConeComponent : public OneComponentGPSolver {
public:
    /**
    *@brief Params object that stores all input parameters
    */
    Params p;
    
    BEConeComponent(Params &p);

    /**
    * @brief
    * Initializes the wavefunction \f$\psi\f$
    */
    Psi InitPsi();
    /**
    * @brief
    * Initializes the external potential \f$V_{ext}\f$
    */
    Vext InitVext();

    // functions, which have be overriden 
    /**
    * @brief Calculates both \f$\hat{\mathcal{H}}|\psi\rangle\f$ and \f$\mathcal{E}[\psi]\f$
    * @note  Requires hamiltonian-specific CUDA kernels to be implemented before (here BECHamiltonian)
    */
    void alg_calcHpsi() override;
    /**
    * @brief Calculates \f$\hat{\mathcal{H}}|\psi\rangle\f$
    * @note  Requires hamiltonian-specific CUDA kernels to be implemented before (here BECHamiltonianMU)
    */
    void alg_calcHpsiMU() override;

    // functions, which can be overriden (or left empty {}) 

    /**
    * @brief
    * Executed before the imaginary time evolution. 
    */
    void call_IM_init() override;
    /**
    * @brief
    * Executed before the imaginary time step. 
    */
    void call_IM_loop_before_step() override;
    /**
    * @brief
    * Executed after the imaginary time step. 
    */
    void call_IM_loop_after_step() override;
    /**
    * @brief
    * Executed at the end of algorithm. 
    */
    void call_IM_loop_convergence() override;
    /**
    * @brief
    * Executed at the end of imaginary time evolution. 
    */
    void call_IM_end() override;



    /**
    * @brief
    * Calls user-defined code before real time evolution.
    */
    void call_RE_init() override;
    /**
    * @brief
    * Calls user-defined code before every single real time evolution step.
    */
    void call_RE_loop_before_step() override;
    /**
    * @brief
    * Calls user-defined code after every single real time evolution step.
    */
    void call_RE_loop_after_step() override;
    /**
    * @brief
    * Calls user-defined code after real time evolution.
    */
    void call_RE_end() override;

    // Define user-specific functions here:
    /**
    * @brief
    * An example of a user-defined function. Calculates an average cosine  \f$\int d\textbf{x}\psi^\dagger(\textbf{x}) cos(x) \psi(\textbf{x})\f$
    * 
    * @param _avgcos  pointer to the double variable to which the result will be returned.
    * @param _psi     the wavefunction  \f$\psi\f$ on which the operation will be done.
    */
    void alg_calcCos(double *_avgcos, cufftDoubleComplex* _psi );

        
};

/**
* @brief CUDA kernel that specifies the hamiltonian of the system \f$\mathcal{H}\f$ and the energy density functional \f$\mathcal{E}\f$
* 
* Calculates \f$\mathcal{H}|\psi\rangle\f$ and \f$\mathcal{E}[\psi]\f$
* @warning This CUDA kernel implements the hamiltonian and the energy density functional of the problem. 
* @param time current time
* @param psi the wavefunction \f$\psi\f$
* @param KEpsi kinetic operator term \f$\nabla^2 \psi\f$ 
* @param vext external potential \f$V_{ext}\f$ 
* @param h_en a pointer to the array where \f$\mathcal{E}[\psi]\psi\f$ will be stored
* @param h_mu a pointer to the array where \f$\mathcal{H}\psi\f$ will be stored
* @param N the length of the wavefunction \f$\psi\f$ array
* @param a parameter array
* @return h_en energy density functional \f$\mathcal{E}\f$
* @return h_mu vector \f$\mathcal{H}|\psi\rangle\f$
*/
__global__ void BECHamiltonian  ( double time, cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_en, cufftDoubleComplex* h_mu, int N, double* a);
/**
* @brief CUDA kernel that specifies the hamiltonian of the system \f$\mathcal{H}\f$.
* 
* Calculates \f$\mathcal{H}|\psi\rangle\f$ 
* @warning This CUDA kernel implements the hamiltonian of the problem. Must be defined alongside  BECHamiltonian<> CUDA kernel
* @param time current time
* @param psi the wavefunction \f$\psi\f$
* @param KEpsi kinetic operator term \f$\nabla^2 \psi\f$ 
* @param vext external potential \f$V_{ext}\f$ 
* @param h_mu a pointer to the array where \f$\mathcal{H}\psi\f$ will be stored
* @param N the length of the wavefunction \f$\psi\f$ array
* @param a parameter array
* @return h_mu vector \f$\mathcal{H}|\psi\rangle\f$
*/
__global__ void BECHamiltonianMU( double time, cufftDoubleComplex* psi, cufftDoubleComplex* KEpsi,  cufftDoubleComplex* vext, cufftDoubleComplex* h_mu, int N, double* a);
/**
* @brief Example: a CUDA kernel that can be called from the Hamiltonian CUDA kernels
* 
* Calculates the scattering length at a given time 
* @param time current time
* @param prms parameter arra
* @return a the scattering length
*/
__device__ double d_quench_as(double time, double* prms);


#endif