#include "baseHamiltonian.hpp"

BaseHamiltonian::BaseHamiltonian(ParamsBase* par): pars(par), Cuda(par->NX, NTHREADS, par->isTwoComponent), output(pars->outpath){
        
	InitializeKspace();
    InitializeSpace();

    // choose the reduction scheme 
    if ( this->gridSize%this->noThreads==0 ) 
    {
        std::cout << "Grid: " << gridSize << " noThreads: " << noThreads << " running with GPU support for reduction kernels." << std::endl;
        this->reduction = &BaseHamiltonian::_parallel_reduction;
        this->reduction2 = &BaseHamiltonian::_parallel_reduction2;
    }
    else 
    {
        std::cout << "Grid: " << gridSize << " noThreads: " << noThreads << " running without GPU support for reduction kernels." << std::endl;

        this->reduction = &BaseHamiltonian::_simple_reduction;
        this->reduction2 = &BaseHamiltonian::_simple_reduction2;
    }
	
}

void BaseHamiltonian::InitializeSpace()
{
    this->x = (double*)malloc( pars->NX[0] * sizeof(double) );
    this->y = (double*)malloc( pars->NX[1] * sizeof(double) );
    this->z = (double*)malloc( pars->NX[2] * sizeof(double) );

    if (pars->DIM==1)
    {
        for(int ix=0; ix < pars->NX[0]; ++ix){ x[ix] = -pars->XMAX[0] + (ix)*pars->DX[0] + 0.5*pars->DX[0];}
        for(int iy=0; iy < pars->NX[1]; ++iy){ y[iy] = 0;}
        for(int iz=0; iz < pars->NX[2]; ++iz){ z[iz] = 0;}
    }else if(pars->DIM==2){
        for(int ix=0; ix < pars->NX[0]; ++ix){ x[ix] = -pars->XMAX[0] + (ix)*pars->DX[0] + 0.5*pars->DX[0];}
        for(int iy=0; iy < pars->NX[1]; ++iy){ y[iy] = -pars->XMAX[1] + (iy)*pars->DX[1] + 0.5*pars->DX[1];}
        for(int iz=0; iz < pars->NX[2]; ++iz){ z[iz] = 0;}
    }else if (pars->DIM==3)
    {
        for(int ix=0; ix < pars->NX[0]; ++ix){ x[ix] = -pars->XMAX[0] + (ix)*pars->DX[0] + 0.5*pars->DX[0];}
        for(int iy=0; iy < pars->NX[1]; ++iy){ y[iy] = -pars->XMAX[1] + (iy)*pars->DX[1] + 0.5*pars->DX[1];}
        for(int iz=0; iz < pars->NX[2]; ++iz){ z[iz] = -pars->XMAX[2] + (iz)*pars->DX[2] + 0.5*pars->DX[2];}
    }

    cudaMemcpy(d_x, this->x, pars->NX[0] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, this->y, pars->NX[1] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, this->z, pars->NX[2] * sizeof(double), cudaMemcpyHostToDevice);
    //output.Write3DMatrix( x, pars->NX[0], "x.wdat" );
    //output.Write3DMatrix( y, pars->NX[1], "y.wdat" );
    //output.Write3DMatrix( z, pars->NX[2], "z.wdat" );

}

void BaseHamiltonian::InitializeKspace()
{
    const double *DKX = pars->DKX;
    
    this->kx = (double*)malloc( pars->NX[0] * sizeof(double) );
    this->ky = (double*)malloc( pars->NX[1] * sizeof(double) );
    this->kz = (double*)malloc( pars->NX[2] * sizeof(double) );
    
    for(int ik=0; ik <= pars->NX[0]/2; ++ik){               kx[ik] = (ik)*DKX[0];}
    for(int ik=pars->NX[0]/2 + 1; ik < pars->NX[0]; ++ik){  kx[ik] = -pars->KXMAX[0] + (ik-(pars->NX[0]/2))*DKX[0];}

    for(int ik=0; ik <= pars->NX[1]/2; ++ik){               ky[ik] = (ik)*DKX[1];}
    for(int ik=pars->NX[1]/2 + 1; ik < pars->NX[1]; ++ik){  ky[ik] = -pars->KXMAX[1] + (ik-(pars->NX[1]/2))*DKX[1];}
    
    for(int ik=0; ik <= pars->NX[2]/2; ++ik){               kz[ik] = (ik)*DKX[2];}
    for(int ik=pars->NX[2]/2 + 1; ik < pars->NX[2]; ++ik){  kz[ik] = -pars->KXMAX[2] + (ik-(pars->NX[2]/2))*DKX[2];}
    
    if (pars->DIM==1)
    {
        for(int ik=0; ik < pars->NX[1]; ++ik){ ky[ik] = 0;}
        for(int ik=0; ik < pars->NX[2]; ++ik){ kz[ik] = 0;}
    }else if(pars->DIM==2){
        for(int ik=0; ik < pars->NX[2]; ++ik){ kz[ik] = 0;}
    }

    cudaMemcpy(d_kx, this->kx, pars->NX[0] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky, this->ky, pars->NX[1] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kz, this->kz, pars->NX[2] * sizeof(double), cudaMemcpyHostToDevice);


//    output.Write3DMatrix( kx, pars->NX[0], "kx.wdat" );
//    output.Write3DMatrix( ky, pars->NX[1], "ky.wdat" );
//    output.Write3DMatrix( kz, pars->NX[2], "kz.wdat" );
}   



void BaseHamiltonian::alg_Laplace(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_hpsi){
   

    if (cufftExecZ2Z(planForward, _d_psi, _d_hpsi, CUFFT_FORWARD) != CUFFT_SUCCESS) {   std::cerr << "CUFFT error: Forward FFT failed" << std::endl; }

    Laplace<<<gridSize, noThreads>>>(_d_hpsi, d_kx, d_ky, d_kz, _d_hpsi, pars->Npoints, pars->NX[0], pars->NX[1], pars->NX[2]);
    
    CCE(cudaGetLastError(), "Laplace kernel launch failed");

    if( cufftExecZ2Z(planBackward, _d_hpsi, _d_hpsi, CUFFT_INVERSE) != CUFFT_SUCCESS) { std::cerr << "CUFFT error: Inverse FFT failed" << std::endl; }

    ScalarMultiply<<<gridSize, noThreads>>>( _d_hpsi, 0.5*std::pow(2*pi, 2)/(pars->Npoints), pars->Npoints);

    CCE(cudaGetLastError(), "Scalar Multiply Kernel launch failed");
}






void BaseHamiltonian::alg_updatePsi(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_psi_new, cufftDoubleComplex* _d_psi_old, cufftDoubleComplex* _d_hpsi, const double &mu, double &dt, double &beta){
    
    BECUpdatePsi<<<gridSize, noThreads>>>( _d_psi, _d_psi_new, _d_psi_old, _d_hpsi, mu, dt, beta, pars->Npoints );
    CCE(cudaGetLastError(), "Update psi Kernel launch failed");

}



void BaseHamiltonian::_simple_reduction(double* norm){
    CCE(cudaMemcpy(h_partialSums, this->d_partial, gridSize * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error: malloc at partial sums failed");
    double n = 0.0;
    for (int i = 0; i < gridSize; ++i) {
        n += h_partialSums[i];
    }
    *norm = n*pars->DV;

}





void BaseHamiltonian::_simple_reduction2(double* _o1, double* _o2, double _norm){
    CCE(cudaMemcpy(this->h_partialSums, this->d_partial, gridSize * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error: malloc at partial sums failed");
    CCE(cudaMemcpy(this->h_partialSums2, this->d_partial2, gridSize * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error: malloc at partial sums failed");
    double o1=0.0;
    double o2=0.0;
    for (int i = 0; i < gridSize; ++i) {
        o1 += this->h_partialSums[i];
        o2 += this->h_partialSums2[i];
    }
    *_o1 = o1 * pars->DV / _norm;
    *_o2 = o2 * pars->DV / _norm;

}

void BaseHamiltonian::_parallel_reduction(double* norm){
    sumReductionKernel<<<gridSize/noThreads, noThreads, noThreads * sizeof(double)>>>(this->d_partial, this->d_final, gridSize);
    CCE(cudaGetLastError(), "Sum Reduction2 Kernel launch failed");
    CCE(cudaMemcpy(h_final, this->d_final, gridSize/noThreads * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error: malloc at partial sums failed");
    double n = 0.0;
    for (int i = 0; i < gridSize/noThreads; ++i) {
        n += h_final[i];
    }
    *norm = n*pars->DV;
}

void BaseHamiltonian::_parallel_reduction2(double* _o1, double* _o2, double _norm){

    sumReductionKernel<<<gridSize/noThreads, noThreads, noThreads * sizeof(double)>>>(this->d_partial, this->d_final, gridSize);
    CCE(cudaGetLastError(), "Sum Reduction2 Kernel launch failed");
    sumReductionKernel<<<gridSize/noThreads, noThreads, noThreads * sizeof(double)>>>(this->d_partial2, this->d_final2, gridSize);
    CCE(cudaGetLastError(), "Sum Reduction2 Kernel launch failed");
    
    CCE(cudaMemcpy(this->h_final, this->d_final, gridSize/noThreads * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error: malloc at partial sums failed");
    CCE(cudaMemcpy(this->h_final2, this->d_final2, gridSize/noThreads * sizeof(double), cudaMemcpyDeviceToHost), "CUDA error: malloc at partial sums failed");
    double o1=0.0;
    double o2=0.0;
    for (int i = 0; i < gridSize/noThreads; ++i) {
        o1 += this->h_final[i];
        o2 += this->h_final2[i];
    }
    *_o1 = o1 * pars->DV / _norm;
    *_o2 = o2 * pars->DV / _norm;
}



void BaseHamiltonian::alg_calcNorm(double *norm, cufftDoubleComplex* _d_psi){
    
    NormalizePsi<<<gridSize, noThreads>>>(_d_psi, this->d_partial, pars->Npoints);
    CCE(cudaGetLastError(), "Sum Reduction Kernel launch failed");
    
    (this->*reduction)(norm);
}

/**
 * @brief Calculates <\psi|P_x|psi>
 * */
void BaseHamiltonian::alg_calcPx(double *px, cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_Pxpsi){

    if (cufftExecZ2Z(planForward, _d_psi, _d_Pxpsi, CUFFT_FORWARD) != CUFFT_SUCCESS) {   std::cerr << "CUFFT error: Forward FFT failed" << std::endl; }
    BECVPx<<<gridSize, noThreads>>>(_d_Pxpsi, d_kx, _d_Pxpsi, pars->Npoints, pars->NX[0]);
    CCE(cudaGetLastError(), "Laplace kernel launch failed");
    if( cufftExecZ2Z(planBackward, _d_Pxpsi, _d_Pxpsi, CUFFT_INVERSE) != CUFFT_SUCCESS) { std::cerr << "CUFFT error: Inverse FFT failed" << std::endl; }

    CalculateObservable<<<gridSize, noThreads>>>( _d_psi, _d_Pxpsi, this->d_partial, pars->Npoints);
    CCE(cudaGetLastError(), "Sum Reduction Kernel launch failed");
    
    (this->*reduction)(px);
}


void BaseHamiltonian::alg_addVPx(cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_aux, cufftDoubleComplex* _d_hpsi, cufftDoubleComplex* _d_hpsi_en, const double &_lm){

    if (cufftExecZ2Z(planForward, _d_psi, _d_aux, CUFFT_FORWARD) != CUFFT_SUCCESS) {   std::cerr << "CUFFT error: Forward FFT failed" << std::endl; }

    BECVPx<<<gridSize, noThreads>>>(_d_aux, d_kx, _d_aux, pars->Npoints, pars->NX[0]);
    CCE(cudaGetLastError(), "Laplace kernel launch failed");

    if( cufftExecZ2Z(planBackward, _d_aux, _d_aux, CUFFT_INVERSE) != CUFFT_SUCCESS) { std::cerr << "CUFFT error: Inverse FFT failed" << std::endl; }
    AppendArray<<<gridSize, noThreads>>>( _d_hpsi,    (-1)*_lm/pars->Npoints, _d_aux, pars->Npoints);
    AppendArray<<<gridSize, noThreads>>>( _d_hpsi_en, (-1)*_lm/pars->Npoints, _d_aux, pars->Npoints);
    
    CCE(cudaGetLastError(), "Scalar Multiply Kernel launch failed");

}



// normalizes the wavefunction
void BaseHamiltonian::alg_updateWavefunctions(double norm, cufftDoubleComplex* _d_psi_old, cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_psi_new){

    CCE(cudaMemcpy( _d_psi_old, _d_psi, pars->Npoints * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice), "CUDA error at malloc: psi_old");

    ScalarMultiply<<<gridSize, noThreads>>>( _d_psi, _d_psi_new, norm, pars->Npoints);
    CCE(cudaGetLastError(), "Scalar Multiply Kernel launch failed");

}



void BaseHamiltonian::alg_calc2Observables(double norm, double *_o1, double *_o2, cufftDoubleComplex* _d_psi, cufftDoubleComplex* _d_o1psi, cufftDoubleComplex* _d_o2psi ){

    Calculate2Observables<<<gridSize, noThreads>>>( _d_psi, _d_o1psi, this->d_partial, _d_o2psi, this->d_partial2, pars->Npoints);
    CCE(cudaGetLastError(), "Chemical Potential and Energy Kernel launch failed");
 
    (this->*reduction2)(_o1, _o2, norm);
}






