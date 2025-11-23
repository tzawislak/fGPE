#include "parallel.hpp"                                                                     


void CCE(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(error) << ")\n";
    }
}


Cuda::Cuda(const int *NX, int nth, bool _is2comp) : isTwoComponent(_is2comp), size(NX[0]*NX[1]*NX[2]), noThreads(nth) {

    PrintGPUInfo();

    gridSize = ((size + noThreads - 1) / noThreads);
    // Hamiltonian parameters
    CCE(cudaMalloc(&d_h_params, NO_HAMIL_PARAMS * sizeof(double)), "CUDA error: malloc d_h_params");
    // an array for parrallel summation
    CCE(cudaMalloc(&d_partial, gridSize * sizeof(double)), "CUDA error: malloc d_partial");
    CCE(cudaMalloc(&d_partial2, gridSize * sizeof(double)), "CUDA error: malloc d_partial2");
    CCE(cudaMalloc(&d_final, gridSize/noThreads * sizeof(double)), "CUDA error: malloc d_final");
    CCE(cudaMalloc(&d_final2, gridSize/noThreads * sizeof(double)), "CUDA error: malloc d_final2");
    h_final = (double*) malloc( gridSize/noThreads * sizeof(double));
    h_final2 = (double*) malloc( gridSize/noThreads * sizeof(double));


    h_partialSums = (double*) malloc( gridSize * sizeof(double));
    h_partialSums2 = (double*) malloc( gridSize * sizeof(double));

    CCE(cudaMalloc((void**)&d_psi_old,  size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_psi_old");
    CCE(cudaMalloc((void**)&d_psi,      size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_psi");
    CCE(cudaMalloc((void**)&d_psi_new,  size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_psi_new");
    CCE(cudaMalloc((void**)&d_hpsi,     size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_hpsi");
    CCE(cudaMalloc((void**)&d_hpsi_en,  size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_hpsi_en");
    CCE(cudaMalloc((void**)&d_vext,     size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_vext");
    CCE(cudaMalloc((void**)&d_aux,      size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_aux");

    if(this->isTwoComponent)
    {
        std::cout << "Allocating memory for the second component..." << std::endl;
        // Allocate memory for the second component
        CCE(cudaMalloc((void**)&d_psi2_old,  size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_psi2_old");
        CCE(cudaMalloc((void**)&d_psi2,      size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_psi2");
        CCE(cudaMalloc((void**)&d_psi2_new,  size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_psi2_new");
        CCE(cudaMalloc((void**)&d_hpsi2,     size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_hpsi2");
        CCE(cudaMalloc((void**)&d_hpsi2_en,  size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_hpsi2_en");
        CCE(cudaMalloc((void**)&d_vext2,     size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_vext2");
        CCE(cudaMalloc((void**)&d_aux2,      size * sizeof(cufftDoubleComplex)), "CUDA malloc: d_aux2");

    }

    CCE(cudaMalloc((void**)&d_x, NX[0] * sizeof(double)), "CUDA malloc: d_x");
    CCE(cudaMalloc((void**)&d_y, NX[1] * sizeof(double)), "CUDA malloc: d_y");
    CCE(cudaMalloc((void**)&d_z, NX[2] * sizeof(double)), "CUDA malloc: d_z");  

    CCE(cudaMalloc((void**)&d_kx, NX[0] * sizeof(double)), "CUDA malloc: d_kx");
    CCE(cudaMalloc((void**)&d_ky, NX[1] * sizeof(double)), "CUDA malloc: d_ky");
    CCE(cudaMalloc((void**)&d_kz, NX[2] * sizeof(double)), "CUDA malloc: d_kz");

    int dim = 3;
    if( (NX[0]*NX[1] == 0) || (NX[1]*NX[2] == 0) )
    {
        if( NX[0]*NX[1] == NX[1]*NX[2] ){
            dim = 1;
        }else{
            dim = 2;
        }
    }

    switch (dim) {
        case 1:
            if (cufftPlan1d(&planForward, NX[0], CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan Forward creation failed" << std::endl;
            }
            if (cufftPlan1d(&planBackward, NX[0], CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan Backword creation failed" << std::endl;
            }
            break;

        case 2:
            if (cufftPlan2d(&planForward, NX[1], NX[0], CUFFT_Z2Z) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan Forward creation failed" << std::endl;
            }
            if (cufftPlan2d(&planBackward, NX[1], NX[0], CUFFT_Z2Z) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan Backword creation failed" << std::endl;
            }
            break;

        case 3:
            if (cufftPlan3d(&planForward, NX[2], NX[1], NX[0], CUFFT_Z2Z) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan Forward creation failed" << std::endl;
            }
            if (cufftPlan3d(&planBackward, NX[2], NX[1], NX[0], CUFFT_Z2Z) != CUFFT_SUCCESS) {
                std::cerr << "CUFFT error: Plan Backword creation failed" << std::endl;
            }
            break;

        default:
            std::cerr << "Dimension has to be 1, 2 or 3." << std::endl;
            break;
    }

    
}




void Cuda::PrintGPUInfo()
{
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "Error retrieving CUDA device count: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "\nDevice " << device << " - " << deviceProp.name << ":" << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Maximum threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum threads dimension (block): ["
                  << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", "
                  << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Maximum grid size: ["
                  << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", "
                  << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  Multi-Processor Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " 
                  << 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
    }
}


// ------------------------
//
// SINGLE COMPONENT KERNELS
//
// ------------------------
/**
 * @brief Calculates Laplace of the input function
 *
 * @param v1 3D function (type: cufftDoubleComplex*)
 * @param kx $x$ component of the $k$ vector represented by an array of length NX (type: double*)
 * @param ky $y$ component of the $k$ vector represented by an array of length NY (type: double*)
 * @param kz $z$ component of the $k$ vector represented by an array of length NZ (type: double*)
 * @param result 3D Laplace of the input function (type: cufftDoubleComplex*)
 * @param N number of mesh points (NX$\cdot$NY$\cdot$NZ) (type: int)
 * @param NX number of mesh points in $x$ direction (type: int)
 * @param NY number of mesh points in $y$ direction (type: int)
 * @param NZ number of mesh points in $z$ direction (type: int)
 */
__global__ void Laplace( cufftDoubleComplex* v1, double* kx,  double* ky, double* kz, cufftDoubleComplex* result, int N, int NX, int NY, int NZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;
    int iZ = idx / (NX*NY);
    
    if ((idx < N) && (iX < NX) && (iY < NY) && (iZ < NZ)) {
        // Perform element-wise complex multiplication
        result[idx].x = v1[idx].x * (kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);
        result[idx].y = v1[idx].y * (kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]); 
    }
}

__global__ void ScalarMultiply( double* vout, double a, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix] *= a ;
    }
}

__global__ void ScalarMultiply( double* vout, double* vin, double a, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix] = a * vin[ix] ;
    }
}

__global__ void ScalarMultiply( cufftDoubleComplex* vout, double a, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix].x *= a ;
        vout[ix].y *= a ;
    }
}

__global__ void ScalarMultiply( cufftDoubleComplex* vout, cufftDoubleComplex* v1, double a, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix].x = a * v1[ix].x;
        vout[ix].y = a * v1[ix].y;
    }
}

__global__ void ScalarMultiply( cufftDoubleComplex* vout, cufftDoubleComplex* v1, cufftDoubleComplex a, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix].x = (a.x * v1[ix].x - a.y*v1[ix].y);
        vout[ix].y = (a.y * v1[ix].x + a.x*v1[ix].y);
    }
}

__global__ void AppendArray( cufftDoubleComplex* vout, cufftDoubleComplex a, cufftDoubleComplex* v1, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix].x +=  (a.x*v1[ix].x - a.y*v1[ix].y);
        vout[ix].y +=  (a.y*v1[ix].x + a.x*v1[ix].y);
    }
}

__global__ void AppendArray( cufftDoubleComplex* vout, double a, cufftDoubleComplex* v1, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix].x +=  (a*v1[ix].x);
        vout[ix].y +=  (a*v1[ix].y);
    }
}

__global__ void SumArrays( cufftDoubleComplex* vout, cufftDoubleComplex* v2, cufftDoubleComplex a, cufftDoubleComplex* v1, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        vout[ix].x = v2[ix].x + (a.x * v1[ix].x - a.y*v1[ix].y);
        vout[ix].y = v2[ix].y + (a.y * v1[ix].x + a.x*v1[ix].y);
    }
}


__global__ void MultiplyArrays(cufftDoubleComplex* a, cufftDoubleComplex* b, cufftDoubleComplex* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        result[idx].x = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
        result[idx].y = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
    }
}

// !!!!!!!!!!!!!!!!!!!! NOT READY !!!!!!!!!!!!!!!!!!!!!!
__global__ void MultiplyArraysX(cufftDoubleComplex* a, double* b, cufftDoubleComplex* result, int N, int NX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;

    if ( (idx < N) && (iX < NX) ) {
        result[idx].x = a[idx].x * b[iX];
        result[idx].y = a[idx].y * b[iX];
    }
}

__global__ void MultiplyArraysY(cufftDoubleComplex* a, double* b, cufftDoubleComplex* result, int N, int NX, int NY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;

    if ( (idx < N) && (iX < NX) && (iY < NY) ) {
        result[idx].x = a[idx].x * b[iY];
        result[idx].y = a[idx].y * b[iY];
    }
}




__global__ void MultiplyArrays(double* a, double* b, double* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void SquareArray( cufftDoubleComplex *in, cufftDoubleComplex *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx].x = in[idx].x*in[idx].x + in[idx].y*in[idx].y;
        out[idx].y = 0;
    }
}

__global__ void SquareArray( cufftDoubleComplex *in, double *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx].x*in[idx].x + in[idx].y*in[idx].y;
    }
}

__global__ void CalcK( cufftDoubleComplex* k, cufftDoubleComplex* hpsi, double dt, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        k[ix].x =        dt * hpsi[ix].y;
        k[ix].y = -1.0 * dt * hpsi[ix].x;
    }
}


__global__ void UpdateRKStep( cufftDoubleComplex* psi, cufftDoubleComplex* psi_old, cufftDoubleComplex* k, double sc, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        psi[ix].x = psi_old[ix].x + sc * k[ix].x;
        psi[ix].y = psi_old[ix].y + sc * k[ix].y;
    }
}

__global__ void UpdateRKStep( cufftDoubleComplex* psi, cufftDoubleComplex* psi_old, cufftDoubleComplex* hpsi, cufftDoubleComplex nIdt, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        psi[ix].x = psi_old[ix].x + ( nIdt.x*hpsi[ix].x - nIdt.y*hpsi[ix].y );
        psi[ix].y = psi_old[ix].y + ( nIdt.x*hpsi[ix].y + nIdt.y*hpsi[ix].x );
    }
}

/**
 * @brief Sums all RK4 steps to a new wavefunction for a spin mixture
 *
 * @param psi New wavefunction (type: (cufftDoubleComplex*, cufftDoubleComplex*) )
 * @param psi_old Old wavefunction (type: (cufftDoubleComplex*, cufftDoubleComplex*) )
 * @param hpsi Last H|psi> to be added as the fourth step of RK (type: cufftDoubleComplex*)
 * 
 * @note Here h12 is meant to be -\hbar\Omega/2 -- a real number. To introduce other coupling, one needs
 *       to replace the double h12 with a complex arrays (one for h12 and the other for h21)
 * @warning Warnings about misuse, performance issues, or other concerns.
 * @pre Precondition that must hold true before calling the function.
 * @post Postcondition that will be true after calling the function.
 */
/*
__global__ void UpdateRKStep( 
    cufftDoubleComplex* psi1, cufftDoubleComplex* psi1_old, cufftDoubleComplex* psi2, cufftDoubleComplex* psi2_old,
    cufftDoubleComplex* h11psi1, cufftDoubleComplex* h22psi2, double h12, cufftDoubleComplex nIdt, int N)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (ix < N) {
        psi1[ix].x = psi1_old[ix].x +    nIdt.x*(h11psi1[ix].x+h12*psi2_old[ix].x) - nIdt.y*(h11psi1[ix].y+h12*psi2_old[ix].y) ;
        psi1[ix].y = psi1_old[ix].y +    nIdt.x*(h11psi1[ix].y+h12*psi2_old[ix].y) + nIdt.y*(h11psi1[ix].x+h12*psi2_old[ix].x) ;

        psi2[ix].x = psi2_old[ix].x +    nIdt.x*(h22psi2[ix].x+h12*psi1_old[ix].x) - nIdt.y*(h22psi2[ix].y+h12*psi1_old[ix].y) ;
        psi2[ix].y = psi2_old[ix].y +    nIdt.x*(h22psi2[ix].y+h12*psi1_old[ix].y) + nIdt.y*(h22psi2[ix].x+h12*psi1_old[ix].x) ;
    }
}
*/


/**
 * @brief Sums all RK4 steps to a new wavefunction
 *
 * @param psi New wavefunction (type: cufftDoubleComplex*)
 * @param psi_old Old wavefunction (type: cufftDoubleComplex*)
 * @param _k Sum of previous RK steps (1-3) yet to be multiplied by I (type: cufftDoubleComplex*)
 * @param hpsi Last H|psi> to be added as the fourth step of RK (type: cufftDoubleComplex*)
 * 
 * @note Additional notes or caveats about the function.
 * @warning Warnings about misuse, performance issues, or other concerns.
 * @pre Precondition that must hold true before calling the function.
 * @post Postcondition that will be true after calling the function.
 */
__global__ void FinalRKStep( cufftDoubleComplex* psi, cufftDoubleComplex* psi_old, cufftDoubleComplex* _k, cufftDoubleComplex* hpsi, cufftDoubleComplex nIdt, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    double aX = 0.0;
    double aY = 0.0;
    if (ix < N) {
        aX = _k[ix].x + hpsi[ix].x;
        aY = _k[ix].y + hpsi[ix].y;
        psi[ix].x = psi_old[ix].x + ( nIdt.x*aX - nIdt.y*aY )/6.0;
        psi[ix].y = psi_old[ix].y + ( nIdt.x*aY + nIdt.y*aX )/6.0;
    }
}


__global__ void SumRK4( cufftDoubleComplex* psi_out, cufftDoubleComplex* psi_in, cufftDoubleComplex* k1, cufftDoubleComplex* k2, cufftDoubleComplex* k3, cufftDoubleComplex* k4, int N){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        psi_out[ix].x = psi_in[ix].x + (k1[ix].x + 2*k2[ix].x + 2*k3[ix].x + k4[ix].x)/6.0;
        psi_out[ix].y = psi_in[ix].y + (k1[ix].y + 2*k2[ix].y + 2*k3[ix].y + k4[ix].y)/6.0;
    }
}


__global__ void BECVPx( cufftDoubleComplex* v1, double* kx, cufftDoubleComplex* result, int N, int NX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    
    if ( (idx < N) && (iX < NX) ) {
        // Perform element-wise complex multiplication
        result[idx].x = v1[idx].x * kx[iX];
        result[idx].y = v1[idx].y * kx[iX]; 
    }
}

__global__ void BECOmegaLz( cufftDoubleComplex* v1, double* kx, double* ky, cufftDoubleComplex* result, int N, int NX, int NY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;
    
    if ( (idx < N) && (iX < NX) && (iY < NY)) {
        // kx \rho - ky \rho
        result[idx].x = kx[iX] * v1[idx].x - ky[iY] * v1[idx].x;
        result[idx].y = kx[iX] * v1[idx].y - ky[iY] * v1[idx].y; 
    }
}


__global__ void BECUpdatePsi( cufftDoubleComplex* psi, cufftDoubleComplex* newpsi,  cufftDoubleComplex* oldpsi, cufftDoubleComplex* hpsi, double mu, double dt, double beta, int N ){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < N) {
        newpsi[ix].x = ( 1.0 + mu * dt ) * psi[ix].x - dt * hpsi[ix].x + beta * ( psi[ix].x - oldpsi[ix].x );
        newpsi[ix].y = ( 1.0 + mu * dt ) * psi[ix].y - dt * hpsi[ix].y + beta * ( psi[ix].y - oldpsi[ix].y );

    }
}


__global__ void DipoleDipoleInteraction( double* kx,  double* ky, double* kz, cufftDoubleComplex* vtilde, double a_dd, double d_x, double d_y, double d_z, int N, int NX, int NY, int NZ) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;
    int iZ = idx / (NX*NY);

    double num=0.0;
    double den=0.0;

    if ((idx < N) && (iX < NX) && (iY < NY) && (iZ < NZ)) {

        
        num = kx[iX]*d_x + ky[iY]*d_y + kz[iZ]*d_z;
        den = sqrt(kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);
        if( den < 1E-6)
        {
            vtilde[idx].x = 0;
            vtilde[idx].y = 0;
        }
        else
        {
            vtilde[idx].x = 12.0*pi*a_dd*( ((num/den)*(num/den)) - 1.0/3.0);
            vtilde[idx].y = 0;
        }   
    }
}


__global__ void SoftCoreInteraction_1D( double* kx,  double* ky, double* kz, cufftDoubleComplex* vtilde, int N, int NX, int NY, int NZ, double* _d_h_params) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;
    int iZ = idx / (NX*NY);

    double k=0.0;
    double a = _d_h_params[0];
    double U = _d_h_params[1];

    if ((idx < N) && (iX < NX) && (iY < NY) && (iZ < NZ)) {

        k = 2*pi*sqrt(kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);
        if( k < 1E-6)
        {
            // repulsive soft-core only: a>0
            vtilde[idx].x = U*2*a*(1 - (a*k)*(a*k)/6.0);
            vtilde[idx].y = 0;
        }
        else
        {
            vtilde[idx].x = U * 2* sin(a*k)/(k);
            vtilde[idx].y = 0;
        }   
    }
}

__global__ void SoftCoreInteraction_2D( double* kx,  double* ky, double* kz, cufftDoubleComplex* vtilde, int N, int NX, int NY, int NZ, double* _d_h_params) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;
    int iZ = idx / (NX*NY);

    double k=0.0;
    double a = _d_h_params[0];
    double U = _d_h_params[1];

    if ((idx < N) && (iX < NX) && (iY < NY) && (iZ < NZ)) {

        k = 2*pi*sqrt(kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);
        if( k < 1E-6)
        {
            // repulsive soft-core only: a>0
            vtilde[idx].x = U*pi*a*a;
            vtilde[idx].y = 0;
        }
        else
        {
            vtilde[idx].x = 2*pi*a*U * j1(k*a) / k;
            vtilde[idx].y = 0;
        }   
    }
}

__global__ void SoftCoreInteraction_3D( double* kx,  double* ky, double* kz, cufftDoubleComplex* vtilde, int N, int NX, int NY, int NZ, double* _d_h_params) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;
    int iZ = idx / (NX*NY);

    double k=0.0;
    double a = _d_h_params[0];
    double U = _d_h_params[1];

    if ((idx < N) && (iX < NX) && (iY < NY) && (iZ < NZ)) {

        k = 2*pi*sqrt(kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);

        if( k < 1E-6)
        {
            vtilde[idx].x = U* (4./3.)*pi*std::pow(a, 3);
            vtilde[idx].y = 0;
        }
        else
        {
            vtilde[idx].x = 4*pi*U * std::pow(a,3) * ( sin(k*a) - k*a*cos(k*a) )/( std::pow(k*a, 3) );;
            vtilde[idx].y = 0;
        }   
    }
}

cufftDoubleComplex complexSqrt(cufftDoubleComplex z) {
    double a = cuCreal(z);  // Real part
    double b = cuCimag(z);  // Imaginary part

    double magnitude = sqrt(a * a + b * b); // |z|
    double sqrtReal = sqrt((magnitude + a) / 2.0);
    double sqrtImag = (b >= 0 ? 1 : -1) * sqrt((magnitude - a) / 2.0);

    cufftDoubleComplex result = make_cuDoubleComplex(sqrtReal, sqrtImag);
    return result;
}




__global__ void NormalizePsi(cufftDoubleComplex *input, double *output, int N) {
    // Allocate shared memory
    __shared__ double sharedData[NTHREADS];

    // Thread index in the block
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    if (globalIdx < N) {
        sharedData[tid] = input[globalIdx].x*input[globalIdx].x + input[globalIdx].y*input[globalIdx].y;
    } else {
        sharedData[tid] = 0.0;  // Padding for out-of-bounds threads
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

/*
 *  this function is user-specific
 *  think of a way to implement it from BEC_oneComponent level
 * 
 */
__global__ void CalcAverageCos(cufftDoubleComplex *psi, double* x, double *output, double L, int NX, int N) 
{
    __shared__ double sharedData[NTHREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int iX = idx % NX;

    /* CALCULATE THE <COS> AND STORE IN sharedData */
    if (idx < N) {
        sharedData[tid] = cos( 2*pi*x[iX]/L )*(psi[idx].x*psi[idx].x + psi[idx].y*psi[idx].y);
    } else {
        sharedData[tid] = 0.0;  // Padding for out-of-bounds threads
    }
    __syncthreads();

    /* BLOCK REDUCTION - DO NOT MODIFY */
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}


__global__ void CalculateRelativePhase(cufftDoubleComplex *psi1, cufftDoubleComplex *psi2, double *output, int N) {
    __shared__ double sharedData[NTHREADS];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;
    
    // Load elements into shared memory
    if (globalIdx < N) {
        sharedData[tid] = atan2(psi1[globalIdx].y, psi1[globalIdx].x) - atan2(psi2[globalIdx].y, psi2[globalIdx].x);
    } else {
        sharedData[tid] = 0.0;  

    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

__global__ void CalculateDszDt(cufftDoubleComplex *psi1, cufftDoubleComplex *psi2, double *output, int N) {
    __shared__ double sharedData[NTHREADS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    double phir=0.0;
    double n1=0.0;
    double n2=0.0;

    // Load elements into shared memory
    if (gid < N) {
        phir = atan2(psi1[gid].y, psi1[gid].x) - atan2(psi2[gid].y, psi2[gid].x);
        n1   = psi1[gid].x*psi1[gid].x + psi1[gid].y*psi1[gid].y;
        n2   = psi2[gid].x*psi2[gid].x + psi2[gid].y*psi2[gid].y;
        sharedData[tid] = sqrt(n1*n2)*sin(phir);
    } else {
        sharedData[tid] = 0.0;  

    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

__global__ void CalculateObservable(cufftDoubleComplex *psi, cufftDoubleComplex *Opsi, double *output, int N) {
    __shared__ double sharedData[NTHREADS];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    if (globalIdx < N) {
        sharedData[tid] = psi[globalIdx].x*Opsi[globalIdx].x + psi[globalIdx].y*Opsi[globalIdx].y;
    } else {
        sharedData[tid] = 0.0;  

    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

__global__ void Calculate2Observables(cufftDoubleComplex *psi, cufftDoubleComplex *O1psi, double *output1, cufftDoubleComplex *O2psi, double *output2, int N){
    __shared__ cufftDoubleComplex sharedData[NTHREADS];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    if (globalIdx < N) {
        // store one observable under x and the other under y
        sharedData[tid].x = psi[globalIdx].x*O1psi[globalIdx].x + psi[globalIdx].y*O1psi[globalIdx].y;
        sharedData[tid].y = psi[globalIdx].x*O2psi[globalIdx].x + psi[globalIdx].y*O2psi[globalIdx].y;
    } else {
        sharedData[tid].x = 0.0;  // Padding for out-of-bounds threads
        sharedData[tid].y = 0.0;  // Padding for out-of-bounds threads

    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid].x += sharedData[tid + stride].x;
            sharedData[tid].y += sharedData[tid + stride].y;
        }
        __syncthreads();
    }

    // Write the result of this block to the output array
    if (tid == 0) {
        output1[blockIdx.x] = sharedData[0].x;
        output2[blockIdx.x] = sharedData[0].y;
    }
}




__global__ void sumReductionKernel(double *input, double *output, int n) {
// Shared memory to store partial sums
    extern __shared__ double sharedData[];

    // Calculate the global thread ID
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory with input values
    if (globalIdx < n) {
        sharedData[tid] = input[globalIdx];
    } else {
        sharedData[tid] = 0.0; // If out of bounds, set to 0
    }

    // Synchronize threads to ensure all data is loaded into shared memory
    __syncthreads();

    // Perform reduction in shared memory (iterative halving)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads(); // Ensure all threads complete the current stride
    }

    // Write the result of this block's sum to the output array
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}



// ---------------------
//
// TWO COMPONENT KERNELS
//
// ---------------------

__global__ void Laplace_2( cufftDoubleComplex* v1, cufftDoubleComplex* v2, double* kx,  double* ky, double* kz, cufftDoubleComplex* result1, cufftDoubleComplex* result2, int N, int NX, int NY, int NZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iX = idx % NX;
    int iY = (idx / NX) % NY;
    int iZ = idx / (NX*NY);
    

    if ((idx < N) && (iX < NX) && (iY < NY) && (iZ < NZ)) {

        result1[idx].x = v1[idx].x * (kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);
        result1[idx].y = v1[idx].y * (kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);

        result2[idx].x = v2[idx].x * (kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]);
        result2[idx].y = v2[idx].y * (kx[iX]*kx[iX] + ky[iY]*ky[iY] + kz[iZ]*kz[iZ]); 
    }
}




__global__ void CalcAverageCos( cufftDoubleComplex *psi1, cufftDoubleComplex *psi2, double* x, double *output, double L, int NX, int N, int sign) 
{
    extern __shared__ double sharedData[NTHREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int iX = idx % NX;

    double psix=0.0;
    double psiy=0.0;

    if (idx < N) {
        psix = psi1[idx].x + sign*psi2[idx].x;
        psiy = psi1[idx].y + sign*psi2[idx].y;

        sharedData[tid] = cos( 2*pi*x[iX]/L )*(psix*psix + psiy*psiy);
    } else {
        sharedData[tid] = 0.0;  // Padding for out-of-bounds threads
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}





