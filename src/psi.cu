#include "psi.hpp"




// 1 COMPONENT
Psi::Psi(const Params &p, const std::string &type) : generator( p.getInt("rndseed")), distribution(-1.0,1.0){
    
    this->psi = (complex*)malloc( p.Npoints * sizeof(complex) );

    // decide what type of initial wavefunction to construct    
    if ( p.getBool("read_psi") ) 
    {
        std::cout << "# Reading initial wavefunction from: " << p.inpath << std::endl;
        readPsi(p.inpath, p.Npoints);
    }else{
        // check if there should be periodic boundary conditions somewhere:
        bool result = std::any_of(p.omega, p.omega + 3, [](double x) { return std::abs(x) < 1E-6; });

        if ( result && abs(p.omega[0]) < 1E-6 )
        {
            std::cout << "# Initializing tubular psi_init..." << std::endl;
            initTube(p, 0);
        }else{
            std::cout << "# Initializing harmonic psi_init..." << std::endl;
            initHarmonic(p, 0);
        }
        normalize(p);
    }

    
}

Psi& Psi::operator=(const Psi &other){
    if (this == &other) {  // Self-assignment check
        return *this;
    }

    return *this;
}

Psi::Psi(const Psi& obj) {
    this->psi = obj.getPsi();
    this->generator = obj.getGenerator();
    this->distribution = obj.getDistribution();
}


// 2 COMPONENT
Psi::Psi(const ParamsBase &p, const int &type) : generator( p.getInt("rndseed")+type), distribution(-1.0,1.0){
    
    this->psi = (complex*)malloc( p.Npoints * sizeof(complex) );

    // decide what type of initial wavefunction to construct
    if ( p.getBool("read_psi") ) 
    {
        std::cout << "# Reading initial wavefunction from: " << p.inpath << std::endl;
        readPsi( p.inpath+std::to_string(type), p.Npoints);

    }else{
        const double *omega = (type ==1 ) ? p._omega(1) :  p._omega(2);

        bool result = std::any_of(omega, omega + 3, [](double x) { return std::abs(x) < 1E-6; });

        if ( result && abs(omega[0]) < 1E-6 )
        {
            std::cout << "# Initializing tubular psi_init..." << std::endl;
            initTube(p, type);
        }else{
            std::cout << "# Initializing harmonic psi_init..." << std::endl;
            initHarmonic(p, type);
        }
        normalize(p, type);
    }

    
}

// initializes a tube init wavefunction along the x-axis
void Psi::initTube(const ParamsBase &p, const int &type){

    double rrnd = p.getDouble("rand_pct");
    double crnd = p.getDouble("crand_pct"); 

    const double *ax = (type ==1 ) ? p._ax(1) : p._ax(2);


    // no need to parallelize
    for (int i = 0; i < p.Npoints; ++i)
    {
        double y = getY(p, i)/ ax[1];
        double z = getZ(p, i)/ ax[2];
        complex noise = complex(rrnd*distribution(generator), crnd*distribution(generator));

        psi[i] = (complex(1,0) + noise) * complex( exp( -0.5 * 0.1 * ((y*y)+(z*z))), 0) ;
    }
}

// initializes a 3D Gaussian
void Psi::initHarmonic(const ParamsBase &p, const int &type){

    double rrnd = p.getDouble("rand_pct");
    double crnd = p.getDouble("crand_pct"); 

    const double *ax = (type ==1 ) ? p._ax(1) : p._ax(2);

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i)/ ax[0];
        double y = getY(p, i)/ ax[1];
        double z = getZ(p, i)/ ax[2];
        complex noise = complex(rrnd*distribution(generator), crnd*distribution(generator));

        psi[i] = (complex(1,0) + noise) * complex( exp( -0.5 * 0.1 * ((y*y)+(z*z)+(x*x))), 0) ;
    }
}

// empty init
void Psi::initVoid(const ParamsBase &p, const int &type){

    double rrnd = p.getDouble("rand_pct");
    double crnd = p.getDouble("crand_pct"); 

    for (int i = 0; i < p.Npoints; ++i)
    {
        complex noise = complex(rrnd*distribution(generator), crnd*distribution(generator));

        psi[i] = (complex(1,0) + noise)  ;
    }
}



double Psi::getX(const ParamsBase &p, int i)
{
    int ix = iX( i, p.NX[0], p.NX[1]);
    return -p.XMAX[0] + (ix)*p.DX[0] + 0.5*p.DX[0];
}

double Psi::getY(const ParamsBase &p, int i)
{
    int iy = iY( i, p.NX[0], p.NX[1]);
    return -p.XMAX[1] + (iy)*p.DX[1] + 0.5*p.DX[1];
}

double Psi::getZ(const ParamsBase &p, int i)
{
    int iz = iZ( i, p.NX[0], p.NX[1]);
    return -p.XMAX[2] + (iz)*p.DX[2] + 0.5*p.DX[2];
}



// imprint Ncirc quanta of circulation in the phase pattern
void Psi::imprintVortexTube(const ParamsBase &p, const double &Ncirc){

    double xLen = 2*p.XMAX[0];

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i);
        psi[i] = complex(std::cos(Ncirc *2*pi/xLen *x), std::sin(Ncirc * 2*pi/xLen *x )) * psi[i];
    }
}

// add a random noise to the wavefunction
void Psi::addNoise(complex &noise_amp, const int &length){

    for (int i = 0; i < length; ++i)
    {
        complex noise = complex(std::real(noise_amp)*distribution(generator), std::imag(noise_amp)*distribution(generator));

        psi[i] = (complex(1,0) + noise) * psi[i];
    }
}



void Psi::force1DLattice( const ParamsBase &p, const double& amp, const double& k, const double& off, const int &type)
{
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    double avg = std::pow( npart/ (p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i) - off;

        double force = amp * ( 1- cos(k*x) )/2.; // from 0 to amp
        psi[i] = force*avg + psi[i];
    }
    
    normalize(p, 1);
}


void Psi::force2DTriangularLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const int &type)
{
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    double avg = std::pow( npart/ (p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i) - offx;
        double y = getY(p, i) - offy;
        double force = amp * ( cos(k * x) + 
                               cos(k * (-0.5 * x + ( sqrt(3)/2) * y)) + 
                               cos(k * (-0.5 * x - ( sqrt(3)/2) * y)))/3;
        psi[i] = force*avg + psi[i];
    }
    
    normalize(p, 1);
} 


 

void Psi::force2DSquareLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const int &type)
{
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    double avg = std::pow( npart/ (p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i) - offx;
        double y = getY(p, i) - offy;
        double force = amp * (cos(k*y) + cos(k*x))/2;
        psi[i] = force*avg + psi[i];
    }
    
    normalize(p, 1);
} 



// HEX
void Psi::force3DHEXLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type)
{
    double kk = k;
    
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    double avg = std::pow( npart/ (p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i) - offx;
        double y = getY(p, i) - offy;
        double z = getZ(p, i) - offz;

        double x1 = (-x + sqrt(3.)*y)/2.;
        double x4 = (-x - sqrt(3.)*y)/2.;
        double x2 = x ;
        double x3 = z ;


        double force = amp * (cos(kk*x1) + cos(kk*x2) + cos(kk*x3) + cos(kk*x4))/4.0;
                      
        psi[i] = force*avg + psi[i];
    }
    
    normalize(p, 1);
}


// FCC
void Psi::force3DFCCLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type)
{
    double kk =  k ;
    
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    double avg = std::pow( npart/ (p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i) - offx;
        double y = getY(p, i) - offy;
        double z = getZ(p, i) - offz;
     
        double x1 = (x+y-z)/sqrt(3.);
        double x2 = (y+z-x)/sqrt(3.);
        double x3 = (z+x-y)/sqrt(3.);
        double x4 = (x-y-z)/sqrt(3.);
        double x5 = (y-z-x)/sqrt(3.);
        double x6 = (z-x-y)/sqrt(3.);
        double x7 = (x+y+z)/sqrt(3.);
        double x8 = (y+z+x)/sqrt(3.);
        double x9 = (z+x+y)/sqrt(3.);

        double force = amp * (cos(kk*x1) + cos(kk*x2) + cos(kk*x3) +
                              cos(kk*x4) + cos(kk*x5) + cos(kk*x6) +
                              cos(kk*x7) + cos(kk*x8) + cos(kk*x9) )/9.0;
                      
        psi[i] = force*avg + psi[i];
    }
    
    normalize(p, 1);
}



// BCC
void Psi::force3DBCCLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type)
{
    double kk = k;
    
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    double avg = std::pow( npart/ (p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i) - offx;
        double y = getY(p, i) - offy;
        double z = getZ(p, i) - offz;

        double x1 = (x+y)/sqrt(2);
        double x2 = (z+y)/sqrt(2);
        double x3 = (z+x)/sqrt(2);
        double x4 = (x-y)/sqrt(2);
        double x5 = (z-y)/sqrt(2);
        double x6 = (z-x)/sqrt(2);

        double force = amp * (cos(kk*x1) + cos(kk*x2) + cos(kk*x3) + cos(kk*x4) + cos(kk*x5) + cos(kk*x6))/6;
                      
        psi[i] = force*avg + psi[i];
    }
    
    normalize(p, 1);
 }
 

 // simple cubic
 void Psi::force3DSCLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type)
 {
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    double avg = std::pow( npart/ (p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;

    for (int i = 0; i < p.Npoints; ++i)
    {
        double x = getX(p, i) - offx;
        double y = getY(p, i) - offy;
        double z = getZ(p, i) - offz;
        double force = amp * (cos(k*z) + cos(k*y) + cos(k*x))/3;
        psi[i] = force*avg + psi[i];
    }
    
    normalize(p, 1);
 }


// normalize the wavefunction

void Psi::normalize(const ParamsBase &p, const int &type){
    double norm=0;
    double npart = (type ==1 ) ? p._npart(1) : p._npart(2);
    

    for (int i = 0; i < p.Npoints; ++i)
    {
        norm += std::pow(std::abs(psi[i]), 2) ;
    }
    
    norm = std::pow( npart/ (norm*p.DX[0]*p.DX[1]*p.DX[2]), 0.5) ;
    
    for (int i = 0; i < p.Npoints; ++i) psi[i] = psi[i] * norm;


    
    norm=0;
    for (int i = 0; i < p.Npoints; ++i){
        norm += std::pow(std::abs(psi[i]), 2) * p.DX[0] * p.DX[1] * p.DX[2];
    }
    std::cout << "# Init Psi norm: " << norm  << std::endl;
    

}


// reads the initial wavefunction from the prefix file
void Psi::readPsi(const std::string& prefix, const int& length){

    std::string end = "psi_final.wdat";
   
    // Open the binary file in read mode
    FILE *file = fopen( (prefix+end).c_str(), "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file!\n");
    }

    size_t read_elements = fread( psi, sizeof(complex), length, file);
    if (read_elements != length) {
        fprintf(stderr, "Error: Only %zu elements were read instead of %d!\n", read_elements, length);
        fclose(file);
    }

    fclose(file);
}
