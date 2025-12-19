#ifndef __psi__
#define __psi__

#include "params.hpp"
#include "parallel.hpp"
#include "header.hpp"
#include <random>

/**
* @brief The Gross-Pitaevskii wavefunction (the order parameter)
* 
* @param p parsed input file class object
*/
class Psi {
    complex *psi;

private:
    std::mt19937 generator;  // Use Mersenne Twister as the random number generator
    std::uniform_real_distribution<double> distribution; // Uniform distribution for doubles
    
public:
    //Psi(const std::string *filename, const int &length);
    Psi(const Params &p, const std::string &type);
    Psi(const ParamsBase &p, const int &type);
    Psi(const Psi& obj);
    Psi& operator=(const Psi &other);
    
    // initial state preparators
    void initTube(const ParamsBase &p, const int &type);
    void initHarmonic(const ParamsBase &p, const int &type);
    void initVoid(const ParamsBase &p, const int &type);

    void normalize(const ParamsBase &p, const int &type=-1);

    void addNoise(complex &noise_amp, const int &length);
    void imprintVortexTube(const ParamsBase &p, const double &Ncirc);
    void readPsi(const std::string& prefix, const int& length);

    // soft-core structures: force the initial shape to speed the convergence up
    void force3DHEXLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type=1);
    void force3DFCCLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type=1);
    void force3DBCCLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type=1);
    void force3DSCLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const double& offz, const int &type=1);
    void force2DSquareLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const int &type=1);
    void force2DTriangularLattice( const ParamsBase &p, const double& amp, const double& k, const double& offx, const double& offy, const int &type=1);
    void force1DLattice( const ParamsBase &p, const double& amp, const double& k, const double& off, const int &type=1);

    // index functions
    inline int iX(int index, int NX, int NY) { return index % NX; };
    inline int iY(int index, int NX, int NY) { return (index / NX) % NY; };
    inline int iZ(int index, int NX, int NY) { return index / (NX * NY); };
    
    // get real position
    double getX(const ParamsBase &p, int i);
    double getY(const ParamsBase &p, int i);
    double getZ(const ParamsBase &p, int i);


    complex* getPsi() const { return psi; };
    std::mt19937 getGenerator() const { return generator; }
    std::uniform_real_distribution<double> getDistribution() const { return distribution; }
    ~Psi(){ free(psi); }
};



#endif