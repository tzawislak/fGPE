#ifndef __vext__
#define __vext__

#include "params.hpp"
#include "params2.hpp"
#include "parallel.hpp"
#include "header.hpp"
#include <cmath>  // For pow

class Vext {

    complex *vext;
public:
    void initTube(const ParamsBase &p, const int &type);
    void initHarmonic(const ParamsBase &p, const int &type);
    void initBox(const ParamsBase &p, const int &type);
    void initVoid(const ParamsBase &p, const int &type);


    Vext(const Params &p, const std::string &type);
    Vext(const Params2 &p, const int &type);

    ~Vext() { free(vext); };
    complex *getVext() { return vext; };
    void readVext(const std::string& prefix, const int& length);
    void addProtocolPotential(const ParamsBase &p, const int &type, const int=1);
    void addWeightedProtocolPotential(const Params2 &p, const int &type);

    void addOpticalLattice(const ParamsBase &p, const int &type);


    inline int iX(int index, int NX, int NY) { return index % NX; };
    inline int iY(int index, int NX, int NY) { return (index / NX) % NY; };
    inline int iZ(int index, int NX, int NY) { return index / (NX * NY); };

    double getX(const ParamsBase &p, int i);
    double getY(const ParamsBase &p, int i);
    double getZ(const ParamsBase &p, int i);
};


#endif