#ifndef __params__
#define __params__

#include "header.hpp"
#include "input_variables.hpp"


class ParamsBase{
protected:

    InputVariables iv;

public:

    std::string outpath="", inpath="", vpath="";
    bool isTwoComponent=false;
    // Those values are stored in natural units
    int NX[3] = {0,0,0};
    int DIM = 3;
    int Npoints = 0;
    double V = 0;
    double XMAX[3] = {0,0,0};
    double DX[3] = {0,0,0};
    double DV=0;
    double KXMAX[3] = {0,0,0};
    double DKX[3] = {0,0,0};
    double hamiltonianParams[8] = {0,0,0,0,0,0,0,0};

    virtual const double* _omega(int type) const = 0;  // Pure virtual method to be overridden
    virtual const double* _ax   (int type) const = 0;  // Pure virtual method to be overridden
    virtual double _aho  (int type) const = 0;  
    virtual double _omho  (int type) const = 0;  
    virtual double _mass  (int type) const = 0;  
    virtual double _npart  (int type) const = 0;  
    virtual double _a  (int type) const = 0;  
    virtual double _add  (int type) const = 0;  
    virtual double _edd  (int type) const = 0;  

    double dt, epsilon_e, time0;   
    int niter, itmod, iter0;

    int getInt(const std::string &varName) const;
    double getDouble(const std::string &varName) const;
    std::string getString(const std::string &varName) const;
    bool getBool(const std::string &varName) const; 
    InputVariables getIV() const { return iv; }
    virtual ~ParamsBase() = default;  // Virtual destructor
};





class Params : public ParamsBase {
private:
    
    
    void calculateLengthScales();
    void InitializeParams(const Params *par);
    void RedefineLatticeSoft(const Params *par);

public:
    // Paths for I/O operations
    
    double omega[3] = {0,0,0};
    double ax[3] = {0,0,0};


    double aho, omho;

    double mass;
    double npart;
    double a, add, edd;


    const double* _omega(int type) const  override { return omega; }
    const double* _ax   (int type) const  override { return ax; }
    double _aho  (int type) const  override { return aho; }
    double _omho  (int type) const  override { return omho; }
    double _mass  (int type) const  override { return mass; }
    double _npart  (int type) const  override { return npart; }
    double _a  (int type) const  override { return a; }
    double _add  (int type) const  override { return add; }
    double _edd  (int type) const  override { return edd; }



    Params( ) {};
    Params(const Params& other);
    Params& operator=(const Params& other);
    Params( const InputVariables &iv );
   


};



#endif
