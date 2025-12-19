#ifndef __params2__
#define __params2__

#include "header.hpp"
#include "params.hpp"
#include "input_variables.hpp"


/**
* @brief A class for storing parsed input parameters for two-component Hamiltonian simulations.
* 
* Access:
* get an integer from the input file: Params::getInt(std::string &name)
* get a  double  from the input file: Params::getDouble(std::string &name)
* get a bool     from the input file: Params::getBool(std::string &name)
* get a string   from the input file: Params::getString(std::string &name)
* 
* @param iv InputVariables reference with parsed input parameters
*/
class Params2 : public ParamsBase {
private:
    
    
    void calculateLengthScales();
    void InitializeParams(const Params2 *par);


public:
    double omega1[3] = {0,0,0};
    double omega2[3] = {0,0,0};
    double ax1[3] = {0,0,0};
    double ax2[3] = {0,0,0};


    double aho1, omho1;
    double aho2, omho2;

    double mass1, mass2;
    double npart1, npart2;
    double a11, a22, a12;

    const double* _omega(int type) const  override { return (type==1) ? omega1 : omega2 ; }
    const double* _ax   (int type) const  override { return (type==1) ? ax1 : ax2; }
    double _aho  (int type) const  override { return (type==1) ? aho1 : aho2; }
    double _omho  (int type) const  override { return (type==1) ? omho1 : omho2; }
    double _mass  (int type) const  override { return (type==1) ? mass1 : mass2; }
    double _npart  (int type) const  override { return (type==1) ? npart1 : npart2; }
    double _a  (int type) const  override { return (type==1) ? a11 : a22; }
    double _add  (int type) const  override { return 0; }
    double _edd  (int type) const  override { return 0; }
    double _a12  (int type) const { return a12; }


    Params2( ) {};
    Params2(const Params2& other);
    Params2& operator=(const Params2& other);
    Params2( const InputVariables &iv );
   

};



#endif
