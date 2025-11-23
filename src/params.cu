#include "params.hpp"

Params::Params( const InputVariables &iv ) {
    this->iv = iv;

    InitializeParams(this);

}



Params::Params(const Params& other){
    this->iv = other.getIV();
    InitializeParams(&other);
}


Params& Params::operator=(const Params& other){
    if (this == &other) {  // Self-assignment check
        return *this;
    }
    this->iv = other.getIV();
    InitializeParams(&other);
    return *this;
}

void Params::RedefineLatticeOL(const Params *par)
{
    std::cout << "Old numerical domain lengths: \nLX: " 
              << 2*this->XMAX[0] << "\nLY: " 
              << 2*this->XMAX[1] << "\nLZ: "
              << 2*this->XMAX[2] << std::endl;
    

}

void Params::RedefineLatticeSoft(const Params *par)
{
    std::cout << "Old numerical domain lengths: \nLX: " 
              << 2*this->XMAX[0] << "\nLY: " 
              << 2*this->XMAX[1] << "\nLZ: "
              << 2*this->XMAX[2] << std::endl;
    try
    {
        double akrot = par->getIV().getDouble("akrot") / (2*pi);
        double a     = par->getIV().getDouble("a_soft");
        double idd   = a/akrot; // inter-droplet distance
        double triangular  = 2* std::sqrt(3.)/2.;
        double tetrahedral = std::sqrt(2./3.);

        double iddx = idd;
        double iddy = idd;
        double iddz = idd;

        std::string latticetype = par->getIV().getString("Latticetype");
        if( latticetype == "triangular")
        {
            iddy = idd * triangular;
        }

        if( latticetype == "HEX")
        {
            //iddx = idd * 4/sqrt(3);
            iddy = idd * triangular;
        }

        // this is required, because at the moment the structure is not rotated so that all atoms are along principal axes x,y,z - TODO?
        if( latticetype == "BCC" ){
            iddx = idd * sqrt(2);
            iddy = idd * sqrt(2);
            iddz = idd * sqrt(2);
        }

        if( latticetype == "SC" ){
            // I think this is correct somehow, but who cares, SC is definitely too excited to be metastable
            iddx = idd / sqrt(2);
            iddy = idd / sqrt(2);
            iddz = idd / sqrt(2);
        }

        if( latticetype == "FCC" ){
            iddx = idd * sqrt(3);
            iddy = idd * sqrt(3);
            iddz = idd * sqrt(3);
        }

        switch( this->DIM )
        {
            case 1:{
                double _LX = 2* this->XMAX[0];
                int _NX = std::round( _LX/iddx );
                this->XMAX[0] = 0.5 * _NX * iddx;
                break;
            }
            case 2:{
                double _LX = 2* this->XMAX[0];
                double _LY = 2* this->XMAX[1];
                int _NX = std::round( _LX/iddx );
                int _NY = std::round( _LY/iddy ); // check what kind of lattice you expect to get - if triangular, this needs to be adjusted
                this->XMAX[0] = 0.5 * _NX * iddx;
                this->XMAX[1] = 0.5 * _NY * iddy;
                break;
            }
            case 3:{
                double _LX = 2* this->XMAX[0];
                double _LY = 2* this->XMAX[1];
                double _LZ = 2* this->XMAX[2];
                int _NX = std::round( _LX/iddx );
                int _NY = std::round( _LY/iddy ); 
                int _NZ = std::round( _LZ/iddz ); 

                if( latticetype == "BCC" ){
                    this->XMAX[0] = 0.5 * (_NX+0.5) * iddx;
                    this->XMAX[1] = 0.5 * (_NY+0.5) * iddy;
                    this->XMAX[2] = 0.5 * (_NZ+0.5) * iddz;
                }
                
                this->XMAX[0] = 0.5 * _NX * iddx;
                this->XMAX[1] = 0.5 * _NY * iddy;
                this->XMAX[2] = 0.5 * _NZ * iddz;
              
            }

        }
        
    }
    catch( const std::exception &e)
    {
        std::cout << "Error at redefnintion of the numerical domain: " << e.what() << std::endl;
    }

    std::cout << "New numerical domain lengths: \nLX: " << 2* this->XMAX[0] << "\nLY: " << 2* this->XMAX[1] << "\nLZ: " << 2*this->XMAX[2] << std::endl;
}


void Params::InitializeParams(const Params *par){
    try {
        this->outpath = par->getIV().getString("outprefix");
        this->inpath = par->getIV().getString("inprefix");
        this->vpath = par->getIV().getString("vprefix");

        this->NX[0] = par->getIV().getInt("nx");
        this->NX[1] = par->getIV().getInt("ny");
        this->NX[2] = par->getIV().getInt("nz");

        // if not 3D
        this->DIM = 3;
        if( ((this->NX[0]-1)*(this->NX[1]-1) == 1) ||
            ((this->NX[1]-1)*(this->NX[2]-1) == 0) )
        {
            if( (this->NX[0]-1)*(this->NX[1]-1) == (this->NX[1]-1)*(this->NX[2]-1)  ){
                this->DIM = 1;
            }else{
                this->DIM = 2;
            }
        }

        this->Npoints = NX[0]*NX[1]*NX[2];

        this->XMAX[0] = par->getIV().getDouble("xmax");
        this->XMAX[1] = par->getIV().getDouble("ymax");
        this->XMAX[2] = par->getIV().getDouble("zmax");

        if( par->getIV().getString("hamiltonian") == "soft"){
            RedefineLatticeSoft(par);
        }

        this->DX[0] = 2*this->XMAX[0] / NX[0];
        this->DX[1] = 2*this->XMAX[1] / NX[1];
        this->DX[2] = 2*this->XMAX[2] / NX[2];

        


        this->omega[0] = 2*pi*par->getIV().getDouble("omx");
        this->omega[1] = 2*pi*par->getIV().getDouble("omy");
        this->omega[2] = 2*pi*par->getIV().getDouble("omz");

        this->mass = par->getIV().getDouble("mass");

        if (par->getIV().getVariable("n1D") != "None")
        {
            this->npart = par->getIV().getDouble("n1D") * (2*this->XMAX[0]);
        }else{
            this->npart = par->getIV().getDouble("npart");
        }
        this->a = par->getIV().getDouble("a");
        if (par->getIV().getVariable("add") != "None")
        {
            this->add = par->getIV().getDouble("add");
            this->edd = this->add / this->a ; 
        }else{
            this->add = 0.0;
            this->edd = 0.0;
        }

        calculateLengthScales();

        // time
        this->dt = par->getIV().getDouble("dt");
        this->time0 = par->getIV().getDouble("time0");
        this->epsilon_e = par->getIV().getDouble("epsilon_e");
        this->niter = par->getIV().getInt("niter");
        this->itmod = par->getIV().getInt("itmod");

    
        // Mesh in Fourier space
        this->KXMAX[0] = 1.0/(2.0* DX[0]);
        this->KXMAX[1] = 1.0/(2.0* DX[1]);
        this->KXMAX[2] = 1.0/(2.0* DX[2]);
        this->DKX[0] = 1.0/(NX[0]*DX[0]);
        this->DKX[1] = 1.0/(NX[1]*DX[1]);
        this->DKX[2] = 1.0/(NX[2]*DX[2]);

        this->DV=DX[0]*DX[1]*DX[2];
        this->V = this->DV * this->Npoints;
        
    }catch( const std::exception &e){
        std::cout << "Something went terribly wrong!" << std::endl;
    }
}



void Params::calculateLengthScales() {
    if( (abs(omega[0]) < 1E-6) && (abs(omega[1]) < 1E-6) && (abs(omega[2]) < 1E-6) )
    {
        this->aho = std::sqrt( hbar/(mass) );
        this->omho = 1.0;
    }
    else if( (abs(omega[0]) < 1E-6) && (abs(omega[1]) < 1E-6) )
    {
        this->ax[2] = std::sqrt( hbar/(mass*omega[2]) );
        this->aho = ax[2];
        ax[2] /= aho;
        this->omho = omega[2];
    }
    else if( (abs(omega[0]) < 1E-6) )
    {
        this->ax[1] = std::sqrt( hbar/(mass*omega[1]) );
        this->ax[2] = std::sqrt( hbar/(mass*omega[2]) );
        this->aho = std::sqrt( ax[1]*ax[2] );
        ax[1] /= aho;
        ax[2] /= aho;
        this->omho = std::sqrt(omega[2]*omega[1]);
    }
    else
    {
        this->ax[0] = std::sqrt( hbar/(mass*omega[0]) );
        this->ax[1] = std::sqrt( hbar/(mass*omega[1]) );
        this->ax[2] = std::sqrt( hbar/(mass*omega[2]) );
        this->aho = std::pow( ax[0]*ax[1]*ax[2], 1./3. );
        ax[0] /= aho;
        ax[1] /= aho;
        ax[2] /= aho;
        this->omho = std::pow( omega[0]*omega[1]*omega[2], 1./3. );
    }


    a = a * abohr / aho;
    
    // dipolar
    if (this->add != 0.0)
    {
        add = add * abohr / aho;

        if ( (abs(add)<1E-6) && (abs(a)<1E-6)){
            edd = 0.0;
        }else{
            edd = add / a;
        }
    }
    

    for (int i = 0; i < 3; ++i){
        XMAX[i] = XMAX[i] / aho;
        DX[i] = DX[i] / aho;
    }
       
    
}









int ParamsBase::getInt(const std::string &varName) const {
    return iv.getInt(varName);
}

double ParamsBase::getDouble(const std::string &varName) const {
    return iv.getDouble(varName);
}

std::string ParamsBase::getString(const std::string &varName) const {
    return iv.getString(varName);
}

bool ParamsBase::getBool(const std::string &varName) const {
    return iv.getBool(varName);
}