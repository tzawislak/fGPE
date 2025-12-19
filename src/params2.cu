#include "params2.hpp"

Params2::Params2( const InputVariables &iv ) {
    this->iv = iv;

    InitializeParams(this);

}


Params2::Params2(const Params2& other){
    this->iv = other.getIV();
    InitializeParams(&other);
}


Params2& Params2::operator=(const Params2& other){
    if (this == &other) {  // Self-assignment check
        return *this;
    }
    this->iv = other.getIV();
    InitializeParams(&other);
    return *this;
}



void Params2::InitializeParams(const Params2 *par){
    try {
        this->isTwoComponent = true;
        this->outpath = par->getIV().getString("outprefix");
        this->inpath = par->getIV().getString("inprefix");
        this->vpath = par->getIV().getString("vprefix");

        this->NX[0] = par->getIV().getInt("nx");
        this->NX[1] = par->getIV().getInt("ny");
        this->NX[2] = par->getIV().getInt("nz");

        // if not 3D
        this->DIM = 3;
        if( (this->NX[0]*this->NX[1] == 0) || (this->NX[1]*this->NX[2] == 0) )
        {
            if( this->NX[0]*this->NX[1] == this->NX[1]*this->NX[2]  ){
                this->DIM = 1;
            }else{
                this->DIM = 2;
            }
        }

        this->Npoints = NX[0]*NX[1]*NX[2];

        this->XMAX[0] = par->getIV().getDouble("xmax");
        this->XMAX[1] = par->getIV().getDouble("ymax");
        this->XMAX[2] = par->getIV().getDouble("zmax");

        
        this->DX[0] = 2*this->XMAX[0] / NX[0];
        this->DX[1] = 2*this->XMAX[1] / NX[1];
        this->DX[2] = 2*this->XMAX[2] / NX[2];
        this->omega1[0] = 2*pi*par->getIV().getDouble("omx1");
        this->omega1[1] = 2*pi*par->getIV().getDouble("omy1");
        this->omega1[2] = 2*pi*par->getIV().getDouble("omz1");
        this->omega2[0] = 2*pi*par->getIV().getDouble("omx2");
        this->omega2[1] = 2*pi*par->getIV().getDouble("omy2");
        this->omega2[2] = 2*pi*par->getIV().getDouble("omz2");

        this->mass1 = par->getIV().getDouble("mass1");
        this->npart1 = par->getIV().getDouble("npart1");
        this->a11 = par->getIV().getDouble("a11");
        this->mass2 = par->getIV().getDouble("mass2");
        this->npart2 = par->getIV().getDouble("npart2");
        this->a22 = par->getIV().getDouble("a22");
        this->a12 = par->getIV().getDouble("a12");

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



void Params2::calculateLengthScales() {

    // COMPONENT 1

    if( (abs(omega1[0]) < 1E-6) && (abs(omega1[1]) < 1E-6) )
    {
        this->ax1[2] = std::sqrt( hbar/(mass1*omega1[2]) );
        this->aho1 = ax1[2];
        ax1[2] /= aho1;
        this->omho1 = omega1[2];
    }
    else if( (abs(omega1[0]) < 1E-6) )
    {
        this->ax1[1] = std::sqrt( hbar/(mass1*omega1[1]) );
        this->ax1[2] = std::sqrt( hbar/(mass1*omega1[2]) );
        this->aho1 = std::sqrt( ax1[1]*ax1[2] );
        ax1[1] /= aho1;
        ax1[2] /= aho1;
        this->omho1 = std::sqrt(omega1[2]*omega1[1]);
    }
    else
    {
        this->ax1[0] = std::sqrt( hbar/(mass1*omega1[0]) );
        this->ax1[1] = std::sqrt( hbar/(mass1*omega1[1]) );
        this->ax1[2] = std::sqrt( hbar/(mass1*omega1[2]) );
        this->aho1 = std::pow( ax1[0]*ax1[1]*ax1[2], 1./3. );
        ax1[0] /= aho1;
        ax1[1] /= aho1;
        ax1[2] /= aho1;
        this->omho1 = std::pow( omega1[0]*omega1[1]*omega1[2], 1./3. );
    }


    // COMPONENT 2
    
    if( (abs(omega2[0]) < 1E-6) && (abs(omega2[1]) < 1E-6) )
    {
        this->ax2[2] = std::sqrt( hbar/(mass2*omega2[2]) );
        this->aho2 = ax2[2];
        ax2[2] /= aho2;
        this->omho2 = omega2[2];
    }
    else if( (abs(omega2[0]) < 1E-6) )
    {
        this->ax2[1] = std::sqrt( hbar/(mass2*omega2[1]) );
        this->ax2[2] = std::sqrt( hbar/(mass2*omega2[2]) );
        this->aho2 = std::sqrt( ax2[1]*ax2[2] );
        ax2[1] /= aho2;
        ax2[2] /= aho2;
        this->omho2 = std::sqrt(omega2[2]*omega2[1]);
    }
    else
    {
        this->ax2[0] = std::sqrt( hbar/(mass2*omega2[0]) );
        this->ax2[1] = std::sqrt( hbar/(mass2*omega2[1]) );
        this->ax2[2] = std::sqrt( hbar/(mass2*omega2[2]) );
        this->aho2 = std::pow( ax2[1]*ax2[2], 1./3. );
        ax2[0] /= aho2;
        ax2[1] /= aho2;
        ax2[2] /= aho2;
        this->omho2 = std::pow( omega2[0]*omega2[1]*omega2[2], 1./3. );
    }

    if( abs(this->omho2-this->omho1) > 1E-6 ){
            std::cerr << "ERROR: Different traps for each spin component currently not supported. Exiting..." << std::endl;
            exit(1);
    } 


    a11 = a11 * abohr / aho1;
    a22 = a22 * abohr / aho1;
    a12 = a12 * abohr / aho1;
    

    
    

    for (int i = 0; i < 3; ++i){
        XMAX[i] = XMAX[i] / aho1;
        DX[i] = DX[i] / aho1;
    }
       
    
}









