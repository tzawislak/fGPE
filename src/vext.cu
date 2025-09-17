#include "vext.hpp"




Vext::Vext(const Params &p, const std::string &type)
{
    vext = (complex*) malloc( p.Npoints *sizeof(complex) );

    if( p.getBool("read_vext") == true )
    {
        this->readVext(p.vpath, p.Npoints);
    }
    else
    {
        // decide which type of vext to use:
        bool result = std::any_of(p.omega, p.omega + 3, [](double x) { return std::abs(x) < 1E-6; });

        if ( result && abs(p.omega[0]) < 1E-6 )
        {
            std::cout << " Initializing tubular Vext..." << std::endl;
            initTube(p, 0);
        }
        else
        {
            std::cout << " Initializing harmonic Vext..." << std::endl;
            initHarmonic(p, 0);
        }
    }
}



Vext::Vext(const Params2 &p, const int &type)
{
    vext = (complex*) malloc( p.Npoints *sizeof(complex) );

    if( p.getBool("read_vext") == true )
    {
        readVext(p.vpath+std::to_string(type), p.Npoints);
    }
    else
    {
        // decide which type of vext to use:
        const double *omega = (type ==1 ) ? p._omega(1) :  p._omega(2);

        bool result = std::any_of(omega, omega + 3, [](double x) { return std::abs(x) < 1E-6; });

        if ( result && abs(omega[0]) < 1E-6 )
        {
            initTube(p, type);
        }
        else
        {
            initHarmonic(p, type);
        }
    }
}


void Vext::initTube(const ParamsBase &p, const int &type)
{
    const double *ax = (type ==1 ) ? p._ax(1) : p._ax(2);

    for (int i = 0; i < p.Npoints; ++i)
    {       
        double y = getY(p, i)/ax[1];
        double z = getZ(p, i)/ax[2];
        vext[i] = 0.5 * (  pow(y,2)/pow(ax[1],2) + pow(z,2)/pow(ax[2],2) );
    }
}

void Vext::initHarmonic(const ParamsBase &p, const int &type)
{
    const double *ax = (type ==1 ) ? p._ax(1) : p._ax(2);

    for (int i = 0; i < p.Npoints; ++i)
    {       
        double x = getX(p, i)/ax[0];
        double y = getY(p, i)/ax[1];
        double z = getZ(p, i)/ax[2];
        vext[i] = 0.5 * (  pow(x,2)/pow(ax[0],2) + 
                           pow(y,2)/pow(ax[1],2) +
                           pow(z,2)/pow(ax[2],2) );
    }
}


void Vext::initBox(const ParamsBase &p, const int &type)
{
    const double *ax = (type ==1 ) ? p._ax(1) : p._ax(2);

    double V = p.getDouble("Vbox");
    double xx = p.getDouble("xbox")/p._aho(0);
    double yy = p.getDouble("ybox")/p._aho(0);
    double zz = p.getDouble("zbox")/p._aho(0);
    double B = p.getDouble("Bbox");


    for (int i = 0; i < p.Npoints; ++i)
    {       
        double x = getX(p, i);
        double y = getY(p, i);
        double z = getZ(p, i);

        
        vext[i] = 0.5*V * ( (tanh(B*(x-xx)) - tanh(B*(x+xx)))* 
                        (tanh(B*(y-yy)) - tanh(B*(y+yy)))*
                        (tanh(B*(z-zz)) - tanh(B*(z+zz)))  );
    }
}

void Vext::initVoid(const ParamsBase &p, const int &type)
{
    for (int i = 0; i < p.Npoints; ++i)
    {       
        vext[i] = 0;
    }
}

void Vext::addWeightedProtocolPotential(const Params2 &p, const int &type)
{
    const double n1 = p.npart1 / p.V;
    const double n2 = p.npart2 / p.V;
    const double c0 = p.getDouble("c0") / (p.aho1*1E-3); // Bring to um/s and then to aho (input should be in mm/s)
    double A=0;

    if( abs(c0) < 1E-6 ){
        A = p.getDouble("aseed");
    }else{
        const double num =   4*pi*(p.a11-p.a12) - c0*c0/( n1 * p.omho1*p.omho1 );
        const double den =   4*pi*(p.a22-p.a12) - c0*c0/( n2 * p.omho1*p.omho1 );

        const double xi = -1*num/den;
        if (xi > 0){        A = (type ==1 ) ?  2*p.getDouble("aseed") *xi/(1+xi) : 2*p.getDouble("aseed")/(1+xi);}
        else       {        A = (type ==1 ) ?  2*p.getDouble("aseed") *xi/(1-xi) : 2*p.getDouble("aseed")/(1-xi);}
    }
    std::cout << "# The lambda imbalance: " << A << std::endl;
   
    double L = 2*p.XMAX[0];
    double n = p.getDouble("nseed");

    for (int i = 0; i < p.Npoints; ++i)
    {
        vext[i] += A*cos( n * 2*pi/L * getX(p, i) );
    }

}

void Vext::addProtocolPotential(const ParamsBase &p, const int &type, const int sign)
{
 
    const double *ax = (type ==1 ) ? p._ax(1) : p._ax(2);

    double L = 2*p.XMAX[0];
    double n = p.getDouble("nseed");
    double A = p.getDouble("aseed") * sign;

    for (int i = 0; i < p.Npoints; ++i)
    {
        vext[i] += A*cos( n * 2*pi/L * getX(p, i) );
    }
}

void Vext::addOpticalLattice(const ParamsBase &p, const int &type)
{
    double L = 2*p.XMAX[0];
    double n = p.getDouble("nopt");;
    double A = p.getDouble("aopt");

    for (int i = 0; i < p.Npoints; ++i)
    {
        vext[i] += A*cos( n * 2*pi/L * getX(p, i) );
    }
}



double Vext::getX(const ParamsBase &p, int i)
{
    int ix = iX( i, p.NX[0], p.NX[1]);
    return -p.XMAX[0] + (ix)*p.DX[0] + 0.5*p.DX[0];
}

double Vext::getY(const ParamsBase &p, int i)
{
    int iy = iY( i, p.NX[0], p.NX[1]);
    return -p.XMAX[1] + (iy)*p.DX[1] + 0.5*p.DX[1];
}

double Vext::getZ(const ParamsBase &p, int i)
{
    int iz = iZ( i, p.NX[0], p.NX[1]);
    return -p.XMAX[2] + (iz)*p.DX[2] + 0.5*p.DX[2];
}

    
void Vext::readVext(const std::string& prefix, const int& length)
{
    std::string end = "vext.wdat";
   
    // Open the binary file in read mode
    FILE *file = fopen( (prefix+end).c_str(), "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open the Vext file!\n");
    }

    size_t read_elements = fread( vext, sizeof(complex), length, file);
    if (read_elements != length) {
        fprintf(stderr, "Error: Only %zu elements were read instead of %d!\n", read_elements, length);
        fclose(file);
    }

    fclose(file);
}
