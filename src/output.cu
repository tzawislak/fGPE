#include "output.hpp"

Output::Output(const std::string& prefix){
    this->prefix = prefix;
}

Output& Output::operator=(const Output& other)
{
    if (this == &other) {  // Self-assignment check
        return *this;
    }
    this->prefix = other.getPrefix();
    return *this;
}


Output::Output(const Output& other)
{
    this->prefix = other.getPrefix();
}





void Output::WriteWtxt(const Params &p, const int &ncycle){

    FILE *file = fopen( (this->prefix+"info.wtxt").c_str(), "w");
    if (file == NULL) {
        perror("Wtxt file opening failed");
    }

    fprintf(file, "cycles                   %d\n\n", ncycle);
    fprintf(file, "dim                      %d\n", p.DIM);

    fprintf(file, "nx                       %d\n", p.NX[0]);
    fprintf(file, "ny                       %d\n", p.NX[1]);
    fprintf(file, "nz                       %d\n", p.NX[2]);
    fprintf(file, "dx                       %lf\n", p.DX[0]);
    fprintf(file, "dy                       %lf\n", p.DX[1]);
    fprintf(file, "dz                       %lf\n", p.DX[2]);
    fprintf(file, "lx                       %lf\n", p.NX[0]*p.DX[0]*p.aho);
    fprintf(file, "ly                       %lf\n", p.NX[0]*p.DX[1]*p.aho);
    fprintf(file, "lz                       %lf\n", p.NX[0]*p.DX[2]*p.aho);
    fprintf(file, "datadim                  %d\n", 3);
    fprintf(file, "prefix                   %s\n", p.outpath.c_str());
    fprintf(file, "t0                       %lf\n", p.time0);
    fprintf(file, "dt                       %lf\n\n", p.dt);


    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "# tag", "name", "value", "unit", "format");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "var", "psi", "complex", "none", "wdat");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "var", "vext", "complex", "none", "wdat");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n\n", "", "", "", "", "");


    fprintf(file, "%-15s %-15s %-15s %-15s\n", "# tag", "name", "value", "unit");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "hbar       ", hbar , "u*(um)^2/s");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "kB         ", kB, "[hbar]/nK");
    fprintf(file, "%-15s %-15s %-15.2f %-15s\n", "const", "npart      ", p.npart, "none");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "mass       ", p.mass, "u");
    fprintf(file, "%-15s %-15s %-15.6E %-15s\n", "const", "abohr      ", abohr, "mu_m");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "a          ", p.a/abohr*p.aho, "a0");
    if (p.getIV().getVariable("add") != "None")
    {
        fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "add        ", p.add/abohr*p.aho, "a0");
        fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "dipole_x   ", p.getDouble("dipole_x"), "1");
        fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "dipole_y   ", p.getDouble("dipole_y"), "1");
        fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "dipole_z   ", p.getDouble("dipole_z"), "1");
    }
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omx        ", p.omega[0]/(2*pi), "2pi");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omy        ", p.omega[1]/(2*pi), "2pi");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omz        ", p.omega[2]/(2*pi), "2pi");
   
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "aho        ", p.aho, "?");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ax         ", p.ax[0], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ay         ", p.ax[1], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "az         ", p.ax[2], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omho       ", p.omho, "?");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "Eho        ", p.omho*hbar, "Energy_scale");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ringRho    ", p.getDouble("ringRho"), "um");
    fprintf(file, "%-15s %-15s %-15d %-15s\n", "const", "itmod      ", p.itmod, ".");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "Rcutoff    ", , "um");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "mu         ", mu, "!");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "en_h       ", en, "!");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "convergence", ediff, "Energy");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omega      ", p.getDouble("omega"), "?");

    fclose(file);
}



void Output::WriteWtxt_2c(const Params2 &p, const int &ncycle){

    FILE *file = fopen( (this->prefix+"info.wtxt").c_str(), "w");
    if (file == NULL) {
        perror("Wtxt file opening failed");
    }

    fprintf(file, "cycles                   %d\n\n", ncycle);
    fprintf(file, "dim                      %d\n", p.DIM);

    fprintf(file, "nx                       %d\n", p.NX[0]);
    fprintf(file, "ny                       %d\n", p.NX[1]);
    fprintf(file, "nz                       %d\n", p.NX[2]);
    fprintf(file, "dx                       %lf\n", p.DX[0]);
    fprintf(file, "dy                       %lf\n", p.DX[1]);
    fprintf(file, "dz                       %lf\n", p.DX[2]);
    fprintf(file, "lx                       %lf\n", p.NX[0]*p.DX[0]*p.aho1);
    fprintf(file, "ly                       %lf\n", p.NX[1]*p.DX[1]*p.aho1);
    fprintf(file, "lz                       %lf\n", p.NX[2]*p.DX[2]*p.aho1);
    fprintf(file, "datadim                  %d\n", 3);
    fprintf(file, "prefix                   %s\n", p.outpath.c_str());
    fprintf(file, "t0                       %lf\n", p.time0);
    fprintf(file, "dt                       %lf\n\n", p.dt);


    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "# tag", "name", "value", "unit", "format");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "var", "1psi", "complex", "none", "wdat");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "var", "2psi", "complex", "none", "wdat");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "var", "1vext", "complex", "none", "wdat");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n", "var", "2vext", "complex", "none", "wdat");
    fprintf(file, "%-15s %-15s %-15s %-15s %-15s\n\n", "", "", "", "", "");


    fprintf(file, "%-15s %-15s %-15s %-15s\n", "# tag", "name", "value", "unit");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "hbar       ", hbar , "u*(um)^2/s");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "kB         ", kB, "[hbar]/nK");
    fprintf(file, "%-15s %-15s %-15.6E %-15s\n", "const", "abohr      ", abohr, "mu_m");

    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "npart1      ", p.npart1, "none");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "mass1       ", p.mass1, "u");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "a11         ", p.a11/abohr*p.aho1, "a0");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "npart2      ", p.npart2, "none");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "mass2       ", p.mass2, "u");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "a22         ", p.a22/abohr*p.aho1, "a0");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "a12         ", p.a12/abohr*p.aho1, "a0");
    fprintf(file, "%-15s %-15s %-15.2f %-15s\n", "const", "omx1        ", p.omega1[0]/(2*pi), "2pi");
    fprintf(file, "%-15s %-15s %-15.2f %-15s\n", "const", "omy1        ", p.omega1[1]/(2*pi), "2pi");
    fprintf(file, "%-15s %-15s %-15.2f %-15s\n", "const", "omz1        ", p.omega1[2]/(2*pi), "2pi");
    fprintf(file, "%-15s %-15s %-15.2f %-15s\n", "const", "omx2        ", p.omega2[0]/(2*pi), "2pi");
    fprintf(file, "%-15s %-15s %-15.2f %-15s\n", "const", "omy2        ", p.omega2[1]/(2*pi), "2pi");
    fprintf(file, "%-15s %-15s %-15.2f %-15s\n", "const", "omz2        ", p.omega2[2]/(2*pi), "2pi");
    
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "aho1        ", p.aho1, "?");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ax1         ", p.ax1[0], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ay1         ", p.ax1[1], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "az1         ", p.ax1[2], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omho1       ", p.omho1, "?");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "Eho1        ", p.omho1*hbar, "Energy_scale");

    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "aho2        ", p.aho2, "?");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ax2         ", p.ax2[0], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ay2         ", p.ax2[1], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "az2         ", p.ax2[2], "code");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omho2       ", p.omho2, "?");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "Eho2        ", p.omho2*hbar, "Energy_scale");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "ringRho    ", p.getDouble("ringRho"), "um");
    fprintf(file, "%-15s %-15s %-15d %-15s\n", "const", "itmod      ", p.itmod, ".");

    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "aho        ", p.aho1, "?");
    fprintf(file, "%-15s %-15s %-15.1f %-15s\n", "const", "Eho        ", p.omho1*hbar, "Energy_scale");
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omho       ", p.omho1, "?");

    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "Rcutoff    ", , "um");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "mu         ", mu, "!");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "en_h       ", en, "!");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "convergence", ediff, "Energy");
    //fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", "omega      ", p.getDouble("omega"), "?");

    fclose(file);
}

void Output::WriteVariable2WTXT( const char *name, const double &value, const char *unit){
    FILE *file = fopen( (this->prefix+"info.wtxt").c_str(), "a");
    if (file == NULL) {
        perror("Wtxt file opening failed");
    }
    fprintf(file, "%-15s %-15s %-15.6f %-15s\n", "const", name, value, unit);
    fclose(file);
}

void Output::WriteVariable2WTXT( const char *name, const int &value, const char *unit){
    FILE *file = fopen( (this->prefix+"info.wtxt").c_str(), "a");
    if (file == NULL) {
        perror("Wtxt file opening failed");
    }
    fprintf(file, "%-15s %-15s %-15d %-15s\n", "const", name, value, unit);
    fclose(file);
}

void Output::WritePsi(const complex* psi, const int length, const std::string number){
    // append previous data
    std::string end = number+"psi.wdat";
    this->Write3DMatrix( psi, length, end, "ab");
}

void Output::WritePsiInit(const complex* psi, const int length, const std::string number){
    std::string end = number+"psi_init.wdat";
    this->Write3DMatrix( psi, length, end);
}

void Output::WritePsiFinal(const complex* psi, const int length, const std::string number){
    std::string end = number+"psi_final.wdat";
    this->Write3DMatrix( psi, length, end);
}

void Output::WriteVext(const complex *vext, const int length, const std::string number){
    std::string end = number+"vext.wdat";
    this->Write3DMatrix( vext, length, end);
}


void Output::WriteInputFile(){
    //TODO: copy the input file
}


void Output::CreateStatusFile(const char* header){
    this->status = fopen( (prefix+"status.out").c_str(), "w");
    fprintf(this->status, header);
}

void Output::CreateRealStatusFile(const char* header){
    this->status_real = fopen( (prefix+"status.out").c_str(), "w");
    fprintf(this->status_real, header);
}

void Output::CloseStatusFile(){
    fclose(this->status);
}

void Output::CloseRealStatusFile(){
    fclose(this->status_real);
}

void Output::WriteStatus(const char* line){
    static int count=0; // Close and open the file to update it during the time evolution
    count++;
    if( count%10 == 1 ){
        CloseStatusFile();
        this->status = fopen( (prefix+"status.out").c_str(), "a");
    }
    fprintf(this->status, "%s", line);
}

void Output::WriteStatusReal(const char* line){
    static int count=0; // Close and open the file to update it during the time evolution
    count++;
    if( count%10 == 1 ){
        CloseRealStatusFile();
        this->status_real = fopen( (prefix+"status.out").c_str(), "a");
    }
    fprintf(this->status_real, "%s", line);
}

void Output::WriteObservable(const std::string& name){

}

void Output::Write3DMatrix(const complex *matrix, const int length, const std::string& filename, const char *mode){
    // Open the binary file in read mode
    FILE *file = fopen( (this->prefix+filename).c_str(), mode);
    if (file == NULL) {
        fprintf(stderr, filename.c_str());
        fprintf(stderr, "Error: Could not open file!\n");
    }

    size_t read_elements = fwrite( matrix, sizeof(complex), length, file);
    if (read_elements != length) {
        fprintf(stderr, "Error: Only %zu elements were written instead of %d!\n", read_elements, length);
        fclose(file);
    }

    fclose(file);
}

void Output::Write3DMatrix(const double  *matrix, const int length, const std::string& filename, const char *mode){
    // Open the binary file in read mode
    FILE *file = fopen( (this->prefix+filename).c_str(), mode);
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file!\n");
    }

    size_t read_elements = fwrite( matrix, sizeof(double), length, file);
    if (read_elements != length) {
        fprintf(stderr, "Error: Only %zu elements were written instead of %d!\n", read_elements, length);
        fclose(file);
    }

    fclose(file);
}

Output::~Output(){

    
}
