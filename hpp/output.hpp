#ifndef __output__
#define __output__
#include "input_variables.hpp"
#include "header.hpp"
#include "params.hpp"
#include "params2.hpp"



class Output {

private:
    std::string prefix;
    FILE* status;
    FILE* status_real;

public:
    std::string getPrefix() const { return prefix; };
    Output(const std::string& prefix);
    Output(const Output& other);
    Output& operator=(const Output& other);

    void WriteVariable2WTXT( const char *name, const double &value, const char *unit);
    void WriteVariable2WTXT( const char *name, const int &value, const char *unit);

    void WriteWtxt(const Params &p, const int &ncycle);
    void WriteWtxt_2c(const Params2 &p, const int &ncycle);

    void WriteInputFile();
    void WritePsi(const complex *psi, const int length, const std::string number="");
    void WritePsiInit(const complex *psi, const int length, const std::string number="" );
    void WritePsiFinal(const complex *psi, const int length, const std::string number="" );

    void WriteVext(const complex *vext, const int length, const std::string number="");

    void CreateStatusFile(const char* header);
    void CreateRealStatusFile(const char* header);
    void CloseStatusFile();
    void CloseRealStatusFile();
    void WriteStatus(const char* line);
    void WriteStatusReal(const char* line);
    void WriteObservable(const std::string& name);

    void Write3DMatrix(const double  *matrix, const int length, const std::string& filename, const char* mode = "wb");
    void Write3DMatrix(const complex *matrix, const int length, const std::string& filename, const char* mode = "wb");
    ~Output();
};  

#endif
