#include "input_variables.hpp"
#include "output.hpp"
#include "params.hpp"
#include "params2.hpp"
#include "psi.hpp"
#include "vext.hpp"
#include "header.hpp"


// Single component headers
#include "hamiltonians/BEC_oneComponent.hpp"
#include "hamiltonians/dBEC_oneComponent.hpp"
#include "hamiltonians/BEC_soft.hpp"

// Two component headers
#include "hamiltonians/BEC_twoComponent.hpp"
#include "hamiltonians/BEC_Rabi.hpp"



 


int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];

    InputVariables reader(inputFile);   
	
    
    
    if( !reader.getBool("t2component") ){

        std::cout << "# Single component mode." << std::endl;
        Params params(reader);
        //params.getIV().printVariables("Ints");
        //params.getIV().printVariables("Doubles");
        //params.getIV().printVariables("Strings");
        //params.getIV().printVariables("Bools");

         std::string hamiltonian = params.getString("hamiltonian");

        if (hamiltonian == "bec") 
        {
            std::cout << "BEC hamiltonian" << std::endl;
            BEConeComponent bec(params);
        } 
        else if (hamiltonian == "dbec") 
        {
            std::cout << "Dipolar BEC hamiltonian" << std::endl;
            dBEConeComponent bec(params);
        } 
        else if (hamiltonian == "soft") 
        {
            std::cout << "Soft-core BEC hamiltonian" << std::endl;
            BECsoft bec(params);
        } 
        else 
        {
            std::cerr << "No hamiltonian available for: " << hamiltonian << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cout << "# Two component mode." << std::endl;
        Params2 params2(reader);
        //params2.getIV().printVariables("Ints");
        //params2.getIV().printVariables("Doubles");
        //params2.getIV().printVariables("Strings");
        //params2.getIV().printVariables("Bools");
        std::string hamiltonian = params2.getString("hamiltonian");

        if (hamiltonian == "2bec") 
        {
            std::cout << "BEC hamiltonian" << std::endl;
            BECtwoComponent bec(params2);
        } 
        else if (hamiltonian == "2rabi") 
        {
            std::cout << "Rabi BEC hamiltonian" << std::endl;
            BECRabi bec(params2);
        } 
        else 
        {
            std::cerr << "No hamiltonian available for: " << hamiltonian << std::endl;
            exit(2);
        }

    }   


    





   
    return 0;
}
