#include "input_variables.hpp"
#include "output.hpp"
#include "params.hpp"
#include "params2.hpp"
#include "psi.hpp"
#include "vext.hpp"
#include "header.hpp"


// Single component headers
#include "hamiltonians/BEC_oneComponent.hpp"


// Two-component headers
#include "hamiltonians/BEC_twoComponent.hpp"

// User-defined headers:


int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];

    InputVariables reader(inputFile);   
	
    
    // SINGLE COMPONENT HAMILTONIANS
    if( !reader.getBool("t2component") ){

        std::cout << "# Single component mode." << std::endl;
        Params params(reader);
      
         std::string hamiltonian = params.getString("hamiltonian");

        if (hamiltonian == "bec") 
        {
            std::cout << "# BEC hamiltonian" << std::endl;
            BEConeComponent bec(params);
        } 
        else 
        {
            std::cerr << "No hamiltonian available for: " << hamiltonian << std::endl;
            exit(1);
        }
    }
    else
    // TWO-COMPONENT HAMILTONIANS
    {
        std::cout << "# Two component mode." << std::endl;
        Params2 params2(reader);
        
        std::string hamiltonian = params2.getString("hamiltonian");

        if (hamiltonian == "2bec") 
        {
            std::cout << "# 2BEC hamiltonian" << std::endl;
            BECtwoComponent bec(params2);
        }
        else
        {
            std::cerr << "No hamiltonian available for: " << hamiltonian << std::endl;
            exit(2);
        }

    }   


    





   
    return 0;
}
