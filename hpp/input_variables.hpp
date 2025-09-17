#ifndef __input__variables
#define __input__variables

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <typeinfo>

#include <memory>


class InputVariables {

private:
    std::map<std::string, std::string> variables;
    std::map<std::string, std::string> strings;
    std::map<std::string, double> doubles;
    std::map<std::string, bool> bools;
    std::map<std::string, int> ints;


    std::string  trim(const std::string &str);
    bool isCommentOrEmpty(const std::string &line) ;
    void processLine(const std::string &line);
    void convertTypes();
    bool isInteger(const std::string &str);
    bool isDouble(const std::string &str);


public:
    InputVariables(){};
    InputVariables( const InputVariables &other);
    InputVariables& operator=(const InputVariables& other);

    InputVariables(const std::string filename);
    void readFile(const std::string &filename);
    void printVariables(const std::string _name) const ;

    std::string getVariable(const std::string &varName) const;
    int getInt(const std::string &varName) const;
    double getDouble(const std::string &varName) const;
    std::string getString(const std::string &varName) const;
    bool getBool(const std::string &varName) const; 

    std::map<std::string, std::string> getVariables() const { return variables; };
    std::map<std::string, std::string> getStrings() const { return strings; };
    std::map<std::string, double> getDoubles() const { return doubles; };
    std::map<std::string, bool> getBools() const { return bools; };
    std::map<std::string, int> getInts() const { return ints; };
};


#endif 