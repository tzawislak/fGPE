#include "input_variables.hpp"

InputVariables::InputVariables(const std::string filename){
    std::cout << "#INFO   Reading " << filename << std::endl;
    readFile(filename);

    convertTypes();
}

InputVariables::InputVariables( const InputVariables &other){
    variables = other.getVariables();
    strings = other.getStrings();
    doubles = other.getDoubles();
    bools = other.getBools();
    ints = other.getInts();
}


InputVariables& InputVariables::operator=(const InputVariables& other){
    if (this == &other) {  // Self-assignment check
        return *this;
    }
    variables = other.getVariables();
    strings = other.getStrings();
    doubles = other.getDoubles();
    bools = other.getBools();
    ints = other.getInts();

    return *this;
}


bool InputVariables::isDouble(const std::string &str) {
    try {
        std::stod(str); // Attempt to convert to float
        return true; // Conversion succeeded
    } catch (const std::invalid_argument &) {
        return false; // Conversion failed (invalid argument)
    } catch (const std::out_of_range &) {
        return false; // Conversion failed (out of range)
    }
}

bool InputVariables::isInteger(const std::string &str) {
    try {
        std::stoi(str); // Attempt to convert to float
        return true; // Conversion succeeded
    } catch (const std::invalid_argument &) {
        return false; // Conversion failed (invalid argument)
    } catch (const std::out_of_range &) {
        return false; // Conversion failed (out of range)
    }
}

void InputVariables::convertTypes()
{
    for (const auto &entry : variables) {
        // check if the variable is a bool
        if( entry.second == "false" || entry.second == "true" )
            bools[entry.first] = (entry.second == "false") ? false : true;

        // check if the variable is an int
        if( isInteger(entry.second) )
            ints[entry.first] = std::stoi(entry.second);

        // check if the variable is a string
        if( entry.second.front() == '"' && entry.second.back() == '"' )
            strings[entry.first] = entry.second.substr(1, entry.second.length() - 2);
        // check if the variable is a float
        if( isDouble(entry.second) )
            doubles[entry.first] = std::stod(entry.second);
        
    }
}

// Trim function to remove leading and trailing whitespaces
std::string InputVariables::trim(const std::string &str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return ""; // Empty or only whitespace
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

// Helper to check if a string is a comment or empty
bool InputVariables::isCommentOrEmpty(const std::string &line) {
    std::string trimmedLine = trim(line);
    return trimmedLine.empty() || trimmedLine[0] == '#';
}

// Extract variable type (bool, int, float)
void InputVariables::processLine(const std::string &line) {
    std::istringstream iss(line);
    std::string variableName, value;
    
    // Read the variable name and value
    if (!(iss >> variableName >> value)) {
        // Invalid line, do nothing
        return;
    }

    // Store the variable name and value as strings in a map
    variables[variableName] = value;
}

void InputVariables::readFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(2);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove anything after '#' (the comment)
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        // Trim the line and check if it's a comment or empty
        if (isCommentOrEmpty(line)) {
            continue; // Skip empty or comment-only lines
        }

        // Process the valid line
        processLine(line);
    }

    file.close();

}

void InputVariables::printVariables(const std::string _name) const {
    std::cout << _name << std::endl;
    if( _name == "Bools" ){
        auto vars = this->bools;
        for (const auto &entry : vars) {
            std::cout << entry.first  << "\t" << entry.second << "\t" << typeid(entry.second).name() << std::endl;
        }
    }else if( _name == "Ints" ){
        auto vars = this->ints;
        for (const auto &entry : vars) {
            std::cout << entry.first  << "\t" << entry.second << "\t" << typeid(entry.second).name() << std::endl;
        }
    }else if( _name == "Doubles" ){
        auto vars = this->doubles;
        for (const auto &entry : vars) {
            std::cout << entry.first  << "\t" << entry.second << "\t" << typeid(entry.second).name() << std::endl;
        }
    }else if( _name == "Strings" ){
        auto vars = this->strings;
        for (const auto &entry : vars) {
            std::cout << entry.first  << "\t" << entry.second << "\t" << typeid(entry.second).name() << std::endl;
        }
    }

    
}


// Getter to retrieve variable by name
std::string InputVariables::getVariable(const std::string &varName) const {
    auto it = variables.find(varName);
    if (it != variables.end()) {
        return it->second;
    }
    std::cerr << "Variables: No such variable " << varName << std::endl; 
    return "None";
}

int InputVariables::getInt(const std::string &varName) const {
    auto it = ints.find(varName);
    if (it != ints.end()) {
        return it->second;
    }
    std::cerr << "Ints: No such variable " << varName << std::endl; 
    exit(3);    
    return 0;
}

double InputVariables::getDouble(const std::string &varName) const {
    auto it = doubles.find(varName);
    if (it != doubles.end()) {
        return it->second;
    }
    std::cerr << "Doubles: No such variable " << varName << std::endl; 
    exit(3);
    return 0;
}

std::string InputVariables::getString(const std::string &varName) const {
    auto it = strings.find(varName);
    if (it != strings.end()) {
        return it->second;
    }
    std::cerr << "Strings: No such variable " << varName << std::endl; 
    exit(3);
    return 0;
}

bool InputVariables::getBool(const std::string &varName) const {
    auto it = bools.find(varName);
    if (it != bools.end()) {
        return it->second;
    }
    std::cerr << "Bools: No such variable " << varName << std::endl; 
    exit(3);
    return 0;
}


