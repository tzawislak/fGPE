# Compiler
CXX = nvcc 

# Compiler flags
CXXFLAGS = -Ihpp -std=c++17 -lcufft -O3 -Wno-deprecated-gpu-targets

# Target executable
TARGET = gpe

HAMILTONIANS_DIR = hamiltonians
SRC_DIR = src
INCLUDE_DIR = hpp
OBJ_DIR = obj

MAIN_SRC = main.cu

# Find all .cu files in the directory
SRCS = $(wildcard $(SRC_DIR)/*.cu)
HMTS = $(wildcard $(HAMILTONIANS_DIR)/*.cu)

# Combine all source files (main.cpp + other .cpp files from src/)
ALL_SRCS = $(MAIN_SRC) $(SRCS) $(HMTS)

# Create corresponding .o (object) files in the obj/ directory, including subdirectories
OBJS = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(ALL_SRCS))

# Rule to build the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to compile .cpp files into .o files
$(OBJ_DIR)/%.o: %.cu | create_dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create necessary directories for object files
.PHONY: create_dirs
create_dirs:
	mkdir -p $(OBJ_DIR)/$(SRC_DIR)
	mkdir -p $(OBJ_DIR)/$(HAMILTONIANS_DIR)

# Clean build files
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(TARGET)
