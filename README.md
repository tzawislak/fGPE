# fGPE — A Fast Gross–Pitaevskii Equation Solver 

version: 0.3

fGPE is a high-performance solver for the **Gross–Pitaevskii equation (GPE)**, designed for fast and accurate simulation of Bose–Einstein condensates.

The solver can be extended to user-specific Hamiltonians that include e.g. dipolar interactions of soft-core potential.

---

## Features

- ⚡ **High-performance GPE integration** using optimized numerical schemes  
- 🧮 Supports **imaginary-time evolution** and **real-time dynamics**  
- 📦 Modular architecture for extending potentials  
- 📈 Stable for long-time simulations thanks to RK4 integration algorithm. 
- 🔧 Easy configuration via input files


---

## Solvers
 
### Imaginary-time evolution
The stationary solution of a given problem is found by minimizing the energy of a system defined with user-defined (or predefined) energy density functional \( \mathcal{E} \).



### Real-time evolution
The solver integrates the **time-dependent Gross–Pitaevskii equation**:

$$ i\hbar \frac{\partial \psi}{\partial t} = \left( -\frac{\hbar^2}{2m} \nabla^2 + V(\mathbf{r}) + g|\psi|^2 \right)\psi $$

Includes options for:

- Harmonic, box, or custom potentials  
- Interaction strengths  
- Normalized wavefunctions  
- 1D, 2D, or 3D domains (depending on your implementation)

---

## Dependencies
- CUDA version > 11

## Usage

- Imaginary-time evolution:
```bash
make
./gpe input.txt
```

- Real-time evolution:
```bash
make
./gpe rinput.txt
```
