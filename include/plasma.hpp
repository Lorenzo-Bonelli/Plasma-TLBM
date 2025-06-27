#pragma once

#include "collisions.hpp"
#include "poisson.hpp"
#include "streaming.hpp"
#include "utils.hpp"
#include "visualize.hpp"

#include <array>
#include <vector>

//--------------------------------------------------------------------------------
// LBmethod: performs a two‐species (electron + ion) D2Q9 LBM under an electric field.
// All “physical” parameters are passed in SI units to the constructor, and inside
// the constructor they are converted to lattice units.  After that, Run_simulation()
// can be called to execute the time loop.
//--------------------------------------------------------------------------------
class LBmethod {
public:
    // Constructor: pass in *all* physical/SI parameters + grid‐size + time‐steps
    //
    //   NSTEPS       : number of time steps to run
    //   NX, NY       : number of lattice nodes in x and y (grid size)
    //   n_cores      : number of OpenMP threads (optional, can be ignored)
    //   Z_ion, A_ion : ionic charge‐number and mass‐number (for computing ion mass)
    //
    //   T_e_SI, T_i_SI : electron and ion temperatures [K]
    //   Ex_SI, Ey_SI   : uniform external E‐field [V/m] (can be overridden by Poisson solver)
    //
    //   poisson_type   : which Poisson solver to use (NONE, GAUSS_SEIDEL, or SOR)
    //   bc_type        : which streaming/BC to use (PERIODIC or BOUNCE_BACK)
    //   omega_sor      : over‐relaxation factor for SOR (only used if poisson_type==SOR)
    //
    LBmethod(const int    NSTEPS,
             const int    NX,
             const int    NY,
             const size_t    n_cores,
             const int    Z_ion,
             const int    A_ion,
             const double    Ex_SI,
             const double    Ey_SI,
             const double    T_e_SI_init,
             const double    T_i_SI_init,
             const double    T_n_SI_init,
             const double    n_e_SI_init,
             const double    n_n_SI_init,
             const poisson::PoissonType poisson_type,
             const streaming::BCType      bc_type,
             const double    omega_sor);

    // Run the complete simulation
    void Run_simulation();


private:
    //──────────────────────────────────────────────────────────────────────────────
    // 1) “Raw” (SI) Inputs
    //──────────────────────────────────────────────────────────────────────────────
    const int  NSTEPS;       // total number of time steps
    const int  NX, NY;       // grid dimensions
    const size_t  n_cores;      // # of OpenMP threads (optional)
    const int  Z_ion;        // ionic atomic number (e.g. Z=1 for H+)
    const int  A_ion;        // ionic mass # (e.g. A=1 for H+)
    const poisson::PoissonType  poisson_type; // which Poisson solver to run
    const streaming::BCType       bc_type;      // which streaming/BC we use
    const double       omega_sor;    // over‐relaxation factor for SOR
    
    //──────────────────────────────────────────────────────────────────────────────
    // 1) LU quantities
    //──────────────────────────────────────────────────────────────────────────────

    std::vector<double> q_species;
    std::vector<double> m_species;
    double Kb;

    static constexpr double cs2 = 1.0/3.0;
    
    //──────────────────────────────────────────────────────────────────────────────
    // 5) Per‐Node (“lattice‐unit”) Fields
    //──────────────────────────────────────────────────────────────────────────────
    // Distribution functions: f_e[i + Q*(x + NX*y)], f_i[i + Q*(x + NX*y)]
    std::vector<std::vector<double>> f_species;   
    // Equilibrium distribution functions
    std::vector<std::vector<double>> f_eq_species; 
                         
    // Thermal distribution function
    std::vector<std::vector<double>> g_species;  
    // Equilibrium distribution functions
    std::vector<std::vector<double>> g_eq_species;
    
    // Themporal distribution functions
    std::vector<std::vector<double>> temp_species;

    // Macroscopic moments (per cell)
    std::vector<std::vector<double>> rho_species;      // densities
    std::vector<std::vector<double>> ux_species;       // velocities
    std::vector<std::vector<double>> uy_species;
    
    // Temperature vectors
    std::vector<std::vector<double>> T_species;

    // Electric potential in lattice units
    std::vector<double>   Ex,    Ey;         // self‐consistent E (overwrites Ex_latt_init)

    // Charge density (per cell in lattice units)
    std::vector<double>   rho_q; // dimensionless (#/cell * e_charge)

    //──────────────────────────────────────────────────────────────────────────────
    // 6) Private Methods
    //──────────────────────────────────────────────────────────────────────────────

    // (a) Initialize all fields (set f = f_eq at t=0, zero φ, set E=Ex_latt_init)
    void Initialize(const double T_e_SI_init, const double T_i_SI_init, const double T_n_SI_init,
                    const double n_e_SI_init, const double n_n_SI_init,
                    const double Ex_SI, const double Ey_SI);

    // (b) Compute equilibrium f_eq
    void ComputeEquilibrium();
    
    // (d) Macroscopic update:  ρ = Σ_i f_i,   ρ u = Σ_i f_i c_i + ½ F  T=Σ_i g_i
    void UpdateMacro();
  
};
