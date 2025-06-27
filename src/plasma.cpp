#include "plasma.hpp"

#include <cmath>
#include <omp.h>
#include <iostream>

//──────────────────────────────────────────────────────────────────────────────
//  Physical constants:
//──────────────────────────────────────────────────────────────────────────────
static constexpr double kB_SI       = 1.380649e-23;   // [J/K]
static constexpr double e_charge_SI = 1.602176634e-19;// [C]
static constexpr double epsilon0_SI = 8.854187817e-12;// [F/m]
static constexpr double m_e_SI      = 9.10938356e-31; // [kg]
static constexpr double u_SI        = 1.66053906660e-27; // [kg]

//──────────────────────────────────────────────────────────────────────────────
//  Constructor
//──────────────────────────────────────────────────────────────────────────────
LBmethod::LBmethod(const int    _NSTEPS,
                   const int    _NX,
                   const int    _NY,
                   const size_t    _n_cores,
                   const int    _Z_ion,
                   const int    _A_ion,
                   const double    _Ex_SI,
                   const double    _Ey_SI,
                   const double    _T_e_SI_init,
                   const double    _T_i_SI_init,
                   const double    _T_n_SI_init,
                   const double    _n_e_SI_init,
                   const double    _n_n_SI_init,
                   const poisson::PoissonType _poisson_type,
                   const streaming::BCType      _bc_type,
                   const double    _omega_sor)
    : NSTEPS      (_NSTEPS),
      NX          (_NX),
      NY          (_NY),
      n_cores     (_n_cores),
      Z_ion       (_Z_ion),
      A_ion       (_A_ion),
      poisson_type(_poisson_type),
      bc_type     (_bc_type),
      omega_sor   (_omega_sor)
{
    Initialize(_T_e_SI_init, _T_i_SI_init, _T_n_SI_init,
               _n_e_SI_init, _n_n_SI_init,
               _Ex_SI, _Ey_SI);
}

//──────────────────────────────────────────────────────────────────────────────
//  1) Converto from Si to LU
//  2) Allocate vectors
//  3) Impose initial conditions chosen
//  No need to initialize the macroscopic quantities 
//  since UpdateMacro will do the work
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Initialize(double T_e_SI_init, double T_i_SI_init, double T_n_SI_init,
                          double n_e_SI_init, double n_n_SI_init,
                          double Ex_SI, double Ey_SI) {
    //──────────────────────────────────────────────────────────────────────────────
    //  Converto from Si to LU
    //──────────────────────────────────────────────────────────────────────────────
    // Reference quantities:
    const double n0_SI = n_e_SI_init;
    const double M0_SI = m_e_SI; // physical mass [kg]
    const double T0_SI = T_e_SI_init; // physical temperature [K]
    const double Q0_SI = e_charge_SI; // physical charge [C]
    const double L0_SI = std::sqrt(epsilon0_SI * kB_SI * T0_SI / (n0_SI * Q0_SI * Q0_SI))*1e-2; // physical lenght = lambda_D/100 [m]
    const double t0_SI = std::sqrt(epsilon0_SI * M0_SI / (3.0 * n0_SI * Q0_SI * Q0_SI))  *1e-2; // physical time = rad(3)/w_p/100 [s]
    
    //other useful obtained scaling quantities
    const double E0_SI = M0_SI * L0_SI / (Q0_SI * t0_SI * t0_SI); // physical electric field [V/m]
    //Quantities in LU:
    Kb = kB_SI* (t0_SI * t0_SI * T0_SI)/(L0_SI * L0_SI * M0_SI);
    
    // Converted E‐field in lattice units:
    const double Ex_ext = Ex_SI / E0_SI, 
                 Ey_ext = Ey_SI / E0_SI; // external intial E‐field in lattice units

    // Converted temperatures in lattice units:
    const double T_e_init = T_e_SI_init / T0_SI, 
                 T_i_init = T_i_SI_init / T0_SI,
                 T_n_init = T_n_SI_init / T0_SI; // initial temperatures in lattice units

    // mass in lattice units:
    const double m_e = m_e_SI / M0_SI; // electron mass in lattice units
    const double m_i = (u_SI * A_ion - m_e_SI * Z_ion) / M0_SI; // ion mass in lattice masses
    const double m_n = u_SI * A_ion / M0_SI; // neutrals mass in lattice units
    m_species = {m_e, m_i, m_n};

    // Converted charge in lattice units:
    const double q_e = - e_charge_SI / Q0_SI; // electron charge in lattice units
    const double q_i = Z_ion * e_charge_SI / Q0_SI; // ion charge in lattice units
    q_species = {q_e, q_i, 0.0};

    // Initial density in lattice unit
    const double rho_e_init = m_e * n_e_SI_init / n0_SI, // electron density in lattice units
                 rho_i_init = m_i * n_e_SI_init / n0_SI / static_cast<double>(Z_ion), // ion density in lattice units. The idea behind /Z_ion is overall neutrality of the plamsa at the start
                 rho_n_init = m_n * n_n_SI_init / n0_SI; // neutrals density in lattice units

    //──────────────────────────────────────────────────────────────────────────────
    //  Allocate all the vectors
    //──────────────────────────────────────────────────────────────────────────────
    const int size_distr = NX * NY * Q;
    //in order to allow deciding initial conditions this has to be set to zero
    //in this way we can define the function only where we want it
    f_species.assign(N_species, std::vector<double>(size_distr, 0.0));
    
    f_eq_species.resize(N_species * N_species, std::vector<double>(size_distr, 0.0));

    g_species.assign(N_species, std::vector<double>(size_distr, 0.0));

    g_eq_species.resize(N_species * N_species, std::vector<double>(size_distr, 0.0));

    temp_species.resize(N_species, std::vector<double>(size_distr, 0.0));

    const int size_macro = NX * NY;
    const int num_symmetric_pairs = N_species * (N_species + 1) / 2;

    rho_species.resize(N_species, std::vector<double>(size_macro, 0.0));

    ux_species.resize(num_symmetric_pairs, std::vector<double>(size_macro, 0.0));
    uy_species.resize(num_symmetric_pairs, std::vector<double>(size_macro, 0.0));

    T_species.resize(N_species, std::vector<double>(size_macro, 0.0));
    // Pulsed initial eleectric field
    Ex.assign(size_macro, Ex_ext); Ey.assign(size_macro, Ey_ext);

    rho_q.resize(size_macro);
    //──────────────────────────────────────────────────────────────────────────────
    //  Impose initial conditions chosen
    //──────────────────────────────────────────────────────────────────────────────               
    // Assign electrons andd ions only in the center
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = (NX / 4) + 1; x < (3 * NX / 4); ++x) {
        for (int y = (NY / 4) + 1; y < (3 * NY / 4); ++y) {
            for (int i = 0; i < Q; ++i) {
                const int idx_3 = INDEX(x, y, i, NX);
                const double weight = w[i];
                f_species[0][idx_3] = weight * rho_e_init; // Electron
                g_species[0][idx_3] = weight * T_e_init;
                f_species[1][idx_3] = weight * rho_i_init; // Ion
                g_species[1][idx_3] = weight * T_i_init;
            }
        }
    }
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i) {
                const int idx_3 = INDEX(x, y, i, NX);
                const double weight = w[i];
                f_species[2][idx_3] = weight * rho_n_init; // Neutrals
                g_species[2][idx_3] = weight * T_n_init;
            }
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Compute D2Q9 equilibrium for f_eq and g_eq
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::ComputeEquilibrium() {
    const double invcs2 = 1.0 / cs2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y, NX);

            for (int a = 0; a < N_species; ++a) {
                const double rho_a = rho_species[a][idx];
                const double T_a   = T_species[a][idx];

                for (int b = 0; b < N_species; ++b) {
                    const int eq_idx = a * N_species + b;
                    const int u_idx = symmetric_index(a, b, N_species);

                    const double ux_ab = ux_species[u_idx][idx];
                    const double uy_ab = uy_species[u_idx][idx];
                    const double u2_ab = ux_ab * ux_ab + uy_ab * uy_ab;

                    for (int i = 0; i < Q; ++i) {
                        const int idx_3 = INDEX(x, y, i, NX);
                        const double cu_ab = cx[i] * ux_ab + cy[i] * uy_ab;
                        const double weight = w[i];

                        // Maxwell–Boltzmann approximation
                        f_eq_species[eq_idx][idx_3] = weight * rho_a * (
                            1.0 + cu_ab * invcs2 +
                            0.5 * (cu_ab * cu_ab) * invcs2 * invcs2 -
                            0.5 * u2_ab * invcs2
                        );
                        g_eq_species[eq_idx][idx_3] = weight * T_a * (
                            1.0 + cu_ab * invcs2 +
                            0.5 * (cu_ab * cu_ab) * invcs2 * invcs2 -
                            0.5 * u2_ab * invcs2
                        );
                    }
                }
            }
        }
    }
}

//──────────────────────────────────────────────────────────────────────────────
//  Update macroscopic variables for both species:
//    ρ = Σ_i f_i,
//    ρ u = Σ_i f_i c_i + (1/2)*F
//  where F = qom_latt * (Ex_cell, Ey_cell)
//    T = Σ_i g_i
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::UpdateMacro() {

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y, NX);

            std::vector<double> rho_loc(N_species, 0.0);
            std::vector<double> ux_loc(N_species, 0.0);
            std::vector<double> uy_loc(N_species, 0.0);
            std::vector<double> T_loc(N_species, 0.0);

            for (int i = 0; i < Q; ++i) {
                const int idx_3 = INDEX(x, y, i, NX);
                for (int s = 0; s < N_species; ++s) {
                    const double fi = f_species[s][idx_3];
                    rho_loc[s] += fi;
                    ux_loc[s] += fi * cx[i];
                    uy_loc[s] += fi * cy[i];
                    T_loc[s]  += g_species[s][idx_3];
                }
            }

            // Densità, temperatura e velocità per specie
            for (int s = 0; s < N_species; ++s) {
                if (rho_loc[s] < 1e-10) {
                    rho_species[s][idx] = 0.0;
                    T_species[s][idx] = 0.0;
                    ux_species[symmetric_index(s, s, N_species)][idx] = 0.0;
                    uy_species[symmetric_index(s, s, N_species)][idx] = 0.0;
                } else {
                    rho_species[s][idx] = rho_loc[s];
                    T_species[s][idx] = T_loc[s];
                    double ux_val = ux_loc[s] / rho_loc[s];
                    double uy_val = uy_loc[s] / rho_loc[s];

                    // Aggiungi effetto campo elettrico solo se la carica è ≠ 0
                    ux_val += 0.5 * q_species[s] * Ex[idx] / m_species[s];
                    uy_val += 0.5 * q_species[s] * Ey[idx] / m_species[s];

                    ux_species[symmetric_index(s, s, N_species)][idx] = ux_val;
                    uy_species[symmetric_index(s, s, N_species)][idx] = uy_val;
                }
            }

            // Velocità di interazione (simmetriche)
            for (int a = 0; a < N_species; ++a) {
                for (int b = a + 1; b < N_species; ++b) {
                    const double rho_a = rho_loc[a], rho_b = rho_loc[b];
                    double ux_ab = 0.0, uy_ab = 0.0;
                    if (rho_a + rho_b > 1e-10) {
                        ux_ab = (rho_a * ux_species[symmetric_index(a, a, N_species)][idx] +
                                 rho_b * ux_species[symmetric_index(b, b, N_species)][idx]) / (rho_a + rho_b);
                        uy_ab = (rho_a * uy_species[symmetric_index(a, a, N_species)][idx] +
                                 rho_b * uy_species[symmetric_index(b, b, N_species)][idx]) / (rho_a + rho_b);
                    }
                    int ab = symmetric_index(a, b, N_species);
                    ux_species[ab][idx] = ux_ab;
                    uy_species[ab][idx] = uy_ab;
                }
            }

            // Calcolo della densità di carica totale
            double rhoq = 0.0;
            for (int s = 0; s < N_species; ++s) {
                if (m_species[s] > 0.0) {
                    rhoq += q_species[s] * rho_species[s][idx] / m_species[s];
                }
            }
            rho_q[idx] = (rhoq < 1e-15) ? 0.0 : rhoq;
        }
    }
}



void LBmethod::Run_simulation() {
    
    // Set threads for this simulation
    omp_set_num_threads(n_cores);

    // Initialize visualize stuff
    visualize::InitVisualization(NX, NY, NSTEPS);

    //──────────────────────────────────────────────────────────────────────────────
    //  Main loop: for t = 0 … NSTEPS−1,
    //    [1] Update macros (ρ, u)
    //    [2] Calculate equilibrium distribution functions
    //    [3] Collisions (BGK + forcing)
    //    [4] Streaming (+ BC)
    //    [5] Solve Poisson → update Ex, Ey
    //    [6] Visualization
    //──────────────────────────────────────────────────────────────────────────────
    for (int t=0; t<NSTEPS; ++t){
        // Macroscopic update:  ρ = Σ_i f_i,   ρ u = Σ_i f_i c_i + ½ F  T=Σ_i g_i
        UpdateMacro(); 
        ComputeEquilibrium();

        // g(x,y,t)_postcoll=g(x,y,t) + (g_eq - g)/tau + Source
        // f(x,y,t)_postcoll=f(x,y,t) + (f_eq - f)/tau + dt*F
        collisions::Collide(g_species, g_eq_species,
                            f_species, f_eq_species,
                            rho_species, ux_species, uy_species,
                            Ex, Ey, q_species, m_species,
                            temp_species, NX, NY, Kb, cs2); 

        // f(x+cx,y+cx,t+1)=f(x,y,t)
        // +BC applyed
        streaming::Stream(f_species,
                          temp_species,
                          g_species,
                          NX, NY, bc_type);
        // Solve the poisson equation with the method chosen
        // Also BCs are important
        poisson::SolvePoisson(Ex,
                              Ey,
                              rho_q,
                              NX, NY,
                              omega_sor,
                              poisson_type,
                              bc_type);
    
        // Update video and data for plot
        visualize::UpdateVisualization(t, NX, NY,
                                       ux_species[0], uy_species[0],
                                       ux_species[1], uy_species[1],
                                       ux_species[2], uy_species[2],
                                       T_species[0], T_species[1], T_species[2],
                                       rho_species[0], rho_species[1], rho_species[2],
                                       rho_q, Ex, Ey);
    }
    //Close Visualize stuff
    visualize::CloseVisualization();


    std::cout << "Simulation ended " << std::endl;
}
