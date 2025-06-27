#include "collisions.hpp"


namespace collisions {
// Setting values for tau based on previous knowledge
static const std::vector<double> tau_species = [] {
    std::vector<double> tau(N_species * (N_species + 1) / 2, 0.0);

    // self
    tau[symmetric_index(0, 0, N_species)] = 5.0;  // tau_e
    tau[symmetric_index(1, 1, N_species)] = 3.0;  // tau_i
    tau[symmetric_index(2, 2, N_species)] = 1.0;  // tau_n

    // cross
    tau[symmetric_index(0, 1, N_species)] = 6.0;  // tau_e_i
    tau[symmetric_index(0, 2, N_species)] = 4.0;  // tau_e_n
    tau[symmetric_index(1, 2, N_species)] = 2.0;  // tau_i_n

    return tau;
}();
//Thermal tau are considered equal to the f ones

//──────────────────────────────────────────────────────────────────────────────
// Collide function to call in the simulation
// 1) Thermal collision so that we can evealete energy loss
// 2) mass collisions
//──────────────────────────────────────────────────────────────────────────────
void Collide(std::vector<std::vector<double>>& g_species,
             const std::vector<std::vector<double>>& g_eq_species,
             std::vector<std::vector<double>>& f_species,
             const std::vector<std::vector<double>>& f_eq_species,
             std::vector<std::vector<double>>& rho_species,
             std::vector<std::vector<double>>& ux_species,
             std::vector<std::vector<double>>& uy_species,
             const std::vector<double>& Ex,
             const std::vector<double>& Ey,
             const std::vector<double>& q_species,
             const std::vector<double>& m_species,
             std::vector<std::vector<double>>& temp_species,
             const int NX, const int NY,
             const double Kb,
             const double cs2)
{
    ThermalCollisions(g_species, g_eq_species,
                      f_eq_species, rho_species, ux_species, uy_species,
                      temp_species, NX, NY, Kb);
    Collisions(f_species, f_eq_species,
               rho_species, ux_species, uy_species,
               Ex, Ey, q_species, m_species,
               temp_species, NX, NY, cs2);
}

//──────────────────────────────────────────────────────────────────────────────
//  Thermal Collision step for both species:
//    g_e_post = g_e - (1/τ_Te)(g_e - g_e^eq) + Source
//    g_i_post = g_i - (1/τ_Ti)(g_i - g_i^eq) + Source
//  As source term we consider the enrgy losses from f
//──────────────────────────────────────────────────────────────────────────────
void ThermalCollisions(std::vector<std::vector<double>>& g_species,
                       const std::vector<std::vector<double>>& g_eq_species,
                       const std::vector<std::vector<double>>& f_eq_species,
                       const std::vector<std::vector<double>>& rho_species,
                       const std::vector<std::vector<double>>& ux_species,
                       const std::vector<std::vector<double>>& uy_species,
                       std::vector<std::vector<double>>& temp_species,
                       const int NX, const int NY,
                       const double Kb)
{
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y, NX);

            std::vector<double> Sx_eq(N_species, 0.0), Sy_eq(N_species, 0.0);
            std::vector<double> rho_star(N_species, 0.0);

            for (int i = 0; i < Q; ++i) {
                const int idx_3 = INDEX(x, y, i, NX);
                for (int s = 0; s < N_species; ++s) {
                    for (int b = 0; b < N_species; ++b) {
                        const double tau_sb = tau_species[symmetric_index(s, b, N_species)];
                        const double feq_sb = f_eq_species[flat_index(s, b, N_species)][idx_3];
                        Sx_eq[s] += feq_sb / tau_sb * cx[i];
                        Sy_eq[s] += feq_sb / tau_sb * cy[i];
                        rho_star[s] += feq_sb / tau_sb;
                    }
                }
            }

            for (int s = 0; s < N_species; ++s) {
                const double ux = ux_species[symmetric_index(s, s, N_species)][idx];
                const double uy = uy_species[symmetric_index(s, s, N_species)][idx];
                const double Sx = rho_species[s][idx] * ux;
                const double Sy = rho_species[s][idx] * uy;
                double tau_star = 1.0;
                for (int b = 0; b < N_species; ++b) tau_star -= 1.0 / tau_species[symmetric_index(s, b, N_species)];

                const double deltaE =
                    (tau_star * (Sx + Sy) + Sx_eq[s] + Sy_eq[s]) * (tau_star * (Sx + Sy) + Sx_eq[s] + Sy_eq[s]) /
                    (2.0 * (tau_star * rho_species[s][idx] + rho_star[s])) -
                    (Sx + Sy) * (Sx + Sy) / (2.0 * rho_species[s][idx]);

                const double deltaT = -deltaE / Kb;

                for (int i = 0; i < Q; ++i) {
                    const int idx_3 = INDEX(x, y, i, NX);
                    double coll = 0.0;
                    for (int b = 0; b < N_species; ++b) {
                        const int eq_idx = flat_index(s, b, N_species);
                        coll -= (g_species[s][idx_3] - g_eq_species[eq_idx][idx_3]) / tau_species[symmetric_index(s, b, N_species)];
                    }
                    temp_species[s][idx_3] = g_species[s][idx_3] + coll + deltaT;
                }
            }
        }
    }

    for (int s = 0; s < N_species; ++s) {
        g_species[s].swap(temp_species[s]);
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Collision step (BGK + Guo forcing) for both species:
//    f_e_post = f_e - (1/τ_e)(f_e - f_e^eq) + F_e
//    f_i_post = f_i - (1/τ_i)(f_i - f_i^eq) + F_i
//──────────────────────────────────────────────────────────────────────────────
void Collisions(std::vector<std::vector<double>>& f_species,
                const std::vector<std::vector<double>>& f_eq_species,
                const std::vector<std::vector<double>>& rho_species,
                const std::vector<std::vector<double>>& ux_species,
                const std::vector<std::vector<double>>& uy_species,
                const std::vector<double>& Ex,
                const std::vector<double>& Ey,
                const std::vector<double>& q_species,
                const std::vector<double>& m_species,
                std::vector<std::vector<double>>& temp_species,
                const int NX, const int NY,
                const double cs2)
{
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y, NX);
            const double Ex_loc = Ex[idx];
            const double Ey_loc = Ey[idx];

            for (int i = 0; i < Q; ++i) {
                const int idx_3 = INDEX(x, y, i, NX);
                for (int s = 0; s < N_species; ++s) {
                    const double rho_s = rho_species[s][idx];
                    const double ux_s = ux_species[symmetric_index(s, s, N_species)][idx];
                    const double uy_s = uy_species[symmetric_index(s, s, N_species)][idx];
                    const double tau_s = tau_species[symmetric_index(s, s, N_species)];

                    double forcing = 0.0;
                    if (q_species[s] != 0.0 && m_species[s] != 0.0) {
                        const double cu = cx[i] * ux_s + cy[i] * uy_s;
                        const double cE = cx[i] * Ex_loc + cy[i] * Ey_loc;
                        const double uE = ux_s * Ex_loc + uy_s * Ey_loc;
                        forcing = w[i] * q_species[s] * rho_s / m_species[s] / cs2 *
                                  (1.0 - 1.0 / (2 * tau_s)) * (cE + cu * cE / cs2 - uE);
                    }

                    // collision term: sommatoria su b (specie di interazione)
                    double coll = 0.0;
                    for (int b = 0; b < N_species; ++b) {
                        const int idx_eq = flat_index(s, b, N_species);
                        coll -= (f_species[s][idx_3] - f_eq_species[idx_eq][idx_3]) / tau_species[symmetric_index(s, b, N_species)];
                    }

                    temp_species[s][idx_3] = f_species[s][idx_3] + coll + forcing;
                }
            }
        }
    }

    for (int s = 0; s < N_species; ++s) {
        f_species[s].swap(temp_species[s]);
    }
}


} // namespace collisions
