#pragma once

#include "utils.hpp"

#include <vector>
#include <array>

namespace collisions {

void Collide(
    std::vector<std::vector<double>>& g_species,
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
    const double cs2
);
//──────────────────────────────────────────────────────────────────────────────
//  Thermal Collision step for both species:
//    g_e_post = g_e - (1/τ_Te)(g_e - g_e^eq) + Source
//    g_i_post = g_i - (1/τ_Ti)(g_i - g_i^eq) + Source
//  As source term we consider the enrgy losses from f
//──────────────────────────────────────────────────────────────────────────────
void ThermalCollisions(
    std::vector<std::vector<double>>& g_species,
    const std::vector<std::vector<double>>& g_eq_species,
    const std::vector<std::vector<double>>& f_eq_species,
    const std::vector<std::vector<double>>& rho_species,
    const std::vector<std::vector<double>>& ux_species,
    const std::vector<std::vector<double>>& uy_species,
    std::vector<std::vector<double>>& temp_species,
    const int NX, const int NY,
    const double Kb
);
//──────────────────────────────────────────────────────────────────────────────
//  Collision step (BGK + Guo forcing) for both species:
//    f_e_post = f_e - (1/τ_e)(f_e - f_e^eq) + F_e
//    f_i_post = f_i - (1/τ_i)(f_i - f_i^eq) + F_i
//──────────────────────────────────────────────────────────────────────────────
void Collisions(
    std::vector<std::vector<double>>& f_species,
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
    const double cs2
);


} // namespace collisions
