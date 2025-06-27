#pragma once

#include "utils.hpp"

#include <vector>
#include <array>

namespace streaming {

enum class BCType {
    Periodic,
    BounceBack
};
//──────────────────────────────────────────────────────────────────────────────
//  Streaming dispatcher:
//──────────────────────────────────────────────────────────────────────────────
void Stream(std::vector<std::vector<double>>& f_species,
            std::vector<std::vector<double>>& temp_species,
            std::vector<std::vector<double>>& g_species,
            const int NX, const int NY, const BCType type);

//──────────────────────────────────────────────────────────────────────────────
//  Streaming + PERIODIC boundaries:
//    f_{i}(x + c_i, y + c_i, t+1) = f_{i}(x,y,t)
//  or
//    g_{i}(x + c_i, y + c_i, t+1) = g_{i}(x,y,t)
//──────────────────────────────────────────────────────────────────────────────
void StreamingPeriodic(std::vector<std::vector<double>>& f_species,
                       std::vector<std::vector<double>>& temp_species,
                       const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + BOUNCE‐BACK walls at x=0, x=NX−1, y=0, y=NY−1.
//    If (x + c_i,x) or (y + c_i,y) is out‐of‐bounds, reflect:
//       f_{i*}(x,y) += f_{i}(x,y)
//  or
//       g_{i*}(x,y) += g_{i}(x,y)
//    where i* = opp[i].
//──────────────────────────────────────────────────────────────────────────────
void StreamingBounceBack(std::vector<std::vector<double>>& f_species,
                         std::vector<std::vector<double>>& temp_species,
                         const int NX, const int NY);

}  // namespace streaming

