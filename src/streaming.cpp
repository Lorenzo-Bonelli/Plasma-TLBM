#include "streaming.hpp"


#include <stdexcept>

namespace streaming {
// Opposite directions for bounce‐back:
constexpr std::array<int, Q> opp = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

//──────────────────────────────────────────────────────────────────────────────
//  Streaming dispatcher:
//──────────────────────────────────────────────────────────────────────────────
void Stream(std::vector<std::vector<double>>& f_species,
            std::vector<std::vector<double>>& temp_species,
            std::vector<std::vector<double>>& g_species,
            const int NX, const int NY, const BCType type) {
    switch (type) {
        case BCType::Periodic:
            StreamingPeriodic(f_species, temp_species, NX, NY);
            StreamingPeriodic(g_species, temp_species, NX, NY);
            break;
        case BCType::BounceBack:
            StreamingBounceBack(f_species, temp_species, NX, NY);
            StreamingBounceBack(g_species, temp_species, NX, NY);
            break;
        default:
            throw std::runtime_error("Type of streaming not supported.");
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + PERIODIC boundaries:
//    f_{i}(x + c_i, y + c_i, t+1) = f_{i}(x,y,t)
//  or
//    g_{i}(x + c_i, y + c_i, t+1) = g_{i}(x,y,t)
//──────────────────────────────────────────────────────────────────────────────
void StreamingPeriodic(std::vector<std::vector<double>>& f_species,
                       std::vector<std::vector<double>>& temp_species,
                       const int NX, const int NY) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i) {
                const int x_str = (x + NX + cx[i]) % NX;
                const int y_str = (y + NY + cy[i]) % NY;

                const int idx_to   = INDEX(x_str, y_str, i, NX);
                const int idx_from = INDEX(x, y, i, NX);

                for (int s = 0; s < N_species; ++s) {
                    temp_species[s][idx_to] = f_species[s][idx_from];
                }
            }
        }
    }

    for (int s = 0; s < N_species; ++s) {
        f_species[s].swap(temp_species[s]);
    }
}
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
                         const int NX, const int NY) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i) {
                const int x_str = x + cx[i];
                const int y_str = y + cy[i];
                const int idx_from = INDEX(x, y, i, NX);

                if (x_str >= 0 && x_str < NX && y_str >= 0 && y_str < NY) {
                    const int idx_to = INDEX(x_str, y_str, i, NX);
                    for (int s = 0; s < N_species; ++s) {
                        temp_species[s][idx_to] = f_species[s][idx_from];
                    }
                } else if (x_str >= 0 && x_str < NX) {
                    const int idx_to = INDEX(x_str, y, opp[i], NX);
                    for (int s = 0; s < N_species; ++s) {
                        temp_species[s][idx_to] = f_species[s][idx_from];
                    }
                } else if (y_str >= 0 && y_str < NY) {
                    const int idx_to = INDEX(x, y_str, opp[i], NX);
                    for (int s = 0; s < N_species; ++s) {
                        temp_species[s][idx_to] = f_species[s][idx_from];
                    }
                } else {
                    const int idx_to = INDEX(x, y, opp[i], NX);
                    for (int s = 0; s < N_species; ++s) {
                        temp_species[s][idx_to] = f_species[s][idx_from];
                    }
                }
            }
        }
    }

    for (int s = 0; s < N_species; ++s) {
        f_species[s].swap(temp_species[s]);
    }
}


}  // namespace streaming
