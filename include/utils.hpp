#pragma once

#include <array>

constexpr int Q = 9;
const std::array<int, Q> cx = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
const std::array<int, Q> cy = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
const std::array<double, Q> w = {
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

constexpr int N_species=3;

//Overload function to recover the index
inline int INDEX(const int x, const int y, const int i, const int NX) {
    return i + Q * (x + NX * y);
}
inline int INDEX(const int x, const int y, const int NX) {
    return x + NX * y;
}

// Indicizzazione non simmetrica (matrice piena N x N)
inline int flat_index(const int a, const int b, const int N) {
    return a * N + b;
}

// Indicizzazione simmetrica (triangolo superiore, a ≤ b)
inline int symmetric_index(int a, int b, const int N) {
    if (a > b) std::swap(a, b);
    return a * N - (a * (a + 1)) / 2 + b;
}
