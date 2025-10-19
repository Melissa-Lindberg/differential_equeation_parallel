#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "log.h"

constexpr double EPS_GEOM = 1e-12;

struct Point {
    double x;
    double y;
};

class HalfPlane {
public:
    double a;
    double b;
    double c;

    double eval(const Point &p) const {
        return a * p.x + b * p.y - c;
    }

    bool contains(const Point &p) const {
        return eval(p) <= EPS_GEOM;
    }
};

class Grid {
public:
    double A1;
    double B1;
    double A2;
    double B2;
    int M;
    int N;
    double h1;
    double h2;

    Grid(double a1, double b1, double a2, double b2, int m, int n)
        : A1(a1), B1(b1), A2(a2), B2(b2), M(m), N(n) {
        h1 = (B1 - A1) / static_cast<double>(M);
        h2 = (B2 - A2) / static_cast<double>(N);
    }

    double x(int i) const {
        return A1 + h1 * static_cast<double>(i);
    }

    double y(int j) const {
        return A2 + h2 * static_cast<double>(j);
    }

    double x_mid(int i) const {
        return A1 + (static_cast<double>(i) - 0.5) * h1;
    }

    double y_mid(int j) const {
        return A2 + (static_cast<double>(j) - 0.5) * h2;
    }

    std::size_t index(int i, int j) const {
        return static_cast<std::size_t>(j) * static_cast<std::size_t>(M + 1) + static_cast<std::size_t>(i);
    }
};

struct ProblemData {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> F;
    std::vector<double> diag;
};

struct RunConfig {
    Grid grid;
    double delta;
    double tau;
    long long maxIt;
    double epsilon;
};

struct RunResult {
    std::vector<double> solution;
    std::size_t iterations = 0;
    double residual_norm = 0.0;
    double diff_norm = 0.0;
    double rhs_norm = 0.0;
    std::string stop_reason;
    std::vector<IterationLogEntry> iteration_log;
};
