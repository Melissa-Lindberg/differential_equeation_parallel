#pragma once

#include <cstddef>
#include <string>
#include <utility>
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

  double eval(const Point &p) const { return a * p.x + b * p.y - c; }

  bool contains(const Point &p) const { return eval(p) <= EPS_GEOM; }
};

inline const std::vector<HalfPlane> &domain_halfplanes() {
  static const std::vector<HalfPlane> planes = {{+1.0, +1.0, 2.0},
                                                {-1.0, +1.0, 2.0},
                                                {+1.0, -1.0, 2.0},
                                                {-1.0, -1.0, 2.0},
                                                {0.0, +1.0, 1.0}};
  return planes;
}

constexpr double A1 = -2.0;
constexpr double B1 = 2.0;
constexpr double A2 = -2.0;
constexpr double B2 = 2.0;
constexpr double DELTA = 1e-8;
constexpr double TAU = 1e-8;

inline const std::pair<int, int> &default_grids(size_t number) {
  static const std::vector<std::pair<int, int>> grids = {
      {10, 10}, {20, 20}, {40, 40}, {400, 600}, {800, 1200}};
  return grids.at(number);
}

inline std::pair<int, int> default_partition_for(int M, int N) {
  if (M == 10 && N == 10)
    return {2, 2};
  if (M == 20 && N == 20)
    return {4, 4};
  if (M == 40 && N == 40)
    return {8, 8};
  if (M == 400 && N == 600)
    return {8, 12};
  if (M == 800 && N == 1200)
    return {16, 24};
  return {0, 0};
}

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

  double x(int i) const { return A1 + h1 * static_cast<double>(i); }

  double y(int j) const { return A2 + h2 * static_cast<double>(j); }

  double x_mid(int i) const { return A1 + (static_cast<double>(i) - 0.5) * h1; }

  double y_mid(int j) const { return A2 + (static_cast<double>(j) - 0.5) * h2; }

  std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(M + 1) +
           static_cast<std::size_t>(i);
  }
};

struct ProblemData {
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> F;
  std::vector<double> diag;
};

struct Config {
  Grid grid;
  double delta;
  double tau;
  long long maxIt;
  double epsilon;
};

struct DomainBlock {
  int nodes_x;
  int nodes_y;
};

struct DomainRange {
  int ix0, ix1;
  int iy0, iy1;
  int ii0, ii1;
  int jj0, jj1;
  int ai0, ai1, aj0, aj1;
  int bi0, bi1, bj0, bj1;
};

struct Partition {
  int Px;
  int Py;
  std::vector<DomainBlock> blocks;
  std::vector<DomainRange> ranges;
};

struct PartitionCheckResult {
  Partition partition;
  std::string report;
};

struct Result {
  std::vector<double> solution;
  std::size_t iterations = 0;
  double residual_norm = 0.0;
  double diff_norm = 0.0;
  double rhs_norm = 0.0;
  std::string stop_reason;
  std::vector<IterationLogEntry> iteration_log;
};

std::vector<int> make_blocks(int total_nodes, int parts);
Partition build_partition_with_ranges(int M, int N, int Px, int Py);
PartitionCheckResult check_partition(int M, int N, int Px, int Py,
                                     double rmin = 0.5, double rmax = 2.0);
