#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "MPI.h"

static inline std::pair<int, int> get_cart_dims(int world_size) {
  int dims[2] = {0, 0};
  int rc = MPI_Dims_create(world_size, 2, dims);
  if (rc != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Dims_create failed");
  }
  if (dims[0] <= 0 || dims[1] <= 0 || dims[0] * dims[1] != world_size) {
    throw std::runtime_error("MPI_Dims_create не удалось разложить world_size");
  }
  return {dims[0], dims[1]};
}

struct DistributedVector {
  int global_ix0 = 0;
  int global_iy0 = 0;
  int local_nx = 0;
  int local_ny = 0;
  int halo = 0;
  int pitch_x = 0;
  std::vector<double> data;

  DistributedVector() = default;

  DistributedVector(int ix0, int iy0, int nx, int ny, int halo_width)
      : global_ix0(ix0), global_iy0(iy0), local_nx(nx), local_ny(ny),
        halo(halo_width) {
    pitch_x = local_nx + 2 * halo;
    int total_y = local_ny + 2 * halo;
    data.assign(static_cast<std::size_t>(pitch_x * total_y), 0.0);
  }

  inline std::size_t idx(int i_with_halo, int j_with_halo) const {
    return static_cast<std::size_t>(j_with_halo + halo) *
               static_cast<std::size_t>(pitch_x) +
           static_cast<std::size_t>(i_with_halo + halo);
  }

  inline double &at(int i_local, int j_local) {
    return data[idx(i_local, j_local)];
  }

  inline const double &at(int i_local, int j_local) const {
    return data[idx(i_local, j_local)];
  }

  inline double &at_with_halo(int i_with_halo, int j_with_halo) {
    return data[idx(i_with_halo, j_with_halo)];
  }

  inline const double &at_with_halo(int i_with_halo, int j_with_halo) const {
    return data[idx(i_with_halo, j_with_halo)];
  }

  inline void fill(double value) { std::fill(data.begin(), data.end(), value); }
};

struct LocalProblemData {
  DistributedVector a;
  DistributedVector b;
  DistributedVector F;
  DistributedVector diag;
};
bool in_D(double x, double y) {
  Point p{x, y};
  for (const auto &hp : domain_halfplanes()) {
    if (!hp.contains(p)) {
      return false;
    }
  }
  return true;
}

Point intersect_segment_with_plane(const Point &p0, const Point &p1,
                                   const HalfPlane &hp) {
  double v0 = hp.eval(p0);
  double v1 = hp.eval(p1);
  double denom = v0 - v1;
  if (std::abs(denom) < EPS_GEOM) {
    return p0;
  }
  double t = v0 / denom;
  t = std::max(0.0, std::min(1.0, t));
  Point res;
  res.x = p0.x + t * (p1.x - p0.x);
  res.y = p0.y + t * (p1.y - p0.y);
  return res;
}

std::vector<Point> clip_polygon_by_halfplane(const std::vector<Point> &poly,
                                             const HalfPlane &hp) {
  std::vector<Point> result;
  if (poly.empty()) {
    return result;
  }
  Point prev = poly.back();
  bool prev_inside = hp.contains(prev);
  for (const Point &curr : poly) {
    bool curr_inside = hp.contains(curr);
    if (curr_inside) {
      if (!prev_inside) {
        Point inter = intersect_segment_with_plane(prev, curr, hp);
        result.push_back(inter);
      }
      result.push_back(curr);
    } else if (prev_inside) {
      Point inter = intersect_segment_with_plane(prev, curr, hp);
      result.push_back(inter);
    }
    prev = curr;
    prev_inside = curr_inside;
  }
  return result;
}

double polygon_area(const std::vector<Point> &poly) {
  if (poly.size() < 3) {
    return 0.0;
  }
  double area = 0.0;
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const Point &p0 = poly[i];
    const Point &p1 = poly[(i + 1) % poly.size()];
    area += p0.x * p1.y - p1.x * p0.y;
  }
  return 0.5 * std::abs(area);
}

double cell_area_in_D(double x0, double x1, double y0, double y1) {
  std::vector<Point> poly = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
  for (const auto &hp : domain_halfplanes()) {
    poly = clip_polygon_by_halfplane(poly, hp);
    if (poly.empty()) {
      break;
    }
  }
  return polygon_area(poly);
}

double segment_length_in_D(const Point &p0, const Point &p1) {
  double total_len = std::hypot(p1.x - p0.x, p1.y - p0.y);
  if (total_len <= EPS_GEOM) {
    return 0.0;
  }
  double t_low = 0.0;
  double t_high = 1.0;
  for (const auto &hp : domain_halfplanes()) {
    double v0 = hp.eval(p0);
    double v1 = hp.eval(p1);
    bool inside0 = v0 <= EPS_GEOM;
    bool inside1 = v1 <= EPS_GEOM;
    if (inside0 && inside1) {
      continue;
    }
    if (!inside0 && !inside1) {
      // сегмент полностью вне относительно текущей полуплоскости
      return 0.0;
    }
    double denom = v0 - v1;
    if (std::abs(denom) < EPS_GEOM) {
      // параллельно границе и одна точка снаружи
      return 0.0;
    }
    double t = v0 / denom;
    t = std::max(0.0, std::min(1.0, t));
    if (!inside0 && inside1) {
      // входим в область
      t_low = std::max(t_low, t);
    } else if (inside0 && !inside1) {
      // выходим из области
      t_high = std::min(t_high, t);
    }
    if (t_low - t_high >= EPS_GEOM) {
      return 0.0;
    }
  }
  double length = (t_high - t_low) * total_len;
  if (length < 0.0) {
    return 0.0;
  }
  if (length > total_len) {
    length = total_len;
  }
  return length;
}

DistributedVector make_distributed_vector(const DomainRange &range, int halo) {
  int nx = range.ix1 >= range.ix0 ? range.ix1 - range.ix0 + 1 : 0;
  int ny = range.iy1 >= range.iy0 ? range.iy1 - range.iy0 + 1 : 0;
  return DistributedVector(range.ix0, range.iy0, nx, ny, halo);
}

LocalProblemData build_local_problem(const Grid &grid, double epsilon,
                                     const Partition &partition,
                                     const DomainRange &local_range) {
  (void)partition;
  LocalProblemData data;
  data.a = make_distributed_vector(local_range, 1);
  data.b = make_distributed_vector(local_range, 1);
  data.F = make_distributed_vector(local_range, 1);
  data.diag = make_distributed_vector(local_range, 1);

  // коэффициенты a_{i,j}
  for (int j_local = -data.a.halo; j_local < data.a.local_ny + data.a.halo;
       ++j_local) {
    int j_global = local_range.iy0 + j_local;
    for (int i_local = -data.a.halo; i_local < data.a.local_nx + data.a.halo;
         ++i_local) {
      int i_global = local_range.ix0 + i_local;
      double value = 0.0;
      if (i_global >= 1 && i_global <= grid.M && j_global >= 1 &&
          j_global <= grid.N - 1) {
        double xmid = grid.x_mid(i_global);
        Point p0{xmid, grid.y(j_global) - 0.5 * grid.h2};
        Point p1{xmid, grid.y(j_global) + 0.5 * grid.h2};
        double len = segment_length_in_D(p0, p1);
        double frac = len / grid.h2;
        if (frac < 0.0) {
          frac = 0.0;
        }
        if (frac > 1.0) {
          frac = 1.0;
        }
        value = frac + (1.0 - frac) / epsilon;
      }
      data.a.at_with_halo(i_local, j_local) = value;
    }
  }

  // коэффициенты b_{i,j}
  for (int j_local = -data.b.halo; j_local < data.b.local_ny + data.b.halo;
       ++j_local) {
    int j_global = local_range.iy0 + j_local;
    for (int i_local = -data.b.halo; i_local < data.b.local_nx + data.b.halo;
         ++i_local) {
      int i_global = local_range.ix0 + i_local;
      double value = 0.0;
      if (i_global >= 1 && i_global <= grid.M - 1 && j_global >= 1 &&
          j_global <= grid.N) {
        Point p0{grid.x(i_global) - 0.5 * grid.h1, grid.y_mid(j_global)};
        Point p1{grid.x(i_global) + 0.5 * grid.h1, grid.y_mid(j_global)};
        double len = segment_length_in_D(p0, p1);
        double frac = len / grid.h1;
        if (frac < 0.0) {
          frac = 0.0;
        }
        if (frac > 1.0) {
          frac = 1.0;
        }
        value = frac + (1.0 - frac) / epsilon;
      }
      data.b.at_with_halo(i_local, j_local) = value;
    }
  }

  // Правая часть F
  double cell_area = grid.h1 * grid.h2;
  for (int j = local_range.iy0; j <= local_range.iy1; ++j) {
    for (int i = local_range.ix0; i <= local_range.ix1; ++i) {
      double x_left = grid.x(i) - 0.5 * grid.h1;
      double x_right = grid.x(i) + 0.5 * grid.h1;
      double y_bottom = grid.y(j) - 0.5 * grid.h2;
      double y_top = grid.y(j) + 0.5 * grid.h2;
      double area = cell_area_in_D(x_left, x_right, y_bottom, y_top);
      double ratio = area / cell_area;
      if (ratio < 1e-12) {
        ratio = 0.0;
      }
      if (ratio > 1.0 - 1e-12) {
        ratio = 1.0;
      }
      data.F.at(i - local_range.ix0, j - local_range.iy0) = ratio;
    }
  }

  // Диагональный предобуславливатель
  double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
  double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
  if (!(local_range.ii0 > local_range.ii1 ||
        local_range.jj0 > local_range.jj1)) {
    for (int j = local_range.jj0; j <= local_range.jj1; ++j) {
      int j_local = j - local_range.iy0;
      for (int i = local_range.ii0; i <= local_range.ii1; ++i) {
        int i_local = i - local_range.ix0;
        double val = 0.0;
        val += (data.a.at(i_local + 1, j_local) + data.a.at(i_local, j_local)) *
               inv_h1_sq;
        val += (data.b.at(i_local, j_local + 1) + data.b.at(i_local, j_local)) *
               inv_h2_sq;
        data.diag.at(i_local, j_local) = val;
      }
    }
  }

  return data;
}

double inner_product_local(const Grid &grid, const DomainRange &local_range,
                           const DistributedVector &u,
                           const DistributedVector &v) {
  if (local_range.ii0 > local_range.ii1 || local_range.jj0 > local_range.jj1) {
    return 0.0;
  }
  double sum = 0.0;
  for (int j = local_range.jj0; j <= local_range.jj1; ++j) {
    int j_local = j - local_range.iy0;
    for (int i = local_range.ii0; i <= local_range.ii1; ++i) {
      int i_local = i - local_range.ix0;
      sum += u.at(i_local, j_local) * v.at(i_local, j_local);
    }
  }
  return sum * grid.h1 * grid.h2;
}

double inner_product_global(const Grid &grid, const DomainRange &local_range,
                            const DistributedVector &u,
                            const DistributedVector &v, MPI_Comm comm) {
  double local_sum = inner_product_local(grid, local_range, u, v);
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global_sum;
}

double norm_E_global(const Grid &grid, const DomainRange &local_range,
                     const DistributedVector &u, MPI_Comm comm) {
  double sq = inner_product_global(grid, local_range, u, u, comm);
  return std::sqrt(sq);
}

void apply_A_local(const Grid &grid, const DomainRange &local_range,
                   const DistributedVector &a_local,
                   const DistributedVector &b_local,
                   const DistributedVector &w_local,
                   DistributedVector &out_local) {
  out_local.fill(0.0);
  if (local_range.ii0 > local_range.ii1 || local_range.jj0 > local_range.jj1) {
    return;
  }
  double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
  double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
  for (int j = local_range.jj0; j <= local_range.jj1; ++j) {
    int j_local = j - local_range.iy0;
    for (int i = local_range.ii0; i <= local_range.ii1; ++i) {
      int i_local = i - local_range.ix0;
      double w_ij = w_local.at(i_local, j_local);
      double term_x = (a_local.at(i_local + 1, j_local) *
                           (w_local.at(i_local + 1, j_local) - w_ij) -
                       a_local.at(i_local, j_local) *
                           (w_ij - w_local.at(i_local - 1, j_local))) *
                      inv_h1_sq;
      double term_y = (b_local.at(i_local, j_local + 1) *
                           (w_local.at(i_local, j_local + 1) - w_ij) -
                       b_local.at(i_local, j_local) *
                           (w_ij - w_local.at(i_local, j_local - 1))) *
                      inv_h2_sq;
      out_local.at(i_local, j_local) = -(term_x + term_y);
    }
  }
}

void apply_D_inv_local(const Grid &grid, const DomainRange &local_range,
                       const DistributedVector &diag_local,
                       const DistributedVector &in_local,
                       DistributedVector &out_local) {
  (void)grid;
  out_local.fill(0.0);
  if (local_range.ii0 > local_range.ii1 || local_range.jj0 > local_range.jj1) {
    return;
  }
  for (int j = local_range.jj0; j <= local_range.jj1; ++j) {
    int j_local = j - local_range.iy0;
    for (int i = local_range.ii0; i <= local_range.ii1; ++i) {
      int i_local = i - local_range.ix0;
      double d = diag_local.at(i_local, j_local);
      if (d <= 0.0) {
        throw std::runtime_error(
            "Диагональный элемент предобуславливателя не положителен");
      }
      out_local.at(i_local, j_local) = in_local.at(i_local, j_local) / d;
    }
  }
}

void exchange_halo(MPI_Comm cart_comm, const Partition &partition,
                   const DomainRange &local_range, DistributedVector &vec) {
  (void)partition;
  (void)local_range;
  if (vec.halo == 0)
    return;

  int left = MPI_PROC_NULL, right = MPI_PROC_NULL, down = MPI_PROC_NULL,
      up = MPI_PROC_NULL;
  MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 1, 1, &down, &up);

  if (vec.local_ny > 0) {
    std::vector<double> send(vec.local_ny, 0.0), recv(vec.local_ny, 0.0);

    if (left != MPI_PROC_NULL) {
      for (int j = 0; j < vec.local_ny; ++j)
        send[static_cast<std::size_t>(j)] = vec.at(0, j);
      MPI_Sendrecv(send.data(), vec.local_ny, MPI_DOUBLE, left, 0, recv.data(),
                   vec.local_ny, MPI_DOUBLE, left, 1, cart_comm,
                   MPI_STATUS_IGNORE);
      for (int j = 0; j < vec.local_ny; ++j)
        vec.at_with_halo(-1, j) = recv[static_cast<std::size_t>(j)];
    } else {
      for (int j = 0; j < vec.local_ny; ++j)
        vec.at_with_halo(-1, j) = 0.0;
    }

    if (right != MPI_PROC_NULL) {
      for (int j = 0; j < vec.local_ny; ++j)
        send[static_cast<std::size_t>(j)] = vec.at(vec.local_nx - 1, j);
      MPI_Sendrecv(send.data(), vec.local_ny, MPI_DOUBLE, right, 1, recv.data(),
                   vec.local_ny, MPI_DOUBLE, right, 0, cart_comm,
                   MPI_STATUS_IGNORE);
      for (int j = 0; j < vec.local_ny; ++j)
        vec.at_with_halo(vec.local_nx, j) = recv[static_cast<std::size_t>(j)];
    } else {
      for (int j = 0; j < vec.local_ny; ++j)
        vec.at_with_halo(vec.local_nx, j) = 0.0;
    }
  }

  if (vec.local_nx > 0) {
    std::vector<double> send(vec.local_nx, 0.0), recv(vec.local_nx, 0.0);

    if (down != MPI_PROC_NULL) {
      for (int i = 0; i < vec.local_nx; ++i)
        send[static_cast<std::size_t>(i)] = vec.at(i, 0);
      MPI_Sendrecv(send.data(), vec.local_nx, MPI_DOUBLE, down, 2, recv.data(),
                   vec.local_nx, MPI_DOUBLE, down, 3, cart_comm,
                   MPI_STATUS_IGNORE);
      for (int i = 0; i < vec.local_nx; ++i)
        vec.at_with_halo(i, -1) = recv[static_cast<std::size_t>(i)];
    } else {
      for (int i = 0; i < vec.local_nx; ++i)
        vec.at_with_halo(i, -1) = 0.0;
    }

    if (up != MPI_PROC_NULL) {
      for (int i = 0; i < vec.local_nx; ++i)
        send[static_cast<std::size_t>(i)] = vec.at(i, vec.local_ny - 1);
      MPI_Sendrecv(send.data(), vec.local_nx, MPI_DOUBLE, up, 3, recv.data(),
                   vec.local_nx, MPI_DOUBLE, up, 2, cart_comm,
                   MPI_STATUS_IGNORE);
      for (int i = 0; i < vec.local_nx; ++i)
        vec.at_with_halo(i, vec.local_ny) = recv[static_cast<std::size_t>(i)];
    } else {
      for (int i = 0; i < vec.local_nx; ++i)
        vec.at_with_halo(i, vec.local_ny) = 0.0;
    }
  }
}

std::vector<double> gather_global_solution(const Grid &grid, MPI_Comm cart_comm,
                                           const Partition &partition,
                                           int block_id,
                                           const DistributedVector &w_local) {
  int cart_rank = 0;
  int cart_size = 1;
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Comm_size(cart_comm, &cart_size);

  int local_count = w_local.local_nx * w_local.local_ny;
  std::vector<double> local_buffer(static_cast<std::size_t>(local_count), 0.0);
  if (local_count > 0) {
    for (int j = 0; j < w_local.local_ny; ++j) {
      for (int i = 0; i < w_local.local_nx; ++i) {
        local_buffer[static_cast<std::size_t>(j * w_local.local_nx + i)] =
            w_local.at(i, j);
      }
    }
  }

  std::vector<int> counts;
  std::vector<int> displs;
  std::vector<int> block_ids;
  if (cart_rank == 0) {
    counts.resize(static_cast<std::size_t>(cart_size));
    displs.resize(static_cast<std::size_t>(cart_size));
    block_ids.resize(static_cast<std::size_t>(cart_size));
  }

  MPI_Gather(&local_count, 1, MPI_INT, cart_rank == 0 ? counts.data() : nullptr,
             1, MPI_INT, 0, cart_comm);
  MPI_Gather(&block_id, 1, MPI_INT, cart_rank == 0 ? block_ids.data() : nullptr,
             1, MPI_INT, 0, cart_comm);

  std::vector<double> recv_buffer;
  if (cart_rank == 0) {
    int total_entries = 0;
    for (int i = 0; i < cart_size; ++i) {
      displs[static_cast<std::size_t>(i)] = total_entries;
      total_entries += counts[static_cast<std::size_t>(i)];
    }
    recv_buffer.resize(static_cast<std::size_t>(total_entries));
  }

  MPI_Gatherv(local_buffer.data(), local_count, MPI_DOUBLE,
              cart_rank == 0 ? recv_buffer.data() : nullptr,
              cart_rank == 0 ? counts.data() : nullptr,
              cart_rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
              cart_comm);

  std::vector<double> global_solution;
  if (cart_rank == 0) {
    std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) *
                              static_cast<std::size_t>(grid.N + 1);
    global_solution.assign(total_nodes, 0.0);
    for (int r = 0; r < cart_size; ++r) {
      const auto &range = partition.ranges[static_cast<std::size_t>(
          block_ids[static_cast<std::size_t>(r)])];
      int nx = range.ix1 >= range.ix0 ? range.ix1 - range.ix0 + 1 : 0;
      int ny = range.iy1 >= range.iy0 ? range.iy1 - range.iy0 + 1 : 0;
      const double *src =
          recv_buffer.data() + displs[static_cast<std::size_t>(r)];
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          int global_i = range.ix0 + i;
          int global_j = range.iy0 + j;
          global_solution[grid.index(global_i, global_j)] = src[j * nx + i];
        }
      }
    }
  }

  return global_solution;
}

Result solve_problem(const Config &config, const Partition &partition,
                     const LocalProblemData &data, MPI_Comm cart_comm,
                     const DomainRange &local_range, int block_id,
                     bool collect_logs) {
  const Grid &grid = config.grid;
  DistributedVector w_local = make_distributed_vector(local_range, 1);
  w_local.fill(0.0);
  DistributedVector r_local = data.F;
  DistributedVector z_local = make_distributed_vector(local_range, 1);
  DistributedVector p_local = make_distributed_vector(local_range, 1);
  DistributedVector Ap_local = make_distributed_vector(local_range, 1);

  double rhs_norm = norm_E_global(grid, local_range, data.F, cart_comm);
  double rhs_norm_safe = rhs_norm == 0.0 ? 1.0 : rhs_norm;

  double residual_norm = norm_E_global(grid, local_range, r_local, cart_comm);
  double diff_norm = std::numeric_limits<double>::infinity();
  apply_D_inv_local(grid, local_range, data.diag, r_local, z_local);
  double rho =
      inner_product_global(grid, local_range, r_local, z_local, cart_comm);
  p_local = z_local;
  std::vector<IterationLogEntry> iteration_log;
  if (collect_logs) {
    iteration_log.reserve(static_cast<std::size_t>(config.maxIt));
  }
  std::size_t iter = 0;
  std::string stop_reason;

  while (iter < static_cast<std::size_t>(config.maxIt)) {
    if (std::abs(rho) < 1e-30) {
      stop_reason = "rho<=0";
      break;
    }

    exchange_halo(cart_comm, partition, local_range, p_local);
    apply_A_local(grid, local_range, data.a, data.b, p_local, Ap_local);

    double pAp =
        inner_product_global(grid, local_range, p_local, Ap_local, cart_comm);
    if (std::abs(pAp) < 1e-30) {
      stop_reason = "pAp<=0";
      break;
    }
    double alpha = rho / pAp;
    double p_norm_sq =
        inner_product_global(grid, local_range, p_local, p_local, cart_comm);
    if (p_norm_sq < 1e-30) {
      stop_reason = "direction_norm~0";
      break;
    }
    diff_norm = std::sqrt(p_norm_sq) * std::abs(alpha);

    if (!(local_range.ii0 > local_range.ii1 ||
          local_range.jj0 > local_range.jj1)) {
      for (int j = local_range.jj0; j <= local_range.jj1; ++j) {
        int j_local = j - local_range.iy0;
        for (int i = local_range.ii0; i <= local_range.ii1; ++i) {
          int i_local = i - local_range.ix0;
          w_local.at(i_local, j_local) += alpha * p_local.at(i_local, j_local);
          r_local.at(i_local, j_local) -= alpha * Ap_local.at(i_local, j_local);
        }
      }
    }

    residual_norm = norm_E_global(grid, local_range, r_local, cart_comm);
    ++iter;
    if (collect_logs) {
      iteration_log.push_back({iter, residual_norm, alpha});
    }

    if (diff_norm < config.delta) {
      stop_reason = "diff<delta";
      break;
    }
    if (residual_norm / rhs_norm_safe < config.tau) {
      stop_reason = "relative_residual<tau";
      break;
    }

    apply_D_inv_local(grid, local_range, data.diag, r_local, z_local);
    double rho_next =
        inner_product_global(grid, local_range, r_local, z_local, cart_comm);
    if (std::abs(rho_next) < 1e-30) {
      stop_reason = "rho<=0";
      break;
    }
    double beta = rho_next / rho;
    if (!(local_range.ii0 > local_range.ii1 ||
          local_range.jj0 > local_range.jj1)) {
      for (int j = local_range.jj0; j <= local_range.jj1; ++j) {
        int j_local = j - local_range.iy0;
        for (int i = local_range.ii0; i <= local_range.ii1; ++i) {
          int i_local = i - local_range.ix0;
          p_local.at(i_local, j_local) = z_local.at(i_local, j_local) +
                                         beta * p_local.at(i_local, j_local);
        }
      }
    }
    rho = rho_next;
  }

  if (stop_reason.empty()) {
    if (iter >= static_cast<std::size_t>(config.maxIt)) {
      stop_reason = "maxIt";
    } else {
      stop_reason = "stagnation";
    }
  }

  std::vector<double> global_solution =
      gather_global_solution(grid, cart_comm, partition, block_id, w_local);

  Result result;
  result.solution = std::move(global_solution);
  result.iterations = iter;
  result.residual_norm = residual_norm;
  result.diff_norm = diff_norm;
  result.rhs_norm = rhs_norm;
  result.stop_reason = stop_reason;
  if (collect_logs) {
    result.iteration_log = std::move(iteration_log);
  }
  return result;
}

void write_solution_csv(const std::string &filename, const Grid &grid,
                        const std::vector<double> &w) {
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Не удалось открыть файл " + filename);
  }
  out << std::scientific << std::setprecision(12);
  out << "x,y,w\n";
  for (int j = 0; j <= grid.N; ++j) {
    for (int i = 0; i <= grid.M; ++i) {
      std::size_t idx = grid.index(i, j);
      out << grid.x(i) << ',' << grid.y(j) << ',' << w[idx] << '\n';
    }
  }
}

std::vector<MaskEntry> build_mask_entries(const Grid &grid) {
  std::vector<MaskEntry> entries;
  entries.reserve(static_cast<std::size_t>(grid.M + 1) *
                  static_cast<std::size_t>(grid.N + 1));
  for (int j = 0; j <= grid.N; ++j) {
    double y_val = grid.y(j);
    for (int i = 0; i <= grid.M; ++i) {
      double x_val = grid.x(i);
      bool inside = in_D(x_val, y_val);
      entries.push_back(MaskEntry{x_val, y_val, inside});
    }
  }
  return entries;
}

std::vector<int> make_blocks(int total_nodes, int parts) {
  if (parts <= 0) {
    return {};
  }
  std::vector<int> result(static_cast<std::size_t>(parts), 0);
  int q = total_nodes / parts;
  int r = total_nodes % parts;
  for (int i = 0; i < parts; ++i) {
    result[static_cast<std::size_t>(i)] = q + (i < r ? 1 : 0);
  }
  return result;
}

Partition build_partition_with_ranges(int M, int N, int Px, int Py) {
  Partition partition;
  partition.Px = Px;
  partition.Py = Py;
  if (Px <= 0 || Py <= 0) {
    return partition;
  }

  std::vector<int> nx = make_blocks(M + 1, Px);
  std::vector<int> ny = make_blocks(N + 1, Py);
  if (nx.size() != static_cast<std::size_t>(Px) ||
      ny.size() != static_cast<std::size_t>(Py)) {
    return partition;
  }

  partition.blocks.reserve(static_cast<std::size_t>(Px * Py));
  partition.ranges.reserve(static_cast<std::size_t>(Px * Py));

  std::vector<int> xcuts(static_cast<std::size_t>(Px + 1), 0);
  std::vector<int> ycuts(static_cast<std::size_t>(Py + 1), 0);
  for (int i = 0; i < Px; ++i) {
    xcuts[static_cast<std::size_t>(i + 1)] =
        xcuts[static_cast<std::size_t>(i)] + nx[static_cast<std::size_t>(i)];
  }
  for (int j = 0; j < Py; ++j) {
    ycuts[static_cast<std::size_t>(j + 1)] =
        ycuts[static_cast<std::size_t>(j)] + ny[static_cast<std::size_t>(j)];
  }

  for (int j = 0; j < Py; ++j) {
    for (int i = 0; i < Px; ++i) {
      DomainBlock block{nx[static_cast<std::size_t>(i)],
                        ny[static_cast<std::size_t>(j)]};
      partition.blocks.push_back(block);

      DomainRange range{};
      range.ix0 = xcuts[static_cast<std::size_t>(i)];
      range.ix1 = xcuts[static_cast<std::size_t>(i + 1)] - 1;
      range.iy0 = ycuts[static_cast<std::size_t>(j)];
      range.iy1 = ycuts[static_cast<std::size_t>(j + 1)] - 1;

      range.ii0 = std::max(1, range.ix0);
      range.ii1 = std::min(M - 1, range.ix1);
      range.jj0 = std::max(1, range.iy0);
      range.jj1 = std::min(N - 1, range.iy1);

      range.ai0 = std::max(1, range.ix0);
      range.ai1 = std::min(M, range.ix1);
      range.aj0 = std::max(1, range.iy0);
      range.aj1 = std::min(N - 1, range.iy1);

      range.bi0 = std::max(1, range.ix0);
      range.bi1 = std::min(M - 1, range.ix1);
      range.bj0 = std::max(1, range.iy0);
      range.bj1 = std::min(N, range.iy1);

      partition.ranges.push_back(range);
    }
  }

  return partition;
}

PartitionCheckResult check_partition(int M, int N, int Px, int Py, double rmin,
                                     double rmax) {
  if (M < 1 || N < 1) {
    throw std::runtime_error("Размеры сетки должны быть >=1");
  }
  if (Px < 1 || Py < 1) {
    throw std::runtime_error("Px и Py должны быть положительными");
  }
  if (Px > M + 1) {
    throw std::runtime_error("Px превышает M+1");
  }
  if (Py > N + 1) {
    throw std::runtime_error("Py превышает N+1");
  }

  PartitionCheckResult result;
  result.partition = build_partition_with_ranges(M, N, Px, Py);
  std::size_t expected_size = static_cast<std::size_t>(Px * Py);
  if (result.partition.blocks.size() != expected_size ||
      result.partition.ranges.size() != expected_size) {
    throw std::runtime_error("Не удалось построить полное разбиение");
  }

  const double tol = 1e-12;
  int min_nx = result.partition.blocks.front().nodes_x;
  int max_nx = result.partition.blocks.front().nodes_x;
  int min_ny = result.partition.blocks.front().nodes_y;
  int max_ny = result.partition.blocks.front().nodes_y;
  bool ratio_violation = false;

  std::string report;
  report += "Сетка " + std::to_string(M) + "x" + std::to_string(N) +
            ", разбиение " + std::to_string(Px) + "x" + std::to_string(Py) +
            "\n";
  report += "Домены:\n";

  for (int j = 0; j < Py; ++j) {
    for (int i = 0; i < Px; ++i) {
      std::size_t idx = static_cast<std::size_t>(j * Px + i);
      const auto &block = result.partition.blocks[idx];
      const auto &range = result.partition.ranges[idx];
      min_nx = std::min(min_nx, block.nodes_x);
      max_nx = std::max(max_nx, block.nodes_x);
      min_ny = std::min(min_ny, block.nodes_y);
      max_ny = std::max(max_ny, block.nodes_y);
      double ratio = block.nodes_y == 0
                         ? std::numeric_limits<double>::infinity()
                         : static_cast<double>(block.nodes_x) /
                               static_cast<double>(block.nodes_y);
      bool ratio_ok = ratio >= rmin - tol && ratio <= rmax + tol;
      char ratio_buf[64];
      std::snprintf(ratio_buf, sizeof(ratio_buf), "%.6f", ratio);
      report += "  (" + std::to_string(i) + "," + std::to_string(j) +
                "): nx=" + std::to_string(block.nodes_x) +
                " ny=" + std::to_string(block.nodes_y) + " nx/ny=" + ratio_buf +
                " диапазоны i=[" + std::to_string(range.ix0) + ".." +
                std::to_string(range.ix1) + "] j=[" +
                std::to_string(range.iy0) + ".." + std::to_string(range.iy1) +
                "]";
      if (!ratio_ok) {
        report += " [нарушение отношения]";
        ratio_violation = true;
      }
      report += "\n";
    }
  }

  if (ratio_violation) {
    throw std::runtime_error("Нарушено условие отношения узлов");
  }
  if (max_nx - min_nx > 1) {
    throw std::runtime_error("Разброс размеров по x превышает 1");
  }
  if (max_ny - min_ny > 1) {
    throw std::runtime_error("Разброс размеров по y превышает 1");
  }

  report += "Условия выполнены: отношение узлов в пределах [" +
            std::to_string(rmin) + "," + std::to_string(rmax) +
            "] и разброс узлов <= 1." + "\n";
  result.report = std::move(report);
  return result;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::string output_dir = "output";
  if (world_rank == 0) {
    ensure_directory(output_dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  int exit_code = EXIT_SUCCESS;
  std::string error_message;
  std::string current_error_log_path = output_dir + "/mpi_error.log";
  MPI_Comm cart_comm = MPI_COMM_NULL;

  try {
    if (B1 <= A1 || B2 <= A2) {
      throw std::runtime_error("Диапазоны по x или y заданы некорректно");
    }

    int thread_count = parse_thread_argument(argc, argv, 1);
    (void)thread_count;
    std::string output_prefix = output_dir + "/mpi_";
    if (world_rank == 0) {
      current_error_log_path = output_prefix + "error.log";
    }

    const auto grid_dims = parse_grid_arguments(argc, argv, default_grids(0));
    int M = grid_dims.first;
    int N = grid_dims.second;
    if (M < 2 || N < 2) {
      throw std::runtime_error("Сетке нужны M,N >= 2");
    }

    auto used_dims = get_cart_dims(world_size);
    int Px_used = used_dims.first;
    int Py_used = used_dims.second;

    auto part_check = check_partition(M, N, Px_used, Py_used);
    Partition partition = std::move(part_check.partition);
    if (static_cast<int>(partition.ranges.size()) != Px_used * Py_used) {
      throw std::runtime_error("partition size != Px_used*Py_used");
    }

    int dims[2] = {Px_used, Py_used};
    int periods[2] = {0, 0};
    if (MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm) !=
            MPI_SUCCESS ||
        cart_comm == MPI_COMM_NULL) {
      throw std::runtime_error("MPI_Cart_create failed");
    }
    int cart_rank = 0;
    MPI_Comm_rank(cart_comm, &cart_rank);
    int coords[2] = {0, 0};
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    int block_id = coords[1] * Px_used + coords[0];
    if (block_id < 0 || block_id >= static_cast<int>(partition.ranges.size())) {
      throw std::runtime_error("Не валидный block_id");
    }
    const DomainRange &local_range =
        partition.ranges[static_cast<std::size_t>(block_id)];

    std::vector<SummaryEntry> summary;
    std::vector<RuntimeEntry> runtime_entries;
    auto start_time = std::chrono::steady_clock::now();

    Grid grid(A1, B1, A2, B2, M, N);
    double h = std::max(grid.h1, grid.h2);
    double epsilon = h * h;
    LocalProblemData data =
        build_local_problem(grid, epsilon, partition, local_range);
    long long maxIt = static_cast<long long>((M - 1) * (N - 1));
    Config config{grid, DELTA, TAU, maxIt, epsilon};

    auto grid_start = std::chrono::steady_clock::now();
    Result result = solve_problem(config, partition, data, cart_comm,
                                  local_range, block_id, world_rank == 0);
    auto grid_end = std::chrono::steady_clock::now();
    double grid_seconds =
        std::chrono::duration<double>(grid_end - grid_start).count();
    if (world_rank == 0) {
      runtime_entries.push_back({M, N, grid_seconds});
    }

    if (world_rank == 0) {
      std::string suffix = "_p" + std::to_string(world_size) + "_" +
                           std::to_string(M) + "x" + std::to_string(N);
      std::string solution_path = output_prefix + "solution" + suffix + ".csv";
      std::string meta_path = output_prefix + "meta" + suffix + ".txt";
      std::string run_log_path = output_prefix + "run" + suffix + ".log";
      std::string mask_path = output_prefix + "mask" + suffix + ".csv";
      std::string partition_report_path =
          output_prefix + "partition" + suffix + ".txt";

      write_partition_log(partition_report_path, part_check.report);
      write_solution_csv(solution_path, grid, result.solution);
      write_meta_txt(meta_path, grid.A1, grid.B1, grid.A2, grid.B2, grid.M,
                     grid.N, grid.h1, grid.h2, epsilon, config.delta,
                     config.tau, config.maxIt, result.iterations,
                     result.residual_norm, result.diff_norm, result.rhs_norm,
                     result.stop_reason);
      write_run_log(run_log_path, M, N, result.iteration_log, result.iterations,
                    result.residual_norm, result.diff_norm);
      write_mask_csv(mask_path, build_mask_entries(grid));

      summary.push_back({M, N, result.residual_norm});
      write_summary_txt(output_prefix + "summary" + suffix + ".txt", summary);
      auto end_time = std::chrono::steady_clock::now();
      double total_seconds =
          std::chrono::duration<double>(end_time - start_time).count();
      write_runtime(output_prefix + "runtime" + suffix + ".txt",
                    runtime_entries, total_seconds);
    }
  } catch (const std::exception &ex) {
    error_message = ex.what();
    exit_code = EXIT_FAILURE;
  }

  if (cart_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm);
    cart_comm = MPI_COMM_NULL;
  }

  int global_exit_code = 0;
  MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  exit_code = global_exit_code;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  if (world_rank == 0 && exit_code != EXIT_SUCCESS) {
    write_error_log(current_error_log_path, error_message.empty()
                                                ? "Неизвестная ошибка"
                                                : error_message);
  }

  return exit_code;
}
