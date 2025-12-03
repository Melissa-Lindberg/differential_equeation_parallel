#include "hybrid.h"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

bool in_D(double x, double y)
{
    Point p{x, y};
    for (const auto &hp : domain_halfplanes()) {
        if (!hp.contains(p)) {
            return false;
        }
    }
    return true;
}

Point intersect_segment_with_plane(const Point &p0, const Point &p1,
                                   const HalfPlane &hp)
{
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
                                             const HalfPlane &hp)
{
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

double polygon_area(const std::vector<Point> &poly)
{
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

double cell_area_in_D(double x0, double x1, double y0, double y1)
{
    std::vector<Point> poly = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    for (const auto &hp : domain_halfplanes()) {
        poly = clip_polygon_by_halfplane(poly, hp);
        if (poly.empty()) {
            break;
        }
    }
    return polygon_area(poly);
}

double segment_length_in_D(const Point &p0, const Point &p1)
{
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

ProblemData build_problem(const Grid &grid, double epsilon,
                          const Partition &partition)
{
    ProblemData data;
    std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) *
                              static_cast<std::size_t>(grid.N + 1);
    data.a.assign(total_nodes, 0.0);
    data.b.assign(total_nodes, 0.0);
    data.F.assign(total_nodes, 0.0);
    data.diag.assign(total_nodes, 0.0);

    const auto &ranges = partition.ranges;

    // коэффициенты a_{i,j} для вертикальных граней
    for (const auto &d : ranges) {
        if (d.ai0 > d.ai1 || d.aj0 > d.aj1) {
            continue;
        }
        for (int i = d.ai0; i <= d.ai1; ++i) {
            double xmid = grid.x_mid(i);
            for (int j = d.aj0; j <= d.aj1; ++j) {
                Point p0{xmid, grid.y(j) - 0.5 * grid.h2};
                Point p1{xmid, grid.y(j) + 0.5 * grid.h2};
                double len = segment_length_in_D(p0, p1);
                double frac = len / grid.h2;
                if (frac < 0.0) {
                    frac = 0.0;
                }
                if (frac > 1.0) {
                    frac = 1.0;
                }
                double coeff = frac + (1.0 - frac) / epsilon;
                data.a[grid.index(i, j)] = coeff;
            }
        }
    }

    // коэффициенты b_{i,j} для горизонтальных граней
    for (const auto &d : ranges) {
        if (d.bi0 > d.bi1 || d.bj0 > d.bj1) {
            continue;
        }
        for (int i = d.bi0; i <= d.bi1; ++i) {
            for (int j = d.bj0; j <= d.bj1; ++j) {
                Point p0{grid.x(i) - 0.5 * grid.h1, grid.y_mid(j)};
                Point p1{grid.x(i) + 0.5 * grid.h1, grid.y_mid(j)};
                double len = segment_length_in_D(p0, p1);
                double frac = len / grid.h1;
                if (frac < 0.0) {
                    frac = 0.0;
                }
                if (frac > 1.0) {
                    frac = 1.0;
                }
                double coeff = frac + (1.0 - frac) / epsilon;
                data.b[grid.index(i, j)] = coeff;
            }
        }
    }

    // правая часть F_{i,j}
    double cell_area = grid.h1 * grid.h2;
    for (const auto &d : ranges) {
        if (d.ii0 > d.ii1 || d.jj0 > d.jj1) {
            continue;
        }
        for (int i = d.ii0; i <= d.ii1; ++i) {
            for (int j = d.jj0; j <= d.jj1; ++j) {
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
                data.F[grid.index(i, j)] = ratio;
            }
        }
    }

    // диагональный предобуславливатель
    double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
    double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
    for (const auto &d : ranges) {
        if (d.ii0 > d.ii1 || d.jj0 > d.jj1) {
            continue;
        }
        for (int i = d.ii0; i <= d.ii1; ++i) {
            for (int j = d.jj0; j <= d.jj1; ++j) {
                std::size_t idx = grid.index(i, j);
                double val = 0.0;
                val +=
                    (data.a[grid.index(i + 1, j)] + data.a[grid.index(i, j)]) *
                    inv_h1_sq;
                val +=
                    (data.b[grid.index(i, j + 1)] + data.b[grid.index(i, j)]) *
                    inv_h2_sq;
                data.diag[idx] = val;
            }
        }
    }

    return data;
}

double inner_product(const Grid &grid, const Partition &partition,
                     const std::vector<double> &u, const std::vector<double> &v)
{
    double sum = 0.0;
    for (const auto &d : partition.ranges) {
        if (d.ii0 > d.ii1 || d.jj0 > d.jj1) {
            continue;
        }
        for (int j = d.jj0; j <= d.jj1; ++j) {
            for (int i = d.ii0; i <= d.ii1; ++i) {
                std::size_t idx = grid.index(i, j);
                sum += u[idx] * v[idx];
            }
        }
    }
    return sum * grid.h1 * grid.h2;
}

double norm_E(const Grid &grid, const Partition &partition,
              const std::vector<double> &u)
{
    double sq = inner_product(grid, partition, u, u);
    return std::sqrt(sq);
}

void apply_A(const Grid &grid, const Partition &partition,
             const std::vector<double> &a, const std::vector<double> &b,
             const std::vector<double> &w, std::vector<double> &out)
{
    std::fill(out.begin(), out.end(), 0.0);
    double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
    double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
    for (const auto &d : partition.ranges) {
        if (d.ii0 > d.ii1 || d.jj0 > d.jj1) {
            continue;
        }
        for (int j = d.jj0; j <= d.jj1; ++j) {
            for (int i = d.ii0; i <= d.ii1; ++i) {
                std::size_t idx = grid.index(i, j);
                double w_ij = w[idx];
                double term_x =
                    (a[grid.index(i + 1, j)] *
                         (w[grid.index(i + 1, j)] - w_ij) -
                     a[grid.index(i, j)] * (w_ij - w[grid.index(i - 1, j)])) *
                    inv_h1_sq;
                double term_y =
                    (b[grid.index(i, j + 1)] *
                         (w[grid.index(i, j + 1)] - w_ij) -
                     b[grid.index(i, j)] * (w_ij - w[grid.index(i, j - 1)])) *
                    inv_h2_sq;
                out[idx] = -(term_x + term_y);
            }
        }
    }
}

void apply_D_inv(const Grid &grid, const Partition &partition,
                 const std::vector<double> &diag, const std::vector<double> &in,
                 std::vector<double> &out)
{
    std::fill(out.begin(), out.end(), 0.0);
    for (const auto &d_range : partition.ranges) {
        if (d_range.ii0 > d_range.ii1 || d_range.jj0 > d_range.jj1) {
            continue;
        }
        for (int j = d_range.jj0; j <= d_range.jj1; ++j) {
            for (int i = d_range.ii0; i <= d_range.ii1; ++i) {
                std::size_t idx = grid.index(i, j);
                double d = diag[idx];
                if (d <= 0.0) {
                    throw std::runtime_error(
                        "Диагональный элемент предобуславливателя не "
                        "положителен");
                }
                out[idx] = in[idx] / d;
            }
        }
    }
}

Result solve_problem(const Config &config, const Partition &partition,
                     const ProblemData &data)
{
    const Grid &grid = config.grid;
    std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) *
                              static_cast<std::size_t>(grid.N + 1);
    std::vector<double> w(total_nodes, 0.0);
    std::vector<double> r = data.F;
    std::vector<double> z(total_nodes, 0.0);
    std::vector<double> p(total_nodes, 0.0);
    std::vector<double> Ap(total_nodes, 0.0);

    double rhs_norm = norm_E(grid, partition, data.F);
    double rhs_norm_safe = rhs_norm == 0.0 ? 1.0 : rhs_norm;

    double residual_norm = norm_E(grid, partition, r);
    double diff_norm = std::numeric_limits<double>::infinity();
    std::vector<IterationLogEntry> iteration_log;
    std::size_t iter = 0;
    std::string stop_reason;

    while (iter < static_cast<std::size_t>(config.maxIt)) {
        apply_D_inv(grid, partition, data.diag, r, z);
        p = z;
        apply_A(grid, partition, data.a, data.b, p, Ap);

        double numerator = inner_product(grid, partition, z, r);
        double denominator = inner_product(grid, partition, Ap, p);
        if (std::abs(denominator) < 1e-30) {
            stop_reason = "denominator<=0";
            break;
        }
        double alpha = numerator / denominator;
        double p_norm_sq = inner_product(grid, partition, p, p);
        if (p_norm_sq < 1e-30) {
            stop_reason = "direction_norm~0";
            break;
        }
        diff_norm = std::sqrt(p_norm_sq) * std::abs(alpha);

        for (const auto &d : partition.ranges) {
            if (d.ii0 > d.ii1 || d.jj0 > d.jj1) {
                continue;
            }
            for (int j = d.jj0; j <= d.jj1; ++j) {
                for (int i = d.ii0; i <= d.ii1; ++i) {
                    std::size_t idx = grid.index(i, j);
                    w[idx] += alpha * p[idx];
                    r[idx] -= alpha * Ap[idx];
                }
            }
        }

        residual_norm = norm_E(grid, partition, r);
        ++iter;
        iteration_log.push_back({iter, residual_norm, alpha});

        if (diff_norm < config.delta) {
            stop_reason = "diff<delta";
            break;
        }
        if (residual_norm / rhs_norm_safe < config.tau) {
            stop_reason = "relative_residual<tau";
            break;
        }
    }

    if (stop_reason.empty()) {
        if (iter >= static_cast<std::size_t>(config.maxIt)) {
            stop_reason = "maxIt";
        } else {
            stop_reason = "stagnation";
        }
    }

    Result result;
    result.solution = std::move(w);
    result.iterations = iter;
    result.residual_norm = residual_norm;
    result.diff_norm = diff_norm;
    result.rhs_norm = rhs_norm;
    result.stop_reason = stop_reason;
    result.iteration_log = std::move(iteration_log);
    return result;
}

namespace {
struct LocalProblemSlices {
    LocalView view;
    int a_stride = 0;
    int b_stride = 0;
    int plain_stride = 0;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> F;
    std::vector<double> diag;
};

LocalView make_local_view(const DomainRange &range)
{
    LocalView view;
    view.ii0 = range.ii0;
    view.ii1 = range.ii1;
    view.jj0 = range.jj0;
    view.jj1 = range.jj1;
    if (range.ii0 > range.ii1 || range.jj0 > range.jj1) {
        view.nx = 0;
        view.ny = 0;
    } else {
        view.nx = range.ii1 - range.ii0 + 1;
        view.ny = range.jj1 - range.jj0 + 1;
    }
    view.halo_x = 1;
    view.halo_y = 1;
    return view;
}

inline std::size_t halo_index(const LocalView &view, int li, int lj)
{
    return static_cast<std::size_t>(lj) *
               static_cast<std::size_t>(view.nx + 2) +
           static_cast<std::size_t>(li);
}

inline int to_local_i(const LocalView &view, int global_i)
{
    return global_i - view.ii0 + view.halo_x;
}

inline int to_local_j(const LocalView &view, int global_j)
{
    return global_j - view.jj0 + view.halo_y;
}

inline std::size_t plain_index(const LocalProblemSlices &problem, int gi,
                               int gj)
{
    return static_cast<std::size_t>(gj - problem.view.jj0) *
               static_cast<std::size_t>(problem.plain_stride) +
           static_cast<std::size_t>(gi - problem.view.ii0);
}

inline std::size_t a_index(const LocalProblemSlices &problem, int gi, int gj)
{
    return static_cast<std::size_t>(gj - problem.view.jj0) *
               static_cast<std::size_t>(problem.a_stride) +
           static_cast<std::size_t>(gi - problem.view.ii0);
}

inline std::size_t b_index(const LocalProblemSlices &problem, int gi, int gj)
{
    return static_cast<std::size_t>(gj - problem.view.jj0) *
               static_cast<std::size_t>(problem.b_stride) +
           static_cast<std::size_t>(gi - problem.view.ii0);
}

void allocate_halo_buffers(Dist &dist)
{
    const int nx = dist.view.nx;
    const int ny = dist.view.ny;
    dist.halos.send_west.assign(static_cast<std::size_t>(ny), 0.0);
    dist.halos.recv_west.assign(static_cast<std::size_t>(ny), 0.0);
    dist.halos.send_east.assign(static_cast<std::size_t>(ny), 0.0);
    dist.halos.recv_east.assign(static_cast<std::size_t>(ny), 0.0);
    dist.halos.send_south.assign(static_cast<std::size_t>(nx), 0.0);
    dist.halos.recv_south.assign(static_cast<std::size_t>(nx), 0.0);
    dist.halos.send_north.assign(static_cast<std::size_t>(nx), 0.0);
    dist.halos.recv_north.assign(static_cast<std::size_t>(nx), 0.0);
}

LocalProblemSlices build_problem_local(const Grid &grid, double epsilon,
                                       const LocalView &view)
{
    if (view.nx <= 0 || view.ny <= 0) {
        throw std::runtime_error("Локальная область пустая");
    }

    LocalProblemSlices local;
    local.view = view;
    local.a_stride = view.nx + 1;
    local.b_stride = view.nx;
    local.plain_stride = view.nx;
    local.a.assign(static_cast<std::size_t>(local.a_stride) *
                       static_cast<std::size_t>(view.ny),
                   0.0);
    local.b.assign(static_cast<std::size_t>(local.b_stride) *
                       static_cast<std::size_t>(view.ny + 1),
                   0.0);
    local.F.assign(static_cast<std::size_t>(local.plain_stride) *
                       static_cast<std::size_t>(view.ny),
                   0.0);
    local.diag = local.F;

#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        for (int i = 0; i <= view.nx; ++i) {
            int gi = view.ii0 + i;
            int gj = view.jj0 + j;
            double xmid = grid.x_mid(gi);
            Point p0{xmid, grid.y(gj) - 0.5 * grid.h2};
            Point p1{xmid, grid.y(gj) + 0.5 * grid.h2};
            double len = segment_length_in_D(p0, p1);
            double frac = len / grid.h2;
            if (frac < 0.0) {
                frac = 0.0;
            }
            if (frac > 1.0) {
                frac = 1.0;
            }
            local.a[a_index(local, gi, gj)] = frac + (1.0 - frac) / epsilon;
        }
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j <= view.ny; ++j) {
        for (int i = 0; i < view.nx; ++i) {
            int gi = view.ii0 + i;
            int gj = view.jj0 + j;
            Point p0{grid.x(gi) - 0.5 * grid.h1, grid.y_mid(gj)};
            Point p1{grid.x(gi) + 0.5 * grid.h1, grid.y_mid(gj)};
            double len = segment_length_in_D(p0, p1);
            double frac = len / grid.h1;
            if (frac < 0.0) {
                frac = 0.0;
            }
            if (frac > 1.0) {
                frac = 1.0;
            }
            local.b[b_index(local, gi, gj)] = frac + (1.0 - frac) / epsilon;
        }
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        for (int i = 0; i < view.nx; ++i) {
            int gi = view.ii0 + i;
            int gj = view.jj0 + j;
            double x_left = grid.x(gi) - 0.5 * grid.h1;
            double x_right = grid.x(gi) + 0.5 * grid.h1;
            double y_bottom = grid.y(gj) - 0.5 * grid.h2;
            double y_top = grid.y(gj) + 0.5 * grid.h2;
            double area = cell_area_in_D(x_left, x_right, y_bottom, y_top);
            double ratio = area / (grid.h1 * grid.h2);
            if (ratio < 1e-12) {
                ratio = 0.0;
            }
            if (ratio > 1.0 - 1e-12) {
                ratio = 1.0;
            }
            local.F[plain_index(local, gi, gj)] = ratio;
        }
    }

    double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
    double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        for (int i = 0; i < view.nx; ++i) {
            int gi = view.ii0 + i;
            int gj = view.jj0 + j;
            double val = 0.0;
            val += (local.a[a_index(local, gi + 1, gj)] +
                    local.a[a_index(local, gi, gj)]) *
                   inv_h1_sq;
            val += (local.b[b_index(local, gi, gj + 1)] +
                    local.b[b_index(local, gi, gj)]) *
                   inv_h2_sq;
            local.diag[plain_index(local, gi, gj)] = val;
        }
    }

    return local;
}

double inner_product_plain_local(const LocalProblemSlices &problem,
                                 const std::vector<double> &lhs,
                                 const std::vector<double> &rhs)
{
    double sum = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : sum) schedule(static)
    for (int j = 0; j < problem.view.ny; ++j) {
        std::size_t row_offset = static_cast<std::size_t>(j) *
                                 static_cast<std::size_t>(problem.plain_stride);
        for (int i = 0; i < problem.view.nx; ++i) {
            sum += lhs[row_offset + static_cast<std::size_t>(i)] *
                   rhs[row_offset + static_cast<std::size_t>(i)];
        }
    }
    return sum;
}

double reduce_dot_product(double local_sum, const Grid &grid, MPI_Comm comm)
{
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum * grid.h1 * grid.h2;
}

double norm_from_local_sum(double local_sum, const Grid &grid, MPI_Comm comm)
{
    double scaled = reduce_dot_product(local_sum, grid, comm);
    return std::sqrt(scaled);
}

struct GatherLayout {
    std::vector<int> counts;
    std::vector<int> displs;
    int total = 0;
};

GatherLayout build_gather_layout(const Partition &partition)
{
    GatherLayout layout;
    const std::size_t ranks = partition.ranges.size();
    layout.counts.resize(ranks);
    layout.displs.resize(ranks);
    int offset = 0;
    for (std::size_t idx = 0; idx < ranks; ++idx) {
        const DomainRange &range = partition.ranges[idx];
        int nx = range.ii1 >= range.ii0 ? range.ii1 - range.ii0 + 1 : 0;
        int ny = range.jj1 >= range.jj0 ? range.jj1 - range.jj0 + 1 : 0;
        int count = (nx > 0 && ny > 0) ? nx * ny : 0;
        layout.counts[idx] = count;
        layout.displs[idx] = offset;
        offset += count;
    }
    layout.total = offset;
    return layout;
}

std::vector<double> gather_global_solution(const Grid &grid,
                                           const Partition &partition,
                                           const Dist &dist,
                                           const std::vector<double> &w)
{
    const LocalView &view = dist.view;
    int local_count = view.nx * view.ny;
    std::vector<double> local_buffer(static_cast<std::size_t>(local_count),
                                     0.0);
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        int lj = j + view.halo_y;
        std::size_t row_offset =
            static_cast<std::size_t>(j) * static_cast<std::size_t>(view.nx);
        for (int i = 0; i < view.nx; ++i) {
            int li = i + view.halo_x;
            local_buffer[row_offset + static_cast<std::size_t>(i)] =
                w[halo_index(view, li, lj)];
        }
    }

    std::vector<int> recv_counts;
    std::vector<int> displs;
    int total_count = 0;
    if (dist.world_rank == 0) {
        GatherLayout layout = build_gather_layout(partition);
        recv_counts = std::move(layout.counts);
        displs = std::move(layout.displs);
        total_count = layout.total;
    }

    std::vector<double> gathered;
    if (dist.world_rank == 0) {
        gathered.assign(static_cast<std::size_t>(total_count), 0.0);
    }

    MPI_Gatherv(local_buffer.data(), local_count, MPI_DOUBLE,
                dist.world_rank == 0 ? gathered.data() : nullptr,
                dist.world_rank == 0 ? recv_counts.data() : nullptr,
                dist.world_rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                dist.cart_comm);

    std::vector<double> global_solution;
    if (dist.world_rank == 0) {
        std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) *
                                  static_cast<std::size_t>(grid.N + 1);
        global_solution.assign(total_nodes, 0.0);
        for (std::size_t rank = 0; rank < partition.ranges.size(); ++rank) {
            const DomainRange &range = partition.ranges[rank];
            int nx = range.ii1 >= range.ii0 ? range.ii1 - range.ii0 + 1 : 0;
            int ny = range.jj1 >= range.jj0 ? range.jj1 - range.jj0 + 1 : 0;
            if (nx <= 0 || ny <= 0) {
                continue;
            }
            const double *chunk =
                gathered.data() + static_cast<std::size_t>(displs[rank]);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int gi = range.ii0 + i;
                    int gj = range.jj0 + j;
                    global_solution[grid.index(gi, gj)] =
                        chunk[static_cast<std::size_t>(j) *
                                  static_cast<std::size_t>(nx) +
                              static_cast<std::size_t>(i)];
                }
            }
        }
    }
    return global_solution;
}

void exchange_halos(Dist &dist, std::vector<double> &field)
{
    const LocalView &view = dist.view;
    const int stride = view.nx + 2;
    const int nx = view.nx;
    const int ny = view.ny;
    const int horizontal_tag = 100;
    const int vertical_tag = 200;
    std::vector<MPI_Request> requests;
    requests.reserve(8);

    if (dist.west == MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j) {
            field[halo_index(view, 0, j)] = 0.0;
        }
    }
    if (dist.east == MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j) {
            field[halo_index(view, nx + 1, j)] = 0.0;
        }
    }
    if (dist.south == MPI_PROC_NULL) {
        for (int i = 1; i <= nx; ++i) {
            field[halo_index(view, i, 0)] = 0.0;
        }
    }
    if (dist.north == MPI_PROC_NULL) {
        for (int i = 1; i <= nx; ++i) {
            field[halo_index(view, i, ny + 1)] = 0.0;
        }
    }

    if (dist.west != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            int lj = j + 1;
            dist.halos.send_west[static_cast<std::size_t>(j)] =
                field[halo_index(view, 1, lj)];
        }
        MPI_Request req;
        MPI_Irecv(dist.halos.recv_west.data(), ny, MPI_DOUBLE, dist.west,
                  horizontal_tag, dist.cart_comm, &req);
        requests.push_back(req);
        MPI_Isend(dist.halos.send_west.data(), ny, MPI_DOUBLE, dist.west,
                  horizontal_tag, dist.cart_comm, &req);
        requests.push_back(req);
    }

    if (dist.east != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            int lj = j + 1;
            dist.halos.send_east[static_cast<std::size_t>(j)] =
                field[halo_index(view, nx, lj)];
        }
        MPI_Request req;
        MPI_Irecv(dist.halos.recv_east.data(), ny, MPI_DOUBLE, dist.east,
                  horizontal_tag, dist.cart_comm, &req);
        requests.push_back(req);
        MPI_Isend(dist.halos.send_east.data(), ny, MPI_DOUBLE, dist.east,
                  horizontal_tag, dist.cart_comm, &req);
        requests.push_back(req);
    }

    if (dist.south != MPI_PROC_NULL) {
        for (int i = 0; i < nx; ++i) {
            int li = i + 1;
            dist.halos.send_south[static_cast<std::size_t>(i)] =
                field[halo_index(view, li, 1)];
        }
        MPI_Request req;
        MPI_Irecv(dist.halos.recv_south.data(), nx, MPI_DOUBLE, dist.south,
                  vertical_tag, dist.cart_comm, &req);
        requests.push_back(req);
        MPI_Isend(dist.halos.send_south.data(), nx, MPI_DOUBLE, dist.south,
                  vertical_tag, dist.cart_comm, &req);
        requests.push_back(req);
    }

    if (dist.north != MPI_PROC_NULL) {
        for (int i = 0; i < nx; ++i) {
            int li = i + 1;
            dist.halos.send_north[static_cast<std::size_t>(i)] =
                field[halo_index(view, li, ny)];
        }
        MPI_Request req;
        MPI_Irecv(dist.halos.recv_north.data(), nx, MPI_DOUBLE, dist.north,
                  vertical_tag, dist.cart_comm, &req);
        requests.push_back(req);
        MPI_Isend(dist.halos.send_north.data(), nx, MPI_DOUBLE, dist.north,
                  vertical_tag, dist.cart_comm, &req);
        requests.push_back(req);
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE);
    }

    if (dist.west != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            int lj = j + 1;
            field[halo_index(view, 0, lj)] =
                dist.halos.recv_west[static_cast<std::size_t>(j)];
        }
    }
    if (dist.east != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            int lj = j + 1;
            field[halo_index(view, nx + 1, lj)] =
                dist.halos.recv_east[static_cast<std::size_t>(j)];
        }
    }
    if (dist.south != MPI_PROC_NULL) {
        for (int i = 0; i < nx; ++i) {
            int li = i + 1;
            field[halo_index(view, li, 0)] =
                dist.halos.recv_south[static_cast<std::size_t>(i)];
        }
    }
    if (dist.north != MPI_PROC_NULL) {
        for (int i = 0; i < nx; ++i) {
            int li = i + 1;
            field[halo_index(view, li, ny + 1)] =
                dist.halos.recv_north[static_cast<std::size_t>(i)];
        }
    }
}

void apply_A_local(const Grid &grid, const LocalProblemSlices &problem,
                   const std::vector<double> &w, std::vector<double> &out)
{
    std::fill(out.begin(), out.end(), 0.0);
    const LocalView &view = problem.view;
    const int stride = view.nx + 2;
    double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
    double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        int lj = j + view.halo_y;
        int gj = view.jj0 + j;
        for (int i = 0; i < view.nx; ++i) {
            int li = i + view.halo_x;
            int gi = view.ii0 + i;
            double w_ij = w[halo_index(view, li, lj)];
            double term_x = (problem.a[a_index(problem, gi + 1, gj)] *
                                 (w[halo_index(view, li + 1, lj)] - w_ij) -
                             problem.a[a_index(problem, gi, gj)] *
                                 (w_ij - w[halo_index(view, li - 1, lj)])) *
                            inv_h1_sq;
            double term_y = (problem.b[b_index(problem, gi, gj + 1)] *
                                 (w[halo_index(view, li, lj + 1)] - w_ij) -
                             problem.b[b_index(problem, gi, gj)] *
                                 (w_ij - w[halo_index(view, li, lj - 1)])) *
                            inv_h2_sq;
            out[halo_index(view, li, lj)] = -(term_x + term_y);
        }
    }
}

void apply_D_inv_local(const LocalProblemSlices &problem,
                       const std::vector<double> &in, std::vector<double> &out)
{
    const LocalView &view = problem.view;
    const int stride = view.nx + 2;
    int invalid_flag = 0;
#pragma omp parallel for collapse(2) reduction(| : invalid_flag) \
    schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        int lj = j + view.halo_y;
        std::size_t row_offset = static_cast<std::size_t>(j) *
                                 static_cast<std::size_t>(problem.plain_stride);
        for (int i = 0; i < view.nx; ++i) {
            int li = i + view.halo_x;
            double d = problem.diag[row_offset + static_cast<std::size_t>(i)];
            if (d <= 0.0) {
                invalid_flag = 1;
                out[halo_index(view, li, lj)] = 0.0;
            } else {
                out[halo_index(view, li, lj)] =
                    in[halo_index(view, li, lj)] / d;
            }
        }
    }
    if (invalid_flag) {
        throw std::runtime_error(
            "Диагональный элемент предобуславливателя не положителен");
    }
}

double inner_product_local(const LocalView &view,
                           const std::vector<double> &lhs,
                           const std::vector<double> &rhs)
{
    const int stride = view.nx + 2;
    double sum = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : sum) schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        int lj = j + view.halo_y;
        for (int i = 0; i < view.nx; ++i) {
            int li = i + view.halo_x;
            sum +=
                lhs[halo_index(view, li, lj)] * rhs[halo_index(view, li, lj)];
        }
    }
    return sum;
}

Result solve_problem_parallel(const Config &config, const Partition &partition,
                              Dist &dist, const LocalProblemSlices &problem)
{
    const Grid &grid = config.grid;
    const LocalView &view = dist.view;
    const int stride = view.nx + 2;
    std::size_t plane = static_cast<std::size_t>(stride) *
                        static_cast<std::size_t>(view.ny + 2);
    std::vector<double> w(plane, 0.0);
    std::vector<double> r(plane, 0.0);
    std::vector<double> z(plane, 0.0);
    std::vector<double> p(plane, 0.0);
    std::vector<double> Ap(plane, 0.0);

#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < view.ny; ++j) {
        int lj = j + view.halo_y;
        std::size_t row_offset = static_cast<std::size_t>(j) *
                                 static_cast<std::size_t>(problem.plain_stride);
        for (int i = 0; i < view.nx; ++i) {
            int li = i + view.halo_x;
            r[halo_index(view, li, lj)] =
                problem.F[row_offset + static_cast<std::size_t>(i)];
        }
    }

    double rhs_norm = norm_from_local_sum(
        inner_product_plain_local(problem, problem.F, problem.F), grid,
        dist.cart_comm);
    double rhs_norm_safe = rhs_norm == 0.0 ? 1.0 : rhs_norm;

    double residual_norm = norm_from_local_sum(inner_product_local(view, r, r),
                                               grid, dist.cart_comm);
    double diff_norm = std::numeric_limits<double>::infinity();
    std::vector<IterationLogEntry> iteration_log;
    if (dist.world_rank == 0) {
        iteration_log.reserve(static_cast<std::size_t>(config.maxIt));
    }

    std::size_t iter = 0;
    std::string stop_reason;

    while (iter < static_cast<std::size_t>(config.maxIt)) {
        apply_D_inv_local(problem, r, z);
        p = z;
        exchange_halos(dist, p);
        apply_A_local(grid, problem, p, Ap);

        double numerator = reduce_dot_product(inner_product_local(view, z, r),
                                              grid, dist.cart_comm);
        double denominator = reduce_dot_product(
            inner_product_local(view, Ap, p), grid, dist.cart_comm);
        if (std::abs(denominator) < 1e-30) {
            stop_reason = "denominator<=0";
            break;
        }
        double alpha = numerator / denominator;

        double p_norm = reduce_dot_product(inner_product_local(view, p, p),
                                           grid, dist.cart_comm);
        if (p_norm < 1e-30) {
            stop_reason = "direction_norm~0";
            break;
        }
        diff_norm = std::sqrt(p_norm) * std::abs(alpha);

#pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < view.ny; ++j) {
            int lj = j + view.halo_y;
            for (int i = 0; i < view.nx; ++i) {
                int li = i + view.halo_x;
                std::size_t idx = halo_index(view, li, lj);
                w[idx] += alpha * p[idx];
                r[idx] -= alpha * Ap[idx];
            }
        }

        residual_norm = norm_from_local_sum(inner_product_local(view, r, r),
                                            grid, dist.cart_comm);
        ++iter;
        if (dist.world_rank == 0) {
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
    }

    if (stop_reason.empty()) {
        if (iter >= static_cast<std::size_t>(config.maxIt)) {
            stop_reason = "maxIt";
        } else {
            stop_reason = "stagnation";
        }
    }

    Result result;
    result.iterations = iter;
    result.residual_norm = residual_norm;
    result.diff_norm = diff_norm;
    result.rhs_norm = rhs_norm;
    result.stop_reason = stop_reason;
    if (dist.world_rank == 0) {
        result.iteration_log = iteration_log;
    }
    result.solution = gather_global_solution(grid, partition, dist, w);
    return result;
}
}  // namespace

void write_solution_csv(const std::string &filename, const Grid &grid,
                        const std::vector<double> &w)
{
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

std::vector<MaskEntry> build_mask_entries(const Grid &grid)
{
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

std::vector<int> make_blocks(int total_nodes, int parts)
{
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

Partition build_partition_with_ranges(int M, int N, int Px, int Py)
{
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
            xcuts[static_cast<std::size_t>(i)] +
            nx[static_cast<std::size_t>(i)];
    }
    for (int j = 0; j < Py; ++j) {
        ycuts[static_cast<std::size_t>(j + 1)] =
            ycuts[static_cast<std::size_t>(j)] +
            ny[static_cast<std::size_t>(j)];
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
                                     double rmax)
{
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
                      " ny=" + std::to_string(block.nodes_y) +
                      " nx/ny=" + ratio_buf + " диапазоны i=[" +
                      std::to_string(range.ix0) + ".." +
                      std::to_string(range.ix1) + "] j=[" +
                      std::to_string(range.iy0) + ".." +
                      std::to_string(range.iy1) + "]";
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

int main(int argc, char **argv)
{
    const std::string output_dir = "output";
    int exit_code = EXIT_SUCCESS;
    std::string error_message;
    std::string current_error_log_path = output_dir + "/hybrid_error.log";

    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0) {
        ensure_directory(output_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm cart_comm = MPI_COMM_NULL;

    try {
        if (B1 <= A1 || B2 <= A2) {
            throw std::runtime_error("Диапазоны по x или y заданы некорректно");
        }

        int thread_count = parse_thread_argument(argc, argv, 1);
        if (thread_count > 0) {
            omp_set_num_threads(thread_count);
        }

        std::string output_prefix = output_dir + "/hybrid_";

        const auto grid_dims =
            parse_grid_arguments(argc, argv, default_grids(0));
        int M = grid_dims.first;
        int N = grid_dims.second;
        if (M < 2 || N < 2) {
            throw std::runtime_error("Сетке нужны M,N >= 2");
        }

        auto partition_guess = parse_px_py_or_defaults(argc, argv, M, N);
        int Px = partition_guess.first;
        int Py = partition_guess.second;
        if (Px * Py != world_size) {
            throw std::runtime_error(
                "Px*Py не совпадает с числом процессов MPI");
        }
        auto part_check = check_partition(M, N, Px, Py);
        Partition partition = std::move(part_check.partition);

        int dims[2] = {Py, Px};
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

        int coords[2] = {0, 0};
        MPI_Cart_coords(cart_comm, world_rank, 2, coords);

        Dist dist;
        dist.cart_comm = cart_comm;
        dist.world_rank = world_rank;
        dist.world_size = world_size;
        dist.coords[0] = coords[0];
        dist.coords[1] = coords[1];
        dist.Px = Px;
        dist.Py = Py;
        MPI_Cart_shift(cart_comm, 1, 1, &dist.west, &dist.east);
        MPI_Cart_shift(cart_comm, 0, 1, &dist.south, &dist.north);

        std::size_t range_idx =
            static_cast<std::size_t>(coords[0]) * static_cast<std::size_t>(Px) +
            static_cast<std::size_t>(coords[1]);
        dist.range = partition.ranges.at(range_idx);
        dist.view = make_local_view(dist.range);
        if (dist.view.nx <= 0 || dist.view.ny <= 0) {
            throw std::runtime_error(
                "Локальная область пуста для данного процесса");
        }
        allocate_halo_buffers(dist);

        auto start_time = std::chrono::steady_clock::now();

        Grid grid(A1, B1, A2, B2, M, N);
        double h = std::max(grid.h1, grid.h2);
        double epsilon = h * h;
        LocalProblemSlices local_problem =
            build_problem_local(grid, epsilon, dist.view);

        double local_sums[4] = {std::accumulate(local_problem.a.begin(),
                                                local_problem.a.end(), 0.0),
                                std::accumulate(local_problem.b.begin(),
                                                local_problem.b.end(), 0.0),
                                std::accumulate(local_problem.F.begin(),
                                                local_problem.F.end(), 0.0),
                                std::accumulate(local_problem.diag.begin(),
                                                local_problem.diag.end(), 0.0)};
        double global_sums[4] = {0.0, 0.0, 0.0, 0.0};
        MPI_Allreduce(local_sums, global_sums, 4, MPI_DOUBLE, MPI_SUM,
                      cart_comm);
        if (world_rank == 0) {
            std::ostringstream coeff_os;
            coeff_os << std::scientific << std::setprecision(12);
            coeff_os << "[coeff-sum] a=" << global_sums[0]
                     << " b=" << global_sums[1] << " F=" << global_sums[2]
                     << " diag=" << global_sums[3] << '\n';
            std::cout << coeff_os.str() << std::flush;
        }

        long long maxIt = static_cast<long long>((M - 1) * (N - 1));
        Config config{grid, DELTA, TAU, maxIt, epsilon};

        auto grid_start = std::chrono::steady_clock::now();
        Result result =
            solve_problem_parallel(config, partition, dist, local_problem);
        auto grid_end = std::chrono::steady_clock::now();
        double local_seconds =
            std::chrono::duration<double>(grid_end - grid_start).count();
        double grid_seconds = 0.0;
        MPI_Allreduce(&local_seconds, &grid_seconds, 1, MPI_DOUBLE, MPI_MAX,
                      cart_comm);

        if (world_rank == 0) {
            std::vector<SummaryEntry> summary;
            std::vector<RuntimeEntry> runtime_entries;
            summary.push_back({M, N, result.residual_norm});
            runtime_entries.push_back({M, N, grid_seconds});
            std::string suffix =
                "_" + std::to_string(M) + "x" + std::to_string(N);
            std::string solution_path =
                output_prefix + "solution" + suffix + ".csv";
            std::string meta_path = output_prefix + "meta" + suffix + ".txt";
            std::string run_log_path = output_prefix + "run" + suffix + ".log";
            std::string mask_path = output_prefix + "mask" + suffix + ".csv";
            std::string partition_report_path =
                output_prefix + "partition" + suffix + ".txt";

            write_partition_log(partition_report_path, part_check.report);
            write_solution_csv(solution_path, grid, result.solution);
            write_meta_txt(
                meta_path, grid.A1, grid.B1, grid.A2, grid.B2, grid.M, grid.N,
                grid.h1, grid.h2, epsilon, config.delta, config.tau,
                config.maxIt, result.iterations, result.residual_norm,
                result.diff_norm, result.rhs_norm, result.stop_reason);
            write_run_log(run_log_path, M, N, result.iteration_log,
                          result.iterations, result.residual_norm,
                          result.diff_norm);
            write_mask_csv(mask_path, build_mask_entries(grid));
            write_summary_txt(output_prefix + "summary" + suffix + ".txt",
                              summary);
            auto end_time = std::chrono::steady_clock::now();
            double total_seconds =
                std::chrono::duration<double>(end_time - start_time).count();
            write_runtime(output_prefix + "runtime" + suffix + ".txt",
                          runtime_entries, total_seconds);
        }

        if (cart_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&cart_comm);
            cart_comm = MPI_COMM_NULL;
        }
    } catch (const std::exception &ex) {
        error_message = ex.what();
        exit_code = EXIT_FAILURE;
    }

    if (exit_code != EXIT_SUCCESS) {
        if (cart_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&cart_comm);
        }
        if (world_rank == 0) {
            write_error_log(current_error_log_path, error_message.empty()
                                                        ? "Неизвестная ошибка"
                                                        : error_message);
        }
        MPI_Abort(MPI_COMM_WORLD, exit_code);
        return exit_code;
    }

    MPI_Finalize();
    return exit_code;
}
