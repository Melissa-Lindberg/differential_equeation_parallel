#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "consistent.h"
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

ProblemData build_problem(const Grid &grid, double epsilon) {
  ProblemData data;
  std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) *
                            static_cast<std::size_t>(grid.N + 1);
  data.a.assign(total_nodes, 0.0);
  data.b.assign(total_nodes, 0.0);
  data.F.assign(total_nodes, 0.0);
  data.diag.assign(total_nodes, 0.0);

  // коэффициенты a_{i,j} для вертикальных граней
  for (int i = 1; i <= grid.M; ++i) {
    double xmid = grid.x_mid(i);
    for (int j = 1; j <= grid.N - 1; ++j) {
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

  // коэффициенты b_{i,j} для горизонтальных граней
  for (int i = 1; i <= grid.M - 1; ++i) {
    for (int j = 1; j <= grid.N; ++j) {
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

  // правая часть F_{i,j}
  double cell_area = grid.h1 * grid.h2;
  for (int i = 1; i <= grid.M - 1; ++i) {
    for (int j = 1; j <= grid.N - 1; ++j) {
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

  // диагональный предобуславливатель
  double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
  double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
  for (int i = 1; i <= grid.M - 1; ++i) {
    for (int j = 1; j <= grid.N - 1; ++j) {
      std::size_t idx = grid.index(i, j);
      double val = 0.0;
      val +=
          (data.a[grid.index(i + 1, j)] + data.a[grid.index(i, j)]) * inv_h1_sq;
      val +=
          (data.b[grid.index(i, j + 1)] + data.b[grid.index(i, j)]) * inv_h2_sq;
      data.diag[idx] = val;
    }
  }

  return data;
}

double inner_product(const Grid &grid, const std::vector<double> &u,
                     const std::vector<double> &v) {
  double sum = 0.0;
  for (int j = 1; j <= grid.N - 1; ++j) {
    for (int i = 1; i <= grid.M - 1; ++i) {
      std::size_t idx = grid.index(i, j);
      sum += u[idx] * v[idx];
    }
  }
  return sum * grid.h1 * grid.h2;
}

double norm_E(const Grid &grid, const std::vector<double> &u) {
  double sq = inner_product(grid, u, u);
  return std::sqrt(sq);
}

void apply_A(const Grid &grid, const std::vector<double> &a,
             const std::vector<double> &b, const std::vector<double> &w,
             std::vector<double> &out) {
  std::fill(out.begin(), out.end(), 0.0);
  double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
  double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
  for (int j = 1; j <= grid.N - 1; ++j) {
    for (int i = 1; i <= grid.M - 1; ++i) {
      std::size_t idx = grid.index(i, j);
      double w_ij = w[idx];
      double term_x =
          (a[grid.index(i + 1, j)] * (w[grid.index(i + 1, j)] - w_ij) -
           a[grid.index(i, j)] * (w_ij - w[grid.index(i - 1, j)])) *
          inv_h1_sq;
      double term_y =
          (b[grid.index(i, j + 1)] * (w[grid.index(i, j + 1)] - w_ij) -
           b[grid.index(i, j)] * (w_ij - w[grid.index(i, j - 1)])) *
          inv_h2_sq;
      out[idx] = -(term_x + term_y);
    }
  }
}

void apply_D_inv(const Grid &grid, const std::vector<double> &diag,
                 const std::vector<double> &in, std::vector<double> &out) {
  std::fill(out.begin(), out.end(), 0.0);
  for (int j = 1; j <= grid.N - 1; ++j) {
    for (int i = 1; i <= grid.M - 1; ++i) {
      std::size_t idx = grid.index(i, j);
      double d = diag[idx];
      if (d <= 0.0) {
        throw std::runtime_error(
            "Диагональный элемент предобуславливателя не положителен");
      }
      out[idx] = in[idx] / d;
    }
  }
}

RunResult solve_problem(const RunConfig &config, const ProblemData &data) {
  const Grid &grid = config.grid;
  std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) *
                            static_cast<std::size_t>(grid.N + 1);
  std::vector<double> w(total_nodes, 0.0);
  std::vector<double> r = data.F;
  std::vector<double> z(total_nodes, 0.0);
  std::vector<double> p(total_nodes, 0.0);
  std::vector<double> Ap(total_nodes, 0.0);

  double rhs_norm = norm_E(grid, data.F);
  double rhs_norm_safe = rhs_norm == 0.0 ? 1.0 : rhs_norm;

  double residual_norm = norm_E(grid, r);
  double diff_norm = std::numeric_limits<double>::infinity();
  std::vector<IterationLogEntry> iteration_log;
  std::size_t iter = 0;
  std::string stop_reason;

  while (iter < static_cast<std::size_t>(config.maxIt)) {
    apply_D_inv(grid, data.diag, r, z);
    p = z;
    apply_A(grid, data.a, data.b, p, Ap);

    double numerator = inner_product(grid, z, r);
    double denominator = inner_product(grid, Ap, p);
    if (std::abs(denominator) < 1e-30) {
      stop_reason = "denominator<=0";
      break;
    }
    double alpha = numerator / denominator;
    double p_norm_sq = inner_product(grid, p, p);
    if (p_norm_sq < 1e-30) {
      stop_reason = "direction_norm~0";
      break;
    }
    diff_norm = std::sqrt(p_norm_sq) * std::abs(alpha);

    for (int j = 1; j <= grid.N - 1; ++j) {
      for (int i = 1; i <= grid.M - 1; ++i) {
        std::size_t idx = grid.index(i, j);
        w[idx] += alpha * p[idx];
        r[idx] -= alpha * Ap[idx];
      }
    }

    residual_norm = norm_E(grid, r);
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

  RunResult result;
  result.solution = std::move(w);
  result.iterations = iter;
  result.residual_norm = residual_norm;
  result.diff_norm = diff_norm;
  result.rhs_norm = rhs_norm;
  result.stop_reason = stop_reason;
  result.iteration_log = std::move(iteration_log);
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

int main() {
  auto overall_start = std::chrono::steady_clock::now();
  const std::string output_dir = "output";
  const std::string output_prefix = output_dir + "/";
  int exit_code = EXIT_SUCCESS;
  std::string error_message;
  std::vector<SummaryEntry> summary;
  std::vector<RuntimeEntry> runtime_entries;
  try {
    ensure_directory(output_dir);
    if (B1 <= A1 || B2 <= A2) {
      throw std::runtime_error("Диапазоны по x или y заданы некорректно");
    }

    for (const auto &dims : default_grids()) {
      int M = dims.first;
      int N = dims.second;
      if (M < 2 || N < 2) {
        throw std::runtime_error("Сетке нужны M,N >= 2");
      }

      Grid grid(A1, B1, A2, B2, M, N);
      double h = std::max(grid.h1, grid.h2);
      double epsilon = h * h;
      ProblemData data = build_problem(grid, epsilon);
      long long maxIt = static_cast<long long>((M - 1) * (N - 1));
      RunConfig config{grid, DELTA, TAU, maxIt, epsilon};

      auto grid_start = std::chrono::steady_clock::now();
      RunResult result = solve_problem(config, data);
      auto grid_end = std::chrono::steady_clock::now();
      double grid_seconds =
          std::chrono::duration<double>(grid_end - grid_start).count();
      runtime_entries.push_back({M, N, grid_seconds});

      std::string suffix = "_" + std::to_string(M) + "x" + std::to_string(N);
      std::string solution_path = output_prefix + "solution" + suffix + ".csv";
      std::string meta_path = output_prefix + "meta" + suffix + ".txt";
      std::string run_log_path = output_prefix + "run" + suffix + ".log";
      std::string mask_path = output_prefix + "mask" + suffix + ".csv";
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
    }

    write_summary_txt(output_prefix + "summary.txt", summary);
  } catch (const std::exception &ex) {
    error_message = ex.what();
    exit_code = EXIT_FAILURE;
  }
  auto overall_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = overall_end - overall_start;
  double elapsed_seconds = elapsed.count();

  const std::string runtime_path = output_prefix + "runtime.txt";
  const std::string error_log_path = output_prefix + "error.log";

  if (exit_code == EXIT_SUCCESS) {
    try {
      write_runtime(runtime_path, runtime_entries, elapsed_seconds);
    } catch (const std::exception &ex) {
      error_message = ex.what();
      exit_code = EXIT_FAILURE;
    }
  } else {
    std::ofstream runtime_out(runtime_path.c_str());
    if (runtime_out) {
      runtime_out << std::scientific << std::setprecision(6);
      runtime_out << "Total runtime: " << elapsed_seconds << " s\n";
    }
  }

  if (exit_code != EXIT_SUCCESS) {
    write_error_log(error_log_path, error_message.empty() ? "Неизвестная ошибка"
                                                          : error_message);
  }

  return exit_code;
}
