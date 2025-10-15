#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "log.h"

namespace {

constexpr double EPS_GEOM = 1e-12;

struct Point {
    double x;
    double y;
};

struct HalfPlane {
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

const std::vector<HalfPlane> &domain_halfplanes() {
    static const std::vector<HalfPlane> planes = {
        {+1.0, +1.0, 2.0},
        {-1.0, +1.0, 2.0},
        {+1.0, -1.0, 2.0},
        {-1.0, -1.0, 2.0},
        {0.0, +1.0, 1.0}
    };
    return planes;
}

bool in_D(double x, double y) {
    Point p{x, y};
    for (const auto &hp : domain_halfplanes()) {
        if (!hp.contains(p)) {
            return false;
        }
    }
    return true;
}

Point intersect_segment_with_plane(const Point &p0, const Point &p1, const HalfPlane &hp) {
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

std::vector<Point> clip_polygon_by_halfplane(const std::vector<Point> &poly, const HalfPlane &hp) {
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
    std::vector<Point> poly = {
        {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}
    };
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

struct Grid {
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
    std::vector<double> a; // коэффициенты по x
    std::vector<double> b; // коэффициенты по y
    std::vector<double> F; // правая часть
    std::vector<double> diag; // диагональ предобуславливателя
};

ProblemData build_problem(const Grid &grid, double epsilon) {
    ProblemData data;
    std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) * static_cast<std::size_t>(grid.N + 1);
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
            val += (data.a[grid.index(i + 1, j)] + data.a[grid.index(i, j)]) * inv_h1_sq;
            val += (data.b[grid.index(i, j + 1)] + data.b[grid.index(i, j)]) * inv_h2_sq;
            data.diag[idx] = val;
        }
    }

    return data;
}

double inner_product(const Grid &grid, const std::vector<double> &u, const std::vector<double> &v) {
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

void apply_A(const Grid &grid,
             const std::vector<double> &a,
             const std::vector<double> &b,
             const std::vector<double> &w,
             std::vector<double> &out) {
    std::fill(out.begin(), out.end(), 0.0);
    double inv_h1_sq = 1.0 / (grid.h1 * grid.h1);
    double inv_h2_sq = 1.0 / (grid.h2 * grid.h2);
    for (int j = 1; j <= grid.N - 1; ++j) {
        for (int i = 1; i <= grid.M - 1; ++i) {
            std::size_t idx = grid.index(i, j);
            double w_ij = w[idx];
            double term_x = (a[grid.index(i + 1, j)] * (w[grid.index(i + 1, j)] - w_ij)
                            - a[grid.index(i, j)] * (w_ij - w[grid.index(i - 1, j)])) * inv_h1_sq;
            double term_y = (b[grid.index(i, j + 1)] * (w[grid.index(i, j + 1)] - w_ij)
                            - b[grid.index(i, j)] * (w_ij - w[grid.index(i, j - 1)])) * inv_h2_sq;
            out[idx] = -(term_x + term_y);
        }
    }
}

void apply_D_inv(const Grid &grid,
                 const std::vector<double> &diag,
                 const std::vector<double> &in,
                 std::vector<double> &out) {
    std::fill(out.begin(), out.end(), 0.0);
    for (int j = 1; j <= grid.N - 1; ++j) {
        for (int i = 1; i <= grid.M - 1; ++i) {
            std::size_t idx = grid.index(i, j);
            double d = diag[idx];
            if (d <= 0.0) {
                throw std::runtime_error("Диагональный элемент предобуславливателя не положителен");
            }
            out[idx] = in[idx] / d;
        }
    }
}

struct Options {
    double A1 = -2.0;
    double B1 = 2.0;
    double A2 = -2.0;
    double B2 = 2.0;
    int M = 10;
    int N = 10;
    double delta = 1e-8;
    double tau = 1e-8;
    long long maxIt = -1;
    bool batch = false;
    bool writeMask = false;
};

Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch") {
            opt.batch = true;
        } else if (arg == "--mask") {
            opt.writeMask = true;
        } else if (arg == "--A1" || arg == "--B1" || arg == "--A2" || arg == "--B2" ||
                   arg == "--delta" || arg == "--tau" || arg == "--M" || arg == "--N" ||
                   arg == "--maxIt") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Отсутствует значение для аргумента " + arg);
            }
            std::string value = argv[++i];
            try {
                if (arg == "--A1") opt.A1 = std::stod(value);
                else if (arg == "--B1") opt.B1 = std::stod(value);
                else if (arg == "--A2") opt.A2 = std::stod(value);
                else if (arg == "--B2") opt.B2 = std::stod(value);
                else if (arg == "--delta") opt.delta = std::stod(value);
                else if (arg == "--tau") opt.tau = std::stod(value);
                else if (arg == "--M") opt.M = std::stoi(value);
                else if (arg == "--N") opt.N = std::stoi(value);
                else if (arg == "--maxIt") opt.maxIt = std::stoll(value);
            } catch (const std::exception &) {
                throw std::runtime_error("Некорректное значение аргумента " + arg);
            }
        } else {
            throw std::runtime_error("Неизвестный аргумент: " + arg);
        }
    }
    return opt;
}

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

RunResult solve_problem(const RunConfig &config, const ProblemData &data) {
    const Grid &grid = config.grid;
    std::size_t total_nodes = static_cast<std::size_t>(grid.M + 1) * static_cast<std::size_t>(grid.N + 1);
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

void write_solution_csv(const std::string &filename, const Grid &grid, const std::vector<double> &w) {
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
    entries.reserve(static_cast<std::size_t>(grid.M + 1) * static_cast<std::size_t>(grid.N + 1));
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

}

int main(int argc, char **argv) {
    auto overall_start = std::chrono::steady_clock::now();
    int exit_code = EXIT_SUCCESS;
    std::string error_message;
    std::vector<SummaryEntry> summary;
    try {
        Options opt = parse_args(argc, argv);
        if (opt.B1 <= opt.A1 || opt.B2 <= opt.A2) {
            throw std::runtime_error("Диапазоны по x или y заданы некорректно");
        }
        if (opt.M < 2 || opt.N < 2) {
            throw std::runtime_error("Сетке нужны M,N >= 2");
        }
        std::vector<std::pair<int, int>> grids;
        if (opt.batch) {
            grids = {{400, 600}, {800, 1200}};
        } else {
            grids = {{opt.M, opt.N}};
        }
        for (std::size_t run_idx = 0; run_idx < grids.size(); ++run_idx) {
            int M = grids[run_idx].first;
            int N = grids[run_idx].second;
            Grid grid(opt.A1, opt.B1, opt.A2, opt.B2, M, N);
            double h = std::max(grid.h1, grid.h2);
            double epsilon = h * h;
            ProblemData data = build_problem(grid, epsilon);
            long long maxIt = opt.maxIt > 0 ? opt.maxIt : static_cast<long long>((M - 1) * (N - 1));
            RunConfig config{grid, opt.delta, opt.tau, maxIt, epsilon};

            RunResult result = solve_problem(config, data);

            std::string suffix = opt.batch ? ("_" + std::to_string(M) + "x" + std::to_string(N)) : "";
            write_solution_csv("solution" + suffix + ".csv", grid, result.solution);
            write_meta_txt("meta" + suffix + ".txt",
                           grid.A1,
                           grid.B1,
                           grid.A2,
                           grid.B2,
                           grid.M,
                           grid.N,
                           grid.h1,
                           grid.h2,
                           epsilon,
                           config.delta,
                           config.tau,
                           config.maxIt,
                           result.iterations,
                           result.residual_norm,
                           result.diff_norm,
                           result.rhs_norm,
                           result.stop_reason);
            write_run_log("run" + suffix + ".log",
                         M,
                         N,
                         result.iteration_log,
                         result.iterations,
                         result.residual_norm,
                         result.diff_norm);
            if (opt.writeMask) {
                std::vector<MaskEntry> mask_entries = build_mask_entries(grid);
                write_mask_csv("mask" + suffix + ".csv", mask_entries);
            }

            summary.push_back({M, N, result.residual_norm});
        }

        write_summary_txt("summary.txt", summary);
    } catch (const std::exception &ex) {
        error_message = ex.what();
        exit_code = EXIT_FAILURE;
    }
    auto overall_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = overall_end - overall_start;
    double elapsed_seconds = elapsed.count();

    if (exit_code == EXIT_SUCCESS) {
        try {
            write_runtime("runtime.txt", elapsed_seconds);
        } catch (const std::exception &ex) {
            error_message = ex.what();
            exit_code = EXIT_FAILURE;
        }
    } else {
        std::ofstream runtime_out("runtime.txt");
        if (runtime_out) {
            runtime_out << std::scientific << std::setprecision(6);
            runtime_out << "Total runtime: " << elapsed_seconds << " s\n";
        }
    }

    if (exit_code != EXIT_SUCCESS) {
        write_error_log("error.log", error_message.empty() ? "Неизвестная ошибка" : error_message);
    }

    return exit_code;
}
