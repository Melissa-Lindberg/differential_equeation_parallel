#include "log.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>Ы
#include <limits>
#include <stdexcept>

#include "OpenMP.h"

void ensure_directory(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            throw std::runtime_error(
                "Путь существует, но не является каталогом: " + path);
        }
        return;
    }
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("Не удалось создать каталог " + path);
    }
}

void write_run_log(const std::string &filename, int M, int N,
                   const std::vector<IterationLogEntry> &iteration_log,
                   std::size_t iterations, double residual_norm,
                   double diff_norm)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << std::scientific << std::setprecision(6);
    out << "# Run M=" << M << ", N=" << N << '\n';
    for (const auto &entry : iteration_log) {
        out << entry.iteration << ", " << entry.residual << ", " << entry.alpha
            << '\n';
    }
    out << "iters=" << iterations << ", residual=" << residual_norm
        << ", diff=" << diff_norm << '\n';
}

void write_summary_txt(const std::string &filename,
                       const std::vector<SummaryEntry> &summary)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << "Residual summary:\n";
    out << "M,N,||r||_E\n";
    out << std::scientific << std::setprecision(6);
    for (const auto &entry : summary) {
        out << entry.M << ',' << entry.N << ',' << entry.residual << '\n';
    }
}

void write_runtime(const std::string &filename,
                   const std::vector<RuntimeEntry> &entries,
                   double total_seconds)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << std::scientific << std::setprecision(6);
    out << "M,N,runtime_s\n";
    for (const auto &entry : entries) {
        out << entry.M << ',' << entry.N << ',' << entry.seconds << '\n';
    }
    out << "Total runtime: " << total_seconds << " s\n";
}

void write_error_log(const std::string &filename, const std::string &message)
{
    std::ofstream out(filename);
    if (!out) {
        return;
    }
    out << "Ошибка: " << message << '\n';
}

void write_partition_log(const std::string &filename, const std::string &report)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << report;
}

void write_mask_csv(const std::string &filename,
                    const std::vector<MaskEntry> &mask_entries)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << "x,y,inD\n";
    for (const auto &entry : mask_entries) {
        out << entry.x << ',' << entry.y << ',' << (entry.inside ? 1 : 0)
            << '\n';
    }
}

void write_meta_txt(const std::string &filename, double A1, double B1,
                    double A2, double B2, int M, int N, double h1, double h2,
                    double epsilon, double delta, double tau, long long maxIt,
                    std::size_t iterations, double residual_norm,
                    double diff_norm, double rhs_norm,
                    const std::string &stop_reason)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << std::scientific << std::setprecision(6);
    out << "A1=" << A1 << '\n';
    out << "B1=" << B1 << '\n';
    out << "A2=" << A2 << '\n';
    out << "B2=" << B2 << '\n';
    out << "M=" << M << '\n';
    out << "N=" << N << '\n';
    out << "h1=" << h1 << '\n';
    out << "h2=" << h2 << '\n';
    out << "epsilon=" << epsilon << '\n';
    out << "delta=" << delta << '\n';
    out << "tau=" << tau << '\n';
    out << "maxIt=" << maxIt << '\n';
    out << "iterations=" << iterations << '\n';
    out << "residual_norm=" << residual_norm << '\n';
    out << "diff_norm=" << diff_norm << '\n';
    out << "rhs_norm=" << rhs_norm << '\n';
    out << "stop_reason=" << stop_reason << '\n';
}

std::pair<int, int> parse_grid_arguments(int argc, char **argv,
                                         const std::pair<int, int> &defaults)
{
    bool has_override = false;
    int override_M = 0;
    int override_N = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-g") {
            if (i + 2 >= argc) {
                throw std::runtime_error("После -g ожидаются два целых числа");
            }
            int M = std::stoi(argv[++i]);
            int N = std::stoi(argv[++i]);
            if (M < 2 || N < 2) {
                throw std::runtime_error("Размеры сетки должны быть >= 2");
            }
            if (has_override) {
                throw std::runtime_error(
                    "Параметр -g можно указывать только один раз");
            }
            has_override = true;
            override_M = M;
            override_N = N;
        } else if (arg == "-t") {
            if (i + 1 >= argc) {
                throw std::runtime_error("После -t ожидается целое число");
            }
            ++i;
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("Неизвестный аргумент: " + arg);
        }
    }
    if (has_override) {
        return std::make_pair(override_M, override_N);
    }

    return defaults;
}

int parse_thread_argument(int argc, char **argv, int default_threads)
{
    int threads = default_threads;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-t") {
            if (i + 1 >= argc) {
                throw std::runtime_error("После -t ожидается целое число");
            }
            threads = std::stoi(argv[++i]);
            if (threads <= 0) {
                throw std::runtime_error(
                    "Количество потоков должно быть положительным");
            }
        } else if (arg == "-g") {
            if (i + 2 >= argc) {
                throw std::runtime_error("После -g ожидаются два целых числа");
            }
            i += 2;
        } else if (arg.rfind('-', 0) == 0) {
            throw std::runtime_error("Неизвестный аргумент: " + arg);
        }
    }
    return threads;
}

std::pair<int, int> parse_px_py_or_defaults(int argc, char **argv, int M, int N)
{
    auto parse_positive = [](const char *text, int &out) -> bool {
        if (!text) {
            return false;
        }
        char *end = nullptr;
        errno = 0;
        long value = std::strtol(text, &end, 10);
        if (errno != 0 || end == text || (end && *end != '\0')) {
            return false;
        }
        if (value <= 0 || value > std::numeric_limits<int>::max()) {
            return false;
        }
        out = static_cast<int>(value);
        return true;
    };

    int px = 0;
    int py = 0;
    bool px_ok = false;
    bool py_ok = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i] ? argv[i] : "";
        if (arg == "-px") {
            if (i + 1 < argc) {
                int value = 0;
                if (parse_positive(argv[i + 1], value)) {
                    px = value;
                    px_ok = true;
                }
                ++i;
            }
        } else if (arg == "-py") {
            if (i + 1 < argc) {
                int value = 0;
                if (parse_positive(argv[i + 1], value)) {
                    py = value;
                    py_ok = true;
                }
                ++i;
            }
        }
    }

    if (px_ok && py_ok) {
        return {px, py};
    }

    auto defaults = default_partition_for(M, N);
    if (defaults.first > 0 && defaults.second > 0) {
        return defaults;
    }

    return {1, 1};
}
