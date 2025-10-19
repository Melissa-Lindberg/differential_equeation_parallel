#include "log.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

void write_run_log(const std::string &filename,
                   int M,
                   int N,
                   const std::vector<IterationLogEntry> &iteration_log,
                   std::size_t iterations,
                   double residual_norm,
                   double diff_norm) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << std::scientific << std::setprecision(6);
    out << "# Run M=" << M << ", N=" << N << '\n';
    for (const auto &entry : iteration_log) {
        out << entry.iteration << ", " << entry.residual << ", " << entry.alpha << '\n';
    }
    out << "iters=" << iterations << ", residual=" << residual_norm
        << ", diff=" << diff_norm << '\n';
}

void write_summary_txt(const std::string &filename, const std::vector<SummaryEntry> &summary) {
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
                   double total_seconds) {
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

void write_error_log(const std::string &filename, const std::string &message) {
    std::ofstream out(filename);
    if (!out) {
        return;
    }
    out << "Ошибка: " << message << '\n';
}

void write_mask_csv(const std::string &filename, const std::vector<MaskEntry> &mask_entries) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть файл " + filename);
    }
    out << "x,y,inD\n";
    for (const auto &entry : mask_entries) {
        out << entry.x << ',' << entry.y << ',' << (entry.inside ? 1 : 0) << '\n';
    }
}

void write_meta_txt(const std::string &filename,
                    double A1,
                    double B1,
                    double A2,
                    double B2,
                    int M,
                    int N,
                    double h1,
                    double h2,
                    double epsilon,
                    double delta,
                    double tau,
                    long long maxIt,
                    std::size_t iterations,
                    double residual_norm,
                    double diff_norm,
                    double rhs_norm,
                    const std::string &stop_reason) {
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
