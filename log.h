#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

struct IterationLogEntry {
  std::size_t iteration;
  double residual;
  double alpha;
};

struct SummaryEntry {
  int M;
  int N;
  double residual;
};

struct RuntimeEntry {
  int M;
  int N;
  double seconds;
};

struct MaskEntry {
  double x;
  double y;
  bool inside;
};

void ensure_directory(const std::string &path);

void write_run_log(const std::string &filename, int M, int N,
                   const std::vector<IterationLogEntry> &iteration_log,
                   std::size_t iterations, double residual_norm,
                   double diff_norm);

void write_summary_txt(const std::string &filename,
                       const std::vector<SummaryEntry> &summary);

void write_runtime(const std::string &filename,
                   const std::vector<RuntimeEntry> &entries,
                   double total_seconds);

void write_error_log(const std::string &filename, const std::string &message);

void write_partition_log(const std::string &filename,
                         const std::string &report);

void write_mask_csv(const std::string &filename,
                    const std::vector<MaskEntry> &mask_entries);

void write_meta_txt(const std::string &filename, double A1, double B1,
                    double A2, double B2, int M, int N, double h1, double h2,
                    double epsilon, double delta, double tau, long long maxIt,
                    std::size_t iterations, double residual_norm,
                    double diff_norm, double rhs_norm,
                    const std::string &stop_reason);

std::pair<int, int> parse_grid_arguments(int argc, char **argv,
                                         const std::pair<int, int> &defaults);

int parse_thread_argument(int argc, char **argv, int default_threads);

std::pair<int, int> parse_px_py_or_defaults(int argc, char **argv, int M,
                                            int N);
