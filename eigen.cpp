#include <benchmark/benchmark.h>
#include <eigen3/Eigen/Core>

constexpr long m = 6;
constexpr long n = 8;
constexpr long k = 4;

template <typename A, typename B, typename C>
[[gnu::noinline]] void mul(C &c, A const &a, B const &b) {
    c.noalias() = a * b;
}

void eigen_static(benchmark::State &state) {
  Eigen::Matrix<double, m, k> a;
  Eigen::Matrix<double, k, n> b;
  Eigen::Matrix<double, m, n> c;

  a.setOnes();
  b.setOnes();
  c.setOnes();
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());

  for (auto _ : state) {
    mul(c, a, b);
    benchmark::ClobberMemory();
  }
}

void eigen_dynamic(benchmark::State &state) {
  Eigen::Matrix<double, -1, -1> a(m, k);
  Eigen::Matrix<double, -1, -1> b(k, n);
  Eigen::Matrix<double, -1, -1> c(m, n);

  a.setOnes();
  b.setOnes();
  c.setOnes();
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());

  for (auto _ : state) {
    c.noalias() = a * b;
    benchmark::ClobberMemory();
  }
}

BENCHMARK(eigen_static);
BENCHMARK(eigen_dynamic);
BENCHMARK_MAIN();
