#define _POSIX_C_SOURCE 1

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double func(double x, double y) { return x / (y * y); }

double getrand(unsigned int *seed) { return (double)rand_r(seed) / RAND_MAX; }

int main(int argc, char **argv) {
  if (argc < 2) return 1;
  int num_thds = atoi(argv[1]);
  double t = omp_get_wtime();
  const int n = 10000000;
  printf("Numerical integration by Monte Carlo method: n = %d\n", n);
  int in = 0;
  double s = 0;
#pragma omp parallel num_threads(num_thds)
  {
    double s_loc = 0;
    int in_loc = 0;
    unsigned int seed = omp_get_thread_num();
#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      double x = getrand(&seed);         /* x in [0, 1] */
      double y = 2 + getrand(&seed) * 3; /* y in [2, 5] */
      if (y <= 5 && y >= 2 && x >= 0 && x <= 1) {
        ++in_loc;
        s_loc += func(x, y);
      }
    }
#pragma omp atomic
    s += s_loc;
#pragma omp atomic
    in += in_loc;
  }
  double v = 3 * in / n;
  double res = v * s / in;
  printf("Result: %.12f, n %d\n", res, n);
  t = omp_get_wtime() - t;
  printf("Elapsed time (sec.): %.6f\n", t);
  return 0;
}
