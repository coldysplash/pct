#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

double getrand() { return (double)rand() / RAND_MAX; }

double func(double x, double y) { return exp(pow(x + y, 2)); }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    srand(rank);

    double start_time = MPI_Wtime();

    int in = 0;
    double s = 0;
    int n = 10000000;
    for (int i = rank; i < n; i += commsize)
    {
        double x = getrand();           /* x in [0, 1] */
        double y = getrand() * (1 - x); /* y in [0, 1 - x] */
        if (y <= 1 - x)
        {
            in++;
            s += func(x, y);
        }
    }

    double end_time = MPI_Wtime();

    int total_in, total_n;
    MPI_Reduce(&in, &total_in, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&n, &total_n, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed_time = end_time - start_time;
    double max_elapsed_time;

    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0)
    {
        double result = (double)total_in / total_n;
        printf("Result: %.12f\n", result);
        printf("Max elapsed time: %f seconds\n", max_elapsed_time);
    }

    MPI_Finalize();
    return 0;
}