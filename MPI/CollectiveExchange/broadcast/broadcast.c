#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    int rank, size, len;
    int root = 0;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Get_processor_name(procname, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int count = 1024;
    char *buf = malloc(sizeof(*buf) * count);
    double t = MPI_Wtime();
    if (rank == root)
    {
        buf[0] = 'A';
        for (int i = 0; i < size; i++)
        {
            if (i == root)
            {
                continue;
            }
            MPI_Send(buf, count, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(buf, count, MPI_CHAR, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    t = MPI_Wtime() - t;
    printf("Rank %d/%d: buf = %c\n", rank, size, buf[0]);
    printf("Rank %d/%d: time (sec): %.6f\n", rank, size, t);
    free(buf);
    MPI_Finalize();
    return 0;
}