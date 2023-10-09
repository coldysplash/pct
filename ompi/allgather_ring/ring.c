#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

int main(int argc, char **argv)
{
    int rank, size, len, next, prev, recvdatafrom, senddatafrom;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Get_processor_name(procname, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int count = 3;

    char *buf = malloc(sizeof(*buf) * count * size);

    MPI_Request request[2];
    MPI_Status status[2];

    memset(buf + rank * count, 'a' + rank, count);

    double t = MPI_Wtime();

    next = (rank + 1) % size;
    prev = (rank - 1 + size) % size;

    for (int i = 0; i < size - 1; i++)
    {
        recvdatafrom = (rank - i - 1 + size) % size;
        senddatafrom = (rank - i + size) % size;
        MPI_Isend(buf + senddatafrom * count, count, MPI_CHAR, next, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(buf + recvdatafrom * count, count, MPI_CHAR, prev, 0, MPI_COMM_WORLD, &request[1]);
        MPI_Waitall(2, request, status);
    }

    t = MPI_Wtime() - t;

    bool testpassed = true;
    for (size_t i = 0; i < size; i++)
    {
        if (*buf + i * count != 'a' + i * count)
        {
            testpassed = false;
            break;
        }
    }
    if (testpassed)
    {
        printf("Message passing successfully!");
    }
    else
    {
        printf("Error!");
    }

    printf(" Result buf  [ ");
    for (size_t i = 0; i < size * count; i++)
    {
        printf("%c  ", buf[i]);
    }
    printf("]\n");
    printf("\n");
    printf("time (sec): %.6f\n", t);
    free(buf);
    MPI_Finalize();
    return 0;
}
