#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

void all_to_all(char *sbuf, char *rbuf, int count, int commsize, int rank)
{
    int sendto, recvfrom;

    MPI_Request request[2];
    MPI_Status status[2];

    for (int step = 1; step < commsize + 1; step++)
    {
        sendto = (rank + step) % commsize;
        recvfrom = (rank + commsize - step) % commsize;

        MPI_Isend(sbuf, count, MPI_CHAR, sendto, 0, MPI_COMM_WORLD, &request[0]);

        MPI_Irecv(rbuf + recvfrom * count, count, MPI_CHAR, recvfrom, 0, MPI_COMM_WORLD, &request[1]);

        MPI_Waitall(2, request, status);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int count = 2;

    char *sbuf = (char *)malloc(sizeof(*sbuf) * count * commsize);
    char *rbuf = (char *)malloc(sizeof(*rbuf) * count * commsize);

    memset(sbuf, 'a' + rank, count);

    double t = MPI_Wtime();
    all_to_all(sbuf, rbuf, count, commsize, rank);
    t = MPI_Wtime() - t;

    printf("%f\n[", t);
    for (size_t i = 0; i < commsize * count; i++)
    {
        printf("%c ", rbuf[i]);
    }
    printf("]\n");

    bool testpassed = true;
    for (size_t i = 0; i < commsize; i++)
    {
        if (*rbuf + i * count != 'a' + i * count)
        {
            testpassed = false;
            break;
        }
    }
    if (testpassed)
    {
        printf("Message passing successfully!\n");
    }
    else
    {
        printf("Error!");
    }

    // printf("[%d] %s\n", rank, sbuf);

    free(sbuf);
    free(rbuf);

    MPI_Finalize();
}