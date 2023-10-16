#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

void gather(char *sbuf, char *rbuf, int count, int root, int commsize, int rank)
{
    if (rank > root)
    {
        MPI_Send(sbuf, count, MPI_CHAR, root, 0, MPI_COMM_WORLD);
    }
    else
    {
        for (int i = 1; i < commsize; ++i)
        {
            MPI_Recv(rbuf + count * i, count, MPI_CHAR, MPI_ANY_SOURCE, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int commsize, rank;
    int root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int count = 2;

    char *sbuf = (char *)malloc(sizeof(*sbuf) * count);
    char *rbuf = NULL;

    if (rank == root)
    {
        rbuf = (char *)malloc(sizeof(*rbuf) * count * commsize);
        memset(rbuf, 'a', count);
    }
    else
    {
        memset(sbuf, 'a' + rank, count);
    }

    double t = MPI_Wtime();
    gather(sbuf, rbuf, count, root, commsize, rank);
    t = MPI_Wtime() - t;

    if (rank == root)
    {
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
        printf("Time = %f\n", t);
        printf("Root buf [ ");
        for (size_t i = 0; i < commsize * count; i++)
        {
            printf("%c ", rbuf[i]);
        }
        printf("]\n");
    }

    free(sbuf);
    if (rbuf)
        free(rbuf);

    MPI_Finalize();
}