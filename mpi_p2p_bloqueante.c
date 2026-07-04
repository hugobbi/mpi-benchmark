// MPI Matrix Multiplication using Point-to-Point Communication
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void initialize_matrices(int n, double* A, double* B, double* C) {
    for (int i = 0; i < n * n; i++) {
        A[i] = i % 100;
        B[i] = (i % 100) + 1;
        C[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    int rank, size, n = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A, *B, *C;
    A = (double*)malloc(n * n * sizeof(double));
    B = (double*)malloc(n * n * sizeof(double));
    C = (double*)malloc(n * n * sizeof(double));

    if (rank == 0) {
        initialize_matrices(n, A, B, C);
    }

    double* local_A = (double*)malloc((n * n / size) * sizeof(double));
    double* local_C = (double*)malloc((n * n / size) * sizeof(double));

    double comm_local = 0.0;
    double comp_local = 0.0;
    double total_local;
    double comm_max = 0.0;
    double comp_max = 0.0;
    double total_max = 0.0;

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before starting the timer
    double total_start = MPI_Wtime();

    double comm_start = MPI_Wtime();  
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(A + i * (n * n / size), n * n / size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < n * n / size; i++) {
            local_A[i] = A[i];
        }
    } else {
        MPI_Recv(local_A, n * n / size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_local += MPI_Wtime() - comm_start;

    double comp_start = MPI_Wtime();
    for (int i = 0; i < n / size; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }
    comp_local += MPI_Wtime() - comp_start;

    comm_start = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < n * n / size; i++) {
            C[i] = local_C[i];
        }
        for (int i = 1; i < size; i++) {
            MPI_Recv(C + i * (n * n / size), n * n / size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_C, n * n / size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    comm_local += MPI_Wtime() - comm_start;

    total_local = MPI_Wtime() - total_start;

    MPI_Reduce(&comm_local, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_local, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
    // Print the results for benchmarking
    // printf("===\n");
	// printf("blocking\n");
	// printf("n:\t\t%d\n", n);
	// printf("size:\t\t%d\n", size);
	// printf("comm_time:\t%.6f\n", comm_max);
	// printf("comp_time:\t%.6f\n", comp_max);
	// printf("total_time:\t%.6f\n", total_max);
    // printf("===\n");
    printf("blocking,%d,%d,%.6f,%.6f,%.6f\n", n, size, comm_max, comp_max, total_max);
    }


/*    if (rank == 0) {
        printf("Result Matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", C[i * n + j]);
            }
            printf("\n");
        }
    }
*/
    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
