# Makefile for MPI Matrix Multiplication

# size of matrix:
N? = 1024
# num of proc:
SIZE? = 4

all: #mpi_coletiva mpi_p2p_bloqueante mpi_p2p_naobloqueante
	mpicc mpi_coletiva.c -o mpi_coletiva
	mpicc mpi_p2p_bloqueante.c -o mpi_p2p_bloqueante
	mpicc mpi_p2p_naobloqueante.c -o mpi_p2p_naobloqueante

clean:
	rm mpi_coletiva mpi_p2p_bloqueante mpi_p2p_naobloqueante

run: #mpi_coletiva mpi_p2p_bloqueante mpi_p2p_naobloqueante
	mpirun -np $(SIZE) ./mpi_p2p_bloqueante $(N)
	mpirun -np $(SIZE) ./mpi_p2p_naobloqueante $(N)
	mpirun -np $(SIZE) ./mpi_coletiva $(N)
