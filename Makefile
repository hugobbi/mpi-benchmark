# Makefile for MPI Matrix Multiplication

all: #mpi_coletiva mpi_p2p_bloqueante mpi_p2p_naobloqueante
	mpicc mpi_coletiva.c -o mpi_coletiva
	mpicc mpi_p2p_bloqueante.c -o mpi_p2p_bloqueante
	mpicc mpi_p2p_naobloqueante.c -o mpi_p2p_naobloqueante

clean:
	rm mpi_coletiva mpi_p2p_bloqueante mpi_p2p_naobloqueante
