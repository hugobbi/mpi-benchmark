# Makefile for MPI Matrix Multiplication

# size of matrix:
N? = 1024
# num of proc:
SIZE? = 4

all: bin_dir #mpi_coletiva mpi_p2p_bloqueante mpi_p2p_naobloqueante
	mpicc mpi_coletiva.c -o bin/mpi_coletiva
	mpicc mpi_p2p_bloqueante.c -o bin/mpi_p2p_bloqueante
	mpicc mpi_p2p_naobloqueante.c -o bin/mpi_p2p_naobloqueante

bin_dir:
	mkdir -p bin

clean:
	rm -rf bin

run: #mpi_coletiva mpi_p2p_bloqueante mpi_p2p_naobloqueante
	mpirun -np $(SIZE) ./bin/mpi_p2p_bloqueante $(N)
	mpirun -np $(SIZE) ./bin/mpi_p2p_naobloqueante $(N)
	mpirun -np $(SIZE) ./bin/mpi_coletiva $(N)
