#!/bin/bash
#SBATCH --job-name=mpi_benchmark
#SBATCH --partition=hype
#SBATCH --nodes=2
#SBATCH --ntasks=40
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

MACHINEFILE="nodes.$SLURM_JOB_ID"
srun -l hostname | sort -n | awk '{print $2}' > $MACHINEFILE

VARIANTS=(
       "blocking:./bin/mpi_p2p_bloqueante"
       "non-blocking:./bin/mpi_p2p_naobloqueante"
       "collective:./bin/mpi_coletiva"
)

# Keep the benchmark sizes divisible by all selected process counts.
MATRIX_SIZES=(200 400 800 1600 3200)
PROCESS_COUNTS=(1 2 4 5 8 10 20 40)

echo "variant,n,size,comm_time,comp_time,total_time"

for entry in "${VARIANTS[@]}"; do
       variant_name="${entry%%:*}"
       binary_path="${entry##*:}"

       for n in "${MATRIX_SIZES[@]}"; do
              for size in "${PROCESS_COUNTS[@]}"; do
                     if (( n % size != 0 )); then
                            continue
                     fi

                     for i in {1..3}; do
                            mpirun -np "$size" \
                                   -machinefile "$MACHINEFILE" \
                                   --mca btl ^openib \
                                   --mca btl_tcp_if_include eno2 \
                                   --bind-to none \
                                   "$binary_path" "$n"
                     done
              done
       done
done
