#!/bin/bash
#SBATCH --job-name=det_out_air   # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks=224             # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:06:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ltt@princeton.edu
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --account=mueller

module purge
module load intel-mpi/gcc/2021.13

srun ./PeleC3d.gnu.MPI.ex det.inp