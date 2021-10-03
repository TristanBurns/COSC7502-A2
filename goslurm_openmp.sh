#!/bin/bash -l
#
#SBATCH --job-name=Assignment2_openmp_TB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G # memory (MB)
#SBATCH --time=0-00:05 # time (D-HH:MM)


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo 'running with OMP_NUM_THREADS =' $OMP_NUM_THREADS
echo 'running with MKL_NUM_THREADS =' $MKL_NUM_THREADS
echo "This is job '$SLURM_JOB_NAME' (id: $SLURM_JOB_ID) running on the following nodes:"
echo $SLURM_NODELIST
echo "running with OMP_NUM_THREADS= $OMP_NUM_THREADS "
echo "running with SLURM_TASKS_PER_NODE= $SLURM_TASKS_PER_NODE "

if [ ! -f Assignment2_openmp ] ; then
   echo "unable to find Assignment2_openmp"
   echo "you probably need to compile code"
   exit 2
fi

export OMP_NUM_THREADS=8
time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt
# time ./Assignment2_openmp $1 >> outputfile_openmp.txt