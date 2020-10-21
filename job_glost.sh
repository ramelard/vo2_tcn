#!/bin/bash
#SBATCH --gres=gpu:t4:4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=ramelard@uwaterloo.ca
#SBATCH --mail-type=END

#module load nixpkgs/16.09  intel/2016.4  openmpi/2.0.2 glost/0.3.1
#module load cuda cudnn
module load glost

echo "Starting run at: `date`"
echo Start,$(date +"%H:%M:%S") >> ~/scratch/runtime-${SLURM_JOBID}.txt

cwd=$(pwd)

#mkdir $SLURM_TMPDIR/logs
#mkdir $SLURM_TMPDIR/chkpts
mkdir /scratch/ramelard/logs/${SLURM_JOBID}
mkdir /scratch/ramelard/chkpts/${SLURM_JOBID}

srun glost_launch glost_task_list.txt

# Clean up (copy output back to project space)
#cd /scratch/ramelard
#cd $SLURM_TMPDIR
#tar -cf $cwd/logs/logs_${SLURM_JOBID}.tar logs/
#tar -cf $cwd/chkpts/chkpt_${SLURM_JOBID}.tar chkpts/
