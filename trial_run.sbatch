#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=dl_round_4
#SBATCH --qos=short
#SBATCH --time=0-00:10:00

# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2500MB
#SBATCH --gres=gpu:1

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/11.0 cudnn/11.0-8.0.3.33
module list
#
cd /home/nfs/sudharshankuma/
source env/bin/activate
cd python_scripts_aes/
#
srun python3 sca_dl_train_model.py
exit 0