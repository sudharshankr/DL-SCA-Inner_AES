#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=cnn_trial2
#SBATCH --qos=short
#SBATCH --time=0-04:00:00

# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=jobspernode:four:1
#SBATCH --mem-per-cpu=3000MB
#SBATCH --gres=gpu:1

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/11.0 cudnn/11.0-8.0.3.33
module list
#
cd /home/nfs/sudharshankuma/
source env/bin/activate
cd python_scripts/
#
srun python3 sca_dl_train_model.py cnn_hw_1.h5
exit 0