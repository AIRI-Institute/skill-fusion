#!/bin/bash -l
#SBATCH --job-name=skilltron
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32GB
##SBATCH --time=0-72:00:00
#SBATCH --partition=DGX-1v100
#SBATCH --gres=gpu:1


singularity instance start \
                     --nv  \
                     --containall \
                     --home ~/zemskova_ts \
                     --bind /tmp \
                     --bind ~/datasets/habitat_datasets:/data:ro \
                     skilltron.sif skilltron

singularity exec instance://skilltron /bin/bash -c "
      cd ~/skill-fusion/root/skilltron/;
      export PYTHONPATH=/home/zemskova_ts/skill-fusion/root/PONI/:$PYTHONPATH;
      python main.py config_skilltron.yaml;
"

singularity instance stop skilltron