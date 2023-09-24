#!/bin/bash -l
#SBATCH --job-name=segmatron_single_frame
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32GB
##SBATCH --time=0-72:00:00
#SBATCH --partition=DGX-1v100
##SBATCH --array=1-5
#SBATCH --mail-user=zemskova.ts@phystech.edu
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"

singularity instance start \
                     --nv  \
                     --containall \
                     --home ~/zemskova_ts \
                     --bind /tmp \
                     --bind ~/datasets/habitat_datasets:/data:ro \
                     habitat.sif habitat

singularity exec instance://habitat /bin/bash -c "
      cd ~/skill-fusion/root/exploration_ros_free/;
      export PYTHONPATH=/home/AI/yudin.da/zemskova_ts/skill-fusion/root/PONI/:$PYTHONPATH;
      python main.py config_poni_exploration_1_step.yaml;
"

#singularity instance stop habitat
