#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-chbe-rdi
#SBATCH --account=blanca-chbe-rdi
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=plm-1
#SBATCH --output=plm-epistasis_cutoff-0p9_run-full-train_rep-1.%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hapa3524@colorado.edu

# --- configuration ----
export PLM_DATA_DIR=/scratch/alpine/hapa3524/PLM
# ---------------------

echo this is job $SLURM_JOBID
echo $SLURM_NTASKS

# load the needed modules
module purge
module load cuda
module load gcc/11.2.0
module load openmpi/4.1.1

# activate the appropriate conda environment
module load anaconda
conda activate /projects/hapa3524/software/anaconda/envs/protML
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

nvidia-smi

python src/run_full_train.py \
    --task_type classification --criterion CrossEntropy \
    --dropout 0.3 --head_hidden_dim 128 \
    --pooling cls --unfreeze_layers 9 \
    --batch_size 6 --lr 1.68e-5 --epochs 50 \
    --seed 42 \
    --input_csv data/input_VRC01_IC80_cutoff-0p9.csv \
    --val_csv   data/input_VRC01_IC80_cutoff-0p9.csv \
    --output_model_name PLM_classification_cutoff-0p9_rep_1.pt
