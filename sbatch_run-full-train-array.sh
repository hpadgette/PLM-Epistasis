#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-chbe-rdi
#SBATCH --account=blanca-chbe-rdi
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=plm-ens
#SBATCH --array=1-5
#SBATCH --output=plm-epistasis_cutoff-0p9_run-full-train_rep-%a.%A.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hapa3524@colorado.edu

# --- configuration ----
export PLM_DATA_DIR=/scratch/alpine/hapa3524/PLM

CUTOFF=0p9
REP=${SLURM_ARRAY_TASK_ID}

# Draw a fresh random seed at runtime from the kernel entropy pool (/dev/urandom),
# independently in each array task. The seed is echoed to this task's .out file and
# logged by run_full_train.py, so any replicate can be regenerated afterwards by
# re-running with --seed <value from that replicate's log>.
SEED=$(( $(od -An -N4 -tu4 < /dev/urandom | tr -d ' ') % 2147483647 ))
# ---------------------

echo "this is job ${SLURM_ARRAY_JOB_ID}, array task ${REP}"
echo "replicate ${REP}, seed ${SEED}"
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
    --seed ${SEED} \
    --rep ${REP} \
    --input_csv data/input_VRC01_IC80_cutoff-${CUTOFF}.csv \
    --val_csv   data/input_VRC01_IC80_cutoff-${CUTOFF}.csv \
    --output_model_name PLM_classification_cutoff-${CUTOFF}_rep_${REP}.pt
