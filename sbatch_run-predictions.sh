#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --qos=blanca-chbe-rdi
#SBATCH --account=blanca-chbe-rdi
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=plm-pred
#SBATCH --output=plm-epistasis_cutoff-0p9_run-predictions.%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hapa3524@colorado.edu

# --- configuration ----
export PLM_DATA_DIR=/scratch/alpine/hapa3524/PLM

CUTOFF=0p9
N_REPS=5
INPUT_CSV=data/input_VRC01_IC80_cutoff-${CUTOFF}.csv
MODEL_DIR=${PLM_DATA_DIR}/results/full
PRED_DIR=${PLM_DATA_DIR}/results/predictions/classification

# Architecture flags MUST match those used at training time, or the checkpoint
# will not load (run_predict.py loads with strict=True).
HEAD_HIDDEN_DIM=128
DROPOUT=0.3
POOLING=cls
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

# Fail early if any replicate checkpoint is missing, rather than part-way through.
for REP in $(seq 1 ${N_REPS}); do
    MODEL_PATH=${MODEL_DIR}/PLM_classification_cutoff-${CUTOFF}_rep_${REP}.pt
    if [[ ! -f ${MODEL_PATH} ]]; then
        echo "ERROR: missing checkpoint for replicate ${REP}: ${MODEL_PATH}"
        exit 1
    fi
done
echo "All ${N_REPS} replicate checkpoints found."

for REP in $(seq 1 ${N_REPS}); do
    echo "=================================================="
    echo "  Predicting with replicate ${REP}"
    echo "=================================================="

    python src/run_predict.py \
        --task_type classification \
        --model_path ${MODEL_DIR}/PLM_classification_cutoff-${CUTOFF}_rep_${REP}.pt \
        --load_weights all \
        --head_hidden_dim ${HEAD_HIDDEN_DIM} \
        --dropout ${DROPOUT} \
        --pooling ${POOLING} \
        --input_csv ${INPUT_CSV} \
        --result_dir ${PRED_DIR} \
        --output_file train_cutoff-${CUTOFF}_rep_${REP}.csv || {
            echo "ERROR: prediction failed for replicate ${REP}"
            exit 1
        }
done

echo "=================================================="
echo "  All ${N_REPS} replicate predictions complete."
echo "  Output: ${PRED_DIR}"
echo "=================================================="
ls -la ${PRED_DIR}
