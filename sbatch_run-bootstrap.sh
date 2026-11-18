#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --qos=blanca-chbe-rdi
#SBATCH --account=blanca-chbe-rdi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=plm-boot
#SBATCH --output=plm-epistasis_cutoff-0p9_run-bootstrap.%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hapa3524@colorado.edu

# Bootstrap significance test (model vs. DummyClassifier baseline) for each
# replicate's predictions. CPU-only: reads prediction CSVs, no model loading.
# Run AFTER sbatch_run-predictions.sh; output JSONs are read by
# jupyter_notebook/pred_analysis.ipynb (cell 4).

# --- configuration ----
export PLM_DATA_DIR=/scratch/alpine/hapa3524/PLM

CUTOFF=0p9
N_REPS=5
INPUT_CSV=data/input_VRC01_IC80_cutoff-${CUTOFF}.csv
PRED_DIR=${PLM_DATA_DIR}/results/predictions/classification
BOOT_DIR=${PLM_DATA_DIR}/results/bootstrap

N_BOOTSTRAP=5000
BASELINE_STRATEGY=stratified   # most_frequent | stratified | uniform
SEED=42
# ---------------------

echo this is job $SLURM_JOBID

# load the needed modules
module purge
module load anaconda
conda activate /projects/hapa3524/software/anaconda/envs/protML

# Fail early if any replicate's predictions are missing.
for REP in $(seq 1 ${N_REPS}); do
    PRED_CSV=${PRED_DIR}/train_cutoff-${CUTOFF}_rep_${REP}.csv
    if [[ ! -f ${PRED_CSV} ]]; then
        echo "ERROR: missing predictions for replicate ${REP}: ${PRED_CSV}"
        exit 1
    fi
done
echo "All ${N_REPS} replicate prediction files found."

for REP in $(seq 1 ${N_REPS}); do
    echo "=================================================="
    echo "  Bootstrapping replicate ${REP}"
    echo "=================================================="

    python src/run_bootstrap.py \
        --task_type classification \
        --input_csv ${INPUT_CSV} \
        --pred_csv ${PRED_DIR}/train_cutoff-${CUTOFF}_rep_${REP}.csv \
        --result_dir ${BOOT_DIR} \
        --output_file bootstrap_cutoff-${CUTOFF}_rep_${REP}.json \
        --baseline_strategy ${BASELINE_STRATEGY} \
        --n_bootstrap ${N_BOOTSTRAP} \
        --seed ${SEED} || {
            echo "ERROR: bootstrap failed for replicate ${REP}"
            exit 1
        }
done

echo "=================================================="
echo "  All ${N_REPS} replicate bootstrap tests complete."
echo "  Output: ${BOOT_DIR}"
echo "=================================================="
ls -la ${BOOT_DIR}
