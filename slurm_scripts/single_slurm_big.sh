#!/bin/bash
#SBATCH --job-name=llm_expts
#SBATCH --mem=78gb
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=output_from_slurm/%j.out

# Change to working directory
cd /nas/ucb/bplaut/caution-when-ood-llm

# Set up environment
export HF_HOME="/nas/ucb/bplaut/hugging-face"
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate llm-nas

# Capture command-line inputs
model=$1
dataset=$2
question_range=$3
prompt_phrasing=$4

# Run the command
srun run_qa_tests.sh "$model" "$dataset" "$question_range" "$prompt_phrasing" False
