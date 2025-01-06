#!/bin/bash

# Define arrays
models=(Falcon-7b Falcon-40b Llama3-8b Llama3-70b Llama-7b Llama-13b Llama-70b Mistral Mixtral Solar Yi-6b Yi-34b)
datasets=(arc hellaswag mmlu truthfulqa winogrande)
question_ranges=(0-500 500-1000 1000-1500 1500-2000 2000-2500 2500-3000 3000-3500 3500-4000 4000-4500 4500-5000 5000-5500 5500-6000)
prompt_phrasings=(0 1)

# Loop over all combinations
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    
    # Restrict question_ranges for certain datasets
    if [ "$dataset" = "truthfulqa" ]; then
      # Only up to 500-1000
      sub_question_ranges=(0-500 500-1000)
    elif [ "$dataset" = "arc" ]; then
      # Only up to 2500-3000
      sub_question_ranges=(0-500 500-1000 1000-1500 1500-2000 2000-2500 2500-3000)
    else
      # Use the full range for other datasets
      sub_question_ranges=("${question_ranges[@]}")
    fi

    for question_range in "${sub_question_ranges[@]}"; do
      for prompt_phrasing in "${prompt_phrasings[@]}"; do
        # Run the QA tests
        sbatch single_slurm.sh "$model" "$dataset" "$question_range" "$prompt_phrasing"
      done
    done
  done
done
