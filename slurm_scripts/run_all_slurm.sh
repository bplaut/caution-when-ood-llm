#!/bin/bash

# There must be exactly one command line argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <results_dir>"
  exit 1
fi

results_dir=$1

# Define arrays
models=(Falcon-7b Falcon-40b Llama3-8b Llama3-70b Llama-7b Llama-13b Llama-70b Mistral Mixtral Solar Yi-6b Yi-34b)
datasets=(arc hellaswag mmlu truthfulqa winogrande)
question_ranges=(0-500 500-1000 1000-1500 1500-2000 2000-2500 2500-3000 3000-3500 3500-4000 4000-4500 4500-5000 5000-5500 5500-6000)
prompt_phrasings=(0 1)

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    
    # Restrict question ranges for certain datasets
    if [ "$dataset" = "truthfulqa" ]; then
      # Only up to 817 questions total
      sub_question_ranges=(0-500 500-817)
    elif [ "$dataset" = "arc" ]; then
      # Only up to 2590 questions total
      sub_question_ranges=(0-500 500-1000 1000-1500 1500-2000 2000-2500 2500-2590)
    else
      sub_question_ranges=("${question_ranges[@]}")
    fi

    for question_range in "${sub_question_ranges[@]}"; do

      # Replace the dash with 'to' to match your file-naming scheme
      question_range_for_filename="${question_range/-/to}"

      for prompt_phrasing in "${prompt_phrasings[@]}"; do
        
        # Convert prompt_phrasing to string for the filename
        if [ "$prompt_phrasing" -eq 0 ]; then
          prompt_str="first_prompt"
        else
          prompt_str="second_prompt"
        fi

        # Construct the expected filename
        filename="${results_dir}/${dataset}_${model}-q${question_range_for_filename}_no_abst_raw_logits_${prompt_str}.txt"

        # Determine if we should skip or launch
        if [ ! -f "$filename" ]; then
          echo "Launching job for dataset='$dataset', model='$model', range='$question_range', prompt='$prompt_str'"
          sbatch single_slurm.sh "$model" "$dataset" "$question_range" "$prompt_phrasing"
        fi

      done
    done

  done
done
