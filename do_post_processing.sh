#!/bin/bash

# Check if no more than three arguments are provided
if [ $# -gt 3 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <directory> [comma-separated list of probability thresholds] [comma-separated list of raw logit thresholds]"
    exit 1
fi

# Check if at least one argument is provided (the directory)
if [ $# -lt 1 ]; then
    echo "No directory specified. Usage: ./do_post_processing <directory> [comma-separated list of probability thresholds] [comma-separated list of raw logit thresholds]"
    exit 1
fi

# Assign the arguments to variables. If no values are provided for thresholds, default to 0
dir=$1
thresholds1=${2:-0}
thresholds2=${3:-0}

echo -e "Making tables...\n"
output_dir=tables

for abstain in "no_abst" "yes_abst"; do
    for logit_type in "norm" "raw"; do
        for prompt in "first_prompt" "second_prompt"; do
            # Construct the file name part for table command
            filename_part="${abstain}_${logit_type}_logits_${prompt}"
            # Construct the threshold variable
            if [ "$logit_type" = "norm" ]; then
                thresholds=$thresholds1
            else
                thresholds=$thresholds2
            fi
            # Run the table command
            python combine_grades_into_table.py $output_dir/${filename_part}.tex $thresholds True $dir/*${filename_part}.txt
        done
    done
done

output_dir=figs
echo -e "\nMaking figures...\n"

for abstain in "no_abst" "yes_abst"; do
    for prompt in "first_prompt" "second_prompt"; do
        python plot_data.py $output_dir/main_figs True arc,hellaswag,mmlu,truthfulqa,winogrande $dir/*${abstain}*${prompt}.txt
        python plot_data.py $output_dir/piqa True piqa $dir/*${abstain}*${prompt}.txt
    done
done
