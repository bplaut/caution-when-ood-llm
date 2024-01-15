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
python combine_grades_into_table.py tables/no_abstain_normed_logits.tex $thresholds1 True $dir/*no_abst_norm_logits.txt
python combine_grades_into_table.py tables/no_abstain_raw_logits.tex $thresholds2 True $dir/*no_abst_raw_logits.txt
python combine_grades_into_table.py tables/yes_abstain_normed_logits.tex $thresholds1 True $dir/*yes_abst_norm_logits.txt
python combine_grades_into_table.py tables/yes_abstain_raw_logits.tex $thresholds2 True $dir/*yes_abst_raw_logits.txt

echo -e "\nMaking main figures...\n"
python plot_data.py figs/main_figs True arc,hellaswag,mmlu,truthfulqa,winogrande $dir/*yes_abst*.txt
python plot_data.py figs/main_figs True arc,hellaswag,mmlu,truthfulqa,winogrande $dir/*no_abst*.txt

echo -e "\nMaking figures for PIQA...\n"
python plot_data.py figs/piqa/no_abstain_normed_logits True piqa $dir/*no_abst*.txt
python plot_data.py figs/piqa/yes_abstain_normed_logits True piqa $dir/*yes_abst*.txt

echo -e "\nMaking figures excluding winogrande...\n"
python plot_data.py figs/no_winogrande/no_abstain_normed_logits True arc,hellaswag,mmlu,truthfulqa $dir/*no_abst*.txt
python plot_data.py figs/no_winogrande/yes_abstain_normed_logits True arc,hellaswag,mmlu,truthfulqa $dir/*yes_abst*.txt
