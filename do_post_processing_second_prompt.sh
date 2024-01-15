#!/bin/bash

# Check if no more than two arguments are provided
if [ $# -gt 2 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <[omma-separated list of probability thresholds] [comma-separated list of raw logit thresholds]"
    exit 1
fi

# Assign the arguments to variables. If those variables are not provided, default to 0
thresholds1=${1:-0}
thresholds2=${2:-0}

echo -e "Making tables...\n"
python combine_grades_into_table.py tables_second_prompt/no_abstain_normed_logits.tex $thresholds1 True results_second_prompt/*no_abst_norm_logits.txt
python combine_grades_into_table.py tables_second_prompt/no_abstain_raw_logits.tex $thresholds2 True results_second_prompt/*no_abst_raw_logits.txt
python combine_grades_into_table.py tables_second_prompt/yes_abstain_normed_logits.tex $thresholds1 True results_second_prompt/*yes_abst_norm_logits.txt
python combine_grades_into_table.py tables_second_prompt/yes_abstain_raw_logits.tex $thresholds2 True results_second_prompt/*yes_abst_raw_logits.txt

echo -e "\nMaking main figures...\n"
python plot_data.py figs_second_prompt/main_figs True arc,hellaswag,mmlu,truthfulqa,winogrande results_second_prompt/*.txt

echo -e "\nMaking figures for PIQA...\n"
python plot_data.py figs_second_prompt/piqa/no_abstain_normed_logits True piqa results_second_prompt/*.txt

echo -e "\nMaking figures excluding winogrande...\n"
python plot_data.py figs_second_prompt/no_winogrande/no_abstain_normed_logits True arc,hellaswag,mmlu,truthfulqa results_second_prompt/*.txt
