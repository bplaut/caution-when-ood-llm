#!/bin/bash

# Check if two arguments are provided
if [ $# -ne 2 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <comma-separated list of probability thresholds> <comma-separated list of raw logit thresholds>"
    exit 1
fi

# Assign the arguments to variables
thresholds1=$1
thresholds2=$2

echo -e "Making tables...\n"
python combine_grades_into_table.py tables_second_prompt/no_abstain_normed_logits.tex $thresholds1 True results_second_prompt/*no_abst_norm_logits.txt
python combine_grades_into_table.py tables_second_prompt/no_abstain_raw_logits.tex $thresholds2 True results_second_prompt/*no_abst_raw_logits.txt
python combine_grades_into_table.py tables_second_prompt/yes_abstain_normed_logits.tex $thresholds1 True results_second_prompt/*yes_abst_norm_logits.txt
python combine_grades_into_table.py tables_second_prompt/yes_abstain_raw_logits.tex $thresholds2 True results_second_prompt/*yes_abst_raw_logits.txt

echo -e "\nMaking figures for normed logits & no abstain option ...\n"
python plot_data.py figs_second_prompt/main_figs/no_abstain_normed_logits True arc,hellaswag,mmlu,truthfulqa,winogrande results_second_prompt/*no_abst_norm_logits.txt
echo -e "\nMaking figures for raw logits & no abstain option...\n"
python plot_data.py figs_second_prompt/main_figs/no_abstain_raw_logits True arc,hellaswag,mmlu,truthfulqa,winogrande results_second_prompt/*no_abst_raw_logits.txt
echo -e "\nMaking figures for normed logits & yes abstain option...\n"
python plot_data.py figs_second_prompt/main_figs/yes_abstain_normed_logits True arc,hellaswag,mmlu,truthfulqa,winogrande results_second_prompt/*yes_abst_norm_logits.txt
echo -e "\nMaking figures for raw logits & yes abstain option...\n"
python plot_data.py figs_second_prompt/main_figs/yes_abstain_raw_logits True arc,hellaswag,mmlu,truthfulqa,winogrande results_second_prompt/*yes_abst_raw_logits.txt

echo -e "\nMaking figures for normed logits & no abstain option including PIQA...\n"
python plot_data.py figs_second_prompt/with_piqa/no_abstain_normed_logits True all results_second_prompt/*no_abst_norm_logits.txt
echo -e "\nMaking figures for raw logits & no abstain option including PIQA...\n"
python plot_data.py figs_second_prompt/with_piqa/no_abstain_raw_logits True all results_second_prompt/*no_abst_raw_logits.txt
echo -e "\nMaking figures for normed logits & yes abstain option including PIQA...\n"
python plot_data.py figs_second_prompt/with_piqa/yes_abstain_normed_logits True all results_second_prompt/*yes_abst_norm_logits.txt
echo -e "\nMaking figures for raw logits & yes abstain option including PIQA...\n"
python plot_data.py figs_second_prompt/with_piqa/yes_abstain_raw_logits True all results_second_prompt/*yes_abst_raw_logits.txt

echo -e "\nMaking figures for normed logits & no abstain option excluding winogrande...\n"
python plot_data.py figs_second_prompt/no_winogrande/no_abstain_normed_logits True arc,hellaswag,mmlu,truthfulqa results_second_prompt/*no_abst_norm_logits.txt
echo -e "\nMaking figures for raw logits & no abstain option excluding winogrande...\n"
python plot_data.py figs_second_prompt/no_winogrande/no_abstain_raw_logits True arc,hellaswag,mmlu,truthfulqa results_second_prompt/*no_abst_raw_logits.txt
echo -e "\nMaking figures for normed logits & yes abstain option excluding winogrande...\n"
python plot_data.py figs_second_prompt/no_winogrande/yes_abstain_normed_logits True arc,hellaswag,mmlu,truthfulqa results_second_prompt/*yes_abst_norm_logits.txt
echo -e "\nMaking figures for raw logits & yes abstain option excluding winogrande...\n"
python plot_data.py figs_second_prompt/no_winogrande/yes_abstain_raw_logits True arc,hellaswag,mmlu,truthfulqa results_second_prompt/*yes_abst_raw_logits.txt
