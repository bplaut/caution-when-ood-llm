#!/bin/bash

# Check if two arguments are provided
if [ $# -ne 2 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <comma-separated list of probability thresholds> <comma-separated list of raw logit thresholds>"
    exit 1
fi

# Assign the arguments to variables
thresholds1=$1
thresholds2=$2

python combine_grades_into_table.py tables/no_abstain.tex $thresholds1 True results/*no_abstain.txt
python plot_data.py figs/no_abstain True 0-1 results/*no_abstain.txt
python combine_grades_into_table.py tables/no_abstain_raw_logits.tex $thresholds2 True results/*no_abstain_raw_logits.txt
python plot_data.py figs/no_abstain_raw_logits True 0-1 results/*no_abstain_raw_logits.txt
