#!/bin/bash

# Check if exactly one argument is provided
if [ $# -ne 1 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <directory>"
    exit 1
fi

# Assign the argument to a variable
dir=$1

# Loop over the two values of collapse_prompts
for collapse_prompts in True False
do
    # Construct the output directory name
    output_dir="all_figs_collapse_${collapse_prompts}_${dir}"
    echo -e "\nMaking figures and saving them in $output_dir...\n"

    python plot_data.py "$output_dir/main_figs" arc,hellaswag,mmlu,truthfulqa,winogrande "$collapse_prompts" "$dir"/*.txt
    python plot_data.py "$output_dir/arc" arc "$collapse_prompts" "$dir"/*.txt
    python plot_data.py "$output_dir/hellaswag" hellaswag "$collapse_prompts" "$dir"/*.txt
    python plot_data.py "$output_dir/mmlu" mmlu "$collapse_prompts" "$dir"/*.txt
    python plot_data.py "$output_dir/truthfulqa" truthfulqa "$collapse_prompts" "$dir"/*.txt
    python plot_data.py "$output_dir/winogrande" winogrande "$collapse_prompts" "$dir"/*.txt

    echo -e "\nCopying important figures...\n"
    python copy_important_figs.py "$output_dir" "important_figs_collapse_${collapse_prompts}_${dir}"

    # Only run statistical tests for AUROC stuff, which uses collapse_prompts=False
    if [ "$collapse_prompts" = "False" ]; then
        echo -e "\nDoing statistical tests...\n"
        for option in {1..4}; do
            python statistical_tests.py -o "$option" -d "$dir"
        done
    fi
done
