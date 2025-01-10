#!/bin/bash

# Check if exactly one argument is provided
if [ $# -ne 2 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <input_directory> <output_directory>"
    exit 1
fi

# Assign the argument to a variable
input_dir=$1
final_output_dir=$2

# Loop over the two values of collapse_prompts
for collapse_prompts in True False
do
    # Construct the output directory name
    all_figs_output_dir="all_figs_${input_dir}/collapse_${collapse_prompts}"
    echo -e "\nMAKING FIGURES FOR COLLAPSE_PROMPTS=${collapse_prompts}\n"

    python plot_data.py "$all_figs_output_dir/main_figs" arc,hellaswag,mmlu,truthfulqa,winogrande "$collapse_prompts" "$input_dir"/*.txt
    python plot_data.py "$all_figs_output_dir/arc" arc "$collapse_prompts" "$input_dir"/*.txt
    python plot_data.py "$all_figs_output_dir/hellaswag" hellaswag "$collapse_prompts" "$input_dir"/*.txt
    python plot_data.py "$all_figs_output_dir/mmlu" mmlu "$collapse_prompts" "$input_dir"/*.txt
    python plot_data.py "$all_figs_output_dir/truthfulqa" truthfulqa "$collapse_prompts" "$input_dir"/*.txt
    python plot_data.py "$all_figs_output_dir/winogrande" winogrande "$collapse_prompts" "$input_dir"/*.txt

    echo -e "\nCopying important figures...\n"
    python copy_important_figs.py $all_figs_output_dir $final_output_dir

    # Only run statistical tests for AUROC stuff, which uses collapse_prompts=False
    if [ "$collapse_prompts" = "False" ]; then
        echo -e "\nDoing statistical tests...\n"
        for option in {1..4}; do
            python statistical_tests.py -o "$option" -d "$input_dir"
        done
    fi
done
