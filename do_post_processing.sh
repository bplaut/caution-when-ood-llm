#!/bin/bash

# Check if exactly three arguments are provided
if [ $# -ne 1 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <directory>"
    exit 1
fi

# Assign the arguments to variables
dir=$1

output_dir="all_figs"
echo -e "\nMaking figures...\n"

python plot_data.py $output_dir/main_figs arc,hellaswag,mmlu,truthfulqa,winogrande $dir/*.txt
python plot_data.py $output_dir/arc arc $dir/*.txt
python plot_data.py $output_dir/hellaswag hellaswag $dir/*.txt
python plot_data.py $output_dir/mmlu mmlu $dir/*.txt
python plot_data.py $output_dir/truthfulqa truthfulqa $dir/*.txt
python plot_data.py $output_dir/winogrande winogrande $dir/*.txt

echo -e "\nCopying important figures...\n"
python copy_important_figs.py $output_dir paper_figs

echo -e "\nDoing statistical tests...\n"
for option in {1..4}; do
    python statistical_tests.py -o $option -d $dir
done
