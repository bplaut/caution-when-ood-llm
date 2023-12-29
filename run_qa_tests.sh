#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: ./run_qa_tests.sh 'model1,model2' 'dataset1,dataset2' 'question_range1,question_range2' ['two_choices']"
    exit 1
fi

# Read the argument options from the command line input and split them into arrays
IFS=',' read -r -a model_options <<< "$1"
IFS=',' read -r -a dataset_options <<< "$2"
IFS=',' read -r -a question_ranges <<< "$3"
two_choices=${4:-}

# Function to determine batch_size based on model and dataset names
# Function to determine batch_size based on model and dataset names
get_batch_size() {
    local model_name="$1"
    local dataset_name="$2"
    local batch_size

    case "$model_name" in
        "Llama-70b")
            batch_size=16
            ;;
        "Llama-7b")
            batch_size=40
            ;;
        "Llama-13b")
            batch_size=30
            ;;
        "Mistral"|"Zephyr"|"Sakura-Solar")
            batch_size=160
            ;;
        *)
            batch_size=60 # Default value
            ;;
    esac

    # For some reason, mmlu and piqa crash with large batch sizes
    if [ "$dataset_name" = "mmlu" ]; then
        batch_size=$((batch_size / 3))
    fi
    if [ "$dataset_name" = "piqa" ]; then
	batch_size=$((batch_size / 3))
    fi

    echo "$batch_size"
}

# Loop through each combination of model, dataset, and question_range
for question_range in "${question_ranges[@]}"
do
    for dataset in "${dataset_options[@]}"
    do
        for model in "${model_options[@]}"
        do
            # Determine batch_size based on the model and dataset
            batch_size=$(get_batch_size "$model" "$dataset")

            # Modify log file name based on the presence of two_choices
            if [ -z "$two_choices" ]; then
                log_file="logs/${model}_${dataset}_${question_range}_log.txt"
            else
                log_file="logs/${model}_${dataset}_${question_range}_${two_choices}_log.txt"
            fi
	    
            # Running the command with the arguments
            echo -e "\nRunning take_qa_test.py with arguments: --model=$model --dataset=$dataset --question_range=$question_range --batch_size=$batch_size --max_new_tokens=100 $two_choices"
            if [ -z "$two_choices" ]; then
                python take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --max_new_tokens=100 &> "$log_file"
            else
                python take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --max_new_tokens=100 --two_choices="$two_choices" &> "$log_file"
            fi	    
        done
    done
done

echo "Script completed."
