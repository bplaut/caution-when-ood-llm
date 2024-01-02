#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: ./run_qa_tests.sh 'model1,model2' 'dataset1,dataset2' 'question_range1,question_range2' ['abstain_option']"
    exit 1
fi

# Read the argument options from the command line input and split them into arrays
IFS=',' read -r -a model_options <<< "$1"
IFS=',' read -r -a dataset_options <<< "$2"
IFS=',' read -r -a question_ranges <<< "$3"
abstain_option=${4:-}

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
            batch_size=22
            ;;
        "Falcon-40b")
            batch_size=10
            ;;
	"Falcon-7b")
	    batch_size=253
	    ;;
        "Mistral"|"Zephyr"|"Solar")
            batch_size=128
            ;;
        *)
            batch_size=63 # Default value
            ;;
    esac

    # For some reason, mmlu and piqa crash with large batch sizes
    if [ "$dataset_name" = "mmlu" ]; then
        batch_size=$((batch_size / 3))
    fi
    if [ "$dataset_name" = "piqa" ]; then
	batch_size=$((batch_size / 2))
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

            # Modify log file name based on the presence of abstain_option
            if [ -z "$abstain_option" ]; then
                log_file="logs/${model}_${dataset}_${question_range}_no-abstain-option_log.txt"
            else
                log_file="logs/${model}_${dataset}_${question_range}_yes-abstain-option_log.txt"
            fi
	    
            # Running the command with the arguments
            if [ -z "$abstain_option" ]; then
		echo -e "\nRunning take_qa_test.py with arguments: --model=$model --dataset=$dataset --question_range=$question_range --batch_size=$batch_size"
                python take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --max_new_tokens=100 --num_top_tokens=1 &> "$log_file"
            else
		echo -e "\nRunning take_qa_test.py with arguments: --model=$model --dataset=$dataset --question_range=$question_range --batch_size=$batch_size --abstain_option"
                python take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --max_new_tokens=100 --num_top_tokens=1 --abstain_option &> "$log_file"
            fi	    
        done
    done
done

echo "Script completed."
