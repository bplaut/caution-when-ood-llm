#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: ./run_qa_tests.sh 'model1,model2' 'dataset1,dataset2' question_range"
    exit 1
fi

# Read the argument options from the command line input and split them into arrays
IFS=',' read -r -a model_options <<< "$1"
IFS=',' read -r -a dataset_options <<< "$2"
question_range="$3"


# Function to determine batch_size based on model and dataset names
get_batch_size() {
    local model_name="$1"
    local dataset_name="$2"
    local batch_size

    if [ "$model_name" = "Llama-70b" ]; then
        batch_size=20
    elif [ "$dataset_name" = "mmlu" ]; then
        batch_size=100
    elif [ "$model_name" = "Mistral" ] || [ "$model_name" = "Zephyr" ]; then
        batch_size=200
    else
        batch_size=100 # Default value
    fi

    echo "$batch_size"
}

# Loop through each combination of model and dataset (excluding question_range)
for model in "${model_options[@]}"
do
    for dataset in "${dataset_options[@]}"
    do
        # Determine batch_size based on the model
        batch_size=$(get_batch_size "$model")

        # Running the command with the arguments
        echo -e "\nRunning take_qa_test.py with arguments: --model=$model --dataset=$dataset --question_range=$question_range --batch_size=$batch_size"
        python take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --max_new_tokens=100 &> logs/"$model"_"$dataset"_"$question_range"_log.txt
    done
done

echo "Script completed."
