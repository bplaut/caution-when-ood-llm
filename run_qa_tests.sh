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

# Function to determine batch_size based on model name
get_batch_size() {
    local model_name="$1"
    local batch_size

    case "$model_name" in
        "Llama-70b")
            batch_size=20
            ;;
        "Llama-7b")
            batch_size=80
            ;;
        "Mistral"|"Zephyr")
            batch_size=200
            ;;
        *)
            batch_size=100 # Default value
            ;;
    esac

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
        python take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --max_new_tokens=100
    done
done

echo "Script completed."
