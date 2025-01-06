import os
import sys
from collections import defaultdict

def get_parsing_stats(directory, filter_string=None):
    stats_per_model = defaultdict(lambda: defaultdict(int)) # maps each model to another map which maps 0..5 to counts

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filter_string is not None and filter_string not in filename:
            continue

        # Check if it's a file and not a directory, and matches the 'no_abst'/'norm' pattern
        if os.path.isfile(file_path) and 'no_abst' in filename and 'norm' in filename:
            with open(file_path, 'r') as file:
                model_name = filename.split('-q')[0].split('_')[1]
                for line in file:
                    parts = line.strip().split()
                    # line should have three parts: grade, confidence_level, and parsing_issue
                    if len(parts) == 3 and parts[0] != 'grade':  # skip header line
                        stats_per_model[model_name][5] += 1
                        stats_per_model[model_name][int(parts[2])] += 1

    return stats_per_model

def print_parsing_stats(stats_per_model):
    # Labels for printing
    labels = [
        "Response starts with A./B./C. etc",
        "Response contains A./B./C. etc but not at start",
        "Response starts with A/B/C etc",
        "Response contains A/B/C etc but not at start",
        "Could not parse",
        "Total data points"
    ]

    # Calculate overall totals
    overall_totals = [0]*6
    for model_data in stats_per_model.values():
        for i in range(6):
            overall_totals[i] += model_data[i]

    # Print stats per model, sorted alphabetically
    for model_name, model_data in sorted(stats_per_model.items()):
        total = model_data[5]
        print(f"=== {model_name} ===")
        print(f"Total data points: {total}")
        for i in range(5):
            count_i = model_data[i]
            percent_i = (count_i / total * 100) if total else 0
            print(f"{labels[i]}: {count_i} ({percent_i:.2f}%)")
        print()  # Blank line for readability

    # Print overall totals
    grand_total = overall_totals[5]
    print("=== Totals across all models ===")
    print(f"Total data points: {grand_total}")
    for i in range(5):
        count_i = overall_totals[i]
        percent_i = (count_i / grand_total * 100) if grand_total else 0
        print(f"{labels[i]}: {count_i} ({percent_i:.2f}%)")
    print()

def main():
    if len(sys.argv) not in [2, 3]:
        print("Usage: python count_unparseable.py <directory_path> [filter string]")
        sys.exit(1)

    directory_path = sys.argv[1]
    filter_string = sys.argv[2] if len(sys.argv) == 3 else None

    stats_per_model = get_parsing_stats(directory_path, filter_string)
    print_parsing_stats(stats_per_model)

if __name__ == "__main__":
    main()
