import os
import sys
from collections import defaultdict

def count_data_points(directory):
    data_points_per_model = defaultdict(int)
    unparseable_per_model = defaultdict(int)

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path) and 'no_abst' in filename and 'norm' in filename:
            with open(file_path, 'r') as file:
                model_name = filename.split('-q')[0].split('_')[1]
                for line in file:
                    # Splitting each line into grade and status
                    parts = line.strip().split()

                    # Ensure the line has two parts: grade and status
                    if len(parts) == 2 and parts[0] != 'grade':
                        data_points_per_model[model_name] += 1

                        # Check if the status is 'Unparseable'
                        if parts[0] == 'Unparseable':
                            unparseable_per_model[model_name] += 1

    return data_points_per_model, unparseable_per_model

# Get directory path from command line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

directory_path = sys.argv[1]
data_points, unparseable = count_data_points(directory_path)
print(f"Num data points, num unparseable, percentage unparseable: {sum(data_points.values())}, {sum(unparseable.values())}, {100 * round(sum(unparseable.values())/sum(data_points.values()),4)}")

# Print the number of data points and unparseable data points for each model
for model, count in data_points.items():
    print(f"{model}: {count} data points, {unparseable[model]} unparseable, {100* round(unparseable[model]/count,4)} percentage unparseable")
