import os
import sys

def count_data_points(directory):
    total_data_points = 0
    total_unparseable = 0

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path) and 'no_abst' in filename and 'norm' in filename:
            with open(file_path, 'r') as file:
                for line in file:
                    # Splitting each line into grade and status
                    parts = line.strip().split()

                    # Ensure the line has two parts: grade and status
                    if len(parts) == 2 and parts[0] != 'grade':
                        total_data_points += 1

                        # Check if the status is 'Unparseable'
                        if parts[0] == 'Unparseable':
                            total_unparseable += 1

    return total_data_points, total_unparseable

# Get directory path from command line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

directory_path = sys.argv[1]
total, unparseable = count_data_points(directory_path)
print(f"Total data points: {total}")
print(f"Total Unparseable data points: {unparseable}")
