import sys
import os
import datetime

def count_string_per_file(directory, string, date=None):
    """Returns a list of tuples containing the file name and the number of times the string appears in the file, 
    considering only files created after the specified date if provided."""
    file_list = os.listdir(directory)
    result = []
    for file in file_list:
        file_path = os.path.join(directory, file)
        # Make sure it's not a directory and, if a date is provided, that the file was created after that date
        if not os.path.isdir(file_path) and (date is None or file_created_after(file_path, date)):
            with open(file_path, 'r') as f:
                contents = f.read()
                count = contents.count(string)
                result.append((file, count))
    # Sort the list by the second element of each tuple (the count)
    result.sort(key=lambda tup: tup[1])
    return result

def file_created_after(file_path, date):
    """Checks if the file was created after the given date."""
    file_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
    return file_creation_time > date

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 count_string_per_file.py <directory> <string> [date]")
        sys.exit(1)

    directory = sys.argv[1]
    string = sys.argv[2]
    date = None

    if len(sys.argv) == 4:
        date_str = sys.argv[3]
        try:
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            sys.exit(1)

    result = count_string_per_file(directory, string, date)
    for file, count in result:
        # Print the count, followed by a tab, followed by the file name
        print(str(count) + "\t" + file)

if __name__ == "__main__":
    main()
