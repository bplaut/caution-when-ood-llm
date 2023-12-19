import os
import sys

def main():
    # 1st arg is this file's name, 2nd is output dir, the rest are files to merge
    if len(sys.argv) <= 3: # 
        raise Exception('Must provide at least two results files')

    output_filepath = sys.argv[1]
    result_filepaths = sys.argv[2:]
    result_filepaths.sort()
    print("Merging from result filepaths:", result_filepaths)

    print('\nWriting to', output_filepath)
    with open(output_filepath, 'w') as f1:
        for result_filepath in result_filepaths:
            with open(result_filepath, 'r') as f2:
                results = f2.read()
                result_filename = result_filepath.split('/')[-1]
                f1.write(result_filename + '\n')
                f1.write(results + '\n\n')

if __name__ == '__main__':
    main()
    
