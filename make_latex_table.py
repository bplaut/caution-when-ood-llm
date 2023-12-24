import sys
from collections import defaultdict

def parse_and_convert_to_tex(input_file_path, output_file_path):
    """
    Parses the input txt file and converts the data into a .tex file format,
    creating separate tables for each group.

    Args:
    input_file_path (str): Path to the input txt file.
    output_file_path (str): Path where the output .tex file will be saved.
    """
    # Read and parse the input file
    with open(input_file_path, 'r') as f:
        lines = f.readlines()

    # Process the data and prepare for LaTeX format
    data = defaultdict(list)
    section_length = 5 # name, correct, wrong, abstained, newline
    for i in range(0, len(lines), section_length):
        full_name = lines[i].split('.txt')[0]
        group_name, name = full_name.split('_', 1)
        name = name.replace('_',' ')
        correct = int(lines[i+1].split(':')[1].strip())
        wrong = int(lines[i+2].split(':')[1].strip())
        abstained = int(lines[i+3].split(':')[1].strip())
        score_1 = correct - wrong
        score_2 = correct - 2 * wrong
        total = correct + wrong + abstained
        data[group_name].append([name, correct, wrong, abstained, score_1, score_2, total])

    # Convert the data to LaTeX table format
    with open(output_file_path, 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\renewcommand\\arraystretch{1.4}\n')
        f.write('\\begin{document}\n')
        for group_name, rows in data.items():
            f.write(f'\\section*{{{group_name}}}\n')
            f.write('\\begin{tabular}{c|c|c|c|c|c|c}\n')
            f.write('\\hline\n')
            f.write('Name & Correct & Wrong & Abstained & C - W & C - 2*W & Total \\\\\n')
            f.write('\\hline\n')
            for row in rows:
                f.write(f'{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\\n')
                f.write('\\hline\n')
            f.write('\\end{tabular}\n')
            f.write('\\newpage\n')  # Optional: Start each group table on a new page
        f.write('\\end{document}')
    print(f"Successfully wrote to {output_file_path}")

def main():
    if len(sys.argv) != 3:
        raise Exception('Usage: python make_latex_table.py input_file output_file')
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    parse_and_convert_to_tex(input_file, output_file)

if __name__ == "__main__":
    main()
