import sys

def parse_and_convert_to_tex(input_file_path, output_file_path):
    """
    Parses the input txt file and converts the data into a .tex file format.

    Args:
    input_file_path (str): Path to the input txt file.
    output_file_path (str): Path where the output .tex file will be saved.
    """
    # Read and parse the input file
    with open(input_file_path, 'r') as f:
        lines = f.readlines()

    # Process the data and prepare for LaTeX format
    data = []
    section_length = 5 # name, correct, wrong, abstained, newline
    for i in range(0, len(lines), section_length):
        name = lines[i].split('.txt')[0].replace('_', ' ')
        correct = int(lines[i+1].split(':')[1].strip())
        wrong = int(lines[i+2].split(':')[1].strip())
        abstained = int(lines[i+3].split(':')[1].strip())
        score_1 = correct - wrong
        score_2 = correct - 2 * wrong
        data.append([name, correct, wrong, abstained, score_1, score_2])

    # Convert the data to LaTeX table format
    with open(output_file_path, 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\begin{document}\n')
        f.write('\\begin{tabular}{c|c|c|c|c|c|}\n')
        f.write('\\hline\n')
        f.write('Name & Correct & Wrong & Abstained & Score 1 & Score 2 \\\\\n')
        f.write('\\hline\n')
        for row in data:
            f.write(f'{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\\n')
            f.write('\\hline\n')
        f.write('\\end{tabular}\n')
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
