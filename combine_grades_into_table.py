import os
import sys

def get_total_results(grade_filepaths, thresh):
    grades = {'Correct':0, 'Wrong':0, 'Abstained':0}
    for p in grade_filepaths:
        with open(p) as f:
            for line in f:
                grade = line.split(' ')[0]
                confidence = line.split(' ')[1]
                if grade in grades: # this skips the header line
                    if float(confidence) < thresh:
                        grades['Abstained'] += 1
                    else:
                        grades[grade] += 1
    return grades

def write_to_table(output_filepath, all_grades):
    # Convert the data to LaTeX table format                                                               
    with open(output_filepath, 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\renewcommand\\arraystretch{1.4}\n')
        f.write('\\begin{document}\n')
        for dataset_name in sorted(all_grades.keys()):
            f.write(f'\\section*{{Dataset: {dataset_name}}}\n')
            f.write('\\begin{tabular}{c|c|c|c|c|c|c}\n')
            f.write('\\hline\n')
            f.write('Model name & Correct & Wrong & Abstained & C - W & C / W & Total \\\\\n')
            f.write('\\hline\n')
            for row_name in sorted(all_grades[dataset_name].keys()):
                correct = all_grades[dataset_name][row_name]['Correct']
                wrong = all_grades[dataset_name][row_name]['Wrong']
                abstained = all_grades[dataset_name][row_name]['Abstained']
                ratio = round(correct/wrong, 2) if wrong > 0 else '-'
                f.write(f'{row_name} & {correct} & {wrong} & {abstained} & {correct - wrong} & {ratio} & {correct + wrong + abstained} \\\\\n')
                f.write('\\hline\n')
            f.write('\\end{tabular}\n')
            f.write('\\newpage\n')  # Optional: Start each group table on a new page                          
        f.write('\\end{document}')
    print(f"Successfully wrote to {output_filepath}")    

def main():
    if len(sys.argv) <= 3:
        raise Exception('Usage: python add_up_grades.py [output_filepath] [comma-separated list of confidence thresholds] [add least one grades file]')

    output_filepath = sys.argv[1]
    thresholds = [float(t) for t in sys.argv[2].split(',')]
    grade_filepaths = sys.argv[3:]
    
    print("Combining from these grade files:")
    [print(f) for f in grade_filepaths]
    filenames = [p.split('/')[-1] for p in grade_filepaths]
    all_param_sets = set([f.split('-q')[0] for f in filenames])

    all_grades = dict()
    for thresh in thresholds:
        for param_set in all_param_sets:
            paths_to_use = [p for p in grade_filepaths if p.split('/')[-1].split('-q')[0] == param_set]
            grades_for_param_set = get_total_results(paths_to_use, thresh)
            dataset_name = param_set.split('_')[0]
            model_name = param_set.split('_')[1]
            if dataset_name not in all_grades:
                all_grades[dataset_name] = dict()
            params_str = f"{model_name}, thresh={thresh}"
            params_str = params_str.replace('_',' ').replace('thresh=0.0', 'base model')
            all_grades[dataset_name][params_str] = grades_for_param_set
    write_to_table(output_filepath, all_grades)

if __name__ == '__main__':
    main()
    
