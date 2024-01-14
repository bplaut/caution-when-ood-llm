import os
import sys

def get_total_results(grade_filepaths, thresh, incl_unparseable):
    grades = {'Correct':0, 'Wrong':0, 'Abstained':0, 'TP':0, 'FP':0, 'P':0, 'N':0}
    for p in grade_filepaths:
        with open(p) as f:
            for line in f:
                grade = line.split(' ')[0]
                confidence = line.split(' ')[1]
                # If incl_unparseable, we count unparseable responses as wrong
                if grade in grades or (grade == 'Unparseable' and incl_unparseable): # skips header
                    grade = 'Wrong' if grade == 'Unparseable' else grade
                    # For computing the net score
                    if float(confidence) < thresh:
                        grades['Abstained'] += 1
                    else:
                        grades[grade] += 1
                    # For computing the FPR and TPR
                    if grade == 'Correct':
                        grades['P'] += 1
                        if float(confidence) >= thresh:
                            grades['TP'] += 1
                    elif grade == 'Wrong':
                        grades['N'] += 1
                        if float(confidence) >= thresh:
                            grades['FP'] += 1
    return grades

def write_to_table(output_filepath, all_grades):
    # Convert the data to LaTeX table format                                                               
    with open(output_filepath, 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\usepackage[left=1in, right=1in, top=1in, bottom=1in]{geometry}\n')
        f.write('\\renewcommand\\arraystretch{1.35}\n')
        f.write('\\usepackage{amsmath}\n')
        f.write('\\pagestyle{empty}\n')
        f.write('\\begin{document}\n')
        for dataset_name in sorted(all_grades.keys()):
            f.write(f'\\section*{{Dataset: {dataset_name}}}\n')
            f.write('\\begin{tabular}{c|c|c|c|c|c|c|c|c|c}\n')
            f.write('\\hline\n')
            f.write('Model name & Correct & Wrong & Abst. & C - W & $\\frac{\\text{C-W}}{\\text{Total}}$ & Acc. & TPR & FPR & Total \\\\\n')
            f.write('\\hline\n')
            for row_name in sorted(all_grades[dataset_name].keys()):
                correct = all_grades[dataset_name][row_name]['Correct']
                wrong = all_grades[dataset_name][row_name]['Wrong']
                abstained = all_grades[dataset_name][row_name]['Abstained']
                coverage = round((correct + wrong) / (correct + wrong + abstained), 3)
                accuracy = round(correct/(correct + wrong), 3) if (correct + wrong) > 0 else '-'
                net_score = round((correct - wrong)/(correct + wrong + abstained), 3)
                fpr = round(all_grades[dataset_name][row_name]['FP'] / all_grades[dataset_name][row_name]['N'], 3) if all_grades[dataset_name][row_name]['N'] > 0 else '-'
                tpr = round(all_grades[dataset_name][row_name]['TP'] / all_grades[dataset_name][row_name]['P'], 3) if all_grades[dataset_name][row_name]['P'] > 0 else '-'
                # bold rows for base models
                if "base model" in row_name:
                    f.write(f'\\textbf{{{row_name}}} & \\textbf{{{correct}}} & \\textbf{{{wrong}}} & \\textbf{{{abstained}}} & \\textbf{{{correct - wrong}}} & \\textbf{{{net_score}}} & \\textbf{{{accuracy}}}  & \\textbf{{{tpr}}} & \\textbf{{{fpr}}} & {correct + wrong + abstained} \\\\\n')
                else:
                    f.write(f'{row_name} & {correct} & {wrong} & {abstained} & {correct - wrong} & {net_score} & {accuracy} & {tpr} & {fpr} & {correct + wrong + abstained} \\\\\n')
                f.write('\\hline\n')
            f.write('\\end{tabular}\n')
            f.write('\\newpage\n')  # Optional: Start each group table on a new page                          
        f.write('\\end{document}')
    print(f"Successfully wrote to {output_filepath}")    

def main():
    if len(sys.argv) <= 4:
        raise Exception('Usage: python combine_grades_into_table.py [output_filepath] [comma-separated list of confidence thresholds] [include unparseable grades] [add least one grades file]')

    output_filepath = sys.argv[1]
    thresholds = [float(t) for t in sys.argv[2].split(',')]
    incl_unparseable = (False if sys.argv[3].lower() == 'false' else
                        True if sys.argv[3].lower() == 'true' else None)
    if incl_unparseable is None:
        raise Exception('Third argument incl_unparseable must be a boolean (True or False)')
    grade_filepaths = sys.argv[4:]
    
    print(f"Combining {len(grade_filepaths)} grade files into {output_filepath}...")
    filenames = [p.split('/')[-1] for p in grade_filepaths]
    all_param_sets = set([f.split('-q')[0] for f in filenames])

    all_grades = dict()
    for thresh in thresholds:
        for param_set in all_param_sets:
            paths_to_use = [p for p in grade_filepaths if p.split('/')[-1].split('-q')[0] == param_set]
            grades_for_param_set = get_total_results(paths_to_use, thresh, incl_unparseable)
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
    
