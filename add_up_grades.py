import os
import sys

def get_total_results(grade_filepaths, thresh):
    grades_dict = {'Correct':0, 'Wrong':0, 'Abstained':0}
    for p in grade_filepaths:
        with open(p) as f:
            for line in f:
                grade = line.split(' ')[0]
                confidence = line.split(' ')[1]
                if grade in grades_dict: # this skips the header line
                    if float(confidence) < thresh:
                        grades_dict['Abstained'] += 1
                    else:
                        grades_dict[grade] += 1
    return grades_dict
    
def main():
    # 1st arg is this file's name, 2nd is output dir, 3rd is comma-separated list of confidence thresholds, the rest are files to add up
    if len(sys.argv) <= 3:
        raise Exception('Usage: python add_up_grades.py [output_dir] [comma-separated list of confidence thresholds] [add least one grades file]')    

    output_dir = sys.argv[1]
    thresholds = [float(t) for t in sys.argv[2].split(',')]
    grade_filepaths = sys.argv[3:]
    
    print("Combining from these grade files:", grade_filepaths)
    filenames = [p.split('/')[-1] for p in grade_filepaths]
    all_param_sets = set([f.split('-q')[0] for f in filenames])

    for thresh in thresholds:
        for param_set in all_param_sets:
            paths_to_use = [p for p in grade_filepaths if p.split('/')[-1].split('-q')[0] == param_set]
            grades_dict = get_total_results(paths_to_use, thresh)
            output_filename = param_set + f'_thresh-{thresh}.txt'
            output_filepath = os.path.join(output_dir, output_filename)
            print('Writing to', output_filepath)
            with open(output_filepath, 'w') as f:
                for grade_type in grades_dict:
                    f.write(f"{grade_type}: {grades_dict[grade_type]}\n")

if __name__ == '__main__':
    main()
    
