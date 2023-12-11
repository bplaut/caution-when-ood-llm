import os
import sys

def get_total_results(result_filepaths):
    
    (total_correct, total_wrong, total_abstained) = (0,0,0)
    for p in result_filepaths:
        question_range = p.split('-q')[-1]
        start_q = int(question_range.split('to')[0])
        end_q = int(question_range.split('to')[-1].replace('.txt',''))
        with open(p) as f:
            f.readline() # skip first line
            stat_line = f.readline()
            stats = stat_line.split(' | ')
            correct = int(stats[0].split(' ')[-1])
            wrong = int(stats[1].split(' ')[-1])
            abstained = int(stats[2].split(' ')[-1])
            total_correct += correct
            total_wrong += wrong
            total_abstained += abstained
    return (total_correct, total_wrong, total_abstained)
    
def main():
    # 1st arg is this file's name, 2nd is output dir, the rest are files to merge
    if len(sys.argv) <= 3: # 
        raise Exception('Must provide at least two results files')

    output_dir = sys.argv[1]
    result_filepaths = sys.argv[2:]
    print("Combining from result filepaths:", result_filepaths)
    filenames = [p.split('/')[-1] for p in result_filepaths]
    result_params = [f.split('-q')[0] for f in filenames]
    if len(set(result_params)) > 1: # multiple different models
        raise Exception('All results must be from the same model + parameters')

    (correct, wrong, abstained) = get_total_results(result_filepaths)
    output_filename = result_params[0] + '-total.txt'
    output_filepath = os.path.join(output_dir, output_filename)
    print('Writing to', output_filepath)
    with open(output_filepath, 'w') as f:
        f.write("Correct: %d | Wrong: %d | Abstained: %d" % (correct, wrong, abstained))

if __name__ == '__main__':
    main()
    
