import os
import shutil
from typing import List

def copy_files(output_directory: str, filepaths: List[str]):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filepath in filepaths:
        filepath = filepath + '.pdf'
        print("Copying ", filepath)
        if os.path.isfile(filepath):
            # Copy file to output directory
            base_name = os.path.basename(filepath)[:-4]
            dir_name = os.path.dirname(filepath)
            prompt_str = '' if 'prompt' in base_name else '_first_prompt' if 'first_prompt' in dir_name else '_second_prompt' if 'second_prompt' in dir_name else ''
            logit_str = '' if ('logit' in base_name.lower() or 'MSP' in base_name) else '_norm_logits' if 'norm_logits' in dir_name else '_raw_logits' if 'raw_logits' in dir_name else ''
            new_path = os.path.join(output_directory, base_name + prompt_str + logit_str + '.png')
            shutil.copy(filepath, new_path)
        else:
            print(f"File not found: {filepath}")

output_dir = 'paper_figs'
cross_group_dir = 'figs/main_figs/cross_group_plots'
file_list = [cross_group_dir + '/no_abst_all/acc_vs_auc-no_abst_norm_logits-no_abst_raw_logits',
             cross_group_dir + '/no_abst_norm_logits/acc_vs_auc_all_datasets_MSP',
             cross_group_dir + '/no_abst_norm_logits/acc_vs_auc-no_abst_norm_logits_second_prompt-no_abst_norm_logits_first_prompt',
             cross_group_dir + '/no_abst_raw_logits/acc_vs_auc_all_datasets_Max_Logit',
             cross_group_dir + '/no_abst_raw_logits/acc_vs_auc-no_abst_raw_logits_second_prompt-no_abst_raw_logits_first_prompt',
             ]
datasets = ['arc', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande']
middle_dirs = ['_abst_norm_logits_first_prompt', '_abst_norm_logits_second_prompt', '_abst_raw_logits_first_prompt', '_abst_raw_logits_second_prompt']
for middle_dir in middle_dirs:
    file_list += [f'figs/main_figs/no{middle_dir}/roc_curve_{dataset}' for dataset in datasets]
    file_list += [f'figs/piqa/no{middle_dir}/roc_curve_piqa']
    file_list += [f'figs/main_figs/yes{middle_dir}/test/score_vs_conf_all_datasets']
    file_list += [f'figs/main_figs/yes{middle_dir}/test/harsh-score_vs_conf_all_datasets']
copy_files(output_dir, file_list)
print("Copied files to", output_dir)
