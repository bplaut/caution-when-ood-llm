import os
import shutil

def copy_files(output_directory, filepaths):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filepath in filepaths:
        print("Trying to copy ", filepath)
        extensions = ['.pdf', '.png', '.tex']
        for extension in extensions:
            full_path = filepath + extension
            if os.path.isfile(full_path):
                # Copy file to output directory
                base_name = os.path.basename(full_path)[:-4]
                dir_name = os.path.dirname(full_path)
                prompt_str = '' if 'prompt' in base_name else '_first_prompt' if 'first_prompt' in dir_name else '_second_prompt' if 'second_prompt' in dir_name else ''
                logit_str = '' if ('logit' in base_name.lower() or 'MSP' in base_name) else '_norm_logits' if 'norm_logits' in dir_name else '_raw_logits' if 'raw_logits' in dir_name else ''
                new_path = os.path.join(output_directory, base_name + prompt_str + logit_str + extension)
                shutil.copy(full_path, new_path)
                print("Successfully copied to ", new_path)

output_dir = 'paper_figs'
cross_group_dir = 'figs/main_figs/cross_group_plots'
suffixes = ['/no_abst_all/acc_vs_auc-no_abst_norm_logits-no_abst_raw_logits',
            '/no_abst_norm_logits/acc_vs_auc_all_datasets_MSP',
            '/no_abst_norm_logits/acc_vs_auc-no_abst_norm_logits_second_prompt-no_abst_norm_logits_first_prompt',
            '/no_abst_raw_logits/acc_vs_auc_all_datasets_Max_Logit',
            '/no_abst_raw_logits/acc_vs_auc-no_abst_raw_logits_second_prompt-no_abst_raw_logits_first_prompt',
             ]
file_list = [cross_group_dir + suffix for suffix in suffixes] + ['figs/main_figs/' + suffix for suffix in suffixes]
datasets = ['arc', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande', 'piqa', 'no_winogrande']
middle_dirs = ['_abst_norm_logits_first_prompt', '_abst_norm_logits_second_prompt', '_abst_raw_logits_first_prompt', '_abst_raw_logits_second_prompt', '_abst_norm_logits', '_abst_raw_logits']
for middle_dir in middle_dirs:
    file_list += [f'figs/main_figs/yes{middle_dir}/test/score_vs_conf_all_datasets']
    file_list += [f'figs/main_figs/yes{middle_dir}/test/harsh-score_vs_conf_all_datasets']
file_list += [cross_group_dir + '/no_abst_None/auroc_table', cross_group_dir + '/no_abst_None/acc_vs_auc-no_abst_raw_logits-no_abst_norm_logits', cross_group_dir + '/no_abst_None/score_table']
for table_type in ['auroc', 'score']:
    file_list += [f'figs/{dataset}/cross_group_plots/no_abst_None/{dataset}_{table_type}_table' for dataset in datasets]
copy_files(output_dir, file_list)
print("Copied files to", output_dir)
