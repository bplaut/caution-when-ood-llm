import os
import shutil
import sys

def copy_files(output_directory, filepaths):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filepath in filepaths:
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
                print("Successfully copied", new_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python copy_important_figs.py <input_dir> <output_directory>")
        sys.exit(1)
    input_dir = sys.argv[1]
    input_subdir_False = input_dir + '/collapse_False'
    input_subdir_True = input_dir + '/collapse_True'
    output_dir = sys.argv[2]
    file_list = []
    datasets = ['arc', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande']

    # AUROC and calibration plots: use collapse_prompts=False
    main_figs_dir = input_subdir_False + '/main_figs'
    cross_group_dir = main_figs_dir + '/cross_group_plots'
    suffixes = ['no_abst_all/auc_vs_acc-no_abst_norm_logits-no_abst_raw_logits',
                'no_abst_norm_logits/auc_vs_size_all_datasets_MSP',
                'no_abst_raw_logits/auc_vs_size_all_datasets_Max_Logit',
                'no_abst_norm_logits/auc_vs_acc_all_datasets_MSP',
                'no_abst_norm_logits/auc_vs_acc-no_abst_norm_logits_second_prompt-no_abst_norm_logits_first_prompt',
                'no_abst_raw_logits/auc_vs_acc_all_datasets_Max_Logit',
                'no_abst_raw_logits/auc_vs_acc-no_abst_raw_logits_second_prompt-no_abst_raw_logits_first_prompt',
                 ]
    file_list += [cross_group_dir + '/' + suffix for suffix in suffixes]
    suffixes = ['no_abst_dataset',
                'frac-correct_vs_msp_quantile',
                'calibration_table_quantile',
                'calib_vs_acc_all_datasets',
                'calib_vs_size_all_datasets',
                'no_abst_dataset_bar',
                ]
    file_list += [main_figs_dir + '/' + suffix for suffix in suffixes]
    file_list.append(cross_group_dir + '/no_abst_all/auroc_table')
    file_list.append(cross_group_dir + '/no_abst_all/auc_vs_acc-no_abst_norm_logits-no_abst_raw_logits')
    for dataset in datasets:
        file_list.append(f'{input_subdir_False}/{dataset}/cross_group_plots/no_abst_all/{dataset}_auroc_table')
    # Q&A with abstention plots (aka score plots): use collapse_prompts=True
    logit_types = ['norm_logits', 'raw_logits']
    score_types = ['score', 'harsh-score']
    for logit_type in logit_types:
        for score_type in score_types:
            file_list.append(f'{input_subdir_True}/main_figs/no_abst_{logit_type}/{score_type}_vs_conf_all_datasets')
    file_list.append(f'{input_subdir_True}/main_figs/cross_group_plots/no_abst_None/score_table')
    file_list.append(f'{input_subdir_True}/main_figs/cross_group_plots/no_abst_None/pct_abstained_table')
    for dataset in datasets:
        file_list.append(f'{input_subdir_True}/{dataset}/cross_group_plots/no_abst_None/{dataset}_score_table')
        file_list.append(f'{input_subdir_True}/{dataset}/cross_group_plots/no_abst_None/{dataset}_pct_abstained_table')
    copy_files(output_dir, file_list)

main()
