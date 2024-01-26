import scipy.stats as stats
from utils import *
import argparse
import os
import re
import numpy as np
from sklearn.metrics import roc_curve, auc
from collections import defaultdict


ALL_DATASETS = ["arc", "hellaswag", "mmlu", "piqa", "truthfulqa", "winogrande"]
ALL_MODELS = ["Falcon-7b", "Falcon-40b", "Llama-7b", "Llama-13b", "Llama-70b", "Mistral", "Mixtral", "Solar", "Yi-6b", "Yi-34b"]


def test_large_sample(data, threshold = 30):
    return len(data), len(data) >= threshold


def test_normality(data, mode = "ks", threshold = 0.05):
    """
    Shapiro-Wilk: smaller datasets
    Kolmogorov-Smirnov: bigger datasets
    """
    if mode == "sw":
        p_value = stats.shapiro(data).pvalue
    elif mode == "ks":
        p_value = stats.kstest(data, "norm").pvalue
    return p_value, p_value > threshold


def test_equal_variance(data1, data2, mode = "bartlett", threshold = 0.05):
    """
    Levene: less sensitive
    Bartlett: more sensitive
    """
    if mode == "lv":
        p_value = stats.levene(data1, data2).pvalue
    elif mode == "bl":
        p_value = stats.bartlett(data1, data2).pvalue
    return p_value, p_value > threshold


def mann_whitney(data1, data2, threshold = 0.05):
    u_statistic, u_p_value = stats.mannwhitneyu(data1, data2)
    return u_p_value, u_statistic, u_p_value < threshold


def unpaired_z(data1, data2, threshold = 0.05):
    z_statistic, z_p_value = stats.ttest_ind(data1, data2, equal_var = True)
    return z_p_value, z_statistic, z_p_value < threshold


def one_sample_t(data, expected_mean, alternative = "two-sided", threshold = 0.05):
    test_result = stats.ttest_1samp(data, expected_mean, alternative = alternative)
    return test_result.pvalue, test_result.statistic, test_result.df, test_result.pvalue < threshold


def wilcoxon(data, expected_mean, alternative = "two-sided", threshold = 0.05):
    diffs = list(map(lambda x: x - expected_mean, data))
    test_result = stats.wilcoxon(diffs, alternative = alternative)
    return test_result.pvalue, test_result.statistic, -1, test_result.pvalue < threshold


def test_assumptions(data1, data2):
    _, sample1_large = test_large_sample(data1)
    _, sample2_large = test_large_sample(data2)
    large_sample = sample1_large and sample2_large
    _, normal1 = test_normality(data1)
    _, normal2 = test_normality(data2)
    normal = normal1 and normal2
    equal_variance = test_equal_variance(data1, data2)
    return (large_sample or normal) and equal_variance


def _collect_model_and_dataset_data(value, prompt, incl_unparseable):
    results_dir = "./results_first_prompt/" if prompt == 1 else "./results_second_prompt/"
    all_results = os.listdir(results_dir)
    results_suffix = "_raw_logits.txt" if value == "logit" else "_norm_logits.txt"
    all_data = defaultdict(lambda: defaultdict(lambda: ([], [], 0)))
    for dataset in ALL_DATASETS:
        for model in ALL_MODELS:
            results_pattern = fr"{dataset}_{model}-.*{results_suffix}"
            relevant_files = [file for file in all_results if re.match(results_pattern, file)]
            for file in relevant_files:
                labels, conf_levels, total_qs = parse_data(results_dir + file, incl_unparseable)
                old_labels, old_conf_levels, old_total_qs = all_data[dataset][model]
                all_data[dataset][model] = (np.concatenate([old_labels, labels]), np.concatenate([old_conf_levels, conf_levels]), old_total_qs + total_qs)
    return all_data


def conduct_all_average_test(value, prompt, incl_unparseable):
    all_data = _collect_model_and_dataset_data(value, prompt, incl_unparseable)
    aurocs = []
    for dataset in ALL_DATASETS:
        for model in ALL_MODELS:
            labels, conf_levels, _ = all_data[dataset][model]
            if len(labels) > 0:
                fpr, tpr, _ = roc_curve(labels, conf_levels)
                auroc = auc(fpr, tpr)
                aurocs.append(auroc)
    print(f"Results for all_average_test using prompt {prompt} and {value}s")
    _, is_normal = test_normality(aurocs)
    if is_normal:  # one-sample t-test
        print("Data passed t-test assumptions")
        p_val, stat, df, verdict = one_sample_t(aurocs, 0.5, alternative = "greater")
    else:  # wilcoxon signed rank test
        print("Data did not pass t-test assumptions")
        p_val, stat, df, verdict = wilcoxon(aurocs, 0.5, alternative = "greater")  
    print("P-value:", p_val)
    print("Statistic:", stat)
    print("DoF:", df)
    print("Null rejected:", verdict)
    print("\n")


def conduct_model_average_test(value, prompt, incl_unparseable):
    all_data = _collect_model_and_dataset_data(value, prompt, incl_unparseable)
    aurocs = {model: [] for model in ALL_MODELS}
    for model in ALL_MODELS:
        for dataset in ALL_DATASETS:
            labels, conf_levels, _ = all_data[dataset][model]
            if len(labels) > 0:
                fpr, tpr, _ = roc_curve(labels, conf_levels)
                auroc = auc(fpr, tpr)
                aurocs[model].append(auroc)
    print(f"Results for model_average_test using prompt {prompt} and {value}s")
    for model in ALL_MODELS:
        if len(aurocs[model]) > 0:
            print("For model", model)
            _, is_normal = test_normality(aurocs[model])
            if is_normal:  # one-sample t-test
                print("Data passed t-test assumptions")
                p_val, stat, df, verdict = one_sample_t(aurocs[model], 0.5, alternative = "greater")
            else:  # wilcoxon signed rank test
                print("Data did not pass t-test assumptions")
                p_val, stat, df, verdict = wilcoxon(aurocs[model], 0.5, alternative = "greater")  
            print("P-value:", p_val)
            print("Statistic:", stat)
            print("DoF:", df)
            print("Null rejected:", verdict)
            print()
        print()


def conduct_dataset_average_test(value, prompt, incl_unparseable):
    all_data = _collect_model_and_dataset_data(value, prompt, incl_unparseable)
    aurocs = {dataset: [] for dataset in ALL_DATASETS}
    for dataset in ALL_DATASETS:
        for model in ALL_MODELS:
            labels, conf_levels, _ = all_data[dataset][model]
            if len(labels) > 0:
                fpr, tpr, _ = roc_curve(labels, conf_levels)
                auroc = auc(fpr, tpr)
                aurocs[dataset].append(auroc)
    print(f"Results for dataset_average_test using prompt {prompt} and {value}s")
    for dataset in ALL_DATASETS:
        if len(aurocs[dataset]) > 0:
            print("For dataset", dataset)
            _, is_normal = test_normality(aurocs[dataset])
            if is_normal:  # one-sample t-test
                print("Data passed t-test assumptions")
                p_val, stat, df, verdict = one_sample_t(aurocs[dataset], 0.5, alternative = "greater")
            else:  # wilcoxon signed rank test
                print("Data did not pass t-test assumptions")
                p_val, stat, df, verdict = wilcoxon(aurocs[dataset], 0.5, alternative = "greater")  
            print("P-value:", p_val)
            print("Statistic:", stat)
            print("DoF:", df)
            print("Null rejected:", verdict)
            print()
        print()


def conduct_all_combo_test(value, prompt, incl_unparseable):
    all_data = _collect_model_and_dataset_data(value, prompt, incl_unparseable)
    print(f"Results for dataset_average_test using prompt {prompt} and {value}s")
    for dataset in ALL_DATASETS:
        for model in ALL_MODELS:
            labels, conf_levels, _ = all_data[dataset][model]
            if len(labels) > 0:
                labels = np.array(labels)
                conf_levels = np.array(conf_levels)
                conf_levels_right = conf_levels[labels == 1]
                conf_levels_wrong = conf_levels[labels == 0]
                print(f"For dataset {dataset} and model {model}")
                p_val, stat, verdict = mann_whitney(conf_levels_right, conf_levels_wrong)
                print("P-value:", p_val)
                print("Statistic:", stat)
                print("Null rejected:", verdict)
                print()
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options = {
        0: "all_average",
        1: "model_average",
        2: "dataset_average",
        3: "all_combos"
    }
    parser.add_argument("--value", "-v", required = True, type = str, help = "Either 'logit' or 'prob'")
    parser.add_argument("--prompt", "-p", required = True, type = int)
    parser.add_argument("--option", "-o", required = True, type = int, help = f"Choose a number from {list(options.keys())}")
    parser.add_argument("--incl_unparseable", "-i", action = "store_true")
    args = parser.parse_args()
    option = options[args.option]

    if option == "all_average":
        # H0: Average AUROC across all models and datasets ≤ 0.5
        # H1: Average AUROC across all models and datasets > 0.5
        conduct_all_average_test(args.value, args.prompt, args.incl_unparseable)
    elif option == "model_average":
        # H0: Average AUROC across datasets for each model ≤ 0.5
        # H1: Average AUROC across datasets for each model > 0.5
        conduct_model_average_test(args.value, args.prompt, args.incl_unparseable)
    elif option == "dataset_average":
        # H0: Average AUROC across models for each dataset ≤ 0.5
        # H1: Average AUROC across models for each dataset > 0.5
        conduct_dataset_average_test(args.value, args.prompt, args.incl_unparseable)
    elif option == "all_combos":
        # H0: AUROC for each model-dataset pair ≤ 0.5
        # H1: AUROC for each model-dataset pair > 0.5
        conduct_all_combo_test(args.value, args.prompt, args.incl_unparseable)
