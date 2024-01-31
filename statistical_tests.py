import scipy.stats as stats
from utils import *
import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from collections import defaultdict


ALL_DATASETS = ["arc", "hellaswag", "mmlu", "truthfulqa", "winogrande"]
ALL_MODELS = ["Falcon-7b", "Falcon-40b", "Llama-7b", "Llama-13b", "Llama-70b", "Mistral", "Mixtral", "Solar", "Yi-6b", "Yi-34b"]
ALL_PROMPTS = ["first_prompt", "second_prompt"]
ALL_VALUES = ["raw_logits", "norm_logits"]


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
    return test_result.pvalue, test_result.statistic, test_result.pvalue < threshold


def test_assumptions(data1, data2):
    _, sample1_large = test_large_sample(data1)
    _, sample2_large = test_large_sample(data2)
    large_sample = sample1_large and sample2_large
    _, normal1 = test_normality(data1)
    _, normal2 = test_normality(data2)
    normal = normal1 and normal2
    equal_variance = test_equal_variance(data1, data2)
    return (large_sample or normal) and equal_variance


def build_confidence_interval(data, alpha = 0.05):
    lower_percentile = alpha * 50
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return lower_bound, upper_bound


def _collect_model_and_dataset_data(incl_unparseable):
    # index data by data[prompt][value][dataset][model]
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], 0)))))
    for prompt in ALL_PROMPTS:
        results_dir = f"./results_{prompt}/"
        all_results = os.listdir(results_dir)
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    results_pattern = fr"{dataset}_{model}-.*_no_abst_{value}.txt"
                    relevant_files = [file for file in all_results if re.match(results_pattern, file)]
                    for file in relevant_files:
                        labels, conf_levels, total_qs = parse_data(results_dir + file, incl_unparseable)
                        old_labels, old_conf_levels, old_total_qs = all_data[prompt][value][dataset][model]
                        all_data[prompt][value][dataset][model] = (np.concatenate([old_labels, labels]), np.concatenate([old_conf_levels, conf_levels]), old_total_qs + total_qs)
    return all_data


def conduct_mann_whitney_tests(incl_unparseable):
    all_data = _collect_model_and_dataset_data(incl_unparseable)
    test_data = {"prompt": [], "value": [], "dataset": [], "model": [], "p_value": [], "u_stat": [], "reject": []}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    labels, conf_levels, _ = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        labels = np.array(labels)
                        conf_levels = np.array(conf_levels)
                        conf_levels_right = conf_levels[labels == 1]
                        conf_levels_wrong = conf_levels[labels == 0]
                        p_val, stat, verdict = mann_whitney(conf_levels_right, conf_levels_wrong)
                        test_data["prompt"].append(prompt)
                        test_data["value"].append(value)
                        test_data["dataset"].append(dataset)
                        test_data["model"].append(model)
                        test_data["p_value"].append(p_val)
                        test_data["u_stat"].append(stat)
                        test_data["reject"].append(verdict)
                    else:
                        print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
    pd.DataFrame(test_data).to_csv("./results_stat_tests/mann_whitney.csv", index = False)


def construct_confidence_intervals(incl_unparseable):
    all_data = _collect_model_and_dataset_data(incl_unparseable)
    test_data = {"prompt": [], "value": [], "dataset": [], "model": [], "sample_auroc": [], "ci_lb": [], "ci_ub": []}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    labels, conf_levels, _ = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        fpr, tpr, __ = roc_curve(labels, conf_levels)
                        sample_auroc = auc(fpr, tpr)
                        bootstrapped_aurocs = []
                        labels = np.array(labels)
                        conf_levels = np.array(conf_levels)
                        for i in range(1000):
                            indices = resample(np.arange(len(labels)))
                            bootstrapped_labels = labels[indices]
                            bootstrapped_conf_levels = conf_levels[indices]
                            fpr, tpr, ___ = roc_curve(bootstrapped_labels, bootstrapped_conf_levels)
                            bootstrapped_aurocs.append(auc(fpr, tpr))
                        lower_bound, upper_bound = build_confidence_interval(bootstrapped_aurocs)
                        test_data["prompt"].append(prompt)
                        test_data["value"].append(value)
                        test_data["dataset"].append(dataset)
                        test_data["model"].append(model)
                        test_data["sample_auroc"].append(sample_auroc)
                        test_data["ci_lb"].append(lower_bound)
                        test_data["ci_ub"].append(upper_bound)
                    else:
                        print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
    pd.DataFrame(test_data).to_csv("./results_stat_tests/confidence_intervals.csv", index = False)


def conduct_model_summary_tests(incl_unparseable):
    all_data = _collect_model_and_dataset_data(incl_unparseable)
    test_data = {"model": [], "prompt": [], "value": [], "p_value": [], "t_stat": [], "t_dof": [], "w_stat": [], "reject": []}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for model in ALL_MODELS:
                model_aurocs = []
                for dataset in ALL_DATASETS:
                    labels, conf_levels, _ = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        fpr, tpr, __ = roc_curve(labels, conf_levels)
                        auroc = auc(fpr, tpr)
                        model_aurocs.append(auroc)
                    else:
                        print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
                _, is_normal = test_normality(model_aurocs)
                if is_normal:
                    p_val, stat, df, verdict = one_sample_t(model_aurocs, 0.5, alternative = "greater")
                    test_data["p_value"].append(p_val)
                    test_data["t_stat"].append(stat)
                    test_data["t_dof"].append(df)
                    test_data["w_stat"].append(np.nan)
                    test_data["reject"].append(verdict)
                else:
                    p_val, stat, verdict = wilcoxon(model_aurocs, 0.5, alternative = "greater")
                    test_data["p_value"].append(p_val)
                    test_data["t_stat"].append(np.nan)
                    test_data["t_dof"].append(np.nan)
                    test_data["w_stat"].append(stat)
                    test_data["reject"].append(verdict)
                test_data["model"].append(model)
                test_data["prompt"].append(prompt)
                test_data["value"].append(value)
    pd.DataFrame(test_data).to_csv("./results_stat_tests/summary_tests.csv", index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###
    # Options:
    # 0: Mann-Whitney U test on every combination of model-dataset-prompt (200)
    # 1: Construct confidence interval on AUROC for every combination of model-dataset-prompt (200)
    # 2: Summary t-tests/Wilcoxon for average AUROC across models (40)
    ###
    parser.add_argument("--option", "-o", required = True, type = int, help = f"Choose a number from 0 to 3")
    parser.add_argument("--incl_unparseable", "-i", action = "store_true")
    args = parser.parse_args()

    if args.option == 0:
        conduct_mann_whitney_tests(args.incl_unparseable)
    elif args.option == 1:
        construct_confidence_intervals(args.incl_unparseable)
    elif args.option == 2:
        conduct_model_summary_tests(args.incl_unparseable)
