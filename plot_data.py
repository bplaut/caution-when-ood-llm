import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sys
import os
from collections import defaultdict
from adjustText import adjust_text
from scipy.stats import linregress

def parse_file_name(file_name):
    # filename looks like <dataset>_<model>-q<startq>to<endq>_<group>.txt. Assume endq ends with 0
    second_half = file_name[file_name.find('to'):]
    group = second_half[second_half.find('_')+1:-4] # remove initial underscore and .txt
    parts = file_name.split('_')
    dataset = parts[0]
    model = parts[1].split('-q')[0]
    return dataset, model, group

def parse_data(file_path, incl_unparseable):
    labels = []
    conf_levels = []
    total_qs = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # skip header line
                # skip Abstained answers, which don't affect the score or auc
                # skip Unparseable lines if incl_unparseable is False
                if parts[0] in ("Correct", "Wrong") or (incl_unparseable and parts[0] == "Unparseable"):
                    labels.append(1 if parts[0] == "Correct" else 0)
                    conf_levels.append(float(parts[1]))
                if parts[0] in ("Correct", "Wrong", 'Abstained') or (incl_unparseable and parts[0] == "Unparseable"):
                    # Abstentions don't affect the score, but we still want them for normalization
                    total_qs += 1
    except IOError:
        print(f"Error opening file: {file_path}")
        sys.exit(1)
    return labels, conf_levels, total_qs

def expand_model_name(name):
    return ('Mistral-7B' if name == 'Mistral' else
            'Mixtral-8x7B' if name == 'Mixtral' else
            'SOLAR-10.7B' if name == 'Solar' else
            'Llama2-13B' if name == 'Llama-13b' else
            'Llama2-7B' if name == 'Llama-7b' else
            'Llama2-70B' if name == 'Llama-70b' else
            'Yi-6B' if name == 'Yi-6b' else
            'Yi-34B' if name == 'Yi-34b' else
            'Falcon-7B' if name == 'Falcon-7b' else
            'Falcon-40B' if name == 'Falcon-40b' else name)

def expand_label(label):
        return ('Confidence Threshold' if label == 'conf' else
                'Score' if label == 'score' else
                'Score (harsh)' if label == 'harsh-score' else
                'Model Size' if label == 'size' else
                'Average AUC' if label == 'auc' else
                'Average Accuracy' if label == 'acc' else label)    

def model_size(name):
    full_name = expand_model_name(name)
    size_term = full_name.split('-')[1]
    end_of_size_term = size_term.rfind('B')
    return 46.7 if 'Mixtral' in name else float(size_term[:end_of_size_term])

def make_and_sort_legend():
    # Each name is of the form "<model_series>-<size>: <stuff>". Sort by model_series, then by size
    handles, names = plt.gca().get_legend_handles_labels()
    model_series = lambda name: name.split('-')[0]
    zipped = zip(handles, names)
    sorted_zipped = sorted(zipped, key=lambda x: (model_series(x[1]), model_size(x[1])))
    sorted_handles, sorted_names = zip(*sorted_zipped)
    plt.legend(handles=sorted_handles, labels=sorted_names, fontsize='small')

def plot_roc_curves(all_data, output_dir, dataset):
    plt.figure()
    aucs = dict()
    for model, (labels, conf_levels, _) in all_data[dataset].items():
        fpr, tpr, _ = roc_curve(labels, conf_levels)
        roc_auc = auc(fpr, tpr)
        aucs[model] = roc_auc
        plt.plot(fpr, tpr, lw=2, label=f'{expand_model_name(model)} (area = {roc_auc:.2f})')

    make_and_sort_legend()
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset}')
    # Make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"roc_curve_{dataset}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve for {dataset} saved --> {output_path}")
    return aucs
    
def generic_finalize_plot(output_dir, xlabel, ylabel, dataset='all datasets', normalize=True, title_suffix=''):
    # No need to go too negative
    min_y_normed, min_y_unnormed = -0.15, -300
    curr_min_y = plt.ylim()[0]
    new_min_y = max(curr_min_y, min_y_normed) if normalize else max(curr_min_y, min_y_unnormed)
    plt.ylim(bottom = new_min_y)

    plt.xlabel(expand_label(xlabel))
    plt.ylabel(expand_label(ylabel))
    plt.title(f'{expand_label(ylabel)} vs {expand_label(xlabel)}: {dataset}{title_suffix}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{dataset.replace(' ','_')}_{ylabel}_vs_{xlabel}.png")
    
    plt.savefig(output_path)
    plt.close()
    print(f"{ylabel} vs {xlabel} for {dataset} saved --> {output_path}")

def scatter_plot(xs, ys, output_dir, model_names, xlabel, ylabel):
    plt.figure()
    xs, ys = np.array(xs), np.array(ys)
    scatter = plt.scatter(xs, ys)
    texts = []
    for i in range(len(model_names)):
        texts.append(plt.text(xs[i], ys[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7))    
    adjust_text(texts)

    # Fit and plot linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
    plt.plot(xs, intercept + slope * xs, 'r-')

    generic_finalize_plot(output_dir, xlabel, ylabel, title_suffix=f' (r = {r_value:.2f})')
           
def auc_acc_plots(all_data, all_aucs, output_dir):
    # Main three meta metrics are: model size, avg AUC, avg accuracy
    # Create a scatter plot for each pair of metrics
    model_aucs, model_accs = dict(), dict()
    for dataset in all_aucs:
        # Same set of models in each dict, so we can just iterate over one dict
        for model in all_aucs[dataset]:
            if model not in model_aucs:
                model_aucs[model] = []
                model_accs[model] = []    
            model_aucs[model].append(all_aucs[dataset][model])
            (labels, _, _) = all_data[dataset][model]
            model_accs[model].append(np.mean(labels))

    avg_aucs, avg_accs, model_names = [], [], []
    for model in model_aucs:
        avg_aucs.append(np.mean(model_aucs[model]))
        avg_accs.append(np.mean(model_accs[model]))
        model_names.append(model)

    scatter_plot(avg_aucs, avg_accs, output_dir, model_names, 'auc', 'acc')
    return avg_aucs, avg_accs, model_names # We'll use these for the cross-group plots

def mcc_score(labels, conf_levels, thresh):
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    # TP is when label is 1 and conf >= thresh
    # TN is when label is 0 and conf < thresh
    # FP is when label is 0 and conf >= thresh
    # FN is when label is 1 and conf < thresh
    TP, TN, FP, FN = 0, 0, 0, 0
    for label, conf in zip(labels, conf_levels):
        if conf < thresh:
            if label == 0:
                TN += 1
            else:
                FN += 1
        else:
            if label == 1:
                TP += 1
            else:
                FP += 1
    denom = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return (TP*TN - FP*FN) / denom if denom != 0 else 0
    
def subtractive_score(labels, conf_levels, total_qs, thresh, normalize, wrong_penalty=1):
    # Score = num correct - num wrong, with abstaining when confidence < threshold
    score = sum([0 if conf < thresh else (1 if label == 1 else -wrong_penalty) for label, conf in zip(labels, conf_levels)])
    return score / total_qs if normalize else score

def score_plot(data, output_dir, xlabel, ylabel, dataset, thresholds_to_mark=dict(), yscale='linear'):
    plt.figure()
    plt.yscale(yscale)

    for (model, xs, ys) in data:
        # Mark the provided threshold if given, else mark the threshold with the best score
        if model in thresholds_to_mark:
            thresh_to_mark = thresholds_to_mark[model]
            thresh_idx = np.where(xs == thresh_to_mark)[0][0] # First zero idx is because np.where returns a tuple, second zero idx is because we only want the first index (although there should only be one)
            score_to_mark = ys[thresh_idx]
        else:
            thresh_to_mark_idx = np.argmax(ys)
            thresh_to_mark = xs[thresh_to_mark_idx]
            score_to_mark = max(ys)
        # zorder determines which objects are on top
        plt.scatter([thresh_to_mark], [score_to_mark], color='black', marker='o', s=20, zorder=3)
        base_score = ys[0] # threshold of 0 is equivalent to the base model
        plt.plot(xs, ys, label=f"{expand_model_name(model)}: {base_score} to {score_to_mark}", zorder=2)

    # Add dashed black line at y=0
    overall_min_x = min([min(xs) for _, xs, _ in data])
    overall_max_x = max([max(xs) for _, xs, _ in data])
    plt.plot([overall_min_x, overall_max_x], [0, 0], color='black', linestyle='--')

    make_and_sort_legend()
    plt.legend(loc='upper left')
    generic_finalize_plot(output_dir, xlabel, ylabel, dataset)
    
def plot_score_vs_thresholds(data, output_dir, datasets, normalize=True, wrong_penalty=1, thresholds_to_mark=dict(), score_type='subtractive'):
    # Inner max is for one model + dataset, middle max is for one dataset, outer max is overall
    max_conf = max([max([max(conf_levels) for _, (_,conf_levels,_) in data[dataset].items()])
                    for dataset in datasets])
    thresholds = np.linspace(0, max_conf, 200) # 200 data points per plot
    if abs(max_conf - 1) < 0.01: # We're dealing with probabilities: add more points near 1
        thresholds = np.append(thresholds, np.linspace(0.99, 1, 100))
    # Add all keys in thresholds_to_mark to thresholds, and sort
    thresholds = np.sort(np.unique(np.append(thresholds, list(thresholds_to_mark.values()))))

    # For each model and dataset, compute the score for each threshold
    results = defaultdict(lambda: defaultdict(list))        
    for dataset in datasets:
        for model, (labels, conf_levels, total_qs) in data[dataset].items():
            scores  = []
            scores_harsh = []
            for thresh in thresholds:
                if score_type == 'subtractive':
                    score = subtractive_score(labels, conf_levels, total_qs, thresh, normalize, wrong_penalty)
                elif score_type == 'mcc':
                    score = mcc_score(labels, conf_levels, thresh)
                else:
                    raise Exception(f'Unknown score type {score_type}')
                scores.append(score)
            results[model][dataset] = scores
            
    # Now for each model and threshold, average the scores across datasets
    overall_results = []
    optimal_thresholds = dict()
    for model in results:
        results_for_model = []
        for i in range(len(thresholds)):
            # Some models might not have results for all datasets (although eventually they should)
            scores_for_thresh = [results[model][dataset][i] for dataset in results[model]]
            precision = 3
            round_fn = lambda x: round(x, precision) if normalize else x
            avg_score = round_fn(np.mean(scores_for_thresh))
            results_for_model.append(avg_score)
        overall_results.append((model, thresholds, results_for_model))
        optimal_thresh_idx = np.argmax(results_for_model)
        optimal_thresholds[model] = thresholds[optimal_thresh_idx]
            
    dataset_name = 'all datasets' if len(datasets) > 1 else datasets[0]
    ylabel = 'MCC' if score_type == 'mcc' else 'score' if wrong_penalty == 1 else 'harsh-score' if wrong_penalty == 2 else 'unknown'
    score_plot(overall_results, output_dir, 'conf', ylabel, dataset_name, thresholds_to_mark)
    return optimal_thresholds # We use this return value in the train/test context

def train_and_test_score_plots(test_data, train_data, output_dir, datasets, normalize=True, wrong_penalty=1, score_type='subtractive'):
    thresholds_to_mark = plot_score_vs_thresholds(train_data, os.path.join(output_dir, 'train'), datasets, wrong_penalty=wrong_penalty, score_type=score_type)
    plot_score_vs_thresholds(test_data, os.path.join(output_dir, 'test'), datasets, wrong_penalty=wrong_penalty, thresholds_to_mark=thresholds_to_mark, score_type=score_type)

def plots_for_group(data, output_dir):
    # Split into train and test. We don't have to shuffle, since question order is already randomized
    train_data, test_data = defaultdict(dict), defaultdict(dict)
    for dataset in data:
        for model in data[dataset]:
            labels, conf_levels, total_qs = data[dataset][model]
            n = len(labels)
            train_data[dataset][model] = (labels[:n//2], conf_levels[:n//2], total_qs/2)
            test_data[dataset][model] = (labels[n//2:], conf_levels[n//2:], total_qs/2)
            
    # Generating and saving plots
    all_aucs = dict()
    for dataset in data:
        # Plots for all the data together
        all_aucs[dataset] = plot_roc_curves(data, output_dir, dataset)
        # score = (correct - wrong * wrong_penalty) when score_type isn't given
        plot_score_vs_thresholds(data, output_dir, [dataset], wrong_penalty=1)
        plot_score_vs_thresholds(data, output_dir, [dataset], wrong_penalty=2)
        # plot_score_vs_thresholds(data, output_dir, [dataset], score_type='mcc')

        # Plots for train and test splits
        train_and_test_score_plots(test_data, train_data, output_dir, [dataset], wrong_penalty=1)
        train_and_test_score_plots(test_data, train_data, output_dir, [dataset], wrong_penalty=2)
        # train_and_test_score_plots(test_data, train_data, output_dir, [dataset], score_type='mcc')

    # Same plots as before, but for all datasets together
    datasets = list(data.keys())
    plot_score_vs_thresholds(data, output_dir, datasets, wrong_penalty=1)
    plot_score_vs_thresholds(data, output_dir, datasets, wrong_penalty=2)
    # plot_score_vs_thresholds(data, output_dir, datasets, score_type='mcc')
    train_and_test_score_plots(test_data, train_data, output_dir, datasets, wrong_penalty=1)
    train_and_test_score_plots(test_data, train_data, output_dir, datasets, wrong_penalty=2)
    #train_and_test_score_plots(test_data, train_data, output_dir, datasets, score_type='mcc')

    return auc_acc_plots(data, all_aucs, output_dir) # We'll use the return value for cross-group plots
        
def main():
    # Setup
    if len(sys.argv) < 5:
        print("Usage: python plot_data.py <output_directory> <incl_unparseable> <dataset1,dataset2,...> <data_file1> [<data_file2> ...]")
        sys.exit(1)
    output_dir = sys.argv[1]
    incl_unparseable = (False if sys.argv[2].lower() == 'false' else
                        True if sys.argv[2].lower() == 'true' else None)
    if incl_unparseable is None:
        raise Exception('Second argument incl_unparseable must be a boolean (True or False)')
    file_paths = sys.argv[4:]
    print(f"Reading from {len(file_paths)} files...")
    datasets_to_analyze = sys.argv[3].split(',')
    if any([dataset not in ('arc', 'hellaswag', 'mmlu', 'piqa', 'truthfulqa', 'winogrande','all') for dataset in datasets_to_analyze]):
        raise Exception(f'Third argument must be a comma-separated list of datasets or "all"')
    if 'all' in datasets_to_analyze:
        datasets_to_analyze = ['arc', 'hellaswag', 'mmlu', 'piqa', 'truthfulqa', 'winogrande']

    # Data aggregation. We want all_data[group][dataset][model] = (labels, conf_levels, total_qs)
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], 0))))
    for file_path in file_paths:
        dataset, model, group = parse_file_name(os.path.basename(file_path))
        if dataset in datasets_to_analyze:
            labels, conf_levels, total_qs = parse_data(file_path, incl_unparseable)
            old_labels, old_conf_levels, old_total_qs = all_data[group][dataset][model]
            all_data[group][dataset][model] = (np.concatenate([old_labels, labels]), np.concatenate([old_conf_levels, conf_levels]), old_total_qs + total_qs)

    group_data = dict()
    for group in all_data:
        print(f"\nGenerating plots for {group}\n")
        group_data[group] = plots_for_group(all_data[group], os.path.join(output_dir, group))

    # Cross-group plots
    # Plot AUC vs accuracy, but with different colors for each group
    plt.figure()
    texts = []
    for group in group_data:
        (xs, ys, model_names) = group_data[group]
        xs, ys = np.array(xs), np.array(ys)
        plt.scatter(xs, ys, label=group)
        slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
        # plt.plot(xs, intercept + slope * xs, 'r-')
        for i in range(len(model_names)):
            texts.append(plt.text(xs[i], ys[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7))
            
    adjust_text(texts)
    plt.legend(loc='lower right')
    generic_finalize_plot(output_dir, 'auc', 'acc', normalize=True)
    
if __name__ == "__main__":
    main()
