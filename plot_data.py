import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sys
import os
from collections import defaultdict
from adjustText import adjust_text

def parse_file_name(file_name):
    parts = file_name.split('_')
    dataset = parts[0]
    model = parts[1].split('-q')[0]
    return dataset, model

def parse_data(file_path, incl_unparseable):
    labels = []
    scores = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # skip Abstained answers, which don't affect the score or auc
                # skip Unparseable lines if incl_unparseable is False
                if parts[0] in ("Correct", "Wrong") or (incl_unparseable and parts[0] == "Unparseable"):
                    labels.append(1 if parts[0] == "Correct" else 0)
                    scores.append(float(parts[1]))
    except IOError:
        print(f"Error opening file: {file_path}")
        sys.exit(1)
    return labels, scores

def expand_model_name(name):
    return ('Mistral-7B' if name == 'Mistral' else
            'Mixtral-8x7B' if name == 'Mixtral' else
            'SOLAR-10.7B' if name == 'Solar' else
            'Llama2-13B' if name == 'Llama-13b' else
            'Llama2-7B' if name == 'Llama-7b' else
            'Llama2-70B' if name == 'Llama-70b' else
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
    size_term = full_name.split('-')[-1]
    return 46.7 if name == 'Mixtral' else float(size_term[:-1])

def plot_roc_curves(all_data, output_dir, dataset, fpr_range=(0.0, 1.0)):
    plt.figure()
    aucs = dict()
    for model, (labels, scores) in all_data[dataset].items():
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Filter the FPR and TPR based on the fpr_range
        valid_range = (fpr >= fpr_range[0]) & (fpr <= fpr_range[1])
        fpr_filtered = fpr[valid_range]
        tpr_filtered = tpr[valid_range]

        # Calculate the partial AUROC
        roc_auc = auc(fpr_filtered, tpr_filtered)
        aucs[model] = roc_auc
        plt.plot(fpr_filtered, tpr_filtered, lw=2, label=f'{expand_model_name(model)} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([fpr_range[0], fpr_range[1]])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset}')
    plt.legend(loc="lower right")
    fpr_str = f"_fpr_{fpr_range[0]}_{fpr_range[1]}" if fpr_range != (0, 1) else ""
    output_path = os.path.join(output_dir, f"roc_curve_{dataset}{fpr_str}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve for {dataset} saved --> {output_path}")
    return aucs
    
def compute_accuracy_per_confidence_bin(labels, scores, n_bins=10, min_conf=0):
    bins = np.linspace(min_conf, 1, n_bins + 1)
    accuracies = []
    for i in range(len(bins) - 1): 
        idx = (scores >= bins[i]) & (scores < bins[i + 1])
        bin_mid = (bins[i] + bins[i+1]) / 2
        if np.sum(idx) > 0:
            acc = np.mean(labels[idx] == 1)
            accuracies.append((bin_mid, acc))
        else:
            accuracies.append((bin_mid, None))
    return accuracies

def generic_finalize_plot(output_dir, xlabel, ylabel, dataset='all datasets', normalize=True):
    # No need to go too negative
    min_y_normed, min_y_unnormed = -0.15, -300
    curr_min_y = plt.ylim()[0]
    new_min_y = max(curr_min_y, min_y_normed) if normalize else max(curr_min_y, min_y_unnormed)
    plt.ylim(bottom = new_min_y)

    plt.xlabel(expand_label(xlabel))
    plt.ylabel(expand_label(ylabel))
    plt.title(f'{expand_label(ylabel)} vs {expand_label(xlabel)}: {dataset}')
    output_path = os.path.join(output_dir, f"{ylabel}_vs_{xlabel}_{dataset.replace(' ','_')}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"{ylabel} vs {xlabel} for {dataset} saved --> {output_path}")

def plot_accuracy_vs_confidence(data, output_dir, dataset):
    plt.figure()
    for model, (labels, scores) in data.items():
        accuracies = compute_accuracy_per_confidence_bin(labels, scores)
        bins, accs = zip(*accuracies)
        accs = [a if a is not None else 0 for a in accs]  # Replace None with 0
        plt.plot(bins, accs, label=expand_model_name(model), marker='o')
    plt.legend()
    generic_finalize_plot(output_dir, 'conf', 'acc', dataset)

def scatter_plot(xs, ys, output_dir, model_names, xlabel, ylabel, log_scale=True):
    plt.figure()
    # Set x-axis scale based on log_scale parameter
    if log_scale:
        plt.xscale('log')
        x_for_fit = np.log(xs)  # log transform for fitting
    else:
        x_for_fit = xs

    scatter = plt.scatter(xs, ys)
    
    # Adjust x-ticks for log-scale
    if log_scale:
        max_x = max(xs)
        tick_values = [x for x in range(10, int(max_x) + 10, 10)]
        plt.xticks(tick_values, [f'{x}B' for x in tick_values])

    texts = []
    for i in range(len(model_names)):
        texts.append(plt.text(xs[i], ys[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7))    
    adjust_text(texts)

    # Fit and plot regression line appropriate to the scale
    z = np.polyfit(x_for_fit, ys, 1)
    p = np.poly1d(z)
    if log_scale:
        plt.plot(xs, p(np.log(xs)), "r-")
    else:
        plt.plot(xs, p(xs), "r-")

    generic_finalize_plot(output_dir, xlabel, ylabel)
           
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
            (labels, _) = all_data[dataset][model]
            model_accs[model].append(np.mean(labels))

    model_sizes, avg_aucs, avg_accs, model_names = [], [], [], []
    for model in model_aucs:
        model_sizes.append(model_size(model))
        avg_aucs.append(np.mean(model_aucs[model]))
        avg_accs.append(np.mean(model_accs[model]))
        model_names.append(model)

    # scatter_plot(model_sizes, avg_aucs, output_dir, model_names, 'size', 'auc', log_scale=True)
    # scatter_plot(model_sizes, avg_accs, output_dir, model_names, 'size', 'acc', log_scale=True)
    scatter_plot(avg_aucs, avg_accs, output_dir, model_names, 'auc', 'acc', log_scale=False)

def compute_score(labels, conf_levels, thresh, normalize, wrong_penalty=1):
    # Score = num correct - num wrong, with abstaining when confidence < threshold
    total = sum([0 if conf < thresh else (1 if label == 1 else -wrong_penalty) for label, conf in zip(labels, conf_levels)])
    return total / len(labels) if normalize else total

def score_plot(data, output_dir, xlabel, ylabel, dataset, yscale='linear'):
    plt.figure()
    plt.yscale(yscale)

    for (model, xs, ys) in data:
        base_score = ys[0] # threshold of 0 is equivalent to the base model
        max_y = max(ys)
        max_x = xs[ys.index(max_y)]
        plt.plot(xs, ys, label=f"{expand_model_name(model)}: {base_score} to {max_y}")

    # Add dashed black line at y=0
    plt.plot([min(xs), max(xs)], [0, 0], color='black', linestyle='--')

    plt.legend(fontsize='small')
    generic_finalize_plot(output_dir, xlabel, ylabel, dataset)
    
def plot_score_vs_conf_thresholds(all_data, output_dir, datasets, normalize=True):
    # Inner max is for one model + dataset, middle max is for one dataset, outer max is overall
    max_conf = max([max([max(conf_levels) for _, (_, conf_levels) in all_data[dataset].items()])
                    for dataset in datasets])
    thresholds = np.linspace(0, max_conf, 200) # 200 data points per plot
    if abs(max_conf - 1) < 0.01:
        # We're dealing with probabilities: add more points near 1
        thresholds = np.append(thresholds, np.linspace(0.99, 1, 100))

    # For each model and dataset, compute the score for each threshold
    results = defaultdict(lambda: defaultdict(list))        
    results_harsh = defaultdict(lambda: defaultdict(list))        
    for dataset in datasets:
        for model, (labels, conf_levels) in all_data[dataset].items():
            scores  = []
            scores_harsh = []
            for thresh in thresholds:
                score = compute_score(labels, conf_levels, thresh, normalize, wrong_penalty=1)
                scores.append(score)
                score_harsh = compute_score(labels, conf_levels, thresh, normalize, wrong_penalty=2)
                scores_harsh.append(score_harsh)
            results[model][dataset] = scores
            results_harsh[model][dataset] = scores_harsh
            
    # Now for each model and threshold, average the scores across datasets
    overall_results = []
    overall_results_harsh = []
    for model in results:
        results_for_model, results_for_model_harsh = [], []
        for i in range(len(thresholds)):
            # Some models might not have results for all datasets (although eventually they should)
            scores_for_thresh = [results[model][dataset][i] for dataset in results[model]]
            scores_for_thresh_harsh = [results_harsh[model][dataset][i] for dataset in results[model]]
            precision = 3
            round_fn = lambda x: round(x, precision) if normalize else x
            avg_score = round_fn(np.mean(scores_for_thresh))
            avg_score_harsh = round_fn(np.mean(scores_for_thresh_harsh))
            results_for_model.append(avg_score)
            results_for_model_harsh.append(avg_score_harsh)
        overall_results.append((model, thresholds, results_for_model))
        overall_results_harsh.append((model, thresholds, results_for_model_harsh))
            
    dataset_name = 'all datasets' if len(datasets) > 1 else datasets[0]
    score_plot(overall_results, output_dir, 'conf', 'score', dataset_name)
    score_plot(overall_results_harsh, output_dir, 'conf', 'harsh-score', dataset_name)
    
def main():
    if len(sys.argv) < 5:
        print("Usage: python plot_data.py <output_directory> <incl_unparseable> <fpr_range> <data_file1> [<data_file2> ...]")
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    incl_unparseable = (False if sys.argv[2].lower() == 'false' else
                        True if sys.argv[2].lower() == 'true' else None)
    if incl_unparseable is None:
        raise Exception('Second argument incl_unparseable must be a boolean (True or False)')
        
    file_paths = sys.argv[4:]
    print(f"Reading from {len(file_paths)} files...")
    fpr_range_str = sys.argv[3]
    fpr_range = tuple(float(x) for x in fpr_range_str.split('-'))

    # Data aggregation
    all_data = defaultdict(lambda: defaultdict(list))
    for file_path in file_paths:
        dataset, model = parse_file_name(os.path.basename(file_path))
        labels, scores = parse_data(file_path, incl_unparseable)
        old_labels, old_scores = all_data[dataset][model] if len(all_data[dataset][model]) > 0 else (np.array([]), np.array([]))
        all_data[dataset][model] = (np.concatenate([old_labels, labels]), np.concatenate([old_scores, scores]))
            
    # Generating and saving plots
    all_aucs = dict()
    for dataset in all_data:
        all_aucs[dataset] = plot_roc_curves(all_data, output_dir, dataset, fpr_range=fpr_range)
        plot_score_vs_conf_thresholds(all_data, output_dir, [dataset])
        # plot_accuracy_vs_confidence(data, output_dir, dataset)
    auc_acc_plots(all_data, all_aucs, output_dir)
    plot_score_vs_conf_thresholds(all_data, output_dir, all_data.keys())

if __name__ == "__main__":
    main()
