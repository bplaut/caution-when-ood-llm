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
            'Falcon-40B' if name == 'Falcon-40b' else
            name)

def model_size(name):
    full_name = expand_model_name(name)
    size_term = full_name.split('-')[-1]
    return 46.7 if name == 'Mixtral' else float(size_term[:-1])

def plot_and_save_roc_curves(data, output_dir, dataset, fpr_range=(0.0, 1.0)):
    plt.figure()
    aucs = dict()
    for model, values in data.items():
        labels, scores = zip(*values)
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)
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
    fpr_str = f"fpr_{fpr_range[0]}_{fpr_range[1]}" if fpr_range != (0, 1) else ""
    output_path = os.path.join(output_dir, f"roc_curve_{dataset}{fpr_str}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve for {dataset} saved to {output_path} for FPR range {fpr_range}")
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

def plot_accuracy_vs_confidence(data, output_dir, dataset):
    plt.figure()
    for model, values in data.items():
        # Aggregate all labels and scores for this model
        all_labels = np.concatenate([labels for labels, _ in values])
        all_scores = np.concatenate([scores for _, scores in values])
        accuracies = compute_accuracy_per_confidence_bin(all_labels, all_scores)
        bins, accs = zip(*accuracies)
        accs = [a if a is not None else 0 for a in accs]  # Replace None with 0

        plt.plot(bins, accs, label=expand_model_name(model), marker='o')

    plt.xlabel('Confidence Level')
    plt.ylabel('Average Accuracy')
    plt.title(f'Average Accuracy per Confidence Level - {dataset}')
    plt.legend()
    output_path = os.path.join(output_dir, f"calibration_{dataset}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Calibration plot for {dataset} saved to {output_path}")

def scatter_plot(xs, ys, output_dir, model_names, xlabel, ylabel, log_scale=True):
    plt.figure()
    
    # Set x-axis scale based on log_scale parameter
    if log_scale:
        plt.xscale('log')
        x_for_fit = np.log(xs)  # log transform for fitting
    else:
        x_for_fit = xs

    scatter = plt.scatter(xs, ys)

    full_label = lambda label: 'Model Size' if label == 'size' else 'Average AUC' if label == 'auc' else 'Average Accuracy' if label == 'acc' else label
    
    plt.xlabel(full_label(xlabel))
    plt.ylabel(full_label(ylabel))
    plt.title(f'{full_label(ylabel)} vs {full_label(xlabel)}')

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

    output_path = os.path.join(output_dir, f"{ylabel}_vs_{xlabel}_logscale={log_scale}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"{ylabel} vs {xlabel} plot for saved to {output_path}")
       
def model_size_plots(aggregated_data, all_aucs, output_dir):
    # For each model, compute (1) avg AUC across datasets and (2) avg accuracy across datasets
    # Then plot AUC vs model size and accuracy vs model size
    model_aucs = dict()
    model_accs = dict()
    for dataset in all_aucs:
        # Same set of models in each dict, so we can just iterate over one dict
        for model in all_aucs[dataset]:
            if model not in model_aucs:
                model_aucs[model] = []
                model_accs[model] = []    
            model_aucs[model].append(all_aucs[dataset][model])
            all_labels = np.concatenate([labels for labels, _ in aggregated_data[dataset][model]])
            model_accs[model].append(np.mean(all_labels))

    model_sizes = []
    avg_aucs = []
    avg_accs = []
    model_names = []
    for model in model_aucs:
        model_sizes.append(model_size(model))
        avg_aucs.append(np.mean(model_aucs[model]))
        avg_accs.append(np.mean(model_accs[model]))
        model_names.append(model)

    scatter_plot(model_sizes, avg_aucs, output_dir, model_names, 'size', 'auc', log_scale=True)
    scatter_plot(model_sizes, avg_accs, output_dir, model_names, 'size', 'acc', log_scale=True)
    scatter_plot(model_sizes, avg_aucs, output_dir, model_names, 'size', 'auc', log_scale=False)
    scatter_plot(model_sizes, avg_accs, output_dir, model_names, 'size', 'acc', log_scale=False)
    scatter_plot(avg_aucs, avg_accs, output_dir, model_names, 'auc', 'acc', log_scale=False)

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
    fpr_range_str = sys.argv[3]
    fpr_range = tuple(float(x) for x in fpr_range_str.split('-'))

    # Data aggregation
    aggregated_data = defaultdict(lambda: defaultdict(list))
    for file_path in file_paths:
        dataset, model = parse_file_name(os.path.basename(file_path))
        labels, scores = parse_data(file_path, incl_unparseable)
        aggregated_data[dataset][model].append((np.array(labels), np.array(scores)))

    # Generating and saving plots
    all_aucs = dict()
    for dataset, data in aggregated_data.items():
        all_aucs[dataset] = plot_and_save_roc_curves(data, output_dir, dataset, fpr_range=fpr_range)
        # plot_and_save_aupr_curves(data, output_dir, dataset)
        # plot_accuracy_vs_confidence(data, output_dir, dataset)
    model_size_plots(aggregated_data, all_aucs, output_dir)

if __name__ == "__main__":
    main()
