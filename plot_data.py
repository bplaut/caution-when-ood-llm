import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sys
import os
from collections import defaultdict

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

def plot_and_save_roc_curves(data, output_dir, dataset):
    plt.figure()
    for model, values in data.items():
        labels, scores = zip(*values)
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
                
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset}')
    plt.legend(loc="lower right")
    output_path = os.path.join(output_dir, f"roc_curve_{dataset}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve for {dataset} saved to {output_path}")

def plot_and_save_aupr_curves(data, output_dir, dataset):
    plt.figure()
    for model, values in data.items():
        labels, scores = zip(*values)
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)
        precision, recall, thresholds = precision_recall_curve(labels, scores)
                
        aupr = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{model} (area = {aupr:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision Recall curve - {dataset}')
    plt.legend(loc="lower right")
    output_path = os.path.join(output_dir, f"aupr_curve_{dataset}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"AUPR curve for {dataset} saved to {output_path}")
    
def compute_accuracy_per_confidence_bin(labels, scores, n_bins=10, min_conf=0):
    bins = np.linspace(min_conf, 1, n_bins + 1)
    accuracies = []

    for i in range(len(bins) - 1): 
        idx = (scores >= bins[i]) & (scores < bins[i + 1])
        bin_mid = (bins[i] + bins[i+1]) / 2
        if np.sum(idx) > 0:
            acc = np.mean(labels[idx] == 1)
            print(f"Bin from {round(bins[i],3)} to {round(bins[i+1],3)} contains {np.sum(idx)} points with average accuracy {round(acc,3)}")
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
        print("Model:", model, "dataset:", dataset)
        accuracies = compute_accuracy_per_confidence_bin(all_labels, all_scores)
        bins, accs = zip(*accuracies)
        accs = [a if a is not None else 0 for a in accs]  # Replace None with 0

        plt.plot(bins, accs, label=model, marker='o')

    plt.xlabel('Confidence Level')
    plt.ylabel('Average Accuracy')
    plt.title(f'Average Accuracy per Confidence Level - {dataset}')
    plt.legend()
    output_path = os.path.join(output_dir, f"calibration_{dataset}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Calibration plot for {dataset} saved to {output_path}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_data.py <output_directory> <incl_unparseable> <data_file1> [<data_file2> ...]")
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    incl_unparseable = (False if sys.argv[2].lower() == 'false' else
                        True if sys.argv[2].lower() == 'true' else None)
    if incl_unparseable is None:
        raise Exception('Second argument incl_unparseable must be a boolean (True or False)')
        
    file_paths = sys.argv[3:]

    # Data aggregation
    aggregated_data = defaultdict(lambda: defaultdict(list))
    for file_path in file_paths:
        dataset, model = parse_file_name(os.path.basename(file_path))
        labels, scores = parse_data(file_path, incl_unparseable)
        aggregated_data[dataset][model].append((np.array(labels), np.array(scores)))

    # Generating and saving plots
    for dataset, data in aggregated_data.items():
        plot_and_save_roc_curves(data, output_dir, dataset)
#        plot_and_save_aupr_curves(data, output_dir, dataset)
        plot_accuracy_vs_confidence(data, output_dir, dataset)

if __name__ == "__main__":
    main()
