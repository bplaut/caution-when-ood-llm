import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
import os
from collections import defaultdict

def parse_file_name(file_name):
    parts = file_name.split('_')
    dataset = parts[0]
    model = parts[1].split('-q')[0]
    return dataset, model

def parse_data(file_path):
    labels = []
    scores = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] not in ("grade", "Abstained"): # Ignore header and Abstained lines
                if parts[0] not in ("Correct", "Wrong"):
                    raise Exception(f"Invalid grade: {parts[0]}")
                labels.append(1 if parts[0] == "Correct" else 0)
                scores.append(float(parts[1]))
    return labels, scores

def plot_and_save_roc_curves(data, output_dir, dataset):
    plt.figure()
    for model, values in data.items():
        labels, scores = zip(*values)
        labels = np.concatenate(labels)
        scores = np.concatenate(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # For each dataset and model, print fpr and tpr when thresholds[i] is in 0.5, 0.75,0.85,0.95
        x = [0.5, 0.75, 0.85, 0.95]
        # for each value in x, find the index of the closest value in thresholds
        indices = [np.argmin(np.abs(thresholds - i)) for i in x]
        for i in indices:
            print(f"{dataset} {model}, thresh = {round(thresholds[i],3)}: FPR = {round(fpr[i],3)}, TPR = {round(tpr[i],3)}")
                
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset}')
    plt.legend(loc="lower right")
    output_path = os.path.join(output_dir, f"{dataset}_roc_curve.png")
    plt.savefig(output_path)
    print(f"ROC curve for {dataset} saved to {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <output_directory> <data_file1> [<data_file2> ...]")
        sys.exit(1)

    output_dir = sys.argv[1]
    file_paths = sys.argv[2:]

    # Data aggregation
    aggregated_data = defaultdict(lambda: defaultdict(list))
    for file_path in file_paths:
        dataset, model = parse_file_name(os.path.basename(file_path))
        labels, scores = parse_data(file_path)
        aggregated_data[dataset][model].append((np.array(labels), np.array(scores)))

    # Generating and saving plots
    for dataset, data in aggregated_data.items():
        plot_and_save_roc_curves(data, output_dir, dataset)

if __name__ == "__main__":
    main()
