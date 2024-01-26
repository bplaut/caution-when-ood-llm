import sys
import numpy as np

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
