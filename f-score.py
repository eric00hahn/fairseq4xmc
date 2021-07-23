#import numpy as np
from numpy.core.numeric import argwhere
import matplotlib.pyplot as plt

def build_label_dictionary(tst_or_trn : str) :
    if tst_or_trn == 'trn' :
        labels = np.load("./AttentionXML-AmazonCat-13K-labels.npy", allow_pickle=True)
    if tst_or_trn == 'tst' :
        labels = np.load("../data/AmazonCat-13K/test_labels.npy", allow_pickle=True)
    N_tst = labels.shape[0]
    labels_dict = {}
    for i in range(N_tst) :
        labels_current = set(labels[i])
        for l in labels_current :
            if l not in labels_dict :
                labels_dict[l] = {}
                labels_dict[l]['frequency'] = 1
                labels_dict[l]['present_predicted'] = 0
                labels_dict[l]['present_not_predicted'] = 0
                labels_dict[l]['precision'] = 0
            else :
                labels_dict[l]['frequency'] += 1
    return labels_dict

def compute_accuracy(CUTOFF_VALUE : float) : 
    labels = np.load("./AttentionXML-AmazonCat-13K-labels.npy", allow_pickle=True)
    scores = np.load("./AttentionXML-AmazonCat-13K-scores.npy", allow_pickle=True)
    target_labels = np.load("../data/AmazonCat-13K/test_labels.npy", allow_pickle=True)
    N_tst = labels.shape[0]

    precision_cumulated = 0
    recall_cumulated = 0
   
    labels_dict = build_label_dictionary('tst')
    for i in range(N_tst) :
        t = np.argwhere(scores[i] > CUTOFF_VALUE)
        labels_pred = set([labels[i][t][x][0] for x in range(labels[i][t].shape[0])])
        labels_targ = set(target_labels[i])

        for l in labels_targ :
            if l in labels_pred :
                labels_dict[l]['present_predicted'] += 1
            else :
                labels_dict[l]['present_not_predicted'] += 1

        if len(labels_targ) > 0 :
            precision = len(labels_pred.intersection(labels_targ)) / len(labels_targ)
        else :
            precision = 1
        
        if len(labels_pred) > 0 :
            recall = len(labels_pred.intersection(labels_targ)) / len(labels_pred)
        else :
            if len(labels_targ) == 0 :
                recall = 1
            else :
                recall = 0

        precision_cumulated += precision
        recall_cumulated += recall

    freqs_v_precision = []
    for l in labels_dict.keys() :
        labels_dict[l]['precision'] = labels_dict[l]['present_predicted'] / labels_dict[l]['frequency']
        freqs_v_precision += [(labels_dict[l]['frequency'], labels_dict[l]['precision'])]

    freqs_v_precision.sort(key=lambda x : x[0])
    precision = precision_cumulated / N_tst
    recall = recall_cumulated / N_tst
    f1 = (2 * precision * recall ) / ( precision + recall )
    return freqs_v_precision

def plot_freqs_v_precision(X : list) :
    x_values = []
    y_values = []
    c = 0
    for x in X :
        print(x)
        x_values += [c]
        y_values += [x[1]]
        c += 1
    plt.style.use('seaborn-whitegrid')
    plt.plot(x_values, 
             y_values, 
             'o', 
             color='black',
             markersize=0.125)
    plt.title("Label-wise precision")
    plt.xlabel("Label ID ordered by increasing frequency")
    plt.ylabel("Precision")
    plt.savefig("./plots/amazoncat13k_label_precision.png", dpi=600)
    return None 

def plot_freqs_v_precision_smoothed(X : list, 
                                    window_size : float) :
    x_values = []
    y_values = []
    X_windowed = [i * window_size for i in range(int( len(X) /  window_size))]
    c = 1
    for w in X_windowed :
        x_values +=[c]
        y_smoothed = 0
        for i in range(window_size) :
            y_smoothed += X[w + i][1]
        y_smoothed /= window_size
        y_values += [y_smoothed]
        c += 1
    x = np.array(x_values)
    y = np.array(y_values)
    m, b = np.polyfit(x, y, 1)
    plt.style.use('seaborn-whitegrid')
    plt.plot(x_values, 
             y_values, 
             'o', 
             color='black',
             markersize=0.2)
    plt.plot(x, 
             m * x + b,
             color='green',
             linewidth=0.5,
             label="Linear regression")
    plt.title("Smoothed label-wise precision with window size 10")
    plt.xlabel("Label window ID ordered by increasing frequency")
    plt.ylabel("Average precision in window")
    plt.legend(loc="upper right")
    plt.savefig("./plots/amazoncat13k_label_precision_smoothed.png", dpi=600)
    return None

def main() -> int :
    
    freqs_v_precision = compute_accuracy(0.4)
#    plot_freqs_v_precision_smoothed(freqs_v_precision, 10)
    plot_freqs_v_precision(freqs_v_precision)

if __name__ == '__main__' :
    main()
