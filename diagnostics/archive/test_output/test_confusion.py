# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Mon Mar  7 15:10:23 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# IN-BUILT
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})

def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs

def _plot(ax, x=None, y=None, xlabel="Epochs", ylabel="BCE Loss", ls='solid', 
          label="", c=None, scatter=False, hist=False, yscale='linear'):
    
    if scatter:
        ax.scatter(x, y, c=c, label=label)
    elif hist:
        ax.hist(y, bins=100, label=label, alpha=0.8)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)
    
    if ylabel=="True Positive Rate":
        ax.set_xscale(yscale)
    
    ax.set_yscale(yscale)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
        
if __name__ == "__main__":
    
    loss_p1 = np.loadtxt("losses_1e6_closest.txt")
    
    num_epochs = 23
    # All data fields
    epochs = range(1, len(loss_p1[:,0]) + 1)
    train_loss = loss_p1[:,1]
    val_loss = loss_p1[:,2]
    train_accuracy = loss_p1[:,3]
    valid_accuracy = loss_p1[:,4]
    
    ## Loss Curves
    # Figure define
    ax = figure(title="Losses: Closest Dataset 1 (2e6 train, 4e5 valid)")
    # Explicit PyCBC closest to Dataset 1 - plotting experiment
    _plot(ax, epochs, train_loss, label="Train Loss: ePyCBC closest D1", c='red')
    _plot(ax, epochs, val_loss, label="Valid Loss: ePyCBC closest D1", c='red', ls='dashed')
    plt.savefig("loss_curves.png")
    plt.close()
    
    
    ## Accuracy Curves
    # Figure define
    ax = figure(title="Accuracy: Closest Dataset 1 (2e6 train, 4e5 valid)")
    # Explicit PyCBC closest to Dataset 1 - plotting experiment
    _plot(ax, epochs, train_accuracy, label="Max encountered Accuracy: ePyCBC closest D1", c='red', ylabel="Max Accuracy")
    plt.savefig("accuracy_plot.png")
    plt.close()
    
    
    """
    ## Confusion Matrix
    import seaborn as sns
    
    epoch = len(true_negative) - 1
    cf = [[true_negative[epoch], false_positive[epoch]], [false_negative[epoch], true_positive[epoch]]]
    plt.figure(figsize=(12.0, 9.0))
    ax = sns.heatmap(cf, annot=True, cmap='Blues')
    
    ax.set_title('Confusion Matrix for Epoch = {}'.format(epoch+1));
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Noise','Signal'])
    ax.yaxis.set_ticklabels(['Noise','Signal'])
    
    plt.savefig("confusion_matrix_{}.png".format(epoch+1))
    
    
    ## Confusion plot vs epochs
    # Figure define
    ax = figure(title="Confusion Vals: Closest Dataset 1 (2e6 train, 4e5 valid)")
    # Explicit PyCBC closest to Dataset 1 - plotting experiment
    _plot(ax, epochs, true_positive, label="TP: ePyCBC closest D1", c='red', ylabel="Num occurences in Epoch")
    _plot(ax, epochs, true_negative, label="TN: ePyCBC closest D1", c='blue', ylabel="Num occurences in Epoch")
    _plot(ax, epochs, false_positive, label="FP: ePyCBC closest D1", c='green', ylabel="Num occurences in Epoch")
    _plot(ax, epochs, false_negative, label="FN: ePyCBC closest D1", c='m', ylabel="Num occurences in Epoch")
    plt.savefig("confusion_plot.png")
    plt.close()
    
    
    ## Prediction Probabilities
    # Figure define
    ax = figure(title="Pred Prob: Closest Dataset 1 (2e6 train, 4e5 valid)")
    # Load data
    pred_prob_num = 23
    pred_prob_tp = np.load("pred_prob/pred_prob_tp_{}.npy".format(pred_prob_num))
    pred_prob_tn = np.load("pred_prob/pred_prob_tn_{}.npy".format(pred_prob_num))
    # Explicit PyCBC closest to Dataset 1 - plotting experiment
    _plot(ax, y=pred_prob_tp, label="Signals", c='red', ylabel="log10 Number of Occurences", xlabel="Predicted Probabilities", hist=True, yscale='log')
    _plot(ax, y=pred_prob_tn, label="Noise", c='blue', ylabel="log10 Number of Occurences", xlabel="Predicted Probabilities", hist=True, yscale='log')
    plt.savefig("log_pred_prob_{}.png".format(len(train_accuracy)))
    plt.close()
    
    
    ## ROC Curve
    # Figure define
    ax = figure(title="ROC: Closest Dataset 1 (2e6 train, 4e5 valid)")
    # Load data
    roc_num = 23
    output_data = np.load("output_save/output_save_{}.npy".format(roc_num))
    labels_data = np.load("labels_save/labels_save_{}.npy".format(roc_num))
    # Calculating ROC
    fpr, tpr, threshold = metrics.roc_curve(labels_data[:,0], output_data[:,0])
    roc_auc = metrics.auc(fpr, tpr)
    # Explicit PyCBC closest to Dataset 1 - plotting experiment
    _plot(ax, fpr, tpr, label='AUC = %0.2f' % roc_auc, c='red', ylabel="True Positive Rate", xlabel="False Positive Rate", yscale='log')
    _plot(ax, [0, 1], [0, 1], label="Random Classifier", c='blue', ylabel="True Positive Rate", xlabel="False Positive Rate", ls="dashed", yscale='log')
    plt.savefig("roc_curve_{}.png".format(len(train_accuracy)))
    plt.close()
    """
    
    """
    ## Precision-Recall Trade off
    ax = figure(title="PR Curve: Closest Dataset 1 (2e4 train, 2e3 valid)")
    # apply_thresh = lambda x, thresh: np.around(x - thresh + 0.5)
    # output_tmp = apply_thresh(output_data, 0.5)
    pr, re, _ = metrics.precision_recall_curve(labels_data[:,0], output_data[:,0])
    re_score = metrics.recall_score(labels_data[:,0], output_data[:,0])
    pr_score = metrics.precision_score(labels_data[:,0], output_data[:,0])
    text = "precision = {}\nrecall = {}".format(pr_score, re_score)
    # Explicit PyCBC closest to Dataset 1 - plotting experiment
    _plot(ax, re, pr, c='red', label=text, ylabel="Precision", xlabel="Recall")
    plt.savefig("precision_recall_{}.png".format(roc_num))
    plt.close()
    """
