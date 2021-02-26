import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
import argparse

from sklearn.metrics import roc_curve, precision_recall_curve, auc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell_name",
        type=str,
        default="GM12878",
        help="The cell type of dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepcfp",
        help="The name of testing model."
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="ROC",
        help="The name of testing model."
    )
    return parser.parse_args()

def auroc(labels, data):
    FPR, TPR, thresholds = roc_curve(labels, data)
    roc_auc = auc(FPR, TPR)
    return roc_auc

def aupr(labels, data):
    precision, recall, thresholds = precision_recall_curve(labels, data)
    pr_auc = auc(recall, precision)
    return pr_auc

def standardization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def AUC_on_test_set(curve, cell_name, model_name):
    AUROC = []
    AUPR = []
    
    data = pd.read_table(os.path.join('./compare', cell_name, cell_name+'_'+model_name+'_datacmp.txt'))
    
    labels = np.array(data['labels'])
    fri = np.array(data['std(fri)'])
    gnm = np.array(data['GNM'])
    
    AUROC.append(round(auroc(labels, standardization(fri)), 4))
    AUPR.append(round(aupr(labels, standardization(fri)), 4))
    
    AUROC.append(round(auroc(labels, standardization(gnm)), 4))
    AUPR.append(round(aupr(labels, standardization(gnm)), 4))

    prediction = np.array(data['prediction'])
    AUROC.append(round(auroc(labels, standardization(prediction)), 4))
    AUPR.append(round(aupr(labels, standardization(prediction)), 4))

    model_name = ['FRI','GNM','DeepCFP']
    plt.figure(figsize=(6,5))
    bar_width = 0.6

    color = ['#4473C5','#A5A5A5','#FEBF00']
    if(curve=='ROC'):
        plt.bar(np.arange(3), AUROC, color=color, width=bar_width)
        for i in range(len(AUROC)):
            plt.text(i, AUROC[i] + 0.01, AUROC[i], ha='center')
        plt.title('Compare AUROC on the test set', fontsize=15)
    elif(curve=='P-R'):
        plt.bar(np.arange(3), AUPR, color=color, width=bar_width)
        for i in range(len(AUPR)):
            plt.text(i, AUPR[i] + 0.01, AUPR[i], ha='center')
        plt.title('Compare AUPR on the test set', fontsize=15)

    plt.xlabel('Model', fontsize=15)
    plt.ylabel('The area under ROC curve', fontsize=15)
    plt.xticks(np.arange(3), model_name, fontsize=12, rotation=20)
    plt.ylim([0.7, 1.05])

    plt.show()

def AUC_on_each_chromosome(curve, cell_name, model_name):
    data = pd.read_table(os.path.join('./compare',cell_name, cell_name+'_'+model_name+'_datacmp.txt'))

    chrr = ['chr' + str(i) for i in range(1, 23)]
    chrr.append('chrX')

    chrr1 = []
    fri_auc = []
    gnm_auc = []
    prediction_auc = []
    for i in chrr:
        d = data[data['chr'].isin([i])]
        labels = np.array(d['labels'])
        fri = np.array(d['FRI'])
        gnm = np.array(d['GNM'])
        prediction = np.array(d['prediction'])
        chrr1.append(i)
        if(curve=='ROC'):
            fri_auc.append(auroc(labels, fri))
            gnm_auc.append(auroc(labels, gnm))
            prediction_auc.append(auroc(labels, prediction))
        elif(curve=='P-R'):
            fri_auc.append(aupr(labels, fri))
            gnm_auc.append(aupr(labels, gnm))
            prediction_auc.append(aupr(labels, prediction))

    print(np.mean(fri_auc))
    print(np.mean(gnm_auc))
    print(np.mean(prediction_auc))

    plt.figure(figsize=(15, 8))
    bar_width = 0.2

    plt.bar(np.arange(23), fri_auc, label='FRI', color='#4473C5', alpha=0.8, width=bar_width)
    plt.bar(np.arange(23) + bar_width + 0.05, gnm_auc, label='GNM', color='#A5A5A5', alpha=0.8, width=bar_width)
    plt.bar(np.arange(23) + 2 * bar_width + 0.1, prediction_auc, label='DeepCFP', color='#FEBF00', alpha=0.8,
            width=bar_width)

    plt.xlabel('Chromosome', fontsize=15)
    plt.ylabel('The area under '+curve+' curve', fontsize=15)
    plt.xticks(np.arange(23) + 0.25, chrr, fontsize=12, rotation=20)
    plt.ylim([0.0, 1.2])
    plt.legend()
    plt.show()

if __name__=='__main__':
    args = parse_args()
    AUC_on_test_set(args.curve, args.cell_name, args.model_name) #'ROC' or 'P-R'
    AUC_on_each_chromosome(args.curve, args.cell_name, args.model_name) #'ROC' or 'P-R'
