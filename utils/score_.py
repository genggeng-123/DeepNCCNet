import warnings
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics import multilabel_confusion_matrix
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix,precision_score


def specificity(Labels,Predictions):

    metrics_out = confusion_matrix(Labels, Predictions)

    if (metrics_out.shape ==(1,1)):
        if (set(Labels)=={1}) :
            tp = len(Labels)
            tn = 0
            fp = 0
            fn = 0
        elif  set(Labels)=={0}:
            tp = 0
            tn = len(Labels)
            fp = 0
            fn = 0
    else :
            tp = metrics_out[0][0]
            fp = metrics_out[0][1]
            fn = metrics_out[1][0]
            tn = metrics_out[1][1]

    Specificity = np.sum(tn)/(np.sum(tn+fp) + 1e-7)

    return  Specificity

def sensitivity(Labels, Predictions):
    
    metrics_out = confusion_matrix(Labels, Predictions)
    if (metrics_out.shape ==(1,1)):
            if (set(Labels)=={1}) :
                tp = len(Labels)
                tn = 0
                fp = 0
                fn = 0
            elif  set(Labels)=={0}:
                tp = 0
                tn = len(Labels)
                fp = 0
                fn = 0
    else :
        tp = metrics_out[0][0]
        fp = metrics_out[0][1]
        fn = metrics_out[1][0]
        tn = metrics_out[1][1]

    Sensitivity = np.sum(tp) / (np.sum(tp+fn) + + 1e-7)

    return Sensitivity


def ppv( Labels, Predictions):
    metrics_out = confusion_matrix(Labels, Predictions)

    if (metrics_out.shape ==(1,1)):
            if (set(Labels)=={1}) :
                tp = len(Labels)
                tn = 0
                fp = 0
                fn = 0
            elif  set(Labels)=={0}:
                tp = 0
                tn = len(Labels)
                fp = 0
                fn = 0
    else :
        tp = metrics_out[0][0]
        fp = metrics_out[0][1]
        fn = metrics_out[1][0]
        tn = metrics_out[1][1]

    PPV = np.sum(tp) / (np.sum(tp+fp)  + 1e-7)

    return PPV

def npv( Labels, Predictions):
    metrics_out = confusion_matrix(Labels, Predictions)
    metrics_out = confusion_matrix(Labels, Predictions)

    if (metrics_out.shape ==(1,1)):
            if (set(Labels)=={1}) :
                tp = len(Labels)
                tn = 0
                fp = 0
                fn = 0
            elif  set(Labels)=={0}:
                tp = 0
                tn = len(Labels)
                fp = 0
                fn = 0
    else :
        tp = metrics_out[0][0]
        fp = metrics_out[0][1]
        fn = metrics_out[1][0]
        tn = metrics_out[1][1]
    NPV = np.sum(tn) / (np.sum(tn+fn) + + 1e-7)

    return NPV

def precision(Labels, Predictions):
    metrics_out = confusion_matrix(Labels, Predictions)

    if (metrics_out.shape ==(1,1)):
            if (set(Labels)=={1}) :
                tp = len(Labels)
                tn = 0
                fp = 0
                fn = 0
            elif  set(Labels)=={0}:
                tp = 0
                tn = len(Labels)
                fp = 0
                fn = 0
    else :
        tp = metrics_out[0][0]
        fp = metrics_out[0][1]
        fn = metrics_out[1][0]
        tn = metrics_out[1][1]
    Precision = np.sum(tp) / (np.sum(tp+fp) + + 1e-7)

    return Precision


def accuracy(Labels, Predictions):
    metrics_out = confusion_matrix(Labels, Predictions)
    if (metrics_out.shape == (1, 1)):
        if (set(Labels) == {1}):
            tp = len(Labels)
            tn = 0
            fp = 0
            fn = 0
        elif set(Labels) == {0}:
            tp = 0
            tn = len(Labels)
            fp = 0
            fn = 0
    else:
        tp = metrics_out[0][0]
        fp = metrics_out[0][1]
        fn = metrics_out[1][0]
        tn = metrics_out[1][1]

    Accuracy = np.sum(tn+tp) / (np.sum(tn + fp + tp +fn) + + 1e-7)

    return Accuracy