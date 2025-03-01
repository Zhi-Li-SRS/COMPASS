import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_fscore_support, roc_curve


def correlation_loss(x, y):
    """Calculate correlation loss between two tensors"""
    if x.dim() == 2:
        x = x.unsqueeze(1)

    if y.dim() == 2:
        y = y.unsqueeze(1)
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)
    eps = 1e-8
    x_std = torch.std(x, dim=1, keepdim=True, unbiased=False) + eps
    y_std = torch.std(y, dim=1, keepdim=True, unbiased=False) + eps

    correlation = torch.mean((x - x_mean) * (y - y_mean) / (x_std * y_std))

    return 1 - correlation


# def cos_sim_loss(x, y):
#     """Cosine Similarity loss between two tensors"""
#     cosine_loss = nn.CosineSimilarity(dim=1)
#     return 1 - cosine_loss(x, y).mean()


def msle_loss(x, y):
    """Mean Squared Logarithmic Error loss between two tensors"""
    x_log = torch.log1p(x)
    y_log = torch.log1p(y)
    return f.mse_loss(x_log, y_log)


class Denoise_Loss(nn.Module):
    """Custom scale-invariant loss function for denoising
    
    Args:
        corr_weight (float): Weight for cosine similarity loss.
    """

    def __init__(self, corr_weight=0.5):
        super().__init__()
        self.corr_weight = corr_weight

    def forward(self, pred, target):
        cc = self.corr_weight * correlation_loss(pred, target)
        msle = (1 - self.corr_weight) * msle_loss(pred, target)
        return cc + msle


# def denoise_loss(x, y, p=0.5):
#     return p * cos_sim_loss(x, y) + (1 - p) * msle_loss(x, y)

class Loss(nn.Module):
    """
    Custom loss function for classification with optional correlation loss.

    Args:
        use_corr (bool): Whether to use correlation loss.
        corr_weight (float): Weight for correlation loss.
    """

    def __init__(self, use_corr=True, corr_weight=0.2):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.use_corr = use_corr
        self.corr_weight = corr_weight

    def forward(self, pred, target, features=None, true_spectra=None):
        loss = self.ce_loss(pred, target)
        if self.use_corr and features is not None and true_spectra is not None:
            corr_loss = correlation_loss(features, true_spectra)
            loss += self.corr_weight * corr_loss
        return loss


class Metrics:
    @staticmethod
    def compute_accuracy(y_true, y_pred):
        """Compute accuracy score"""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def compute_precision_recall_f1(y_true, y_pred):
        """Compute precision, recall, and F1 score"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        return precision, recall, f1

    @staticmethod
    def compute_roc_auc(y_true, y_pred_proba, n_classes):
        """Compute ROC-AUC score and plot ROC curve"""
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        return fpr, tpr, roc_auc

    @staticmethod
    def compute_spectral_similarity(pred_spectra, true_spectra):
        """Compute cosine similarity between predicted and true spectra"""
        return torch.nn.functional.cosine_similarity(pred_spectra, true_spectra, dim=1).mean()
