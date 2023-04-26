#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


            
class KLContrastiveLoss(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(KLContrastiveLoss, self).__init__()

        self.kl_div = nn.KLDivLoss(reduction="mean")

        
    def __criterion(self, logit, label):
        batchsize = logit.shape[0]
        probs1 = F.log_softmax(logit, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.kl_div(probs1, probs2)
        return loss * batchsize


    def forward(self, logits_image, gt_labels):
        # generate GT matrix
        ground_truth = torch.zeros([len(gt_labels), len(gt_labels)]).float() # size = [bz, bz]
        for i,x in enumerate(gt_labels):
            for j, y in enumerate(gt_labels):
                if x==y:
                    ground_truth[i, j] = 1.0
        ground_truth = ground_truth.to(logits_image.device)

        loss_image = self.__criterion(logits_image, ground_truth)
        loss_text  = self.__criterion(logits_image.t(), ground_truth)

        return (loss_image + loss_text) * 0.5


class MultiLabelKLContrastiveLoss(nn.Module):
    """
    Cross entropy loss with soft target (Multilabel Version).
    """

    def __init__(self):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultiLabelKLContrastiveLoss, self).__init__()

        self.kl_div = nn.KLDivLoss(reduction="mean")

        
    def __criterion(self, logit, label):
        batchsize = logit.shape[0]
        probs1 = F.log_softmax(logit, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.kl_div(probs1, probs2)
        return loss * batchsize


    def forward(self, logits_image, gt_labels, label):   
        # gt_labels: all labels of the image; label: id of the current record
        # generate GT matrix
        ground_truth = torch.zeros([len(gt_labels), len(gt_labels)]).float() # size = [bz, bz]
        # print("gt_labels size:", len(gt_labels))
        # print("label size:", len(label))
        
        # print(gt_labels)
        # print(label)

        # tmp = []
        # labels = []

        for i in range(len(gt_labels)):
            for j in range(len(label)): # TODO: 检验正确性
                if label[j] in gt_labels[i]:
                    ground_truth[i][j] = 1.0

        ground_truth = ground_truth.to(logits_image.device)

        loss_image = self.__criterion(logits_image, ground_truth)
        loss_text  = self.__criterion(logits_image.t(), ground_truth)

        return (loss_image + loss_text) * 0.5


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
