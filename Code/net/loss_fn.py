# python script with loss functions as classes for training
import torch
import torch.nn.functional as F
import torch.nn
import numpy as np
from torchvision.transforms import transforms

# contrastive loss for pretraining -----------------------------------------------------------------------------------------------------


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temp_fac=0.5):
        super().__init__()
        self.temp_fac = temp_fac

    def forward(self, loss, p1, p2, neg):

        # first element of positive pair l(p1,p2)
        # cosine similarity with positive sample
        p1p2_s = F.cosine_similarity(p1, p2)/self.temp_fac

        # cosine similarity with negative sample
        p1neg_s = F.cosine_similarity(p1, neg)/self.temp_fac

        # l(p1,p2)
        loss_1 = -torch.log(torch.exp(p1p2_s) / torch.sum(torch.exp(p1neg_s)))
        loss = loss + loss_1

        # second element of positive pair l(p2,p1)
        # cosine similarity with positive sample
        p2p1_s = F.cosine_similarity(p2, p1)/self.temp_fac

        # cosine similarity with negative sample
        p2neg_s = F.cosine_similarity(p2, neg)/self.temp_fac

        # l(p2,p1)
        loss_2 = -torch.log(torch.exp(p2p1_s) / torch.sum(torch.exp(p2neg_s)))
        loss = loss + loss_2

        return loss

# weighted regression loss for main training -------------------------------------------------------------------------------------------


class RegressionLoss_weighted(torch.nn.Module):
    def __init__(self, counter):
        super().__init__()
        # calculation of weights from number of labels per label type
        self.weights = ((torch.max(counter)/counter) /
                        torch.sum(torch.max(counter)/counter)).cuda()

    def forward(self, y_hat, y):
        # loss vector with size equal to number of labels
        loss_vec = torch.zeros(y_hat.shape[1]).cuda()

        for i in range(0, y_hat.shape[1]):
            for j in range(0, y_hat.shape[0]):
                # MSE for each label
                loss_vec[i] += torch.sum(torch.square(y_hat[j, i, :, :] -
                                         y[j, i, :, :]))/(y_hat.shape[2]*y_hat.shape[3])

        # weighted average over batch
        loss = torch.sum(loss_vec * self.weights) / y_hat.shape[0]

        return loss

# simple classification loss for main training -----------------------------------------------------------------------------------------


class LogisticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        max_pool = torch.nn.MaxPool2d(kernel_size=256)
        bce = torch.nn.BCELoss()

        # max pool over whole image
        y_hat_max = max_pool(y_hat)
        y_max = max_pool(y)

        # binary cross entropy loss
        loss = bce(y_hat_max, y_max)

        return loss

# sectioned classification loss for main training --------------------------------------------------------------------------------------


class LogisticLoss_section(torch.nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_hat, y):
        max_pool = torch.nn.MaxPool2d(kernel_size=32)
        bce = torch.nn.BCELoss()
        thres = torch.nn.Threshold(self.threshold, 0)

        # max pool over image sections
        y_hat_max = max_pool(y_hat)
        y_max = max_pool(y)

        # thresholding (if y_max[i,j,k,l] >= 0.8  then y_max[i,j,k,l] = 1 else y_max[i,j,k,l] =0)
        y_max = thres(y_max).bool().int().float()

        # binary cross entropy loss
        loss = bce(y_hat_max, y_max)

        return loss

# not used anymore ---------------------------------------------------------------------------------------------------------------------

# class Classification_MaxPool(torch.nn.Module):
#     def __init__(self, threshold):
#         super().__init__()
#         self.threshold = threshold

#     def forward(self, y_hat, y):
#         max_pool = torch.nn.MaxPool2d(kernel_size=256)
#         label_present = torch.zeros(
#             (y_hat.shape[0], y_hat.shape[1])).cuda()
#         label_there = torch.zeros(
#             (y_hat.shape[0], y_hat.shape[1])).cuda()

#         y_hat_max = max_pool(y_hat)
#         y_max = max_pool(y)

#         for i in range(0, y_hat.shape[0]):
#             for j in range(0, y_hat.shape[1]):

#                 # is there a label
#                 if y_max[i, j, :, :] == 1:
#                     label_present[i, j] = 1
#                 else:
#                     label_present[i, j] = 0

#                 # doeas the prediction have a strong label probility max. heatmap value bigger than certain threshold
#                 if y_hat_max[i, j, :, :] > self.threshold:
#                     label_there[i, j] = 1
#                 else:
#                     label_there[i, j] = 0

#         # mean suared classification loss
#         loss = torch.sum(torch.square(label_present-label_there)
#                          )/(y_hat.shape[0]*y_hat.shape[1])

#         return loss
