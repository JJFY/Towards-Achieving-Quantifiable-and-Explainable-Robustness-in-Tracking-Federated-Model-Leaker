import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAVG(w, client_data_num, total_data_num):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = client_data_num[0] / total_data_num * w_avg[k]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_data_num[i] / total_data_num
    return w_avg

def FedProx(w, global_weights, mu):

    prox_weights = copy.deepcopy(global_weights) 

    for k in prox_weights.keys():  
        prox_weights[k] = torch.zeros_like(global_weights[k])
        if prox_weights[k].dtype == torch.float32:  
            prox_weights[k] = torch.zeros_like(global_weights[k])
            for i in range(len(w)):  
                prox_weights[k] += w[i][k] - mu * (w[i][k] - global_weights[k])
            prox_weights[k] = torch.div(prox_weights[k], len(w))
        else:
            if "num_batches_tracked" not in k:
                prox_weights[k] = torch.zeros_like(global_weights[k])
                for i in range(len(w)):
                    prox_weights[k] += w[i][k]
                prox_weights[k] = torch.div(prox_weights[k], len(w))

    return prox_weights