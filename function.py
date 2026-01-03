import math
import random
import torch
import numpy
from torch.utils.data import ConcatDataset
from itertools import product


def found_largest_and_second(alist): 
    max = alist[0]
    max_index = 0
    for i in range(1,len(alist)):
        if max <= alist[i]:  
            max = alist[i]
            max_index = i
    if max_index > 0:
        sec = alist[0]
        sec_index = 0
    elif max_index == 0:
        sec = alist[1]
        sec_index = 1
    for j in range(0, len(alist)):
        if j == max_index:
            continue
        if sec <= alist[j]:  
            sec = alist[j]
            sec_index = j
    
    return [(max_index, max),(sec_index, sec)]








