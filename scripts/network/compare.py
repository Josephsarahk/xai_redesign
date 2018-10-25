import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import pickle
import sys
from sklearn.metrics import normalized_mutual_info_score


def unpickle(pathname):
    with open(pathname, 'rb') as file:
        return pickle.load(file)
		
orig = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_fc2_cluster.p')
new  = unpickle('D:\\activation_layers\\fashion_mnist\\clustered_acts_fc2.p')

score=normalized_mutual_info_score(orig,new)
print(score)