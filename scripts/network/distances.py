^import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import pickle 
import random 
import sys, os
import random
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE

def unpickle(pathname):
    with open(pathname, 'rb') as file:
        return pickle.load(file)

centers  = unpickle('D:\\activation_layers\\fashion_mnist\\cluster_centers\\fc1_cluster_centers.p')
centers2 = unpickle('D:\\activation_layers\\fashion_mnist\\cluster_centers\\fc1_cluster_centers.p')

a=pairwise_distances(centers,centers2,metric='euclidean')
print(a)

centers=np.asarray(centers)

centers_embedded = TSNE(n_components=2).fit_transform(centers)
print(centers_embedded)
print(centers_embedded[:,1])

plt.plot(centers_embedded[:,0],centers_embedded[:,1], 'ro')
plt.show()
