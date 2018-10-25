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

def unpickle(pathname):
    with open(pathname, 'rb') as file:
        return pickle.load(file)

#copies the ground truth label for each image and the clusters they were put in throughout the layers
#ground truth labels
true_list = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_mnist_label_list.p')

#gets clusters that each image was put into through out the layers
conv1 = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_conv1_cluster.p')
conv2 = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_conv2_cluster.p')
fc1 = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_fc1_cluster.p')
fc2 = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_fc2_cluster.p')
fc3 = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_fc3_cluster.p')

#gets prediction list 
pred = unpickle('D:\\activation_layers\\fashion_mnist\\fashion_prediction.p')

true_list=np.asarray(true_list)
print(true_list)
conv1_list=np.asarray(conv1)
print(conv1)
conv2_list=np.asarray(conv2)
print(conv2)
fc1_list=np.asarray(fc1)
print(fc1)
fc2_list=np.asarray(fc2)
print(fc2)
pred=np.asarray(pred)
print(pred)


full=np.column_stack((true_list,conv1,conv2,fc1,fc2,pred))
print(full[0:20])
print(full.shape)


#pickle.dump(full, open( "pathway.p", "wb" ) )
