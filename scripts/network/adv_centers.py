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
from path import strip_list, path_distributions

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
def unpickle(pathname):
    with open(pathname, 'rb') as file:
        return pickle.load(file)
"""		
conv1_centers = unpickle('D:\\activation_layers\\fashion_mnist\\cluster_centers\\conv1_cluster_centers.p')
conv2_centers = unpickle('D:\\activation_layers\\fashion_mnist\\cluster_centers\\conv2_cluster_centers.p')
fc1_centers = unpickle('D:\\activation_layers\\fashion_mnist\\cluster_centers\\fc1_cluster_centers.p')
fc2_centers = unpickle('D:\\activation_layers\\fashion_mnist\\cluster_centers\\fc2_cluster_centers.p')

print(conv1_centers.shape)
print(conv2_centers.shape)
print(fc1_centers.shape)
print(fc2_centers.shape)
"""
adv_inputs = unpickle('D:\\activation_layers\\fashion_adv\\fashion_adv_inputs_1000target8.p')
adv_inputs=adv_inputs.astype(float)
adv_data = unpickle('D:\\activation_layers\\fashion_adv\\fashion_adv_data_1000target8.p')
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
		# an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.r = nn.Parameter(data=torch.zeros(28,28), requires_grad=True)

    def forward(self, x):
		# Max pooling over a (2, 2) window
        x = F.max_pool2d(self.relu1(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number
        x = F.max_pool2d(self.relu2(self.conv2(x)), 2)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x
net = Net()
net.load_state_dict(torch.load('./Misc/fashion_adv.pth'))

#initalizes usage of gpu and syncs it to cpu clock 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net=net.to(device)
torch.cuda.synchronize()

"""
active=[]
label_list=[]
def hook_function(module, grad_in, grad_out):
	# Gets the conv output of the selected filter (from selected layer)
    l=len(grad_out)#number of images in the batch
    l2=len(grad_out[0])#number of channel activations
    l3=len(grad_out[0][0])
    #print(grad_out.size())
    #print(l3)
    #print(l2)
    #print(l)
    for idx in range(0, l):
        img_filter=grad_out[idx].view(-1,l2*l3*l3)
        img_filter=img_filter.cpu()
        active.append(img_filter.data.numpy())

def hook_function_fc(module, grad_in, grad_out):
	# Gets the conv output of the selected filter (from selected layer)
    l=len(grad_out)
    #print(l)
    for idx in range(0, l):
        img_filter=grad_out[idx].view(-1)
        img_filter=img_filter.cpu()
        active.append(img_filter.data.numpy())		


net.conv1.register_forward_hook(hook_function)	
#net.fc1.register_forward_hook(hook_function_fc)

length = len(adv_inputs)
for i in range(0,length) :
    # get the inputs
    input = adv_inputs[i]
    input=torch.from_numpy(input)
    input=input.float()
    #print(input.dtype)
    input=input.view(1,1,28,28)
    input=input.to(device)
    out = net(input)
    
pickle.dump( active, open( "fashion_adv_conv1_1000target8.p", "wb" ) )
"""

"""
act=unpickle('D:\\activation_layers\\fashion_adv\\1000_target8\\fashion_adv_fc2_1000target8.p')
centers=unpickle('D:\\activation_layers\\fashion_mnist\\cluster_centers\\fc2_cluster_centers.p')
l=len(act)
print(l)
l2=len(centers)
min_value=np.zeros(shape=(l,1))
min_index=np.zeros(shape=(l,1))
for k in range(0,l):
    min_value[k]=sys.maxsize
    #print(sys.maxsize)
    for j in range(0,l2):
        dist=np.linalg.norm(act[k]-centers[j])
        if(dist<min_value[k]):
            min_value[k]=dist
            min_index[k]=j

print(min_value)	
print(min_index)

pickle.dump(min_index,open('fashion_adv_fc2_1000target8_points.p',"wb"))


index_count=np.zeros(shape=(l2,1))
min_index=min_index.astype(int)
for i in range(0,l):
    index_count[min_index[i]]+=1
print('\n')
print(index_count)
"""


conv1 = unpickle('D:\\activation_layers\\fashion_adv\\fashion_adv_conv1_1000target8_points.p')
conv2 = unpickle('D:\\activation_layers\\fashion_adv\\fashion_adv_conv2_1000target8_points.p')
fc1 = unpickle('D:\\activation_layers\\fashion_adv\\fashion_adv_fc1_1000target8_points.p')
fc2 = unpickle('D:\\activation_layers\\fashion_adv\\fashion_adv_fc2_1000target8_points.p')

print(len(adv_data[:,0]))
print(len(conv1))
full=np.column_stack((adv_data[:,0],conv1,conv2,fc1,fc2,adv_data[:,3]))
print(full)
print(full.shape)
classes=list(classes)
full=full.astype(int)
full=list(map(strip_list,full))
path_distributions(full,classes, 'adv_path_1000target8.txt')




