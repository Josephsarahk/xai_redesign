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
#from fashion_mnist_cnn import Net


#Downloads and transforms the image set
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,.5,.5), (.5,.5,.5))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=0)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

#Class List
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
		   
#loads model and weight into variable net		   
#checkpoint = torch.load("net.checkpoint.pth.tar")
#net = checkpoint["model"]

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
net.load_state_dict(torch.load('./Misc/fashion_epoch1_14000.pth'))
print(net)

#initalizes usage of gpu and syncs it to cpu clock 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net=net.to(device)
torch.cuda.synchronize()

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
        #img_filter=grad_out[idx,:,i,j].view(-1)
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


#net.conv1.register_forward_hook(hook_function)	
#net.relu2.register_forward_hook(hook_function)	
net.relu4.register_forward_hook(hook_function_fc)


for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data
    inputs=inputs.to(device)
    out = net(inputs)
    
    labels=labels.numpy()
    l=len(labels)
    for idx in range(0,l):
        label_list.append(labels[idx])
	
	

print(len(active))
#print(len(label_list))
pickle.dump( active, open( "fashion_epoch1_14000_relu4.p", "wb" ) )

#pickle.dump(label_list,open('fashion_mnist_label_list.p','wb'))


"""
weight=net.conv1.weight.data
#print(weight[0,:,:].size())

fig = plt.figure()
plt.figure(figsize=(15,15))

for idx in range(0,6):
	#print(filt[0, :, :])
	plt.subplot(2,5, idx + 1)
	plt.imshow(weight[idx,0, :, :], cmap="gray")
	plt.axis('off')
fig.show

plt.imshow(weight[0,0, :, :], cmap="gray")
plt.axis('off')
fig.show
"""
