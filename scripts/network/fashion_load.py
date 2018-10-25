import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from fashion_mnist_cnn import Net
from channel_reducer import ChannelReducer
import matplotlib.backends.backend_pdf

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,.5,.5), (.5,.5,.5))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

#net.load_state_dict(torch.load('fashion_mnist.pth'))
checkpoint = torch.load("./Misc/net.checkpoint.pth.tar")
net = checkpoint["model"]
print(net)

active=[]
def hook_function(module, grad_in, grad_out):
	# Gets the conv output of the selected filter (from selected layer)
	out=grad_out[0]
	length=len(grad_out[0])
	print(length)
	print(out.size())
	
	for idx in range(0,length):
		#fig=plt.figure()
		conv_output = grad_out[0, idx]
		active.append(conv_output.data.numpy())
		"""
		#print(conv_input.size())
		#print(conv_output.data.size())
		plt.figure(figsize=(10,10))
		plt.imshow(conv_output.data[:,:], cmap="gray")
		plt.axis('off')
		fig.show	
		"""
	
	

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())


#net.conv1.register_forward_hook(printnorm)
net.conv1.register_forward_hook(hook_function)	
#net.conv2.register_forward_hook(hook_function)	

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.size())
print(images[0].unsqueeze(0).size())
#input = torch.randn(1, 1, 28, 28)
#print(input.size())

input=images[0].unsqueeze(0)

fig11=plt.figure()
plt.figure(figsize=(15,15))
plt.imshow(input[0,0,:,:], cmap='gray')
plt.axis('off')
fig11.show


out = net(input)
n_groups=2
active=np.asarray(active)
active=torch.from_numpy(active).unsqueeze(0)
active=F.relu(active)
nmf = ChannelReducer(n_groups, "NMF")
print(nmf.fit_transform(active).shape)
spatial_factors = nmf.fit_transform(active)[0].transpose(2, 0, 1).astype("float32")
channel_factors = nmf._reducer.components_.astype("float32")

print(spatial_factors.shape)
print(channel_factors.shape)

spatial_memes=spatial_factors.transpose(1,2,0).astype("float32")
channel_memes=channel_factors

print(spatial_memes.shape)
print(channel_memes.shape)

"""
group_zero=np.outer(spatial_memes[:,:,0],channel_memes[0,:])
print(group_zero.shape)
group_zero=group_zero.reshape(10,24,24)
print(group_zero.shape)
print(group_zero)

for idx in range(0,10):
	fig=plt.figure()
	#print(conv_input.size())
	#print(conv_output.data.size())
	plt.figure(figsize=(10,10))
	plt.imshow(group_zero[idx,:,:], cmap="gray")
	plt.axis('off')
	fig.show
"""


pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
group=[]
for idx in range(0,n_groups):
	var=np.outer(spatial_memes[:,:,idx],channel_memes[idx,:])
	group.append(var)
group=np.asarray(group)
print(group.shape)
group=group.reshape(n_groups,10,24,24)

for idx in range(0,n_groups):
    for idx2 in range(0,10):
        fig=plt.figure()
		#print(conv_input.size())
		#print(conv_output.data.size())
        plt.figure(figsize=(10,10))
        plt.imshow(group[idx,idx2,:,:], cmap="gray")
        plt.axis('off')
        pdf.savefig()
        fig.show
pdf.close()
