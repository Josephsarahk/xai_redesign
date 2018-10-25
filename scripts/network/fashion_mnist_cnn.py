import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms


transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5,0.5,0.5), (.5,.5,.5))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
										download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
										  shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
									   download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										 shuffle=False, num_workers=0)
										 
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
		   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
		   

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
net=Net()


#
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

"""
for epoch in range(3):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
"""
#print('Finished Training')
#print(net)
#torch.save(net,'fashion_mnist.pth')
#torch.save(net.state_dict(),'fashion_mnist.pth')
#
"""
torch.save(dict(
	model=net, 
	model_state=net.state_dict()), 
		   'C:\\Users\\Karthik\\Documents\\Python Scripts\\net.checkpoint.pth.tar')
"""
"""
weight=net.conv1.weight.data
#print(weight[0,:,:].size())

fig = plt.figure()
plt.figure(figsize=(10,10))

for idx in range(0,10):
	#print(filt[0, :, :])
	plt.subplot(2,5, idx + 1)
	plt.imshow(weight[idx,0, :, :], cmap="gray")
	plt.axis('off')
fig.show


weight2=net.conv2.weight.data
#print(weight[0,:,:].size())

fig2 = plt.figure()
plt.figure(figsize=(10,10))

for idx in range(0,20):
	#print(filt[0, :, :])
	plt.subplot(4,5, idx + 1)
	plt.imshow(weight2[idx,0, :, :], cmap="gray")
	plt.axis('off')
fig2.show
"""
"""
def hook_function(module, grad_in, grad_out):
	# Gets the conv output of the selected filter (from selected layer)
	
	conv_output = grad_out[0, 3]
	#print(conv_input.size())
	print(conv_output.data.size())
	fig4=plt.figure()
	plt.figure(figsize=(15,15))
	plt.imshow(conv_output.data[:,:], cmap="gray")
	plt.axis('off')
	fig4.show
"""
"""
def hook_function(module, grad_in, grad_out):
	# Gets the conv output of the selected filter (from selected layer)
	
	fig=[None]*10
	for idx in range(0,10):
		fig[idx]=plt.figure()
		conv_output = grad_out[0, idx]
		#print(conv_input.size())
		#print(conv_output.data.size())
		plt.figure(figsize=(15,15))
		plt.imshow(conv_output.data[:,:], cmap="gray")
		plt.axis('off')
		fig[idx].show
"""	

	
def hook_function(module, grad_in, grad_out):
	# Gets the conv output of the selected filter (from selected layer)
	
	
	for idx in range(0,6):
		fig=plt.figure()
		conv_output = grad_out[0, idx]
		#print(conv_input.size())
		#print(conv_output.data.size())
		plt.figure(figsize=(10,10))
		plt.imshow(conv_output.data[:,:], cmap="gray")
		plt.axis('off')
		fig.show	
	
#conv_output.data[img_idx,:,i,j]
"""
#net.conv1.register_forward_hook(printnorm)
net.conv1.register_forward_hook(hook_function)	


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
"""

net.load_state_dict(torch.load('./Misc/fashion_mnist.txt'))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
	
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
net.conv1.weight.data[0,:,:]=torch.zeros([5, 5], dtype=torch.float32)
#net.conv1.weight.data[1,:,:]=torch.zeros([5, 5], dtype=torch.float32)
net.conv1.weight.data[2,:,:]=torch.zeros([5, 5], dtype=torch.float32)
net.conv1.weight.data[3,:,:]=torch.zeros([5, 5], dtype=torch.float32)
net.conv1.weight.data[4,:,:]=torch.zeros([5, 5], dtype=torch.float32)
net.conv1.weight.data[5,:,:]=torch.zeros([5, 5], dtype=torch.float32)
print('\n')
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
	
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

