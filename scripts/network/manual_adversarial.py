# -*- coding: utf-8 -*-

import torch
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

#Declares Network Architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.r = nn.Parameter(data=torch.zeros(28,28), requires_grad=True)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x + self.r
        x = torch.clamp(x,0,1)
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
"""
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #self.r = nn.Parameter(data=torch.zeros(28,28), requires_grad=True)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x + self.r
        x = torch.clamp(x,0,1)
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net2=Net2()
net2.load_state_dict(torch.load('./Misc/fashion_mnist.txt'))

net.conv1.weight.data=net2.conv1.weight.data
net.conv2.weight.data=net2.conv2.weight.data
net.fc1.weight.data=net2.fc1.weight.data
net.fc2.weight.data=net2.fc2.weight.data
net.fc3.weight.data=net2.fc3.weight.data
net.r.data=torch.zeros(28,28)

with open('fashion_adv.pth', 'wb') as f:
    torch.save(net.state_dict(), f)
"""
#Manipulations to the images where x'=(x-mu)/sigma
transform = transforms.Compose(
    [transforms.Resize((28,28)),
            transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])
#gets data set and initalizes batch size and training data

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('T-shirt/top', 'trouser', 'pullover', 'dress',
           'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#dataiter = iter(testloader)
#images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#outputs = net(images)

#declares lose function and optimizer as well as training for the network
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

"""
for epoch in range(5):  # loop over the dataset multiple times

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

with open('fashion_adv.pth', 'wb') as f:
    torch.save(net.state_dict(), f)
"""		
	

#Loss gradient attack	
class Attack:
    def __init__(self, weights):
        self.net = Net() 
        self.softmaxwithxent = nn.CrossEntropyLoss()
        self.optimizer       = optim.SGD(params=[self.net.r], lr=0.008)
        self.load_weights(weights) 

    def load_weights(self, weights=None):
        assert os.path.isfile(weights), "Error: weight file {} is invalid".format(weights)
        # LOAD PRE-TRAINED WEIGHTS 
        self.net.load_state_dict(torch.load(weights))
        print("Weights Loaded!")

    def attack(self, x, y_true, y_target, regularization=None):
        """
        This method uses the method described in the paper
        "Intriguing properties of neural networks" to find a 
        noise vector 'r' that misclassifies 'x' as 'y_target'. 
        Parameters
        ----------
        x: a numpy array containing an mnist example 
        y_target: target label for attack. (int) 
        y_true: true label for x (int)
        Returns
        -------
        noise: Numpy array (1x784) of the noise to be added to x 
        y_pred: Prediction before adversarial optimization  
        y_pred_adversarial: Prediction after adversarial optimization 
        """

        _x = Variable(torch.FloatTensor(x))
        _y_target = Variable(torch.LongTensor([y_target]))

        # Reset value of r 
        self.net.r.data = torch.zeros(28,28) 

        # Classification before modification 
        y_pred =  np.argmax(self.net(_x).data.numpy())
        incorrect_classify = False  
        # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
        if y_true != y_pred:
            incorrect_classify = True
            print("WARNING: IMAGE WAS NOT CLASSIFIED CORRECTLY")

        # Optimization Loop 
        for iteration in range(1000):

            self.optimizer.zero_grad() 
            outputs = self.net(_x)
            xent_loss = self.softmaxwithxent(outputs, _y_target) 

            if regularization == "l1":
                adv_loss = xent_loss + torch.mean(torch.abs(self.net.r))
            elif regularization == "l2":
                adv_loss  = xent_loss + torch.mean(torch.pow(self.net.r,2))
            elif regularization == None:
                adv_loss = xent_loss
            else:
                raise Exception("regularization method {} is not implemented, please choose one of l1, l2 or None".format(regularization))

            adv_loss.backward() 
            self.optimizer.step() 

            # keep optimizing Until classif_op == _y_target
            y_pred_adversarial = np.argmax(self.net(_x).data.numpy())
            if y_pred_adversarial == y_target:
                break 

        if iteration == 999:
            print("Warning: optimization loop ran for 1000 iterations. The result may not be correct")

        return self.net.r.data.numpy(), y_pred, y_pred_adversarial 

"""
def show_adv_example(img, r):
    fig=plt.figure(figsize=(1,3))
    fig.add_subplot(1,3,1)
    plt.imshow(torchvision.utils.make_grid(img))
    fig.add_subplot(1,3,2)
    imshow(torchvision.utils.make_grid(torch.from_numpy(r)))
    fig.add_subplot(1,3,3)
    imshow(torchvision.utils.make_grid(img+torch.from_numpy(r)))
    plt.show()
"""
#criterion.zero_grad()


active=[]
avg_act=[]
def hook_function(module, grad_in, grad_out):
	# Gets the conv output of the selected filter (from selected layer)
    out=grad_out[0]
    length=len(grad_out[0])
	#print(length)
    print(out[0][0].size())
	
    for idx in range(0,length):
        fig=plt.figure()
        conv_output = grad_out[0, idx]
        active.append(conv_output.data.numpy())
		#print(conv_input.size())
        #print(conv_output.data[:,:])
        plt.figure(figsize=(10,10))
        plt.imshow(conv_output.data[:,:], cmap="gray")
        plt.axis('off')
        fig.show
        sum=0
        for idx2 in range(0,len(out[0][0])):
            for idx3 in range(0,len(out[0][0])):
                sum+=conv_output.data[idx2,idx3].numpy()
        avg=sum/(len(out[0][0])**2)
        avg_act.append(avg)

attack=Attack('./Misc/fashion_adv.pth')


net.conv1.register_forward_hook(hook_function)	
#net.conv2.register_forward_hook(hook_function)	
batch=1000
trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=False, num_workers=0)
dataiter = iter(trainloader2)
images, labels = dataiter.next()
"""
print(labels.size())
print(images.size())
print(images[0].unsqueeze(0).size())
#input = torch.randn(1, 1, 28, 28)
#print(input.size())
labels=labels.numpy()
adv=np.zeros(shape=(batch,28,28))
target=np.zeros(shape=(batch,1))
pred=np.zeros(shape=(batch,1))
adv_pred=np.zeros(shape=(batch,1))

for i in range(0,batch):
    input=images[i].unsqueeze(0)
    adv_input=input.numpy()
    n=labels[i]

    end=10
    r=list(range(0,n))+list(range(n+1,end))
    target_class=random.choice(r)

    target_class=8
    target[i]=target_class
    adv_out,pred[i],adv_pred[i]=attack.attack(adv_input,n,target_class, regularization='l1')
    adv[i]=adv_out+adv_input
    #print(adv[i].shape)
    adv[i]=torch.from_numpy(adv[i])
print(adv.shape)
pickle.dump(adv, open( "fashion_adv_inputs_1000target8.p", "wb" ) )
c=np.column_stack((labels,target,pred,adv_pred))
print(c)
pickle.dump(c, open( "fashion_adv_data_1000target8.p", "wb" ) )
"""


input=images[0].unsqueeze(0)
adv_input=input.numpy()
n=labels[0].numpy()
end=10
r=list(range(0,n))+list(range(n+1,end))
target_class=random.choice(r)

adv_out,y_pred,y_bad=attack.attack(adv_input,n,target_class, regularization='l1')
adv=adv_out+adv_input
adv=torch.from_numpy(adv)
adv_out=torch.from_numpy(adv_out)
#print(adv_out.size())

print(input[0,0,:,:].size())
fig11=plt.figure()
plt.figure(figsize=(15,15))
plt.imshow(input[0,0,:,:], cmap='gray')
plt.axis('off')
fig11.show
print(n)
print(y_pred)

fig20=plt.figure()
plt.figure(figsize=(15,15))
plt.imshow(adv[0,0,:,:], cmap='gray')
plt.axis('off')
fig20.show


out = net(input)
reg_act=np.asarray(active)
reg_act=torch.from_numpy(reg_act)
#print(reg_act.size())
print((avg_act))
del active[:]


fig12=plt.figure()
plt.figure(figsize=(15,15))
plt.imshow(adv[0,0,:,:], cmap='gray')
plt.axis('off')
fig12.show



out2=net(adv)
adv_act=np.asarray(active)
adv_act=torch.from_numpy(adv_act)
diff=adv_act-reg_act
#print(diff.size())
diff2=diff.unsqueeze(0)
#print(diff2.size())


fig = plt.figure()
plt.figure(figsize=(10,10))
for idx in range(0,6):
    #print(filt[0, :, :])
    plt.subplot(2,3, idx + 1)
    plt.imshow(diff[idx,:, :], cmap="gray")
    plt.axis('off')
fig.show


weight=net.conv1.weight.data
weight2=net.conv2.weight.data
#print(weight.size())
#print(weight[0,:,:])
#print(weight2.size())
#print(weight2[0,:,:,:])

"""
class mini_deconv(nn.Module):

    def __init__(self):
        super(mini_deconv, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.ConvTranspose2d(10, 10, 5,stride=1,output_padding=0,groups=10,dilation=1,bias=False)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x =self.conv1(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data=weight
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)		
deconv=mini_deconv()
deconv.apply(weights_init)

deconv.conv1.register_forward_hook(hook_function)

deconv_out=deconv(diff2)

"""