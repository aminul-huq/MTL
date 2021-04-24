import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,random_split,Dataset
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from wideresnet import *

trainset = datasets.CIFAR100(root='/home/aminul/data1/', train=True, download=False, transform=transforms.ToTensor())
testset = datasets.CIFAR100(root='/home/aminul/data1/', train=False, download=False, transform=transforms.ToTensor())

device = 'cuda:1'
best_acc = 0
model_name = 'color_jitter'

transformations = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.9, hue=0.5)
])

for i in range(len(trainset.targets)):
    if i%2 == 0:
        trainset.targets[i] = 1
    else :
        trainset.targets[i] = 0
        
for i in range(len(testset.targets)):
    if i%2 == 0:
        testset.targets[i] = 1
    else :
        testset.targets[i] = 0         

class NewDataset(Dataset):
    
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self,idx):
        label = self.data[idx][1]
        if label == 1:
            image = self.transform(self.data[idx][0])
        else:
            image = self.data[idx][0]
        return image, label
    
new_trainset = NewDataset(trainset,transformations)
new_testset = NewDataset(testset,transformations)

train_loader = DataLoader(new_trainset, batch_size=100, shuffle=True)
test_loader = DataLoader(new_testset, batch_size=100, shuffle=True)    
    
def train(net,trainloader,optim,criterion,epoch,device):
    net.train()
    train_loss,total,total_correct = 0,0,0
    
    for i,(inputs,targets) in enumerate(tqdm(trainloader)):
        
        inputs,targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        
        output = net(inputs)
        loss = criterion(output,targets)
        
        loss.backward()
        
        optim.step()
        
        train_loss += loss.item()
        _,prediction = torch.max(output.data,1)
        total_correct += (prediction == targets).sum().item()
        total += targets.size(0)
        
    print("Epoch: [{}]  loss: [{:.2f}] Orig_Acc [{:.2f}]".format(epoch+1,train_loss/(i+1),
                                                                           (total_correct*100/total)))
    
    return train_loss/(i+1)


def tester(net,testloader,optim,criterion,epoch,modelname,device):
    net.eval()
    global best_acc
    test_loss,total,total_correct = 0,0,0
    
    for i,(inputs,targets) in enumerate(tqdm(testloader)):
        
        inputs,targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        
        output = net(inputs)
        loss = criterion(output,targets)
        
        test_loss += loss.item()
        _,prediction = torch.max(output.data,1)
        total_correct += (prediction == targets).sum().item()
        total += targets.size(0)
        
    acc = 100. * total_correct / total
    
    if acc > best_acc:    
        state = {
                    'model':net.state_dict(),
                    'acc':acc,
                    'epoch':epoch,
                 }      
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+model_name+'.pth')
        best_acc = acc
    
    print("\nTest Epoch #%d Loss: %.4f Orig_Acc: %.2f%%" %(epoch+1,test_loss/(i+1),acc))
    return test_loss/(i+1),best_acc

total_classes = 2
net = WideResNet(depth=28,num_classes = total_classes,widen_factor=10,dropRate=0.3).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)


num_epochs = 100
train_loss,test_loss = [],[]

for epoch in range(num_epochs):
    
    a = train(net,train_loader,optimizer,criterion,epoch,device)
    c,d = tester(net,test_loader,optimizer,criterion,epoch,model_name,device)    
    
    train_loss.append(a), test_loss.append(c)

print("Best Acc %.2f%%" %(d))