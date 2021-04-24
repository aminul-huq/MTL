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

device = 'cuda:0'
best_acc = 0
model_name = 'random_4'

x = 0
for i in range (1,101):
    for j in range(500):
        if (j+x) < (500/4)+x:
            trainset.targets[j+x] = 1
        elif (j+x) < (500/2)+x and (j+x)>=(500/4)+x:
            trainset.targets[j+x] =2
        elif (j+x) < ((3*500)/4)+x and (j+x)>=(500/2)+x:
            trainset.targets[j+x] =3
        else :
            trainset.targets[j+x] = 0
    x += 500

x = 0
for i in range (1,101):
    for j in range(100):
        if (j+x) < (100/4)+x:
            testset.targets[j+x] = 1
        elif (j+x) < (100/2)+x and (j+x)>=(100/4)+x:
            testset.targets[j+x] =2
        elif (j+x) < ((3*100)/4)+x and (j+x)>=(100/2)+x:
            testset.targets[j+x] =3
        else :
            testset.targets[j+x] = 0
    x += 100 
    
# Can be replaced by Tensor Dataset
class NewDataset(Dataset):
    
    def __init__(self,data):
        self.data = data
        
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self,idx):
        label = self.data[idx][1]
        image = self.data[idx][0]
        return image, label
    
new_trainset = NewDataset(trainset)
new_testset = NewDataset(testset)

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
    return test_loss/(i+1), best_acc

total_classes = 4
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