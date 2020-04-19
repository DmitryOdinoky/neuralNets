from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils

# import os
import torch
# from skimage import io, transform
import numpy as np
import matplotlib as plt



import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.utils

import warnings
warnings.filterwarnings("ignore")

plt.pyplot.ion()  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

#%%
    
    
train_dataset = torchvision.datasets.FashionMNIST("./data", download=False, transform= \
                                                transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST("./data", download=False, train=False, transform= \
                                               transforms.Compose([transforms.ToTensor()]))  


train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=100)

#%%

# a = next(iter(train_loader))
# a[0].size()

# image, label = next(iter(train_dataset))
# plt.pyplot.imshow(image.squeeze(), cmap="gray")
# print(label)

# demo_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)

# batch = next(iter(demo_loader))
# images, labels = batch
# print(type(images), type(labels))
# print(images.shape, labels.shape)

# grid = torchvision.utils.make_grid(images, nrow=10)

# plt.pyplot.figure(figsize=(15, 20))
# plt.pyplot.imshow(np.transpose(grid, (1, 2, 0)))
# print("labels: ", end=" ")
# for i, label in enumerate(labels):
#     print(output_label(label), end=", ")


#%%

torch.manual_seed(1234)

# hidden_units = 5

# net = torch.nn.Sequential(
#     torch.nn.Linear(4, hidden_units),
#     torch.nn.ReLU(),
#     torch.nn.Linear(hidden_units, 3),
#     torch.nn.Softmax(dim=1)
# )

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


net = FashionCNN()
net.to(device)

loss = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


number_of_epochs = 15

#np.random.seed(32) # set seed value so that the results are reproduceable

var_dict = {'train_loss': [], \
            'test_loss': [], \
            'train_accuracies': [], \
            'iterationz': [], \
            'train_f1_scores': [], \
            'test_f1_scores': [], \
            'test_accuracies': [], 
            }


counter = 0
stage = ''

for epoch in range(number_of_epochs):
    
    counter += 1
    print('Epoch #{} started', counter)
    
    for loader in [train_loader, test_loader]:
        
        if loader == train_loader:
            stage = 'train'
        else:
            stage = 'test'
        
        loss_epoch = []
        accuracy_epoch = []
        f1_epoch = []
        
        helper = 0
        
        for batch in loader:
            
            helper += 1
            # print(helper)
            
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            train_X = Variable(images.view(100, 1, 28, 28))
            train_y = Variable(labels)
            
            optimizer.zero_grad()
            y_prim = net.forward(train_X)
            
            class_count = y_prim.size(1)
            tmp = torch.arange(class_count).unsqueeze(dim=0)            
            
            
            y = (train_y.unsqueeze(dim=1) == tmp).float()
            
            
            
            loss = torch.mean(-y*torch.log(y_prim))
            
            loss = loss(y_prim, labels)
            loss_epoch.append(loss.item())

            _, predict_y = torch.max(y_prim, 1)
            
            correct = 0
            total = 0
            
            for i in range(len(images)):
                act_label = torch.argmax(y_prim[i]) # act_label = 1 (index)
                pred_label = torch.argmax(y[i]) # pred_label = 1 (index)
            
                if(act_label == pred_label):
                    correct += 1
                total += 1
            
            accuracy = correct/total
            
            #accuracy = accuracy_score(train_y.data, predict_y.data)
            
            
            f1 = sklearn.metrics.f1_score(train_y.data, predict_y.data, average='macro')
            
            accuracy_epoch.append(accuracy)
            f1_epoch.append(f1.item())
            
            if not (helper % 10):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(helper, loss.item(), accuracy))
            
            
            
            if loader == train_loader:
                
                loss.backward()
                optimizer.step()
                
        if stage == 'train':
           var_dict[f'{stage}_loss'].append(np.average(loss_epoch))
           var_dict[f'{stage}_accuracies'].append(np.average(accuracy_epoch))
           var_dict[f'{stage}_f1_scores'].append(np.average(f1_epoch))
            
        else:
           var_dict[f'{stage}_loss'].append(np.average(loss_epoch))
           var_dict[f'{stage}_accuracies'].append(np.average(accuracy_epoch))
           var_dict[f'{stage}_f1_scores'].append(np.average(f1_epoch))
        
    var_dict['iterationz'].append(counter)
    


#%%    
    
f, (ax1, ax2, ax3) = plt.pyplot.subplots(3, 1, sharey=False)

ax1.set_title('1: Loss.  2: F1. 3: Accuracy.')

sc1 = ax1.scatter(var_dict['iterationz'], var_dict['train_loss'])
sc2 = ax1.scatter(var_dict['iterationz'],  var_dict['test_loss'])


sc3 = ax2.scatter(var_dict['iterationz'], var_dict['train_f1_scores'])
sc4 = ax2.scatter(var_dict['iterationz'], var_dict['test_f1_scores'])

sc5 = ax3.scatter(var_dict['iterationz'], var_dict['train_accuracies'])
sc6 = ax3.scatter(var_dict['iterationz'], var_dict['test_accuracies'])


ax1.set_xticks([])
ax2.set_xticks([])


ax1.legend((sc1, sc2), ('test', 'train'), loc='upper right', shadow=True)

#%%


