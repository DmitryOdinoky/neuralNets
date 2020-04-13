import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sklearn.utils

import matplotlib as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import torch.optim as optim

#%%

dataset = pd.read_csv('D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/neuralNets/neuralNets/datasets/iris.csv')

dataset = sklearn.utils.shuffle(dataset)

dataset.loc[dataset.species=='setosa', 'species'] = 0
dataset.loc[dataset.species=='versicolor', 'species'] = 1
dataset.loc[dataset.species=='virginica', 'species'] = 2

dataToTrain = dataset.sample(frac=0.8,random_state=200)
dataToTest = dataset.drop(dataToTrain.index)

batch_size = 5

train_dataset = []    
test_dataset = []

for i in range(0, len(dataToTrain), batch_size):
    train_dataset.append(dataToTrain[i:i+batch_size])
    
for i in range(0, len(dataToTest), batch_size):
    test_dataset.append(dataToTest[i:i+batch_size])
    

    
#%%
    
torch.manual_seed(1234)

hidden_units = 5

net = torch.nn.Sequential(
    torch.nn.Linear(4, hidden_units),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_units, 3),
    torch.nn.Softmax(dim=1)
)

#%%

#myModel = Net()

#loss = nn.CrossEntropyLoss()# cross entropy loss

optimizer = optim.Adam(net.parameters(), lr=0.001)


number_of_epochs = 50

#np.random.seed(32) # set seed value so that the results are reproduceable


losses_train = []
losses_test = []
train_accuracies = []
test_accuracies = []
iterationz = []

train_f1_scores = []
test_f1_scores = []

counter = 0

for epoch in range(number_of_epochs):
    
    counter+=1
    
    for dataset in [train_dataset, test_dataset]:
        
        loss_epoch = []
        accuracy_epoch = []
        f1_epoch = []
        
        for batch in dataset:
            
            train_X = batch[batch.columns[0:4]].values
            train_y = batch.species.values
            
            train_X = Variable(torch.Tensor(train_X).float())
            train_y = Variable(torch.Tensor(train_y).long())
            
            optimizer.zero_grad()
            y_prim = net.forward(train_X)
            
            class_count = y_prim.size(1)
            tmp = torch.arange(class_count).unsqueeze(dim=0)
            y = (train_y.unsqueeze(dim=1) == tmp).float()
            
            loss = torch.mean(-y*torch.log(y_prim))
            loss_epoch.append(loss.item())

            
            _, predict_y = torch.max(y_prim, 1)
            
            accuracy = accuracy_score(train_y.data, predict_y.data)
            f1 = sklearn.metrics.f1_score(train_y.data, predict_y.data, average='micro')
            
            accuracy_epoch.append(accuracy.item())
            f1_epoch.append(f1.item())
            
            if dataset == train_dataset:
                
                loss.backward()
                optimizer.step()
                
        if dataset == train_dataset:
            losses_train.append(np.average(loss_epoch))
            train_accuracies.append(np.average(accuracy_epoch))
            train_f1_scores.append(np.average(f1_epoch))
            
        else:
            losses_test.append(np.average(loss_epoch))
            test_accuracies.append(np.average(accuracy_epoch))
            test_f1_scores.append(np.average(f1_epoch))
        
    iterationz.append(counter)
    


#%%    
    
f, (ax1, ax2) = plt.pyplot.subplots(2, 1, sharey=False)
sc1 = ax1.scatter(iterationz, losses_test)


sc2 = ax1.scatter(iterationz, losses_train)
#ax1.set_title('Loss (top) vs F1 (bottom)')
sc3 = ax2.scatter(iterationz, train_f1_scores)
sc4 = ax2.scatter(iterationz, test_f1_scores)

ax1.set_xticks([])
#ax2.set_xticks([])

ax1.legend((sc1, sc2), ('test', 'train'), loc='upper right', shadow=True)
            
    
            
        
            
            
   

