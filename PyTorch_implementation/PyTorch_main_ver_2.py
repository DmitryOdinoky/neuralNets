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


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X
    
# load IRIS dataset
dataset = pd.read_csv('D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/neuralNets/neuralNets/datasets/iris.csv')

dataset = sklearn.utils.shuffle(dataset)



#%%


# transform species to numerics
dataset.loc[dataset.species=='setosa', 'species'] = 0
dataset.loc[dataset.species=='versicolor', 'species'] = 1
dataset.loc[dataset.species=='virginica', 'species'] = 2

dataToTrain = dataset.sample(frac=0.8,random_state=200)
dataToTest = dataset.drop(dataToTrain.index)

#train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values, dataset.species.values, test_size=0.8)

batch_size = 5

train_dataset = []    
test_dataset = []

for i in range(0, len(dataToTrain), batch_size):
    train_dataset.append(dataToTrain[i:i+batch_size])
    
for i in range(0, len(dataToTest), batch_size):
    test_dataset.append(dataToTest[i:i+batch_size])




#%%


net = Net()

criterion = nn.CrossEntropyLoss()# cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


number_of_epochs = 150

#np.random.seed(32) # set seed value so that the results are reproduceable


train_costs = []
test_costs = []
train_accuracies = []
test_accuracies = []
iterationz = []

train_f1_scores = []
test_f1_scores = []

counter = 0

for epoch in range(number_of_epochs):
    
    counter+=1
    
    for dataset in [train_dataset, test_dataset]:
        
        for batch in train_dataset:
            
            train_X = batch[batch.columns[0:4]].values
            train_y = batch.species.values
            
            train_X = Variable(torch.Tensor(train_X).float())
            train_y = Variable(torch.Tensor(train_y).long())
            
            optimizer.zero_grad()
            train_predict_out = net(train_X)
            
            train_loss = criterion(train_predict_out, train_y)
            
            
            _, train_predict_y = torch.max(train_predict_out, 1)
            
            accuracy_train = accuracy_score(train_y.data, train_predict_y.data)
            f1_train = sklearn.metrics.f1_score(train_y.data, train_predict_y.data, average='micro')
            
            
            
            train_loss.backward()
            optimizer.step()
            
        for batch in test_dataset:
            
            test_X = batch[batch.columns[0:4]].values
            test_y = batch.species.values
            
            test_X = Variable(torch.Tensor(test_X).float())
            test_y = Variable(torch.Tensor(test_y).long())
            
            #optimizer.zero_grad()
            test_predict_out = net(test_X)
            
            test_loss = criterion(test_predict_out, test_y)
            
            
            _, test_predict_y = torch.max(test_predict_out, 1)
            
            accuracy_test = accuracy_score(test_y.data, test_predict_y.data)
            f1_test = sklearn.metrics.f1_score(test_y.data, test_predict_y.data, average='micro')
            
            
    if epoch % 5 == 0:
        
        print('number of epoch', epoch, 'loss', train_loss.item())
        
        train_costs.append(train_loss.item())
        test_costs.append(test_loss.item())
   
        train_accuracies.append(accuracy_train.item())
        test_accuracies.append(accuracy_test.item())
        
        train_f1_scores.append(f1_train.item())
        test_f1_scores.append(f1_test.item())

        #extractionz.append(extracted)

        iterationz.append(counter)
        
        
#%%
        
f, (ax1, ax2) = plt.pyplot.subplots(2, 1, sharey=False)
sc1 = ax1.scatter(iterationz, test_costs)


sc2 = ax1.scatter(iterationz, train_costs)
ax1.set_title('Loss (top) vs F1 (bottom)')
sc3 = ax2.scatter(iterationz, test_f1_scores)
sc4 = ax2.scatter(iterationz, train_f1_scores)

ax1.set_xticks([])
#ax2.set_xticks([])

ax1.legend((sc1, sc2), ('train', 'test'), loc='upper right', shadow=True)
            
               
        
        
#%%

# predict_out = net(test_X) 


# _, predict_y = torch.max(predict_out, 1)

# print('prediction accuracy', accuracy_score(test_y.data, predict_y.data))

# #print('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
# print('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
# #print('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
# print('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))
# #print('macro f1', sklearn.metrics.f1_score(test_y.data, predict_y.data, average='macro'))
# print('micro f2', sklearn.metrics.f1_score(test_y.data, predict_y.data, average='micro'))
