from __future__ import print_function, division
# import os
import torch
# from skimage import io, transform
import numpy as np
import matplotlib as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

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


#dataset_frame = pd.read_csv('D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/neuralNets/neuralNets/datasets/iris.csv')



#%%



class simple_Dataset(object):
    
    def __init__(self, csv_file, train = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.if_train = True
        self.dataset_frame = pd.read_csv(csv_file)
        
        self.dataToTrain = []
        self.dataToTest = []
 
        self.dataset_frame = sklearn.utils.shuffle(self.dataset_frame)
        self.dataToTrain = self.dataset_frame.sample(frac=0.8,random_state=200)
        self.dataToTest = self.dataset_frame.drop(self.dataToTrain.index)
        
        self.batch_size = 5
        
        self.batchez = []    
      
        
        if self.if_train == True:

            for i in range(0, len(self.dataToTrain), self.batch_size):
                self.batchez.append(self.dataToTrain[i:i+self.batch_size])
        else:
            
            for i in range(0, len(self.dataToTest), self.batch_size):
                self.batchez.append(self.dataToTest[i:i+self.batch_size])

    def __len__(self):
        return len(self.dataset_frame)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        batch = self.batchez[idx]
        
        batch.loc[batch.species=='setosa', 'species'] = 0
        batch.loc[batch.species=='versicolor', 'species'] = 1
        batch.loc[batch.species=='virginica', 'species'] = 2


        return batch
    
    
train_dataset = simple_Dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/neuralNets/neuralNets/datasets/iris.csv', train = True)
test_dataset = simple_Dataset(csv_file = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/neuralNets/neuralNets/datasets/iris.csv', train = False)


sample = train_dataset[0]

#%%

torch.manual_seed(1234)

hidden_units = 5

net = torch.nn.Sequential(
    torch.nn.Linear(4, hidden_units),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_units, 3),
    torch.nn.Softmax(dim=1)
)




optimizer = optim.Adam(net.parameters(), lr=0.001)


number_of_epochs = 50

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
    
    for dataset in [train_dataset, test_dataset]:
        
        if dataset == train_dataset:
            stage = 'train'
        else:
            stage = 'test'
        
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
            
            correct = 0
            total = 0
            
            for i in range(len(batch)):
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
            
            if dataset == train_dataset:
                
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


    