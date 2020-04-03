import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import copy

import pickle
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms


import numpy as np
import numpy.core.defchararray as np_f




with open('D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/neuralNets/neuralNets/datasets/iris.csv', newline='') as csvfile:
     data = list(csv.reader(csvfile))


dataExpected = np.array(data)


dataExpected = np_f.replace(dataExpected[:,:], 'setosa', '0')
dataExpected = np_f.replace(dataExpected[:,:], 'versicolor', '1')
dataExpected = np_f.replace(dataExpected[:,:], 'virginica', '2')
dataLabels = dataExpected[0]

dataExpected = np.delete(dataExpected,0,0)

dataExpected = dataExpected.astype(np.float)

np.random.shuffle(dataExpected)

dataToTrain = dataExpected[0:130, :]
dataToTest = dataExpected[130:150, :]

def convert_to_probdist(vector):
    
    probdists = []

    for row in vector:
        
        blank_probdist = np.zeros(3)
        
        if row == 0:
            blank_probdist[0] = 1
            probdists.append(blank_probdist)
        elif row == 1:
            blank_probdist[1] = 1
            probdists.append(blank_probdist)
        elif row == 2:
            blank_probdist[2] = 1
            probdists.append(blank_probdist)
    
    return probdists

#%%

batch_size = 5

train_dataset = []    
test_dataset = []

for i in range(0, len(dataToTrain), batch_size):
    train_dataset.append(dataToTrain[i:i+batch_size])
    
for i in range(0, len(dataToTest), batch_size):
    test_dataset.append(dataToTest[i:i+batch_size])
    
#%%

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.out = None
        self.learning_rate = 0.0001
        
        self.Z1 = nn.Linear(4,64)
        self.A1 = nn.ReLU()
        self.Z2 = nn.Linear(64,32)
        self.A2 = nn.ReLU()
        self.Z3 = nn.Linear(32,3)
        self.SM = nn.Softmax()
        
        self.graph = [self.Z1,self.A1,self.Z2,self.A2,self.Z3,self.SM]
  
        
    def forward(self, dataset):
        
        self.out = dataset
        
        for layer in self.graph:
          
            self.out = layer.forward(self.out)
        
        return  self.out
    
    def backward(self):
    
       
        rev_layers = self.graph[::-1]
        
        for layer in rev_layers:
          
            layer.backward()
           
           
           
        stuffToUpdate = []
               
        for item in self.graph:
            if isinstance(item, (nn.Linear)):
                stuffToUpdate.append(item)
            elif isinstance(item, str):
                pass
            
            for linear in stuffToUpdate:
                linear.w.value += np.mean(linear.w.grad, axis=0) * self.learning_rate
                linear.b.value += np.mean(linear.b.grad, axis=0) * self.learning_rate
        
        
instance = Net()
print(instance)





#%%

## backprop in 3 epochs

optimizer = optim.Adam(instance.parameters(), lr=0.001)
np.random.seed(32)

number_of_epochs = 4000

costs = []
accuracies = []
iterationz = []
extractionz = []
counter = 0



for epoch in range(number_of_epochs):
    counter+=1
    
    # np.random.shuffle(dataset)
    
    for batch in train_dataset:
        
        X_train = batch[:,0:4]
        Y_train = batch[:,4]
        Y_train = np.array(convert_to_probdist(Y_train))
        
        
        output = instance.forward(torch.FloatTensor(X_train))
        loss = F.nll_loss(output, instance.forward(torch.FloatTensor(X_train)))
     
        #optimizer.step()
        
        costs.append(loss.item())    
        print(loss)
    
        np.random.shuffle(dataToTest)
    
        ground_truth = dataToTest[:,4]
        ground_truth = np.array(convert_to_probdist(ground_truth))
        
        instance_2 = copy.deepcopy(instance)
    
    
    
    
        predict = instance_2.forward(dataToTest[:,0:4])
        predicted = predict.value
        
        correct = 0
        total = 0
        
        for i in range(len(dataToTest)):
            act_label = np.argmax(ground_truth[i]) # act_label = 1 (index)
            pred_label = np.argmax(predicted[i]) # pred_label = 1 (index)
            if(act_label == pred_label):
                correct += 1
            total += 1
        accuracy = (correct/total)
        
        # argmaxed = predicted.argmax(1)
        # extracted = np.nonzero(ground_truth - argmaxed)
        # accuracy = np.size(extracted[0])/np.size(argmaxed)
        
        
        loss.backward()
        
        instance.backward()







    if (epoch % 10) == 0:
        print("Cost at epoch#{}: {}".format(epoch, loss.value))
        print("Accuracy --- > {}".format(accuracy))
    #print("Snapshot --- > {} ---- > {}".format(instance.out.value[0,0],instance_2.out.value[0,0]))
        costs.append(loss.value)
        accuracies.append(accuracy)
    #extractionz.append(extracted)
      
    
        iterationz.append(counter)
    
    



