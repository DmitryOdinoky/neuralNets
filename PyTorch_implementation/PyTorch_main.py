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



import sklearn.metrics

from torch.autograd import Variable

#%%

class CrossEntropyCustom:
    
    #compute stable cross-entropy
    
    def __init__(self):
        self.y = None
        self.y_hat = None
        
        
    def forward(self, y, y_hat):
        
        #m = np.shape(y.value)[0]
        
        self.y = y
        self.y_hat = y_hat
        
        
        self.output = torch.Tensor(-np.sum(self.y*np.log(self.y_hat)))
        

        return self.output
    
    def backward(self):
        
        #m = np.shape(self.y.value)[0]
        

        self.y_hat.grad =  torch.Tensor(self.y/self.y_hat)





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
        self.SM = nn.Softmax(dim=1)
        
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
        
        
myModel = Net()

print(myModel)





#%%

## backprop in n epochs

optimizer = optim.Adam(myModel.parameters(), lr=0.001)
np.random.seed(32)

number_of_epochs = 200

#loss_func = nn.CrossEntropyLoss()
loss_func = CrossEntropyCustom()


costs = []
accuracies = []
iterationz = []
extractionz = []
counter = 0


for epoch in range(number_of_epochs):
    
    counter+=1
    

    
    for dataset in [train_dataset, test_dataset]:
      
        # --------------- TRAIN DATA, forward, loss & f1, backward

        for batch in train_dataset:
    

    
            X_train = batch[:,0:4]
            Y_train = batch[:,4]
            Y_train = np.array(convert_to_probdist(Y_train))

            
            
            out = myModel.forward(torch.Tensor(X_train))
            train_loss = loss_func.forward(Y_train, out)
            
    #         output = out.value
            
    #         correct = 0
    #         total = 0
    #         true = []
    #         pred = []
           
           
    #         for i in range(len(batch)):
    #             act_label = np.argmax(Y_train[i]) # act_label = 1 (index)
    #             pred_label = np.argmax(output[i]) # pred_label = 1 (index)
    #             true.append(act_label)
    #             pred.append(pred_label)
    #             if(act_label == pred_label):
    #                 correct += 1
    #             total += 1
                
    #         f1_train = sklearn.metrics.f1_score(true, pred, average='macro')
            
            
    #         loss_func.backward()  
    #         myModel.backward()
            
            
    #     # --------------- TEST DATA, forward, loss & f1
            
    #     for batch in test_dataset:

           
    #        X_test = batch[:,0:4]
    #        Y_test = batch[:,4]
    #        Y_test = np.array(toolset_new.convert_to_probdist(Y_test))

           
    #        out = actual_model.forward(X_test)
    #        test_loss = loss_func.forward(Variable(Y_test), out)

           
    #        predict = actual_model.forward(X_test)
    #        predicted = predict.value

           
    #        correct = 0
    #        total = 0
    #        true = []
    #        pred = []
           
           
    #        for i in range(len(batch)):
    #            act_label = np.argmax(Y_test[i]) # act_label = 1 (index)
    #            pred_label = np.argmax(predicted[i]) # pred_label = 1 (index)
    #            true.append(act_label)
    #            pred.append(pred_label)
    #            if(act_label == pred_label):
    #                correct += 1
    #            total += 1
               
        
    #        f1_test = sklearn.metrics.f1_score(true, pred, average='macro')
    #        accuracy = (correct/total)

                
    # if (epoch % 10) == 0:
    #     print("Cost at epoch#{}: {}".format(epoch, train_loss.value))
    #     print("Accuracy --- > {}".format(accuracy))
    #     print("F1 --- > {}".format(f1_test))
       
    #     train_costs.append(train_loss.value)
    #     test_costs.append(test_loss.value)
        
    #     test_accuracies.append(accuracy)
        
    #     train_f1_scores.append(f1_train)
    #     test_f1_scores.append(f1_test)
        
        
     
      
        
    #     iterationz.append(counter)
    
    



