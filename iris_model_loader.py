import csv

import copy

from tools import toolset_new
from tools.toolset_new import Variable, MSE_Loss, CrossEntropy, MyModel

import numpy as np
import numpy.core.defchararray as np_f
import matplotlib as plt
import pickle

import sklearn


with open('datasets/iris.csv', newline='') as csvfile:
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




#%%

batch_size = 5

train_dataset = []    
test_dataset = []

for i in range(0, len(dataToTrain), batch_size):
    train_dataset.append(dataToTrain[i:i+batch_size])
    
for i in range(0, len(dataToTest), batch_size):
    test_dataset.append(dataToTest[i:i+batch_size])


# X = dataset[0][:,0:4]
# Y = dataset[0][:,4]




#%%

number_of_epochs = 400

np.random.seed(32) # set seed value so that the results are reproduceable


#%%

costs = []
accuracies = []
f1_scores = []
iterationz = []
extractionz = []
counter = 0

loss_func = CrossEntropy()

#loss_func = MSE_Loss()

actual_model = MyModel()



for epoch in range(number_of_epochs):
    
    counter+=1
    
    # np.random.shuffle(dataset)
    
    for dataset in [train_dataset, test_dataset]:
      

        for batch in train_dataset:
    
            #np.random.shuffle(batch)
    
            X_train = batch[:,0:4]
            Y_train = batch[:,4]
            Y_train = np.array(toolset_new.convert_to_probdist(Y_train))
    
            # ------------------------- forward-prop -------------------------    
            
            
            out = actual_model.forward(X_train)
            loss = loss_func.forward(Variable(Y_train), out)           
    
            loss_func.backward()  
            actual_model.backward()
            
        for batch in test_dataset:
           
           #np.random.shuffle(batch)
           
           X_test = batch[:,0:4]
           Y_test = batch[:,4]
           Y_test = np.array(toolset_new.convert_to_probdist(Y_test))
           
           # ------------------------- forward-prop -------------------------    
           
           
           out = actual_model.forward(X_test)
           loss = loss_func.forward(Variable(Y_test), out)

           
           #instance_2 = copy.deepcopy(instance)
           
           
          
           
           predict = actual_model.forward(X_test)
           predicted = predict.value
           
           
           
           correct = 0
           total = 0
           true = []
           pred = []
           
           
           for i in range(len(batch)):
               act_label = np.argmax(Y_test[i]) # act_label = 1 (index)
               pred_label = np.argmax(predicted[i]) # pred_label = 1 (index)
               true.append(act_label)
               pred.append(pred_label)
               if(act_label == pred_label):
                   correct += 1
               total += 1
               
        
           f1 = sklearn.metrics.f1_score(true, pred, average='macro')
           accuracy = (correct/total)

                
    if (epoch % 10) == 0:
        print("Cost at epoch#{}: {}".format(epoch, loss.value))
        print("Accuracy --- > {}".format(accuracy))
        print("F1 --- > {}".format(f1))
       
        costs.append(loss.value)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        #extractionz.append(extracted)
      
        
        iterationz.append(counter)
        
#%%

#3plt.pyplot.scatter(iterationz, costs)
#plt.pyplot.scatter(iterationz, accuracies)



# Create two subplots and unpack the output array immediately
f, (ax1, ax2, ax3) = plt.pyplot.subplots(3, 1, sharey=False)
ax1.scatter(iterationz, costs)
ax1.set_title('Loss vs Accuracy vs F1')
ax2.scatter(iterationz, accuracies)
ax3.scatter(iterationz, f1_scores)

ax1.set_xticks([])
ax2.set_xticks([])

