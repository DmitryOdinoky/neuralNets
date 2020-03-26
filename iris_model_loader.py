import csv

import copy

from tools import toolset_new
from tools.toolset_new import Variable, MSE_Loss, CrossEntropy, MyModel

import numpy as np
import numpy.core.defchararray as np_f
import matplotlib as plt
import pickle


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

np.random.shuffle(dataToTest)


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

number_of_epochs = 200

np.random.seed(32) # set seed value so that the results are reproduceable


#%%

costs = []
accuracies = []
iterationz = []
extractionz = []
counter = 0

loss_func = CrossEntropy()

#oss_func = MSE_Loss()

instance = MyModel()
instance_2 = copy.deepcopy(instance)

for epoch in range(number_of_epochs):
    
    counter+=1
    
    # np.random.shuffle(dataset)
      

    for batch in train_dataset:

        np.random.shuffle(batch)

        X_train = batch[:,0:4]
        Y_train = batch[:,4]
        Y_train = np.array(toolset_new.convert_to_probdist(Y_train))

        # ------------------------- forward-prop -------------------------    
        
        
        out = instance.forward(X_train)
        loss = loss_func.forward(Variable(Y_train), out)
        
        ind = np.random.randint(0,len(test_dataset))
        
        ground_truth = test_dataset[ind][:,4]
        ground_truth = np.array(toolset_new.convert_to_probdist(ground_truth))
        
        sample = test_dataset[ind]
        

        predict = instance_2.forward(sample[:,0:4])
        predicted = predict.value
        
        correct = 0
        total = 0
        for i in range(len(sample)):
            act_label = np.argmax(ground_truth[i]) # act_label = 1 (index)
            pred_label = np.argmax(predicted[i]) # pred_label = 1 (index)
            if(act_label == pred_label):
                correct += 1
            total += 1
        accuracy = (correct/total)

        # argmaxed = predicted.argmax(1)
        # extracted = np.nonzero(ground_truth - argmaxed)
        # accuracy = np.size(extracted[0])/np.size(argmaxed)
        

        loss_func.backward()
        
        instance.backward()
        
    
        
        
        
        
            
    if (epoch % 10) == 0:
        print("Cost at epoch#{}: {}".format(epoch, loss.value))
        print("Accuracy --- > {}".format(accuracy))
        #print("Snapshot --- > {} ---- > {}".format(instance.out.value[0,0],instance_2.out.value[0,0]))
        costs.append(loss.value)
        accuracies.append(accuracy)
        #extractionz.append(extracted)
      
        
        iterationz.append(counter)
        
#%%

plt.pyplot.scatter(iterationz, costs)
plt.pyplot.scatter(iterationz, accuracies)




#%% Testing

# X_test = dataToTest[:,0:4]

# predict = instance.forward(X_test)

# predicted = predict.value

# # #%% Evaluation


# argmaxed = predicted.argmax(1)
# groundTruth = dataToTest[:,4]    

# extracted = np.nonzero(groundTruth - argmaxed)

# accuracy = 1 - np.size(extracted)/np.size(argmaxed)

# print("Accuracy --- > {}".format(accuracy))






