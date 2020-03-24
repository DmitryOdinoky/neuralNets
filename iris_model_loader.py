import csv


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


#%%

batch_size = 5

dataset = []    

for i in range(0, len(dataToTrain), batch_size):
    dataset.append(dataToTrain[i:i+batch_size])


# X = dataset[0][:,0:4]
# Y = dataset[0][:,4]




#%%

number_of_epochs = 400

np.random.seed(16) # set seed value so that the results are reproduceable


#%%

costs = []
iterationz = []
counter = 0

loss_func = CrossEntropy()

#oss_func = MSE_Loss()

instance = MyModel()

for epoch in range(number_of_epochs):
    
    counter+=1
    
    # np.random.shuffle(dataset)
      

    for batch in dataset:

        np.random.shuffle(batch)


        X = batch[:,0:4]
        Y = batch[:,4]
        Y = np.array(toolset_new.convert_to_probdist(Y))
      


        # ------------------------- forward-prop -------------------------
        
        
        out = instance.forward(X)
        

        loss = loss_func.forward(Variable(Y), out)

        
        loss_func.backward()
        
        instance.backward()
        
            
    if (epoch % 10) == 0:
        print("Cost at epoch#{}: {}".format(epoch, loss.value))
        costs.append(loss.value)
        iterationz.append(counter)
        
#%%

plt.pyplot.scatter(iterationz, costs)





#%% Testing

X_test = dataToTest[:,0:4]

predict = instance.forward(X_test)

predicted = predict.value

#%% Evaluation


argmaxed = predicted.argmax(1)
groundTruth = dataToTest[:,4]    

extracted = np.nonzero(groundTruth - argmaxed)

errorProbability = np.shape(extracted)[1]/np.shape(argmaxed)[0]

print("Error probability --- > {}".format(errorProbability))






