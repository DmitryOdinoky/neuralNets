import csv


from tools import toolset_new
from tools.toolset_new import Variable, MSE_Loss, CrossEntropy

import numpy as np
import numpy.core.defchararray as np_f
import matplotlib as plt


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

batch_size = 3

dataset = []    

for i in range(0, len(dataToTrain), batch_size):
    dataset.append(dataToTrain[i:i+batch_size])

#%%

X = dataset[0][:,0:4]
Y = dataset[0][:,4]

X_train = X.T
Y_train = Y.T

#%%

# define training constants
learning_rate = 0.0001

number_of_epochs = 500

np.random.seed(18) # set seed value so that the results are reproduceable

# Our network architecture has the shape: 
#               (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid]->[Linear->Sigmoid] -->(output)  

#------ LAYER-1 ----- define hidden layer that takes in training data 
Z1 = toolset_new.LayerLinear(in_features=2, out_features=64)
A1 = toolset_new.LayerSigmoid()

#------ LAYER-2 ----- define output layer that take is values from hidden layer
Z2 = toolset_new.LayerLinear(in_features=64, out_features=32)
A2 = toolset_new.LayerSigmoid()

#------ LAYER-3 ----- define output layer that take is values from 2nd hidden layer
Z3 = toolset_new.LayerLinear(in_features=32, out_features=3)
A3 = toolset_new.LayerSigmoid()

# see what random weights and bias were selected and their shape 
# print(Z1.params)
# print(Z2.params)
# print(Z3.params)


#%%

neural_net = [Z1,A1,Z2,A2,Z3,A3] 
    
#%%

costs = []
iterationz = []
counter = 0

loss_func = CrossEntropy()

#loss_func = MSE_Loss()

for epoch in range(number_of_epochs):
    
    counter+=1
    
    # np.random.shuffle(dataset)
      

    for batch in dataset:

        np.random.shuffle(batch)


        X = batch[:,0:2]
        Y = batch[:,2:3]

        # ------------------------- forward-prop -------------------------
        
    
        out = Variable(X)
        for layer in neural_net:
             out = layer.forward(out)

        # ---------------------- Compute Cost ----------------------------

        loss = loss_func.forward(Variable(Y), out)

        #print(f'loss: {loss.value}')

        # ------------------------- back-prop ----------------------------


        loss_func.backward()

        A3.backward()
        Z3.backward()

        A2.backward()
        Z2.backward()

        A1.backward()
        Z1.backward()

        # ----------------------- Update weights and bias ----------------
        
        stuffToUpdate = []
        
        for item in neural_net:
            if isinstance(item, (toolset_new.LayerLinear)):
                stuffToUpdate.append(item)
            elif isinstance(item, str):
                pass
        
        for linear in stuffToUpdate:
            linear.w.value += np.mean(linear.w.grad, axis=0) * learning_rate
            linear.b.value += np.mean(linear.b.grad, axis=0) * learning_rate
            
    if (epoch % 10) == 0:
        print("Cost at epoch#{}: {}".format(epoch, np.mean(loss.value)))
        costs.append(np.mean(loss.value))
        iterationz.append(counter)
                
                
        
        # See what the final weights and bias are training 
        # print(Z1.params)
        # print(Z2.params)
        # print(Z3.params)
   

#%%

plt.pyplot.scatter(iterationz, costs) # per epoch

#%%


