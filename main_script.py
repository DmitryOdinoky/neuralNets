
from tools import toolset_new
from tools.toolset_new import Variable, MSE_Loss

import numpy as np
import matplotlib as plt


dataExpected = toolset_new.dataGenByExpression('x1**5*x2**4+x1**2+3*x1+0.36*x2',0.25,0.3,100)



#%%


batch_size = 3

dataset = []    

for i in range(0, len(dataExpected), batch_size):
    dataset.append(dataExpected[i:i+batch_size])

#%%

X = dataset[0][:,0:2]


Y = dataset[0][:,2]

X_train = X.T
Y_train = Y.T




#%%

# define training constants
learning_rate = 0.0001

number_of_epochs = 160

np.random.seed(16) # set seed value so that the results are reproduceable


# Our network architecture has the shape: 
#               (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid]->[Linear->Sigmoid] -->(output)  

#------ LAYER-1 ----- define hidden layer that takes in training data 
Z1 = toolset_new.LayerLinear(in_features=2, out_features=64)
A1 = toolset_new.LayerSigmoid()

#------ LAYER-2 ----- define output layer that take is values from hidden layer
Z2 = toolset_new.LayerLinear(in_features=64, out_features=32)
A2 = toolset_new.LayerSigmoid()


#------ LAYER-3 ----- define output layer that take is values from 2nd hidden layer
Z3 = toolset_new.LayerLinear(in_features=32, out_features=1)
A3 = toolset_new.LayerSigmoid()

# see what random weights and bias were selected and their shape 
# print(Z1.params)
# print(Z2.params)
# print(Z3.params)


    
    
#%%

costs = []
iterationz = []
counter = 0

loss_func = MSE_Loss()

for epoch in range(number_of_epochs):
    
    counter+=1
    
    # np.random.shuffle(dataset)
      

    for batch in dataset:

        np.random.shuffle(batch)


        X = batch[:,0:2]
        Y = batch[:,2:3]

        # ------------------------- forward-prop -------------------------
        out = Z1.forward(Variable(X))
        out = A1.forward(out)

        out = Z2.forward(out)
        out = A2.forward(out)

        out = Z3.forward(out)
        out = A3.forward(out)

        # ---------------------- Compute Cost ----------------------------

        loss = loss_func.forward(Variable(Y), out)

        #print(f'loss: {loss.value}')

        # ------------------------- back-prop ----------------------------


        # ------------------------- back-prop ----------------------------

        loss_func.backward()

        A3.backward()
        Z3.backward()

        A2.backward()
        Z2.backward()

        A1.backward()
        Z1.backward()

        # ----------------------- Update weights and bias ----------------
        for linear in [Z3, Z2, Z1]:
            linear.w.value += np.mean(linear.w.grad, axis=0) * learning_rate
            linear.b.value += np.mean(linear.b.grad, axis=0) * learning_rate
            
    if (epoch % 10) == 0:
        print("Cost at epoch#{}: {}".format(epoch, np.mean(loss.value)))
        costs.append(loss.value)
        iterationz.append(counter)
                
                
        
        # See what the final weights and bias are training 
        # print(Z1.params)
        # print(Z2.params)
        # print(Z3.params)
   
    
    
#%%

# see the ouptput predictions
#predicted_outputs, _, rms_error = utilities.predict_dec(X=X_train, Y=Y_train, Zs=[Z1, Z2, Z3], As=[A1, A2, A3])

# print("The expected outputs:\n {}".format(Y_train.T))
# print("The predicted outputs:\n {}".format(np.squeeze(predicted_outputs)))
# print("The RMS error of the model is: {}".format(rms_error))

#%%


plt.pyplot.scatter(iterationz, costs) # per epoch


#%%


    
