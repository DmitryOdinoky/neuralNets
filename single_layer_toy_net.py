
# from sympy import symbols
# from sympy.plotting import plot3d
# from sympy.plotting import plot


from tools import utilities
from tools.utilities import plot_learning_curve
from tools import toolset

import numpy as np


dataExpected = toolset.dataGenByExpression('(x1**2) + (x1*x2**6)+3*x1**5*x2**4',0,0.65,60)

# x1, x2 = symbols(('x1 x2'))

# y = (x1**2) + (x1*x2**2)
# plot3d(y,(x1,1,10),(x2,1,10))


#%%


batch_size = 7

slicez = []    

for i in range(0, len(dataExpected), batch_size):
    slicez.append(dataExpected[i:i+batch_size])

#%%

X = slicez[0].T[:,0:2]

X_train = X

Y = slicez[0].T[:,2]

# add feature cross between 1st and 2nd feature
## X_train = np.c_[X_train, X[:, 0]* X[:, 1]]  # "np.c_" concatenates data column-wise

# now we can set up data in the shape required by the neural net layers
X_train = X_train.T
Y_train = Y.T




#%%

# define training constants
learning_rate = 0.001
number_of_epochs = 2001

np.random.seed(36) # set seed value so that the results are reproduceable


# Our network architecture has the shape: 
#               (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid]->[Linear->Sigmoid] -->(output)  

#------ LAYER-1 ----- define hidden layer that takes in training data 
Z1 = toolset.LinearLayer(input_shape=X_train.shape, n_out=60, ini_type='xavier')
A1 = toolset.SigmoidLayer(Z1.Z.shape)

#------ LAYER-2 ----- define output layer that take is values from hidden layer
Z2= toolset.LinearLayer(input_shape=A1.A.shape, n_out=60, ini_type='xavier')
A2= toolset.SigmoidLayer(Z2.Z.shape)


#------ LAYER-3 ----- define output layer that take is values from 2nd hidden layer
Z3= toolset.LinearLayer(input_shape=A2.A.shape, n_out=60, ini_type='xavier')
A3= toolset.SigmoidLayer(Z3.Z.shape)

# see what random weights and bias were selected and their shape 
# print(Z1.params)
# print(Z2.params)
# print(Z3.params)


    
    
#%%

counter = 1    

for batch in slicez:
    
    print('BATCH ##### -- ' + str(counter))
    counter +=1
    
    X = batch[:,0:2]

    X_train = X
    Y = batch[:,2]

    # add feature cross between 1st and 2nd feature
    ## X_train = np.c_[X_train, X[:, 0]* X[:, 1]]  # "np.c_" concatenates data column-wise

    # now we can set up data in the shape required by the neural net layers
    X_train = X_train.T
    Y_train = Y.T
    

    costs = [] # initially empty list, this will store all the costs after a certian number of epochs
    
    
    # Start training
    for epoch in range(number_of_epochs):
        
        # ------------------------- forward-prop -------------------------
        Z1.forward(X_train)
        A1.forward(Z1.Z)
        
        Z2.forward(A1.A)
        A2.forward(Z2.Z)
        
        Z3.forward(A2.A)
        A3.forward(Z3.Z)
        
        # ---------------------- Compute Cost ----------------------------
        cost, dA3 = utilities.compute_cost(Y=Y_train, Y_hat=A3.A)
        
        # print and store Costs every 100 iterations.
        if (epoch % 100) == 0:
            print("Cost at epoch#{}: {}".format(epoch, cost))
            costs.append(cost)
        
        # ------------------------- back-prop ----------------------------
        A3.backward(dA3)
        Z3.backward(A3.dZ)
        
        A2.backward(Z3.dA_prev)
        Z2.backward(A2.dZ)
        
        A1.backward(Z2.dA_prev)
        Z1.backward(A1.dZ)
        
        # ----------------------- Update weights and bias ----------------
        Z3.update_params(learning_rate=learning_rate)
        Z2.update_params(learning_rate=learning_rate)
        Z1.update_params(learning_rate=learning_rate)
    
    # See what the final weights and bias are training 
    # print(Z1.params)
    # print(Z2.params)
    # print(Z3.params)
   
    
    
#%%

# see the ouptput predictions
predicted_outputs, _, rms_error = utilities.predict(X=X_train, Y=Y_train, Zs=[Z1, Z2, Z3], As=[A1, A2, A3])

print("The expected outputs:\n {}".format(Y_train.T))
print("The predicted outputs:\n {}".format(predicted_outputs))
print("The RMS error of the model is: {}".format(rms_error))

#%%

plot_learning_curve(costs=costs, learning_rate=learning_rate, total_epochs=number_of_epochs)

#%%


    
