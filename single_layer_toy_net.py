
# from sympy import symbols
# from sympy.plotting import plot3d
# from sympy.plotting import plot

from tools.utilities import compute_cost
from tools.utilities import predict

from tools.utilities import plot_learning_curve


from tools.toolset import LinearLayer
from tools.toolset import SigmoidLayer
import numpy as np
import toolset as toolset

dataExpected = toolset.dataGenByExpression('(x1**2) + (x1*x2**2)',0,1,10)

# x1, x2 = symbols(('x1 x2'))

# y = (x1**2) + (x1*x2**2)
# plot3d(y,(x1,1,10),(x2,1,10))

# X = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])


# Y = np.array([
#     [0],
#     [1],
#     [1],
#     [0]
# ])



X = dataExpected[:,0:2]

X_train = X
Y = dataExpected[:,2]

# add feature cross between 1st and 2nd feature
X_train = np.c_[X_train, X[:, 0]* X[:, 1]]  # "np.c_" concatenates data column-wise

# now we can set up data in the shape required by the neural net layers
X_train = X_train.T
Y_train = Y.T



#%%

# define training constants
learning_rate = 1
number_of_epochs = 5000

np.random.seed(48) # set seed value so that the results are reproduceable

# Our network architecture has the shape: 
#                       (input)--> [Linear->Sigmoid] -->(output)  


#------ LAYER-1 ----- define output layer that takes in training data 
Z1 = LinearLayer(input_shape=X_train.shape, n_out=1, ini_type='plain')
A1 = SigmoidLayer(Z1.Z.shape)

#%%

costs = [] # initially empty list, this will store all the costs after a certian number of epochs

# Start training
for epoch in range(number_of_epochs):
    
    # ------------------------- forward-prop -------------------------
    Z1.forward(X_train)
    A1.forward(Z1.Z)
    
    # ---------------------- Compute Cost ----------------------------
    cost, dA1 = compute_cost(Y=Y_train, Y_hat=A1.A)
    
    # print and store Costs every 100 iterations.
    if (epoch % 100) == 0:
        print("Cost at epoch#{}: {}".format(epoch, cost))
        costs.append(cost)
    
    # ------------------------- back-prop ---------------------------- 
    A1.backward(dA1)
    Z1.backward(A1.dZ)
    
    # ----------------------- Update weights and bias ----------------
    Z1.update_params(learning_rate=learning_rate)
    
    
#%%

# see the ouptput predictions
predicted_outputs, _, rms_error = predict(X=X_train, Y=Y_train, Zs=[Z1], As=[A1])

print("The predicted outputs:\n {}".format(predicted_outputs))
print("The RMS error of the model is: {}".format(rms_error))

#%%

plot_learning_curve(costs=costs, learning_rate=learning_rate, total_epochs=number_of_epochs)


    
