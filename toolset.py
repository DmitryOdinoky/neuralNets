import math
import matplotlib.pyplot as plt
import numpy as np
from util.paramInitializer import initialize_parameters  # import function to initialize weights and biases

#import numpy as np

class Unit:
    def __init__(self, value, grad=0.0):
        super().__init__()
        # Value computed in the forward pass
        self.value = value
        # The derivative of circuit output w.r.t. this unit, computed in backward pass
        self.grad = grad
        
        
class LinearLayer:
    """
        This Class implements all functions to be executed by a linear layer
        in a computational graph
        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
                      Opitons are: plain, xavier and he
        Methods:
            forward(A_prev)
            backward(upstream_grad)
            update_params(learning_rate)
    """

    def __init__(self, input_shape, n_out, ini_type="plain"):
        """
        The constructor of the LinearLayer takes the following parameters
        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
        """

        self.m = input_shape[1]  # number of examples in training data
        # `params` store weights and bias in a python dictionary
        self.params = initialize_parameters(input_shape[0], n_out, ini_type)  # initialize weights and bias
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))  # create space for resultant Z output

    def forward(self, A_prev):
        """
        This function performs the forwards propagation using activations from previous layer
        Args:
            A_prev:  Activations/Input Data coming into the layer from previous layer
        """

        self.A_prev = A_prev  # store the Activations/Training Data coming in
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']  # compute the linear function

    def backward(self, upstream_grad):
        """
        This function performs the back propagation using upstream gradients
        Args:
            upstream_grad: gradient coming in from the upper layer to couple with local gradient
        """

        # derivative of Cost w.r.t W
        self.dW = np.dot(upstream_grad, self.A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)

        # derivative of Cost w.r.t A_prev
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learning_rate=0.1):
        """
        This function performs the gradient descent update
        Args:
            learning_rate: learning rate hyper-param for gradient descent, default 0.1
        """

        self.params['W'] = self.params['W'] - learning_rate * self.dW  # update weights
        self.params['b'] = self.params['b'] - learning_rate * self.db  # update bias(es)
        

class SigmoidLayer:
    """
    This class implements the Sigmoid Layer
    Args:
        shape: shape of input to the layer
    Methods:
        forward(Z)
        backward(upstream_grad)
    """

    def __init__(self, shape):
        """
        The consturctor of the sigmoid/logistic activation layer takes in the following arguments
        Args:
            shape: shape of input to the layer
        """
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        """
        This function performs the forwards propagation step through the activation function
        Args:
            Z: input from previous (linear) layer
        """
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        """
        This function performs the  back propagation step through the activation function
        Local gradient => derivative of sigmoid => A*(1-A)
        Args:
            upstream_grad: gradient coming into this layer from the layer above
        """
        # couple upstream gradient with local gradient, the result will be sent back to the Linear layer
        self.dZ = upstream_grad * self.A*(1-self.A)        

class MultiplyGate(object):
    def __init__(self):
        pass

    def forward(self, u0, u1):
        # Store pointers to input Units u0 and u1 and output unit utop
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value * u1.value, 0.0)

        return self.utop

    def backward(self):
        # Take the gradient in output unit and chain  # it with the local gradients, which we
        # derived for multiply gates before, then
        # write those gradients to those Units.
        self.u0.grad += self.u1.value * self.utop.grad
        self.u1.grad += self.u0.value * self.utop.grad

class AddGate(object):
    def __init__(self):
        pass

    def forward(self, u0, u1):
        # Store pointers to input units
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0.0)
        return self.utop

    def backward(self):
        # Add Gate. Derivative wrt both inputs is 1
        self.u0.grad += 1 * self.utop.grad
        self.u1.grad += 1 * self.utop.grad

class SigmoidGate(object):
    def __init__(self):
        pass
    # Helper Function
    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(self.sig(self.u0.value), 0.0)
        return self.utop

    def backward(self):
        s = self.sig(self.u0.value)
        self.u0.grad += (s * (1 - s)) * self.utop.grad

# Create the Input Units
a = Unit(1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)

# Create the gates

mulG0 = MultiplyGate()
mulG1 = MultiplyGate()
addG0 = AddGate()
addG1 = AddGate()
sG0 = SigmoidGate()

# Do the Forward Pass
def forwardNeuron():
    ax = mulG0.forward(a,x) # a*x = -1
    by = mulG1.forward(b, y) # b*y = 6
    axpby = addG0.forward(ax, by) # a*x + b*y = 5
    axpbypc = addG1.forward(axpby, c) # a*x + b*y + c = 2
    s = sG0.forward(axpbypc) # sig(a*x + b*y + c) = 0.8808
    return s

s = forwardNeuron()
print("Results: %f" % s.value)

# Compute Gradient
s.grad = 1.0
sG0.backward() # Writes gradient into axpbypc
addG1.backward() # Writes gradients into axpby and c
addG0.backward() # Writes gradients into ax and by
mulG1.backward() # Writes gradients into b and y
mulG0.backward() # Writes gradients into a and x

step_size = 0.01
a.value += step_size * a.grad # a.grad is -0.105
b.value += step_size * b.grad # b.grad is 0.315
c.value += step_size * c.grad # c.grad is 0.105
x.value += step_size * x.grad # x.grad is 0.105
y.value += step_size * y.grad # y.grad is 0.210

s = forwardNeuron();
print("Circuit output after on Backprop: %f" % s.value)


# Checking the Gradient
def forwardCircuitFast(a,b,c,x,y):
    return 1/(1 + math.exp( - (a*x + b*y + c)))

a,b,c,x,y = 1,2,-3,-1,3

h = 0.0001

a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h
x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h
y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h

gradient_check = [a_grad, b_grad, c_grad, x_grad, y_grad]
print(gradient_check)


class Circuit(object):
    
    def __init__(self):
        
        self.mulG0 = MultiplyGate()
        self.mulG1 = MultiplyGate()
        self.addG0 = AddGate()
        self.addG1 = AddGate()

    def forward(x,y,a,b,c,):
        ax = mulG0.forward(a,x)
        by = mulG1.foward(b,y)
        axpby = addG0.foward(ax, by)
        axpbypc = addG1.forward(axpby, c)
        return axpbypc

    def backward(gradient_top, axpbypc):
        axpbypc.grad = gradient_top
        addG1.backward()
        addG0.backward()
        mulG1.backward()
        mulG0.backward()





#%%
        
def nonDependentDataset(low_bound,high_bound,length):
    
    predictedArray = np.random.uniform(low_bound,high_bound, size=(length, 2))
    theta = np.linspace(low_bound,high_bound,num=length)
    
    
    x1 = theta**2
    x2 = theta**3/5
    
    data = np.column_stack((x1, x2))
    error_abs = np.abs(data - predictedArray)
    fig, ax = plt.subplots(1)

    ax.plot(theta, x1)
    ax.plot(theta, x2)
    plt.stem(theta, error_abs[:,0], markerfmt=' ',linefmt='blue')
    plt.plot(theta, predictedArray[:,0],'o', color = 'b')
    plt.stem(theta, error_abs[:,1], markerfmt=' ',linefmt='red')
    plt.plot(theta, predictedArray[:,1],'o', color = 'r')
    ax.set_aspect(1/high_bound)
    
    
    plt.grid(linestyle='--')
    
    plt.title('Dataset,', fontsize=8)
    

    plt.show()
    return data, predictedArray


def dataGenByExpression(expr,low_bound,high_bound,length):
    
 
    x1_arr = np.random.uniform(low_bound,high_bound, size=(length,))
    x2_arr = np.random.uniform(low_bound,high_bound, size=(length,))

    answerz = []
    
    for i in range(0,length):
        exp_str = expr
          
        exp_str = exp_str.replace('x2', str(x2_arr[i]))
        exp_str = exp_str.replace('x1', str(x1_arr[i]))
        ans = eval(exp_str)
        answerz.append(ans)
        
    output = np.column_stack((x1_arr, x2_arr,answerz))
    
    return output

        
        

