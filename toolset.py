import math
import matplotlib.pyplot as plt
import numpy as np

#import numpy as np



class Unit:
    def __init__(self,value, grad):
        ##value computed in the forward pass
        self.value = value
        ##the derivative of circuit output w.r.t this unit, computed in backward pass
        self.grad  = grad

# node in the network
        
class multiplyGate:
    def forward(self, u0 ,u1):
     
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value * u1.value, 0)
        return self.utop
    
    def backward(self):
        ##take the gradient in output unit and chain it with the
        ##local gradients, which we derived for multiply gate before
        ##then write those gradients to those Units.    
        self.u0.grad += (self.u1.value * self.utop.grad)
        self.u1.grad += (self.u0.value * self.utop.grad)
        print(self.u0.grad)
        print(self.u1.grad)
        
class AddGate(object):

    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0)

    def backward(self):
        self.u0.grad += 1 * self.utop.grad
        self.u1.grad += 1 * self.utop.grad


class SigmoidGate(object):

    def sigmoidGate(self, val):
        return 1 / (1 + math.exp(-val))

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(self.sigmoidGate(self.u0.value), 0.0)

    def backward(self):
        s = self.sigmoidGate(self.u0.value)
        self.u0.grad += (s * (1 - s)) * self.utop.grad
        
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

        
        

