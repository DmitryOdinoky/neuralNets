import math

import numpy as np


class Variable(object):
    def __init__(self, value: np.ndarray, grad: np.ndarray = None):
        
        self.value = value
        self.grad = grad
        
        if self.grad is None:
            self.grad = np.zeros_like(self.value)
        assert self.value.shape == self.grad.shape

class LayerLinear:
    def __init__(self, in_features, out_features):
        
        self.w: Variable = Variable(np.random.uniform(low=-1,size=(in_features, out_features)))
        self.b: Variable = Variable(np.zeros((out_features,)))
        
        self.x: Variable = None
        self.output: Variable = None
        
    def forward(self, x: Variable):
        
        self.x = x
        
        self.output = Variable(
            np.matmul(x.value, self.w.value) + self.b.value)
        return self.output
    
    def backward(self):
        self.x.grad = np.matmul(self.output.grad, np.transpose(self.w.value))
        
        self.w.grad = np.matmul(np.expand_dims(self.x.value, axis=2), np.expand_dims(self.output.grad, axis=1))
        self.b.grad = 1*self.output.grad

        
class LayerSigmoid(object):
    
    def __init__self(self):
        super().__init__()
        self.x = None
        self.output = None
        
    def func(self, x):
        return 1.0 / (1.0 + math.e ** (-x))
    
    def forward(self, x):
        self.x = x
        self.output = Variable(
            self.func(x.value))
        return self.output
    
    def backward(self):
        y = self.func(self.x.value)
        self.x.grad = (y *(1.0 - y)) * self.output.grad
        
class MSE_Loss:
    
    def __init__(self):
        self.x: Variable = None
        self.gradTop: Variable = None
        self.y_prim: Variable = None
        self.y: Variable = None
        
    def forward(self, y: Variable, y_prim: Variable):
        self.y_prim = y_prim
        self.y = y
        self.gradTop = Variable((y.value-y_prim.value)**2)
        return self.gradTop
    
    def backward(self):
        self.y_prim.grad = 2*(self.y.value - self.y_prim.value)
        


class CrossEntropy:
    
    def __init__(self):
        self.y: Variable = None
        self.p_hat: Variable = None
        self.gradTop: Variable = None
        
    def forward(self, y: Variable, p_hat: Variable):
        
        m = np.shape(y.value)[1]
        
        self.y = y
        self.p_hat = p_hat
        
        self.gradTop = Variable((-1/m) * np.sum(np.maximum(self.p_hat.value, 0) - self.p_hat.value * self.y.value + np.log(1+ np.exp(- np.abs(self.p_hat.value)))))
        
        

        return self.gradTop
    
    def backward(self):
        
        m = np.shape(self.y.value)[1]
        
        self.p_hat.grad = (1/m) * ((1/(1+np.exp(- self.p_hat.value))) - self.y.value)

        
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


    
    

        
    
        
        
    
            
    