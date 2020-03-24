import math



import numpy as np


class Variable(object):
    def __init__(self, value: np.ndarray, grad: np.ndarray = None):
        
        self.value = value
        self.grad = grad
        
        if self.grad is None:
            self.grad = np.zeros_like(self.value)
        #assert self.value.shape == self.grad.shape
            
            
class MyModel:
    
    def __init__(self):
        
        self.learning_rate = 0.0001
      
        self.out = None
        
        self.Z1 = LayerLinear(in_features=4, out_features=64)
        self.A1 = LayerSigmoid()

        #------ LAYER-2 ----- 
        self.Z2 = LayerLinear(in_features=64, out_features=32)
        self.A2 = LayerSigmoid()

        #------ LAYER-3 ----- 
        self.Z3 = LayerLinear(in_features=32, out_features=3)

        self.SM = LayerSoftmaxV2()  
        
        #------- GRAPH -------        
        self.graph = [self.Z1, self.A1, self.Z2, self.A2, self.Z3, self.SM]
        
    def forward(self, dataset):
        
        self.out = Variable(dataset)
        
      

        for layer in self.graph:
          
            self.out = layer.forward(self.out)


        
        return self.out
    

    def backward(self):

        
        rev_layers = self.graph[::-1]
        
        for layer in rev_layers:
          
            out = layer.backward()
        
        
        
        stuffToUpdate = []
        
        for item in self.graph:
            if isinstance(item, (LayerLinear)):
                stuffToUpdate.append(item)
            elif isinstance(item, str):
                pass
        
        for linear in stuffToUpdate:
            linear.w.value += np.mean(linear.w.grad, axis=0) * self.learning_rate
            linear.b.value += np.mean(linear.b.grad, axis=0) * self.learning_rate
            
    
        
        









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
        y = self.func(self.output.value)
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
    



        

class LayerSoftmaxV2(object):
    
    def _init_(self):
        
        super()._init_()
        self.x: Variable = None
        self.output: Variable = None

    
    def forward(self, x):
        self.x = x
        exps = np.exp(self.x.value - np.expand_dims(np.max(self.x.value, axis=1), axis=1))
        self.output = Variable(exps / np.expand_dims(np.sum(exps, axis=1), axis=1))

      
        return self.output
        
    def backward(self):
            
            #m = np.shape(self.y.value)[0]
            
        for idx in range(self.output.value.shape[0]):
            
            J = np.zeros((self.output.value.shape[1], self.output.value.shape[1]))
            
            for i in range(self.output.value.shape[1]):
                for j in range(self.output.value.shape[1]):
                    if i == j:
                        J[i,j] = self.output.value[idx][i] * (1-self.output.value[idx][j])
                    else:
                        J[i,j] = -self.output.value[idx][i] * self.output.value[idx][j]
                    
        self.x.grad[idx] = np.matmul(J, self.output.grad[idx])



class CrossEntropy:
    
    #compute stable cross-entropy
    
    def __init__(self):
        self.y = None
        self.y_hat = None
        
        
    def forward(self, y, y_hat):
        
        #m = np.shape(y.value)[0]
        
        self.y = y
        self.y_hat = y_hat
        
        
        self.output = Variable(-np.sum(self.y.value*np.log(self.y_hat.value)))
        

        return self.output
    
    def backward(self):
        
        #m = np.shape(self.y.value)[0]
        

        self.y_hat.grad = self.y.value/self.y_hat.value
        
        
        
        
        
        
        
        
        
        
        
        
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

def convert_to_probdist(vector):
    
    probdists = []

    for row in vector:
        
        blank_probdist = np.zeros(3)
        
        if row == 0:
            blank_probdist[0] = 1
            probdists.append(blank_probdist)
        elif row == 1:
            blank_probdist[1] = 1
            probdists.append(blank_probdist)
        elif row == 2:
            blank_probdist[2] = 1
            probdists.append(blank_probdist)
    
    return probdists


    
    

        
    
        
        
    
            
    