import math
import numpy as np



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
        
        
#%%

u0 = Unit(1,4)
u1 = Unit(2,0.3)        

gate = multiplyGate

aaa = gate.forward(u0,u0,u1)

bbb = gate.backward(u0)

#%%        


def forwardMultiplyGate(x, y):
    return x * y


def forwardAddGate(x, y):
    return x + y


def forwardCircuit(x, y, z):
    q = forwardAddGate(x, y)
    derivative_f_wrt_z = q
    derivative_f_wrt_q = z
    derivative_q_wrt_x = 1.0
    derivative_q_wrt_y = 1.0
    derivative_f_wrt_y = derivative_q_wrt_x * derivative_f_wrt_q
    derivative_f_wrt_x = derivative_q_wrt_y * derivative_f_wrt_q

    gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]
    x = x + step_size * derivative_f_wrt_x
    y = y + step_size * derivative_f_wrt_y
    z = z + step_size * derivative_f_wrt_z
    q = forwardAddGate(x, y)
    f = forwardMultiplyGate(q, z)
    return f


def analytic_gradient(x, y):
    """
    Better than numerical because it is faster
    also allows for no tweaking (direction is always right)
    """
    step_size = 0.01
    out = forwardMultiplyGate(x, y)
    x_derivative = y
    y_derivative = x
    x = x + step_size * x_derivative
    y = y + step_size * y_derivative
    out_new = forwardMultiplyGate(x, y)


def numerical_gradient(x, y):
    h = 0.0001
    xph = x + h
    out = forwardMultiplyGate(x, y)
    out2 = forwardMultiplyGate(xph, y)
    x_derivative = (out2 - out) / h

    yph = y + h
    out3 = forwardMultiplyGate(x, yph)
    y_derivative = (out3 - out) / h


def random_search(x, y):
    tweak_amount = 0.01
    best_out = -10000000
    best_x = x
    best_y = y
    for i in range(100):
        x_try = x + tweak_amount * (Math.random * 2 -1)
        y_try = y + tweak_amount * (Math.random * 2 -1)
        out = forwardMultiplyGate(x_try, y_try)
        if (out > best_out):
            best_out = out
            best_x = x_try
            best_y = y_try


def main():
    x = 2
    y = 3
    #random_search(x, y)
    """
    Back propagation is the chain rule
    """

if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-

