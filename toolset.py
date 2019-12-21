



class Unit:
    def __init__(this,value, grad):
        ##value computed in the forward pass
        this.value = value
        ##the derivative of circuit output w.r.t this unit, computed in backward pass
        this.grad  = grad

# node in the network
        
class multiplyGate:
    def forward(this, u0 ,u1):
     
        this.u0 = u0
        this.u1 = u1
        this.utop = Unit(u0.value * u1.value, 0)
        return this.utop
    
    def backward(this):
        ##take the gradient in output unit and chain it with the
        ##local gradients, which we derived for multiply gate before
        ##then write those gradients to those Units.    
        this.u0.grad += this.u1.value * this.utop.grad
        this.u1.grad += this.u0.value * this.utop.grad
        
#%%

u0 = Unit(1,0.1)
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

