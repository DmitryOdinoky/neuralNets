import tools.toolset as tools




dataExpected = tools.dataGenByExpression('(x1**2) + (x1*x2**2)',0,1,10)


#%%

a = tools.Unit(1.0, 0.0)
b = tools.Unit(2.0, 0.0)
c = tools.Unit(-3.0, 0.0)
x = tools.Unit(-1.0, 0.0)
y = tools.Unit(3.0, 0.0)

# Create the gates

mulG0 = tools.MultiplyGate()
mulG1 = tools.MultiplyGate()
addG0 = tools.AddGate()
addG1 = tools.AddGate()
sG0 = tools.SigmoidGate()

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

### adasdas



