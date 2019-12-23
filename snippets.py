#%% Sollve non-linear equasions
import matplotlib.pyplot as plt

import numpy as np
from gekko import GEKKO

module = GEKKO()
x1,x2 = [module.Var(1) for i in range(2)]
module.Equations([x1**2+x2**2==20,\
             x2-x1**3==0])
module.solve(disp=False)

print(x1.value,x2.value)



    


    

    

#%%
import numpy as np

def generateDataset():
    
    a = np.linspace(0, 1,num=10)
    b = np.linspace(0.5, 1,num=10)
    
    
    y_values = []

    
    for a_point, b_point in a, b:
        y = a_point**2+2*b_point+0.1
        y_values.append(y)
        
    return y_values

set = generateDataset()

#%% generate y values based on whatever x1, x2
import numpy as np

length = 10
x1_arr = np.linspace(0, 1,num=length)
x2_arr = np.linspace(0.5, 1,num=length)



# x1 = 2.5
# x2 = 3

answerz = []

for i in range(0,length):
    expr = '(x1**2) + (x1*x2**2)'
  
    expr = expr.replace('x2', str(x2_arr[i]))
    expr = expr.replace('x1', str(x1_arr[i]))
    ans = eval(expr)
    answerz.append(ans)

#%%
    
import matplotlib.pyplot as plt

import numpy as np

newArray = np.random.uniform(0, 1, size=(10, 2))
#theta = np.linspace(0, 2*np.pi, 10)

theta = np.linspace(0, 1,num=10)

# r = 1

# x1 = r*np.cos(theta)
# x2 = r*np.sin(theta)

x1 = theta**2
x2 = theta**3/5

fig, ax = plt.subplots(1)

ax.plot(theta, x1)
ax.set_aspect(1)
ax.plot(theta, x2)

data = np.column_stack((x1, x2))
error_abs = np.abs(data - newArray)

#plt.stem(theta, error_abs[:,0],markerfmt=' ')

plt.stem(theta, error_abs[:,0], markerfmt=' ',linefmt='blue')
plt.plot(theta, newArray[:,0],'o', color = 'b')
ax.set_aspect(1)
plt.stem(theta, error_abs[:,1], markerfmt=' ',linefmt='red')
plt.plot(theta, newArray[:,1],'o', color = 'r')
ax.set_aspect(1)

# plt.xlim(0,1.25)
# plt.ylim(0,1.25)

plt.grid(linestyle='--')

plt.title('Dataset,', fontsize=8)


plt.show()
