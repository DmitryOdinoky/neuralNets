import matplotlib.pyplot as plt
import numpy as np
import math
import random

from sklearn.metrics import mean_absolute_error


# random weights, x1 and x2

newArray = np.random.uniform(0, 1, size=(10, 2))



#theta = np.linspace(0, 2*np.pi, 10)

theta = np.linspace(0, 1,num=10)

# r = 1

# x1 = r*np.cos(theta)
# x2 = r*np.sin(theta)

x1 = np.power(theta,2)
x2 = np.power(theta,3)/5



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

#plt.savefig("plot_circle_matplotlib_01.png", bbox_inches='tight')

plt.show()


