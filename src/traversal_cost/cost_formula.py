import numpy as np
import matplotlib.pyplot as plt

x = np.random.random(200)*10 - 2
y = np.random.random(200)*8 - 4

cost = x + y

r = np.sqrt(x**2 + y**2)
theta = np.arctan(y/x)

cost_polar = r + theta
cost_polar2 = r*np.sin((theta + np.pi/2)/2)


plt.subplot(121)
plt.plot(0, 0, "ro", markersize=10)
plt.scatter(x, y, c=cost_polar, cmap="jet")

plt.subplot(122)
plt.plot(0, 0, "ro", markersize=10)
plt.scatter(x, y, c=cost_polar2, cmap="jet")

plt.colorbar()

plt.show()
