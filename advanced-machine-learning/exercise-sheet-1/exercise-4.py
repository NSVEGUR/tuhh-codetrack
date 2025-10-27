import numpy as np
import matplotlib.pyplot as plt

# Heaviside function

def heaviside(x):
	return (x >= 0).astype(float)

W1 = np.array([[0,1], [1,-1], [-1, -1]])
b1 = np.array([0,1,1])
W2 = np.array([[1, 1, 1]])
b2 = np.array([-3])

def function(x1, x2):
		x = np.stack([x1, x2], axis=-1)
		a1 = x
		z1 = a1 @ W1.T + b1
		a2 = heaviside(z1)
		z2 = a2 @ W2.T + b2
		a3 = heaviside(z2)
		return a3

# Create grid
xx, yy = np.meshgrid(np.linspace(-3, 3, 400),
                     np.linspace(-3, 3, 400))

zz = function(xx, yy).reshape(xx.shape)

# Plot decision region
plt.contourf(xx, yy, zz, levels=[-0.1, 0.5, 1.1], colors=['white', 'lightgreen'])
plt.axhline(0, color='gray', linestyle='--', label='$z_{1,1}=0$')
plt.plot(xx[0,:], xx[0,:] + 1, 'r--', label='$z_{1,2}=0$')
plt.plot(xx[0,:], -xx[0,:] + 1, 'b--', label='$z_{1,3}=0$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Decision region where f(x)=1')
plt.show()