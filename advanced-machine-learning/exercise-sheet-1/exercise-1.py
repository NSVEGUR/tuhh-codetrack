import numpy as np


class ReLU:
	def __call__(self, x):
		return np.maximum(0, x)
	

class ANDNetwork:
	def __init__(self):
		self.relu = ReLU()
		self.weights = np.array([1.0, 1.0])
		self.bias = -1.0
	
	def __call__(self, x):
		x = np.array(x)
		z = np.dot(self.weights, x) + self.bias
		return self.relu(z)
	
and_net = ANDNetwork()
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(f"AND Network Outputs: ", and_net(inputs.T))
	
class ORNetwork:
	def __init__(self):
		self.relu = ReLU()
		self.weights = np.array([1.0, 1.0])
		self.bias = 0

	def __call__(self, x):
		x = np.array(x)
		z = np.dot(self.weights, x) + self.bias
		return self.relu(z)
	
or_net = ORNetwork()
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(f"OR Network Outputs: ", or_net(inputs.T))

class XORNetwork:
	def __init__(self):
		self.and_net = ANDNetwork()
		self.or_net = ORNetwork()
		self.relu = ReLU()
		self.weights = np.array([1.0, -2.0])
		self.bias = 0.0
	
	def __call__(self, x):
		x = np.array(x)
		and_output = self.and_net(x)
		or_output = self.or_net(x)
		combined = np.array([or_output, and_output])
		z = np.dot(self.weights, combined) + self.bias
		return self.relu(z)
	
xor_net = XORNetwork()
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(f"XOR Network Outputs: ", xor_net(inputs.T))


class AND_OR_XOR_Network:
	def __init__(self):
		self.relu = ReLU()
		self.w1 = np.array([[1,0],[0,1], [1,1]])
		self.b1 = np.array([0,0,-1])
	
	def and_net(self, x):
		z1 = x @ self.w1.T + self.b1
		w2 = np.array([0,0,1])
		b2 = -0.5
		z2 = self.relu(z1) @ w2 + b2
		return self.relu(z2)
	
	def or_net(self, x):
		z1 = x @ self.w1.T + self.b1
		w2 = np.array([1,1,0])
		b2 = -0.5
		z2 = self.relu(z1) @ w2 + b2
		return self.relu(z2)

	def xor_net(self, x):
		z1 = x @ self.w1.T + self.b1
		w2 = np.array([1,1,-2])
		b2 = -0.5
		z2 = self.relu(z1) @ w2 + b2
		return self.relu(z2)
	
and_or_xor_net = AND_OR_XOR_Network()
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(f"AND Network Outputs (shared): ", and_or_xor_net.and_net(inputs))
print(f"OR Network Outputs (shared): ", and_or_xor_net.or_net(inputs))
print(f"XOR Network Outputs (shared): ", and_or_xor_net.xor_net(inputs))