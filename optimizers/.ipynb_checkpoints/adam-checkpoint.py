import numpy as np

class AdamOptimizer():
	"""Stochastic gradient descent optimizer with Adam
	Note: All default values are from the original Adam paper
	Parameters
	----------
	params : list, length = len(coefs_) + len(intercepts_)
		The concatenated list containing coefs_ and intercepts_ in MLP model.
		Used for initializing velocities and updating params
	learning_rate_init : float, default=0.001
		The initial learning rate used. It controls the step-size in updating
		the weights
	beta_1 : float, default=0.9
		Exponential decay rate for estimates of first moment vector, should be
		in [0, 1)
	beta_2 : float, default=0.999
		Exponential decay rate for estimates of second moment vector, should be
		in [0, 1)
	epsilon : float, default=1e-8
		Value for numerical stability
	Attributes
	----------
	learning_rate : float
		The current learning rate
	t : int
		Timestep
	ms : list, length = len(params)
		First moment vectors
	vs : list, length = len(params)
		Second moment vectors
	References
	----------
	Kingma, Diederik, and Jimmy Ba.
	"Adam: A method for stochastic optimization."
	arXiv preprint arXiv:1412.6980 (2014).
	"""

	def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
				 beta_2=0.999, epsilon=1e-8, amsgrad = False):

		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		if type(learning_rate_init) == float:
			self.learning_rate_init = np.ones(len(params))*learning_rate_init
		else:
			self.learning_rate_init = np.array(learning_rate_init)
		self.t = 0
		self.ms = [np.zeros_like(param) for param in params]
		self.vs = [np.zeros_like(param) for param in params]
		self.amsgrad = amsgrad
		self.max_vs = [np.zeros_like(param) for param in params]

	def get_updates(self, grads):
		"""Get the values used to update params with given gradients
		Parameters
		----------
		grads : list, length = len(coefs_) + len(intercepts_)
			Containing gradients with respect to coefs_ and intercepts_ in MLP
			model. So length should be aligned with params
		Returns
		-------
		updates : list, length = len(grads)
			The values to add to params
		"""
		self.t += 1
		self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
					 for m, grad in zip(self.ms, grads)]
		self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
					 for v, grad in zip(self.vs, grads)]
		self.max_vs = [np.maximum(v, max_v) for v, max_v in zip(self.vs, self.max_vs)]
		self.learning_rate = (self.learning_rate_init *
								np.sqrt(1 - self.beta_2 ** self.t) /
								(1 - self.beta_1 ** self.t))
		if self.amsgrad:
			updates = [lr * m / (np.sqrt(max_v) + self.epsilon)
			           for lr, m, max_v in zip(self.learning_rate, self.ms, self.max_vs)]
		else:
			updates = [lr * m / (np.sqrt(v) + self.epsilon)
					 for lr, m, v in zip(self.learning_rate, self.ms, self.vs)]
		return updates