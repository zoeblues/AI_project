import math
import numpy as np
import torch
import torch.nn as nn

#todo: sampling process
#todo: model hyperparameters
#todo: config !!!

'''
Sampling process:
1. INITISLIZATION: sample random noise from Gaussian
2. ITERATIVE REFINEMENT: refine through the learned reverse process
3. COMPLETION: gradually remove noise and add structutre to sample

'''

class GaussianNoise():
	
	def __init__(self, x):
		self.mean = 0
		self.sigma = 1
		self.prob = 1/(self.sigma*np.sqrt(2*math.pi))*math.e**(-((x-self.mean)**2)/(2*self.sigma**2))
	