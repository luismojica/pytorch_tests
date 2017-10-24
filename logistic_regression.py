#!/usr/bin/python3.6 -S
# 10/22/2017
# Luis Mojica
# UTD
# Logistic Regression Using PyTorch


from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

import torch
import torch.nn as nn
from torch.autograd import Variable

# Hyperparameters:
n_features = 10
n_classes = 2
num_empocs = 10
learning_rate = 0.001

# Crate a binary classification problem with 10 featues and 100 instances
X, y = make_classification(n_features=n_features, n_redundant=0, n_informative=5, n_classes=n_classes)

# Model

class LogisticRegression(object):
	"""docstring for LogisticRegression"""
	def __init__(self, input_size, n_classes):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_size,n_classes)
		self.softmax = nn.Softmax()

	def forward(self, x):
		out = self.softmax(self.linear(x))
		return out

# Try the model out:
model = LogisticRegression(n_features,n_classes)


