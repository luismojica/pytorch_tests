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
X, Y = make_classification(n_features=n_features, n_redundant=0, n_informative=5, n_classes=n_classes)
# X=torch.Tensor(X)
# Y=torch.Tensor(Y)
# exit()

# Model
class LogisticRegression(nn.Module):
	"""docstring for LogisticRegression"""
	def __init__(self, input_size, n_classes):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_size,n_classes)
		self.softmax = nn.Softmax()

	def forward(self, x):
		# out = self.softmax(self.linear(x))
		out = self.linear(x)
		return out

# Try the model out:
model = LogisticRegression(n_features,n_classes)

# The loss function and optimization method:
creteria = nn.CrossEntropyLoss()  
# optimzer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train
for epoc in range(num_empocs):
	for x,y in zip(X,Y):
		feats = Variable(torch.Tensor(x))
		label = Variable(torch.Tensor(int(y)))

		# Forward, Backward and Optimize:
		optimizer.zero_grad()
		output = model(feats)
		print (output)
		loss = creteria(output,label)
		# loss.backward()
		# optimizer.step()
		exit()

	# for idx, x in enumerate(X):
	# 	feats = Variable(torch.Tensor(x))
	# 	label = Variable(torch.Tensor([int(Y[idx])]))


