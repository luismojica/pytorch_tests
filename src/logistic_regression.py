#!/usr/bin/python3.6 -S
# 10/22/2017
# Luis Mojica
# UTD
# Logistic Regression Using PyTorch


from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.linear_model import LogisticRegression as LR

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data


# Hyperparameters:
n_features = 100
n_classes = 2
num_epochs = 10
learning_rate = 0.001
batch_size=64
n_samples=1000
cutof=int(n_samples*0.8)


# Crate a binary classification problem with 10 featues and 100 instances
X, Y = make_classification(n_samples=n_samples,n_features=n_features, n_redundant=0, n_informative=n_features, n_classes=n_classes)
X_=torch.from_numpy(X).float()
Y_=torch.from_numpy(Y)
X_train=X_[:cutof]
Y_train=Y_[:cutof]
X_test=X_[cutof:]
Y_test=Y_[cutof:]

trainset=torch.utils.data.TensorDataset(X_train,Y_train)
testset=torch.utils.data.TensorDataset(X_test,Y_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
										  shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
										  shuffle=False, num_workers=2)

def train_test(X,Y,cutof):
	cls = LR()
	X_train=X[:cutof]
	Y_train=Y[:cutof]
	X_test=X[cutof:]
	Y_test=Y[cutof:]
	cls.fit(X_train,Y_train)
	predicted=cls.predict(X_test)
	print((predicted == Y_test).sum()/Y_test.shape[0])

# train_test(X,Y,cutof)
# exit()
# Model
class LogisticRegression(nn.Module):
	"""docstring for LogisticRegression"""
	def __init__(self, input_size, n_classes):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_size,n_classes)
		self.softmax = nn.Softmax()

	def forward(self, x):
		out = self.softmax(self.linear(x))
		# out = self.linear(x)
		return out

# Try the model out:
model = LogisticRegression(n_features,n_classes)

# The loss function and optimization method:
creteria = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train
for epoc in range(num_epochs):
	print ("epoch number %i" %epoc)

	for epoch in range(num_epochs):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data

			# wrap them in Variable
			inputs, labels = Variable(inputs), Variable(labels)
			
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = creteria(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.data[0]
			# if i % 5 == 4:    # print every 2000 mini-batches
			# 	print('[%d, %5d] loss: %.3f' %
			# 		  (epoch + 1, i + 1, running_loss / 5))
			# 	running_loss = 0.0

			if (i+1) % 10 == 0:
				pass
				print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
					% (epoch, num_epochs, i+1, len(trainset)//batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for inputs, labels in testloader:
    inputs = Variable(inputs)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print('Accuracy of the model test inputs: %d %%' % (100 * correct / total))
train_test(X,Y,cutof)

