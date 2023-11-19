import torch.nn as nn
import torch.nn.functional as F

# Define the PyTorch model
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()				# shape: (batch_size, 1 , 28, 28)
		self.conv1 = nn.Conv2d(1, 16, 3)		# shape: (batch_size, 16, 26, 26)
		self.pool = nn.MaxPool2d(2, 2)			# shape: (batch_size, 16, 13, 13)
		self.conv2 = nn.Conv2d(16, 32, 3)		# shape: (batch_size, 32, 11, 11)
		self.fc1 = nn.Linear(32 * 11 * 11, 64)	# shape: (batch_size, 64)
		self.fc2 = nn.Linear(64, 10)			# shape: (batch_size, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = F.dropout(F.relu(self.conv2(x)), 0.25)
		x = x.view(-1, 32 * 11 * 11)
		x = F.dropout(F.relu(self.fc1(x)), 0.5)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
