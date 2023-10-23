import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
from safetensors.torch import save_file
from modules.model import Net
import modules.utils as utils

# Hyperparameters
epochs = 10
batch_size = 64
learning_rate = 1e-4
shuffle = True
num_workers = 4

def main():
	# Check if GPU is available
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	pin_memory = True if device == 'cuda' else False

	# Load the MNIST dataset
	transform = transforms.Compose([transforms.ToTensor(), 
									transforms.Normalize((0.5,), (0.5,))])
	train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
	test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

	# Prepare data loaders
	# train_data, val_data = random_split(train_data, [55000, 5000])
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
	# val_loader = DataLoader(val_data, batch_size=128, shuffle=False, pin_memory=pin_memory,num_workers=num_workers)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

	# Define the model, loss function and optimizer
	model = Net().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

	# Training loop
	for epoch in range(epochs):
		print(f'Epoch {epoch+1}/{epochs}')
		start = False
		for batch_idx, (data, target) in enumerate(train_loader):
			if not start:
				start_time = time.time()
				start = True
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			
			percent_complete = (batch_idx + 1) / len(train_loader)
			bar_length = 50
			progress = '=' * int(bar_length * percent_complete)
			remaining = ' ' * (bar_length - len(progress))
			print(f'\r[{progress}{remaining}] {percent_complete * 100:.2f}%\tLoss: {loss.item():.4f}\ttime: {time.time() - start_time:.4f}s', end='', flush=True)
		print('')
		scheduler.step()
		
	# Validation loop # Only for adjusting hyperparameters
	# model.eval()
	# correct = 0
	# total = 0
	# with torch.no_grad():
	# 	print('Loading validation data...')
	# 	for data, target in val_loader:
	# 		data, target = data.to(device), target.to(device)
	# 		output = model(data)
	# 		_, predicted = torch.max(output.data, dim=1)
	# 		total += target.size(0)
	# 		correct += (predicted == target).sum().item()
	# print(f'Accuracy: {100 * correct / total:.2f} %')

	# Test loop
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		print('Loading test data...')
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			_, predicted = torch.max(output.data, dim=1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
	print(f'Accuracy: {100 * correct / total:.2f} %')

	# Save the model
	utils.ensure_dir('models')
	save_file(model.state_dict(), utils.get_save_file('models'))

if __name__ == '__main__':
    main()