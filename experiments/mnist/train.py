from comet_ml import Experiment
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


DEVICE = 'cuda'

class MLP(nn.Module):
	def __init__(self, num_hidden):
		super(MLP, self).__init__()
		self.W1 = nn.Linear(784, num_hidden)
		self.relu = nn.LeakyReLU(0.01)
		self.W2 = nn.Linear(num_hidden, 10)
	
	def forward(self, x):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		out = self.W2(out)
		# out = self.relu(out)
		return out
	
	def get_layer1_acts(self, x, pre_act=False):
		x = x.view(-1, 784)
		out = self.W1(x)
		if not pre_act:
			out = self.relu(out)
		out = out.detach().numpy().T
		return out
	
	def get_layer2_acts(self, x):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		out = self.W2(out)
		# out = self.relu(out)
		out = out.detach().numpy().T
		return out


def train(num_hidden, iter, experiment):
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('./stash/', train=True, download=True,
								   transform=torchvision.transforms.Compose([
									   torchvision.transforms.ToTensor(),
									   torchvision.transforms.Normalize(
										   (0.1307,), (0.3081,))
								   ])),
		batch_size=128, shuffle=False, num_workers=6)
	
	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('./stash/', train=False, download=True,
								   transform=torchvision.transforms.Compose([
									   torchvision.transforms.ToTensor(),
									   torchvision.transforms.Normalize(
										   (0.1307,), (0.3081,))
								   ])),
		batch_size=128, shuffle=False, num_workers=6)
	
	net = MLP(num_hidden=num_hidden).to(DEVICE)
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
	criterion = nn.CrossEntropyLoss()
	for epoch in range(1, 51):
		net = net.to(DEVICE)
		print("epoch {}".format(epoch))
		net.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			optimizer.zero_grad()
			pred = net(data)
			loss = criterion(pred, target)
			loss.backward()
			optimizer.step()
		
		# eval
		net.eval()
		test_loss = 0
		correct = 0
		crit = nn.CrossEntropyLoss()
		with torch.no_grad():
			for data, target in test_loader:
				data = data.to(DEVICE)
				target = target.to(DEVICE)
				output = net(data)
				test_loss += crit(output, target).item()
				pred = output.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		correct = correct.to('cpu')
		experiment.log_metric(name='val-acc', step=epoch, value=(float(correct.numpy())*100.0)/10000.0)
		experiment.log_metric(name='val-loss', step=epoch, value=test_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))
		if epoch % 5 == 0:
			net = net.cpu()
			torch.save(net.state_dict(), './output/mnist-w-{}-{}-epoch{}.pth'.format(num_hidden, iter, epoch))
			experiment.log_asset('./output/mnist-w-{}-{}-epoch{}.pth'.format(num_hidden, iter, epoch))

	# save net weight
	net = net.cpu()
	torch.save(net.state_dict(), './output/mnist-w-{}-{}.pth'.format(num_hidden, iter))
	experiment.log_asset('./output/mnist-w-{}-{}.pth'.format(num_hidden, iter))


def eval_net(model, dataset):
		model.eval()
		test_loss = 0
		correct = 0
		crit = nn.CrossEntropyLoss()
		with torch.no_grad():
			for data, target in zip(dataset[0], dataset[1]):
				output = model(data)
				test_loss += crit(output, target).item()
				pred = output.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= dataset[0].shape[0]
		correct = correct.to('cpu')
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, dataset[0].shape[0],
			100. * correct / dataset[0].shape[0]))
		return correct / dataset[0].shape[0]

def get_mnist_loaders(args):
	DEVICE = 'cuda' if args.cuda else 'cpu'
	mnist_train = torchvision.datasets.MNIST('./stash/', train=True, download=True,
								   transform=torchvision.transforms.Compose([
									   torchvision.transforms.ToTensor(),
									   torchvision.transforms.Normalize(
										   (0.1307,), (0.3081,))
								   ]))
	mnist_test = torchvision.datasets.MNIST('./stash/', train=False, download=True,
								   transform=torchvision.transforms.Compose([
									   torchvision.transforms.ToTensor(),
									   torchvision.transforms.Normalize(
										   (0.1307,), (0.3081,))
								   ]))

	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=False, num_workers=4)
	
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=4)
	
	mnist_train = []
	mnist_test = []

	for batch_idx, (data, target) in enumerate(train_loader):
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			mnist_train.append((data, target))

	for batch_idx, (data, target) in enumerate(test_loader):
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			mnist_test.append((data, target))

	return mnist_train, mnist_test

def train_net(net, train_loader, test_loader, args):
	DEVICE = 'cuda' if args.cuda else 'cpu'
	
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
	criterion = nn.CrossEntropyLoss()
	best_acc = 0

	net = net.to(DEVICE)
	for epoch in range(1, 4):
		print("epoch {}".format(epoch))
		net.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			optimizer.zero_grad()
			pred = net(data)
			loss = criterion(pred, target)
			loss.backward()
			optimizer.step()
		
		net.eval()
		test_loss = 0
		correct = 0
		crit = torch.nn.CrossEntropyLoss()
		with torch.no_grad():
			for data, target in test_loader:
				output = net(data)
				test_loss += crit(output, target).item()
				pred = output.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		correct = correct.to('cpu')
		# print('float {}'.format(float(correct)))
		best_acc = max(best_acc, float(correct)/len(test_loader.dataset))
		#experiment.log_metric(name='val-acc', step=epoch, value=correct*100.0/len(test_loader.dataset))
		#experiment.log_metric(name='val-loss', step=epoch, value=test_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))
	return best_acc

			
# if __name__ == "__main__":
# 	for n in [4, 8, 16, 32, 64]:
# 		for iter in [0, 1]:
# 			experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg",
# 									project_name="new-opt-1",
# 									workspace="neurozip",
# 									auto_param_logging=False,
# 									auto_metric_logging=False)
# 			#experiment.add_tag('mnist')
# 			experiment.add_tag(n)
# 			experiment.add_tag('lrelu')
# 			experiment.log_parameter('hidden_size', n)
# 			experiment.log_parameter('trial', iter)
# 			train(num_hidden=n, iter=iter, experiment=experiment)
# 			experiment.end()
