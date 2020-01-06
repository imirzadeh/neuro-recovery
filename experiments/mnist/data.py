import torch
import numpy as np
from train import MLP
from torch.utils.data import TensorDataset, DataLoader
from utils import inverse_lrelu

def load_dataset(make_tensors, num_data_points):
	images = np.load('../../dataset/images-5000.npy')[:num_data_points, :, :, :]
	labels = np.load('../../dataset/labels-5000.npy')[:num_data_points].reshape((num_data_points, 1))
	if make_tensors:
		images = torch.from_numpy(images).float()
		labels = torch.from_numpy(labels).long()
	return (images, labels)

def get_input_data(dataset_size):
    dataset = load_dataset(make_tensors=False, num_data_points=dataset_size)
    input_data = []
    for d in dataset[0]:
        d = d.flatten()
        input_data.append(d)
    input_data = np.array(input_data)
    return input_data

def read_model(size, trial=0, epoch=40):
	model = MLP(size)
	model_file_address = "../../models/mnist-w-{}-{}-epoch{}.pth".format(size, trial, epoch)
	model.load_state_dict(torch.load(model_file_address))
	model.eval()
	return model

def get_model_acts(model, dataset):
	l1_acts = None
	l2_acts = None
	with torch.no_grad():
		for data, target in zip(dataset[0], dataset[1]):
			l1_act = model.get_layer1_acts(data)
			l2_act = model.get_layer2_acts(data)
			if l1_acts is None:
				l1_acts = l1_act
				l2_acts = l2_act
			else:
				l1_acts = np.concatenate((l1_acts, l1_act), axis=1)
				l2_acts = np.concatenate((l2_acts, l2_act), axis=1)

	return l1_acts.T, l2_acts.T

def create_pytorch_data_loader(epoch, dataset_size, model_size, cuda=False):
	dataset = load_dataset(make_tensors=True, num_data_points=dataset_size)
	trials = list(range(0, 8))
	l1_acts = []
	l1_res = []
	samples = []
	
	# read data
	for trial in trials:
		model = read_model(size=model_size, trial=trial, epoch=epoch)
		l1, l2 = get_model_acts(model, dataset)
		shifted_l1 = l1 - np.mean(l1, 0, keepdims=True)
		sample = shifted_l1
		samples.append(sample)
		l1_acts.append(sample.T)
		l1_res.append([1.0])
	
	# make tensors from data
	X = torch.FloatTensor(l1_acts)
	# y = torch.FloatTensor(l1_res)
	y = torch.FloatTensor(l1_acts)
	svds = []
	for i in range(8):
		u1, s1, v1 = torch.svd(y[i])
		if cuda:
			s1 = s1.cuda()
		svds.append(s1)
	if cuda:
		X = X.cuda()
		y = y.cuda()
	dataset = TensorDataset(X, y)
	loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
	return loader, samples, svds

def get_acts_and_mean_acts(args, config):
	dataset_size = args.dataset_size
	dataset = load_dataset(make_tensors=True, num_data_points=dataset_size)
	trials = list(range(0, 8))
	l1_acts = []
	l2_acts = []
	r2 = np.vectorize(lambda x: round(x, 2))

	def pre_calculate_activations(model_size, shift_acts=True, pre_acts=False):
		l1_acts = []
		l2_acts = []
		mean_l1_acts = []
		mean_l2_acts = []
		
		epoch = config['target_epoch']
		for trial in trials:
			model = read_model(size=model_size, trial=trial, epoch=epoch)
			l1, l2 = get_model_acts(model, dataset)
			if pre_acts:
				l1 = np.vectorize(inverse_lrelu)(l1)
			if shift_acts:
				mean_l1 = np.mean(l1, 0, keepdims=True)
				mean_l2 = np.mean(l2, 0, keepdims=True)
				l1 = l1 - mean_l1
				l2 = l2 - mean_l2
				mean_l1_acts.append(mean_l1)
				mean_l2_acts.append(mean_l2)
			
			l1_acts.append(r2(l1))
			l2_acts.append(r2(l2))
		return l1_acts, l2_acts, mean_l1_acts, mean_l2_acts
	
	model_acts = {4: {1: [], 2: []},
				  8: {1: [], 2: []},
				  16: {1: [], 2: []},
				  32: {1: [], 2: []}}
	
	mean_acts = {4: {1: [], 2: []},
				 8: {1: [], 2: []},
				 16: {1: [], 2: []},
				 32: {1: [], 2: []}}
	for model_size in [4, 8, 16, 32]:
		l1, l2, mean_l1, mean_l2 = pre_calculate_activations(model_size)
		model_acts[model_size][1] = l1
		model_acts[model_size][2] = l2
		mean_acts[model_size][1] = mean_l1
		mean_acts[model_size][2] = mean_l2
	
	return model_acts, mean_acts
