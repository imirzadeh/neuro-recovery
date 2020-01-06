from comet_ml import Experiment
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from numpy.linalg import lstsq
from torch.utils.data import TensorDataset, DataLoader
from train import MLP, train_net, eval_net
import nni
from utils import parse_arguments, inverse_lrelu, mock_nni_config
from data import create_pytorch_data_loader, get_input_data, get_acts_and_mean_acts, load_dataset


def adjust_learning_rate(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def create_similarity_score_measure(model_size, dataset_size, use_cuda=False):

	class CKA(nn.Module):
		def __init__(self, model_size, dataset_size, use_cuda=False):
			super(CKA, self).__init__()
			self.Y = nn.Linear(dataset_size, model_size)
			self.model_size = model_size
			self.dataset_size = dataset_size
			if use_cuda:
				self.eye = torch.eye(self.dataset_size).cuda()
			else:
				self.eye = torch.eye(self.dataset_size)
		
		def forward(self, x):
			return self.Y(self.eye)
	
	net = CKA(model_size, dataset_size, use_cuda)
	return net


def CKA_loss(pred, truth, config, apply_penalty=False):
	up = torch.pow(torch.norm(torch.matmul(truth, pred), p='fro', dim=(1, 2)), 2)  # ||Y^T X||_F^2
	YY = torch.norm(torch.matmul(truth, truth.transpose(1, 2)), p='fro', dim=(1, 2))
	XX = torch.norm(torch.matmul(pred.transpose(0, 1), pred), p='fro')
	down = XX * YY
	if not apply_penalty:
		penalty = 0.0
	else:
		penalty_fro = torch.mean((torch.norm(pred, 'fro') - torch.norm(truth, p='fro', dim=(1, 2))) ** 2)
		u1, s1, v1 = torch.svd(truth[0])
		u2, s2, v2 = torch.svd(truth[1])
		u3, s3, v3 = torch.svd(truth[2])
		u4, s4, v4 = torch.svd(truth[3])
		u_m, s_m, v_m = torch.svd(pred)
		penalty_svd = (torch.dist(s_m, s1[:8]) + torch.dist(s_m, s2[:8]) + torch.dist(s_m, s3[:8]) + torch.dist(s_m, s4[:8]))
		penalty_minmax = torch.dist(torch.min(truth), torch.min(pred))**2 + torch.dist(torch.max(truth), torch.max(pred))**2
		penalty = config['lambda_fro']*penalty_fro + config['lambda_svd']*penalty_svd + config['lambda_minmax']*penalty_minmax

	return (1.0 - torch.mean(torch.div(up, down))) + penalty


def optimize_pytorch(config, args, expermient):
	loader, samples = create_pytorch_data_loader(config['target_epoch'], args.dataset_size, args.teacher, args.cuda)
	net = create_similarity_score_measure(args.student, args.dataset_size, args.cuda)
	if args.cuda:
		net = net.cuda()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
	iter = 0
	EPOCHS = args.epochs
	for epoch in range(EPOCHS):
	# for epoch in range(24000):
		loss = 0
		net.train()
		apply_penalty = epoch//(EPOCHS//20) % 2
		if apply_penalty:
			adjust_learning_rate(optimizer, config['lr_penalty'])
		else:
			adjust_learning_rate(optimizer, config['lr_no_penalty'])
		for (data, target) in loader:
			iter += 1
			optimizer.zero_grad()
			pred = net(data)
			loss = CKA_loss(pred, target, config, apply_penalty)
			loss.backward()
			optimizer.step()
		if epoch % args.log_every == 0:
			print(epoch, ' => ', loss.item())
			print(np.linalg.norm(net.Y.weight.t().cpu().detach().numpy(), 'fro'), np.linalg.norm(net.Y.weight.t().cpu().detach().numpy(), 'nuc'))
			print('**'*20)
	
	res = net.Y.weight.t().cpu().detach().numpy()
	return res

def from_acts_to_weights(acts, dataset_size, inverse=False):
	input_data = get_input_data(dataset_size)
	if inverse:
		acts = np.vectorize(inverse_lrelu)(acts)
	w, _, _, _ = np.linalg.lstsq(input_data, acts, rcond=-1)
	return w



def evaluate_solution(sol, args, config):
	acts, mean_acts = get_acts_and_mean_acts(args, config)
	dataset = load_dataset(make_tensors=True, num_data_points=args.dataset_size)
	for b in range(8):
		for inv in [True, False]:
			print('===================== {} ====================='.format(b))
			w = from_acts_to_weights(sol+mean_acts[args.student][1][b], args.dataset_size, inv)#+mean_acts[8][1][b], False)
			net = MLP(args.student)
			net.W1.weight = torch.nn.parameter.Parameter(torch.from_numpy(w.T).float()) 
			acc_before = eval_net(net, dataset)
			for param in net.W1.parameters():
				param.requires_grad = False
			acc_after = train_net(net, args)
	return max(acc_before, acc_after)
		

if __name__ == "__main__":
	experiment = Experiment(api_key="1UNrcJdirU9MEY0RC3UCU7eAg",
									project_name="nni-opt-0",
										workspace="neurozip",
									auto_param_logging=False,
									auto_metric_logging=False)
	args = parse_arguments()
	config = nni.get_next_parameter()
	#config = mock_nni_config()
	print(args.teacher, args.student, args.epochs)
	print(config)
	sol = optimize_pytorch(config, args, 12)
	score = evaluate_solution(sol, args, config)
	print("***"*20)
	print(score)
	nni.report_final_result(score)
	experiment.log_parameters(config)
	experiment.log_metric(name='score', value=score)
	experiment.end()
