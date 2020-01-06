import argparse

def inverse_lrelu(x):
	return x if x >=0 else 100*x


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False
	
	
def parse_arguments():
	parser = argparse.ArgumentParser(description='Neuro Zip')
	parser.add_argument('--epochs', default=401, type=int,  help='number of total epochs to run')
	parser.add_argument('--log_every', default=1000, type=int,  help='number of total epochs to run')
	parser.add_argument('--dataset', default='mnist', type=str, help='dataset. can be either cifar10 or cifar100')
	parser.add_argument('--teacher', default=16, type=int, help='teacher size')
	parser.add_argument('--student', '--model', default=8, type=int, help='student size')
	parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--dataset_size', default=500, type=int, help='dataset size')

	args = parser.parse_args()
	return args


def mock_nni_config():
	return  { "lr_no_penalty": 0.01,
			 "lr_penalty": 0.001,
			 "lambda_fro": 0.1,
			 "lambda_svd":0.1,
			 "lambda_minmax":10,
  			 "target_epoch":  40
  			  }