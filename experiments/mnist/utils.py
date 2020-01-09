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
	# parser.add_argument('--epochs', default=21000, type=int,  help='number of total epochs to run')
	parser.add_argument('--log_every', default=1000, type=int,  help='log every?')
	parser.add_argument('--dataset', default='mnist', type=str, help='dataset.')
	parser.add_argument('--teacher', default=16, type=int, help='teacher size')
	parser.add_argument('--student', '--model', default=8, type=int, help='student size')
	parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
	parser.add_argument('--centered', default=False, type=str2bool, help='zero-mean by shift?')
	parser.add_argument('--normalize_penalty', default=True, type=str2bool, help='[0, 1] penalty?')
	parser.add_argument('--dataset_size', default=500, type=int, help='dataset size')
	parser.add_argument('--resume', default='', type=str, help='resume')
	args = parser.parse_args()
	return args


def mock_nni_config():
	return  { "lr_no_penalty": 0.3,
			 "lr_penalty": 0.05,
			 "lambda_fro": 15,
			 "lambda_svd": 5,
			 "lambda_minmax": 2,
  			 "target_epoch": 15,
  			 "centered": False,
  			 "num_epochs": 150000,
  			  }