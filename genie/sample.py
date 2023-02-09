import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm, trange

sys.path.insert(0, '/data/psample/repos/genie')
from genie.utils.model_io import load_model
from genie.utils import data_io

def main(args):

	# device
	device = 'cuda:{}'.format(args.gpu) if args.gpu is not None else 'cpu'

	# model
	model = load_model(args.rootdir, args.model_name, args.model_version, args.model_epoch).to(device)

	# output directory
	outdir = os.path.join(model.rootdir, model.name, 'version_{}'.format(model.version), 'samples')
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	outdir = os.path.join(outdir, 'epoch_{}'.format(model.epoch))
	if os.path.exists(outdir):
		print('Samples existed!')
		sys.exit(0)
	else:
		os.mkdir(outdir)

	# sanity check
	min_length = 50
	max_length = 128
	step_size = 10
	max_n_res = model.config.io['max_n_res']
	assert max_length <= max_n_res

	# sample
	for length in trange(min_length, max_length + 1, step_size):
		for batch_idx in range(args.num_batches):
			mask = torch.cat([
				torch.ones((args.batch_size, length)),
				torch.zeros((args.batch_size, max_n_res - length))
			], dim=1).to(device)
			ts = model.p_sample_loop(mask, verbose=False)[-1]
			for batch_sample_idx in range(ts.shape[0]):
				sample_idx = batch_idx * args.batch_size + batch_sample_idx
				coords = ts[batch_sample_idx].trans.detach().cpu().numpy()
				coords = coords[:length]

				if args.save_pdb:
					pdb = data_io.coordinates_to_pdb(coords)
					with open(os.path.join(outdir, f'{length}_{sample_idx}.pdb'), 'w') as f:
						f.write(pdb)
				else:
					np.savetxt(os.path.join(outdir, f'{length}_{sample_idx}.npy'), coords, fmt='%.3f', delimiter=',')


if __name__ == '__main__':

	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', type=str, help='GPU device to use')
	parser.add_argument('-r', '--rootdir', type=str, help='Root directory (default to runs)', default='runs')
	parser.add_argument('-n', '--model_name', type=str, help='Name of Genie model', required=True)
	parser.add_argument('-v', '--model_version', type=int, help='Version of Genie model')
	parser.add_argument('-e', '--model_epoch', type=int, help='Epoch Genie model checkpointed')
	parser.add_argument('--batch_size', type=int, help='Batch size', default=5)
	parser.add_argument('--num_batches', type=int, help='Number of batches', default=1)
	parser.add_argument("--save_pdb", action="store_true", help="Save PDB file")

	# parser.add_argument('--save_pdb', type=bool, help='Save to PDB file, else .npy', default=False)
	args = parser.parse_args()

	# run
	main(args)