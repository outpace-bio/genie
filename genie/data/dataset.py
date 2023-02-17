import numpy as np
from torch.utils.data import Dataset
import os
from glob import glob
import csv
import pandas as pd

from genie.utils.data_io import load_coord


class SCOPeDataset(Dataset):
	# Assumption: all domains have at least n_res residues

	def __init__(self, filepaths, max_n_res, min_n_res):
		super(SCOPeDataset, self).__init__()
		self.filepaths = filepaths
		self.max_n_res = max_n_res
		self.min_n_res = min_n_res

	def __len__(self):
		return len(self.filepaths)

	def __getitem__(self, idx):
		coords = load_coord(self.filepaths[idx])
		n_res = int(len(coords) / 3)
		if self.max_n_res is not None:
			coords = np.concatenate([coords, np.zeros(((self.max_n_res - n_res) * 3, 3))], axis=0)
			mask = np.concatenate([np.ones(n_res), np.zeros(self.max_n_res - n_res)])
		else:
			assert self.min_n_res is not None
			s_idx = np.random.randint(n_res - self.min_n_res + 1)
			start_idx = s_idx * 3
			end_idx = (s_idx + self.min_n_res) * 3
			coords = coords[start_idx:end_idx]
			mask = np.ones(self.min_n_res)
		return coords, mask


class EnzymeCommission(Dataset):
	"""
	A set of proteins with their Ca coordinates and EC numbers, which describes their
	catalysis of biochemical reactions.

	Statistics (test_cutoff=0.95):
		- #Train: 15,011
		- #Valid: 1,664
		- #Test: 1,840

	Parameters:
		meta_fp (str): the path to store the dataset
		min_n_res (int): the minimum number of residues
		max_n_res (int): the maximum number of residues
		splits (list): the splits to use
	"""

	def __init__(self, meta_fp, min_n_res, max_n_res, splits=["train", "valid", "test"]):
		super(EnzymeCommission, self).__init__()
		self.max_n_res = max_n_res
		self.min_n_res = min_n_res

		self.meta = pd.read_csv(meta_fp, sep="\t")
		self.meta = self.meta[self.meta['split'].isin(splits)]
		self.meta = self.meta[(self.meta['len'] >= min_n_res) & (self.meta['len'] <= max_n_res)].reset_index(drop=True)

	def __len__(self):
		return len(self.meta)

	def __getitem__(self, idx):
		coords = np.loadtxt(self.meta.loc[idx, 'ca_coord_file'], dtype=np.float32)
		n_res = coords.shape[0]
		coords = np.concatenate([coords, np.zeros(((self.max_n_res - n_res), 3))], axis=0)
		mask = np.concatenate([np.ones(n_res), np.zeros(self.max_n_res - n_res)])
		label = self.meta.loc[idx, 'ec_code']
		return coords, mask, label
