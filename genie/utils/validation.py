import logging
import shutil
import numpy as np
import re
import os
import subprocess
import pandas as pd
import altair as alt
import esm
from mpnn import MPNN
import torch
import sys
import tempfile
import random
sys.path.insert(0, '../../')
from genie.utils import data_io, visualize, validation
from genie.diffusion.genie import Genie
from genie.config import Config

def tmalign(pdb1: str, pdb2: str) -> float:
    """
    Calculate the TM-score between two PDB files.

    Parameters
    ----------
    pdb1 : str
        Path to the first PDB file. The reference PDB file.
    pdb2 : str
        Path to the second PDB file. The query PDB file.

    Returns
    -------
    float
        The TM-score between the two PDB files.
    """

    exec = shutil.which("TMalign")
    if not exec:
        raise FileNotFoundError("TMalign not found in PATH")

    try:
        cmd = f"{exec} {pdb1} {pdb2}"
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        logging.warning(f"TMalign failed on {pdb1}|{pdb2}, returning NaN")
        return np.nan

    for line in output.decode().split("\n"):
        if line.startswith("TM-score"):
            if 'Chain_2' in line:
                return float(re.findall(r"=\s+([0-9.]+)", line)[0])



def batch_tmalign_scores(raw_pdb_strs: list, reference_pdb_fp: str, tmp_pdb_fp: str) -> list:
    """
    Calculate the TM-score between all pairs of PDB files in a list.

    Parameters
    ----------
    raw_pdb_strs : list
        List of PDB files as strings.
    reference_pdb_fp : str
        Path to the reference PDB file.
    tmp_pdb_fp : str
        Path to a temporary PDB file. Will be deleted after the function has completed.

    Returns
    -------
    np.ndarray
        The TM-scores between all pairs of PDB files in the list.
    """
    scores = []

    for query_pdb in raw_pdb_strs:
        with open(tmp_pdb_fp, "w") as f:
            f.write(query_pdb)

        scores.append(tmalign(reference_pdb_fp, tmp_pdb_fp))

    os.remove(tmp_pdb_fp)
    return scores

def plot_tm_score_hist(tm_scores):
    '''
    Plot a histogram of TM-scores.

    Parameters
    ----------
    tm_scores : list
        List of TM-scores.

    Returns
    -------
    altair.vegalite.v4.api.Chart
        The histogram of TM-scores.
    '''
    tm_scores = pd.DataFrame(tm_scores, columns=['tmscore'])
    tm_scores['bin'] = pd.cut(tm_scores['tmscore'], bins=np.arange(0, 1.05, 0.05)).apply(lambda x: x.left)

    tm_scores = tm_scores.groupby('bin').agg(count=('tmscore', 'count')).reset_index()

    plot_tms = alt.Chart(tm_scores).mark_bar(width=10, opacity=0.8).encode(
        x=alt.X('bin:Q', scale=alt.Scale(domain=[0, 1]), title='TM-score'),
        y=alt.Y('count', title='Number of structures')
        ).properties(width=300, height=200)
    
    plot_vertical_line = alt.Chart(pd.DataFrame([{'tmscore': 0.5}])).mark_rule(color='red', strokeDash=[10,5], strokeWidth=3).encode(x='tmscore:Q')

    return plot_tms + plot_vertical_line

class StructureValidator:
    def __init__(self, genie_config_fp, genie_ckpt_fp, working_dir='/tmp/genie', device='cuda:0'):
        self.device = device
        self.genie_config_fp = genie_config_fp
        self.genie_ckpt_fp = genie_ckpt_fp
        self.working_dir = working_dir
        self.genie_designs = []
        self.genie_output_dir = os.path.join(self.working_dir, 'genie_output')
        self.mpnn_output = None
        self.esmfold = None
        self.all_tm_scores = []
        self.max_tm_score = 0
        self.mean_tm_score = 0
        
        # Load Genie
        config = Config(self.genie_config_fp)
        self.genie = Genie.load_from_checkpoint(self.genie_ckpt_fp, config=config).to(device)

        # Load ESMFold
        # self.esmfold = esm.pretrained.esmfold_v1().eval().to(self.device)


    def genie_sample(self, aa_len, max_n_res=128, batch_size=1, final_step=1000):
        mask = torch.cat([
            torch.ones((batch_size, aa_len)),
            torch.zeros((batch_size, max_n_res - aa_len))
            ], dim=1).to(self.device)

        ts = self.genie.p_sample_loop(mask, verbose=False)
        ts = np.array([step.trans.cpu().detach().numpy().reshape(batch_size, max_n_res, 3)[:aa_len] for step in ts])
        ts = np.moveaxis(ts, 0, 1)
        self.genie_designs.extend([GenieDesign(ts[i], aa_len, max_n_res) for i in range(ts.shape[0])])

    def run_proteinmpnn_on_designs(self, nstruct=10):
        for design in self.genie_designs:
            with tempfile.TemporaryDirectory() as tmp_out_dir:
                pdb = data_io.coordinates_to_pdb(design.ca_coords)

                with open(os.path.join(tmp_out_dir, f'{np.sum(design.ca_coords)}.pdb'), 'w') as f:
                    f.write(pdb)
            
                design.protmpnn_seqs = MPNN(input_path=tmp_out_dir, nstruct=nstruct, ca_only=True).execute()
                design.protmpnn_seqs = design.protmpnn_seqs[list(design.protmpnn_seqs.keys())[0]] 

    def run_esmfold_on_designs(self):
        if self.esmfold is None:
            print('Loading ESMFold...')
            self.esmfold = esm.pretrained.esmfold_v1().eval().to(self.device)
        for design in self.genie_designs:
            print(f'Folding {len(design.protmpnn_seqs)} sequences...')
            with torch.no_grad():
                design.esm_folds = self.esmfold.infer_pdbs(design.protmpnn_seqs)

    def calc_tm_scores(self):
        self.all_tm_scores = []
        for design in self.genie_designs:
            design.calc_TM_scores()
            self.all_tm_scores.extend(design.tm_scores)
        
        self.max_tm_score = np.max(self.all_tm_scores)
        self.mean_tm_score = np.mean(self.all_tm_scores)


class GenieDesign:
    def __init__(self, timesteps, aa_len, max_n_res, final_step=1000):
        self.timesteps = timesteps
        self.aa_len = aa_len
        self.max_n_res = max_n_res
        self.final_step = final_step
        self.protmpnn_seqs = []
        self.esm_folds = []
        self.tm_scores = []
        self.ca_coords = timesteps[final_step] #.trans.cpu().detach().numpy().reshape(max_n_res, 3)[:aa_len]

    def convert_genie_ca_coords_to_pdb(self):
        return data_io.coordinates_to_pdb(self.ca_coords)
    
    def save_to_pdb(self, fp):
        pdb = data_io.coordinates_to_pdb(self.ca_coords)
        with open(fp, 'w') as f:
            f.write(pdb)

    def calc_TM_scores(self):
        self.tm_scores = []

        with tempfile.TemporaryDirectory() as tmp_out_dir:
            genie_pdb_fp = os.path.join(tmp_out_dir, 'genie_design.pdb')
            esmfold_pdb_fp = os.path.join(tmp_out_dir, 'esmfold.pdb')

            pdb = data_io.coordinates_to_pdb(self.ca_coords)
            with open(genie_pdb_fp, 'w') as f:
                f.write(pdb)

            for fold in self.esm_folds:
                with open(esmfold_pdb_fp, 'w') as f:
                    f.write(fold)
                self.tm_scores.append(tmalign(genie_pdb_fp, esmfold_pdb_fp))

    def best_tm_score(self):
        ind = np.argmax(self.tm_scores)
        return {'seq': self.protmpnn_seqs[ind], 'tm': self.tm_scores[ind], 'fold': self.esm_folds[ind]}


    def __len__(self):
        return len(self.aa_len)

    def __repr__(self) -> str:
        return f'GenieDesign( Len: {self.aa_len}\nProtMPNN Seqs: {self.protmpnn_seqs}\nTM-Scores: {self.tm_scores})'


    