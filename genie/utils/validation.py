import logging
import shutil
import numpy as np
import re
import os
import subprocess
import pandas as pd
import altair as alt

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