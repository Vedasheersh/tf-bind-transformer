from random import choice, randrange
from pathlib import Path
import functools
import polars as pl
import pandas as pd
from collections import defaultdict

import os
import json
import shutil
import numpy as np
import random
import math
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tf_bind_transformer.protein_utils import AA_STR_TO_INT_MAP
from tf_bind_transformer.smiles_utils import SMI_STR_TO_INT_MAP
from transformers import AutoTokenizer

from pyfaidx import Fasta

GLOBAL_VARIABLES = {
    'tokenizer' : AutoTokenizer.from_pretrained('Deepchem/ChemBERTa-77M-MLM')
}

def smi_to_int(smi):
    return GLOBAL_VARIABLES['tokenizer'].encode(smi)

def seq_to_int(seq):
    return [AA_STR_TO_INT_MAP[s] for s in seq]

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def find_first_index(cond, arr):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return -1

def cast_list(val = None):
    if not exists(val):
        return []
    return [val] if not isinstance(val, (tuple, list)) else val

def read_csv(path):
    return pd.read_csv(path)
    
def fetch_experiments_index(path):
    if not exists(path):
        return dict()

    exp_path = Path(path)
    assert exp_path.exists(), 'path to experiments json must exist'

    root_json = json.loads(exp_path.read_text())
    experiments = root_json['experiments']

    index = {}
    for experiment in experiments:
        exp_id = experiment['accession']

        if 'details' not in experiment:
            continue

        details = experiment['details']

        if 'datasets' not in details:
            continue

        datasets = details['datasets']

        for dataset in datasets:
            dataset_name = dataset['dataset_name']
            index[dataset_name] = dataset['peaks_NR']

    return index

# remap dataframe functions

def pl_isin(col, arr):
    equalities = list(map(lambda t: pl.col(col) == t, arr))
    return functools.reduce(lambda a, b: a | b, equalities)

def pl_notin(col, arr):
    equalities = list(map(lambda t: pl.col(col) != t, arr))
    return functools.reduce(lambda a, b: a & b, equalities)

def filter_by_col_isin(df, col, arr, chunk_size = 25):
    """
    polars seem to have a bug
    where OR more than 25 conditions freezes (for pl_isin)
    do in chunks of 25 and then concat instead
    """
    dataframes = []
    for i in range(0, len(arr), chunk_size):
        sub_arr = arr[i:(i + chunk_size)]
        filtered_df = df.filter(pl_isin(col, sub_arr))
        dataframes.append(filtered_df)
    return pl.concat(dataframes)

def mutate(seq, position):
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def augment_data(df, labelcol, N=10, mu=0.01, sigma=0.001):
    allrows = []
    for ind, row in df.iterrows():
        label = row[labelcol]
        label_noisy = np.array(label)*N + np.random.normal(0.1, 0.01, N)
        for j in range(N):
            seq = row.sequence
            s = np.random.normal(mu, sigma, 1)
            mut_rate = s[0]
            times = math.ceil(len(seq) * mut_rate)
            for k in range(times):
                position = random.randint(1 , len(seq) - 1)
                seq = mutate(seq, position)
            seq = seq.replace('*', '<mask>')
            row_now = row.copy()
            row_now.sequence = seq
            row_now.label = label_noisy[j]
            allrows.append(row_now)
    return pd.concat(allrows, axis=1).T

# context string creator class

class ProteinSmilesDataset(Dataset):
    def __init__(
        self,
        csv_file = None,
        df = None,
        smiles_col = 'reaction_smiles',
        seq_col = 'sequence',
        target_col = 'log10_value',
        augment = False,
        df_frac = 1.,
        experiments_json_path = None,
        # balance_sampling_by_target_bin = False,
        **kwargs
    ):
        super().__init__()
        assert exists(csv_file) ^ exists(df), 'either data csv file or dataframe must be passed in'

        if not exists(df):
            df = read_csv(csv_file)

        if df_frac < 1:
            df_frac = df.sample(frac = df_frac)

        # augment dataset by randomly mutating positions in amino acid sequence to <mask>
        if augment: df = augment_data(df, labelcol=target_col)
        
        self.df = df
        self.smiles_col = smiles_col
        self.seq_col = seq_col
        self.target_col = target_col
        
        self.experiments_index = fetch_experiments_index(experiments_json_path)

        # balanced target sampling logic

        # self.balance_sampling_by_target_bin = balance_sampling_by_target_bin

        # if self.balance_sampling_by_target_bin:
        #     self.df_indexed_by_target_bin = []

#             for target in self.df.get_column('target_bin').unique().to_list():
#                 df_by_target_bin = self.df.filter(pl.col('target') == target)
#                 self.df_indexed_by_target_bin.append(df_by_target)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        # if balancing by target bin, randomly draw sample from indexed dataframe

        # if self.balance_sampling_by_target_bin:
        #     filtered_df = self.df_indexed_by_target_bin[ind]
        #     rand_ind = randrange(0, len(filtered_df))
        #     sample = filtered_df.iloc[rand_ind]
        # else:
        sample = self.df.iloc[ind]

        aa_seq, smiles, target = sample[self.seq_col], sample[self.smiles_col], sample[self.target_col]

        # aa_seq = seq_to_int(aa_seq)
        # smiles = smi_to_int(smiles)
        
        # now aggregate all the data

        label = torch.Tensor([target])

        return smiles, aa_seq, label

# dataloader related functions

def collate_fn(data):
    smiles, aa_seq, labels = list(zip(*data))
    return tuple(smiles), tuple(aa_seq), torch.cat(labels, dim = 0)

import ipdb

def collate_dl_outputs(dl_outputs):
    # ipdb.set_trace()
    outputs = list(dl_outputs)
    ret = []
    for entry in outputs:
        if isinstance(entry[0], torch.Tensor):
            # entry = torch.cat(entry, dim = 0)
            pass
        else:
            entry = (el for el in entry)
        ret.append(entry)
    return tuple(ret)

def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_dataloader(ds, cycle_iter = False, **kwargs):
    dataset_len = len(ds)
    batch_size = kwargs.get('batch_size')
    drop_last = dataset_len > batch_size

    dl = DataLoader(ds, collate_fn = collate_fn, drop_last = drop_last, **kwargs)
    wrapper = cycle if cycle_iter else iter
    return wrapper(dl)
