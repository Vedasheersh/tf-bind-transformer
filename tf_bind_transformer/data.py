from Bio import SeqIO
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
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tf_bind_transformer.gene_utils import parse_gene_name
from enformer_pytorch import FastaInterval

from pyfaidx import Fasta

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

def split_df(df, splitby = 'random', test_size = 0.1, seed = 0):
    assert splitby=='random', 'other splitting methods not implemented yet'
    
    df.reset_index(inplace=True)
    train_df, valid_df = train_test_split(df, test_size, random_state = seed)
    
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
    
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

def get_chr_names(ids):
    return set(map(lambda t: f'chr{t}', ids))

CHR_IDS = set([*range(1, 23), 'X'])
CHR_NAMES = get_chr_names(CHR_IDS)

def remap_df_add_experiment_target_cell(df, col = 'column_4'):
    df = df.clone()

    exp_id = df.select([pl.col(col).str.extract(r"^([\w\-]+)\.*")])
    exp_id = exp_id.rename({col: 'experiment'}).to_series(0)
    df.insert_at_idx(3, exp_id)

    targets = df.select([pl.col(col).str.extract(r"[\w\-]+\.([\w\-]+)\.[\w\-]+")])
    targets = targets.rename({col: 'target'}).to_series(0)
    df.insert_at_idx(3, targets)

    cell_type = df.select([pl.col(col).str.extract(r"^.*\.([\w\-]+)$")])
    cell_type = cell_type.rename({col: 'cell_type'}).to_series(0)
    df.insert_at_idx(3, cell_type)

    return df

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

def filter_bed_file_by_(bed_file_1, bed_file_2, output_file):
    # generated by OpenAI Codex

    bed_file_1_bedtool = pybedtools.BedTool(bed_file_1)
    bed_file_2_bedtool = pybedtools.BedTool(bed_file_2)
    bed_file_1_bedtool_intersect_bed_file_2_bedtool = bed_file_1_bedtool.intersect(bed_file_2_bedtool, v = True)
    bed_file_1_bedtool_intersect_bed_file_2_bedtool.saveas(output_file)

def filter_df_by_tfactor_fastas(df, folder):
    files = [*Path(folder).glob('**/*.fasta')]
    present_target_names = set([f.stem.split('.')[0] for f in files])
    all_df_targets = df.get_column('target').unique().to_list()

    all_df_targets_with_parsed_name = [(target, parse_gene_name(target)) for target in all_df_targets]
    unknown_targets = [target for target, parsed_target_name in all_df_targets_with_parsed_name for parsed_target_name_sub_el in parsed_target_name if parsed_target_name_sub_el not in present_target_names]

    if len(unknown_targets) > 0:
        df = df.filter(pl_notin('target', unknown_targets))
    return df

def generate_random_ranges_from_fasta(
    fasta_file,
    *,
    output_filename = 'random-ranges.bed',
    context_length,
    filter_bed_files = [],
    num_entries_per_key = 10,
    keys = None,
):
    fasta = Fasta(fasta_file)
    tmp_file = f'/tmp/{output_filename}'

    with open(tmp_file, 'w') as f:
        for chr_name in sorted(CHR_NAMES):
            print(f'generating ranges for {chr_name}')

            if chr_name not in fasta:
                print(f'{chr_name} not found in fasta file')
                continue

            chromosome = fasta[chr_name]
            chromosome_length = len(chromosome)

            start = np.random.randint(0, chromosome_length - context_length, (num_entries_per_key,))
            end = start + context_length
            start_and_end = np.stack((start, end), axis = -1)

            for row in start_and_end.tolist():
                start, end = row
                f.write('\t'.join((chr_name, str(start), str(end))) + '\n')

    for file in filter_bed_files:
        filter_bed_file_by_(tmp_file, file, tmp_file)

    shutil.move(tmp_file, f'./{output_filename}')

    print('success')

# context string creator class

class ProteinSmilesDataset(Dataset):
    def __init__(
        self,
        *,
        data_folder,
        csv_file = None,
        df = None,
        exclude_targets = None,
        include_targets = None,
        df_frac = 1.,
        experiments_json_path = None,
        balance_sampling_by_target_bin = False,
        **kwargs
    ):
        super().__init__()
        assert exists(csv_file) ^ exists(df), 'either data csv file or dataframe must be passed in'

        if not exists(df):
            df = read_bed(csv_file)

        if df_frac < 1:
            df_frac = df.sample(frac = df_frac)

        self.df = df

        self.experiments_index = fetch_experiments_index(experiments_json_path)

        # balanced target sampling logic

        self.balance_sampling_by_target_bin = balance_sampling_by_target_bin

        if self.balance_sampling_by_target_bin:
            self.df_indexed_by_target_bin = []

            for target in self.df.get_column('target_bin').unique().to_list():
                df_by_target_bin = self.df.filter(pl.col('target') == target)
                self.df_indexed_by_target_bin.append(df_by_target)

    def __len__(self):
        if self.balance_sampling_by_target_bin:
            return len(self.df_indexed_by_target_bin)
        else:
            return len(self.df)

    def __getitem__(self, ind):
        # if balancing by target bin, randomly draw sample from indexed dataframe

        if self.balance_sampling_by_target_bin:
            filtered_df = self.df_indexed_by_target_bin[ind]
            rand_ind = randrange(0, len(filtered_df))
            sample = filtered_df.row(rand_ind)
        else:
            sample = self.df.row(ind)

        aa_seq, smiles, target, *_ = sample

        # now aggregate all the data

        label = torch.Tensor([target])

        return smiles, aa_seq, label

# dataloader related functions

def collate_fn(data):
    smiles, aa_seq, labels = list(zip(*data))
    return tuple(smiles), tuple(aa_seq), torch.cat(labels, dim = 0)

def collate_dl_outputs(*dl_outputs):
    outputs = list(zip(*dl_outputs))
    ret = []
    for entry in outputs:
        if isinstance(entry[0], torch.Tensor):
            entry = torch.cat(entry, dim = 0)
        else:
            entry = (sub_el for el in entry for sub_el in el)
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
