from Bio import SeqIO
from random import choice, randrange
from pathlib import Path
import functools
import polars as pl
from collections import defaultdict

import os
import json
import shutil
import numpy as np

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

def read_bed(path):
    return pl.read_csv(path, sep = '\t', has_headers = False)

def save_bed(df, path):
    df.to_csv(path, sep = '\t', has_header = False)

def parse_exp_target_cell(exp_target_cell):
    experiment, target, *cell_type = exp_target_cell.split('.')
    cell_type = '.'.join(cell_type) # handle edge case where cell type contains periods
    return experiment, target, cell_type

# fetch index of datasets, for providing the sequencing reads
# for auxiliary read value prediction

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

# fetch protein sequences by gene name and uniprot id

class FactorProteinDatasetByUniprotID(Dataset):
    def __init__(
        self,
        folder,
        species_priority = ['human', 'mouse']
    ):
        super().__init__()
        fasta_paths = [*Path(folder).glob('*.fasta')]
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        self.paths = fasta_paths
        self.index_by_id = dict()

        for path in fasta_paths:
            gene, uniprotid, *_ = path.stem.split('.')
            self.index_by_id[uniprotid] = path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, uid):
        index = self.index_by_id

        if uid not in index:
            return None

        entry = index[uid]
        fasta = SeqIO.read(entry, 'fasta')
        return str(fasta.seq)

# fetch

class FactorProteinDataset(Dataset):
    def __init__(
        self,
        folder,
        species_priority = ['human', 'mouse', 'unknown'],
        return_tuple_only = False
    ):
        super().__init__()
        fasta_paths = [*Path(folder).glob('*.fasta')]
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        self.paths = fasta_paths

        index_by_gene = defaultdict(list)
        self.return_tuple_only = return_tuple_only # whether to return tuple even if there is only one subunit

        for path in fasta_paths:
            gene, uniprotid, *_ = path.stem.split('.')
            index_by_gene[gene].append(path)

        # prioritize fasta files of certain species
        # but allow for appropriate fallback, by order of species_priority

        get_species_from_path = lambda p: p.stem.split('_')[-1].lower() if '_' in p.stem else 'unknown'

        filtered_index_by_gene = defaultdict(list)

        for gene, gene_paths in index_by_gene.items():
            species_count = list(map(lambda specie: len(list(filter(lambda p: get_species_from_path(p) == specie, gene_paths))), species_priority))
            species_ind_non_zero = find_first_index(lambda t: t > 0, species_count)

            if species_ind_non_zero == -1:
                continue

            species = species_priority[species_ind_non_zero]
            filtered_index_by_gene[gene] = list(filter(lambda p: get_species_from_path(p) == species, gene_paths))

        self.index_by_gene = filtered_index_by_gene

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, unparsed_gene_name):
        index = self.index_by_gene

        genes = parse_gene_name(unparsed_gene_name)
        seqs = []

        for gene in genes:
            entry = index[gene]

            if len(entry) == 0:
                print(f'no entries for {gene}')
                continue

            path = choice(entry) if isinstance(entry, list) else entry

            fasta = SeqIO.read(path, 'fasta')
            seqs.append(str(fasta.seq))

        seqs = tuple(seqs)

        if len(seqs) == 1 and not self.return_tuple_only:
            return seqs[0]

        return seqs

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
