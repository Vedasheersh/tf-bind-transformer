import torch
from torch import nn
from tf_bind_transformer.optimizer import get_optimizer
from tf_bind_transformer.data import read_csv, collate_dl_outputs, get_dataloader
from tf_bind_transformer.data import ProteinSmilesDataset
from enformer_pytorch.modeling_enformer import pearson_corr_coef
from sklearn.model_selection import train_test_split
import ipdb

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def split_df(df, splitby = 'random', test_size = 0.1, seed = 0):
    assert splitby=='random', 'other splitting methods not implemented yet'
    # ipdb.set_trace()
    df.reset_index(inplace=True, drop=True)
    train_df, valid_df = train_test_split(df, 
                                          test_size = test_size, 
                                          random_state = seed)
    
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

# helpers for logging and accumulating values across gradient steps

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# simple Trainer class

class Trainer(nn.Module):
    def __init__(
        self,
        model,
        *,
        csv_file,
        batch_size,
        seqcol = 'sequence',
        smicol = 'reaction_smiles',
        labelcol = 'log10_value',
        augment_train = False,
        augment_val = False,
        lr = 3e-4,
        wd = 0.1,
        validate_every = 250,
        grad_clip_norm = None,
        grad_accum_every = 1,
        shuffle = False,
        train_sample_frac = 1.,
        valid_sample_frac = 1.,
        df_sample_frac = 1.,
        experiments_json_path = None,
        checkpoint_filename = './checkpoint.pt',
        balance_sampling_by_target_bin = True,
        valid_balance_sampling_by_target_bin = None,
        data_seed = 0
    ):
        super().__init__()
        self.model = model
        valid_balance_sampling_by_target_bin = default(valid_balance_sampling_by_target_bin, balance_sampling_by_target_bin)

        df = read_csv(csv_file)
        df = df.dropna(subset=[seqcol,smicol,labelcol])
        df = df.reset_index(drop=True)
        
        if df_sample_frac < 1:
            df = df.sample(frac = df_sample_frac, seed = data_seed)

        train_df, valid_df = split_df(df, seed = data_seed)
        
        self.ds = ProteinSmilesDataset(
            df = train_df,
            smiles_col = smicol,
            seq_col = seqcol,
            target_col = labelcol,
            df_frac = train_sample_frac,
            augment = augment_train,
            experiments_json_path = experiments_json_path,
            balance_sampling_by_target_bin = balance_sampling_by_target_bin
        )

        self.valid_ds = ProteinSmilesDataset(
            df = valid_df,
            smiles_col = smicol,
            seq_col = seqcol,
            target_col = labelcol,
            augment = augment_val,
            experiments_json_path = experiments_json_path,
            balance_sampling_by_target_bin = valid_balance_sampling_by_target_bin
        )

        self.dl = get_dataloader(self.ds, cycle_iter = False, shuffle = shuffle, batch_size = batch_size)
        self.valid_dl = get_dataloader(self.valid_ds, cycle_iter = False, shuffle = shuffle, batch_size = batch_size)

        self.optim = get_optimizer(model.parameters(), lr = lr, wd = wd)

        self.grad_accum_every = grad_accum_every
        self.grad_clip_norm = grad_clip_norm

        self.validate_every = validate_every
        self.register_buffer('steps', torch.Tensor([0.]))

        self.checkpoint_filename = checkpoint_filename

    def forward(
        self,
        **kwargs
    ):
        grad_accum_every = self.grad_accum_every
        curr_step = int(self.steps.item())
        self.model.train()

        log = {}

        for _ in range(self.grad_accum_every):
            # dl_outputs = [next(self.dl), next(self.neg_dl)]
            # ipdb.set_trace()
            smiles, aa_seq, label = collate_dl_outputs(next(self.dl))
            
            smiles, aa_seq, label = smiles, aa_seq, label.cuda()

            loss = self.model(
                smiles,
                aa_seq = aa_seq,
                label = label,
                **kwargs
            )

            log = accum_log(log, {
                'loss': loss.item() / grad_accum_every,
            })

            (loss / self.grad_accum_every).backward()

        print(f'{curr_step} loss: {log["loss"]}')

        if exists(self.grad_clip_norm):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optim.step()
        self.optim.zero_grad()

        if (curr_step % self.validate_every) == 0:
            self.model.eval()

            for _ in range(self.grad_accum_every):
                smiles, aa_seq, label = collate_dl_outputs(next(self.valid_dl))
            
                smiles, aa_seq, label = smiles, aa_seq, label.cuda()

                valid_preds, loss = self.model(
                    smiles,
                    aa_seq = aa_seq,
                    label = label,
                    return_preds = True
                )

                valid_loss = self.model.loss_fn(valid_preds, label)
                # ipdb.set_trace()
                valid_r = pearson_corr_coef(valid_preds, label, dim = 0)

                log = accum_log(log, {
                    'valid_loss': valid_loss.item() / grad_accum_every,
                    'valid_correlation': valid_r.item() / grad_accum_every
                })

            print(f'{curr_step} valid loss: {log["valid_loss"]}')
            print(f'{curr_step} valid correlation: {log["valid_correlation"]}')

            if curr_step > 0:
                torch.save(self.model.state_dict(), self.checkpoint_filename)

        self.steps += 1
        return log
