import copy
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from functools import wraps

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from contextlib import contextmanager

from enformer_pytorch import Enformer
from enformer_pytorch.modeling_enformer import poisson_loss, pearson_corr_coef
from enformer_pytorch.finetune import freeze_batchnorms_, freeze_all_but_layernorms_, unfreeze_last_n_layers_, unfreeze_all_layers_

from logavgexp_pytorch import logavgexp

from tf_bind_transformer.cache_utils import cache_fn
from tf_bind_transformer.protein_utils import get_protein_embedder
from tf_bind_transformer.smiles_utils import get_smiles_embedder
from tf_bind_transformer.context_utils import get_text_repr, get_contextual_dim

def normal_mve(pred_values, targets):
    """
    Use the negative log likelihood function of a normal distribution as a loss function used for making
    simultaneous predictions of the mean and error distribution variance simultaneously.

    :param pred_values: Combined predictions of means and variances of shape(data, tasks*2).
                        Means are first in dimension 1, followed by variances.
    :return: A tensor loss value.
    """
    # Unpack combined prediction values
    pred_means, pred_var = torch.split(pred_values, pred_values.shape[1] // 2, dim=1)

    return torch.log(2 * np.pi * pred_var) / 2 + (pred_means - targets) ** 2 / (2 * pred_var)
    
from tf_bind_transformer.attention import FeedForward, JointCrossAttentionBlock, CrossAttention, SelfAttentionBlock

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(fn, *args, **kwargs):
    return fn

@contextmanager
def null_context():
    yield

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# FILIP adapter model

class FILIP(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        inner_latent_dim = heads * dim_head

        self.to_latent_w = nn.Parameter(torch.randn(dim, inner_latent_dim))
        self.to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.pre_attn_dropout = dropout

        self.null_context = nn.Parameter(torch.randn(heads, dim_head))
        self.context_to_latent_w = nn.Parameter(torch.randn(context_dim, inner_latent_dim))
        self.context_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

    def forward(
        self,
        x,
        context,
        context_mask = None
    ):
        b, heads, device = x.shape[0], self.heads, x.device

        x = einsum('b n d, d e -> b n e', x, self.to_latent_w)
        x = x + self.to_latent_b

        x = rearrange(x, 'b n (h d) -> b h n d', h = heads)

        context = einsum('b n d, d e -> b n e', context, self.context_to_latent_w)
        context = context + self.context_to_latent_b

        context = rearrange(context, 'b n (h d) -> b h n d', h = heads)

        context, x = map(l2norm, (context, x))

        # fine grained interaction between dna and protein sequences
        # FILIP https://arxiv.org/abs/2111.07783

        if x.shape[0] == 1:
            # in the case one passes in 1 genomic sequence track
            # but multiple factors + contexts, as in enformer training
            x = rearrange(x, '1 ... -> ...')
            einsum_eq = 'h i d, b h j d -> b h i j'
        else:
            einsum_eq = 'b h i d, b h j d -> b h i j'

        # create context mask if not exist

        if not exists(context_mask):
            context_mask = torch.ones((b, context.shape[-1]), device = device).bool()

        # dropout mask by dropout prob

        if self.training:
            keep_mask = prob_mask_like(context_mask, 1 - self.pre_attn_dropout)
            context_mask = context_mask & keep_mask

        # add null context and modify mask

        context_mask = F.pad(context_mask, (1, 0), value = True)
        context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        null_context = repeat(self.null_context, 'h d -> b h 1 d', b = b)
        context = torch.cat((null_context, context), dim = -2)

        # differentiable max, as in FILIP paper

        interactions = einsum(einsum_eq, x, context)
        interactions = logavgexp(interactions, mask = context_mask, dim = -1, temp = 0.05)
        interactions = rearrange(interactions, 'b h i -> b i h')
        return interactions

class AdapterModel(nn.Module):
    def __init__(
        self,
        *,
        smiles_model_dim = 384,
        latent_dim = 64,
        latent_heads = 32,
        aa_embed_dim = 1280,
        aa_embed_encoder = 'esm',
        smiles_embed_encoder = 'chemberta',
        dropout = 0.,
        joint_cross_attn_depth = 1,
        protein_self_attn_depth = 0,
        use_mve_loss = False,
        **kwargs
    ):
        self.norm_smiles_embed = nn.LayerNorm(smiles_model_dim)

        # protein embedding related variables

        self.smiles_embed_config = get_smiles_embedder()
        self.aa_embed_config = get_protein_embedder(aa_embed_encoder)
        self.get_aa_embed = self.aa_embed_config['fn']
        self.get_smiles_embed = self.smiles_embed_config['fn']

        # protein self attn

        self.protein_self_attns = nn.ModuleList([])

        for _ in range(protein_self_attn_depth):
            attn = SelfAttentionBlock(
                dim = enformer_dim,
                dropout = dropout
            )
            self.protein_self_attns.append(attn)

        # joint attn

        self.joint_cross_attns = nn.ModuleList([])

        for _ in range(joint_cross_attn_depth):
            attn = JointCrossAttentionBlock(
                dim = aa_embed_dim,
                context_dim = smiles_model_dim,
                dropout = dropout
            )

            self.joint_cross_attns.append(attn)

        # latents

        self.filip = FILIP(
            dim = aa_embed_dim,
            context_dim = smiles_model_dim,
            dim_head = latent_dim,
            heads = latent_heads,
            dropout = dropout
        )
        
        self.linear_to_logits = nn.Linear(latent_heads, latent_heads)

        # to prediction

        self.loss_fn = normal_mve if use_mve_loss else F.smooth_l1_loss
        out_dim = 2 if use_mve_loss else 1

        self.to_pred = nn.Sequential(
            nn.Linear(latent_heads, out_dim),
            Rearrange(f'... {out_dim} -> ...')
        )

    def forward(
        self,
        smi,
        aa,
        aa_embed = None,
        aa_mask = None,
        target = None
    ):
        device = smi.device

        # smiles embedding
        
        smi_embed, smi_mask = self.get_smi_embed(smi, device = device)
        
        # norm smiles embedding
        
        smi_embed = self.norm_smi_embed(smi_embed)

        # protein embeddings
        
        aa_embed, aa_mask = self.get_aa_embed(aa, device = seq.device)
        for self_attn_block in self.protein_self_attns:
            aa_embed = self_attn_block(aa_embed)
        
        # joint cross attention

        for cross_attn in self.joint_cross_attns:
            seq_embed, aa_embed = cross_attn(
                aa_embed,
                mask = aa_mask,
                context = smi_embed,
                context_mask = smi_mask
            )

        # project both embeddings into shared latent space

        interactions = self.filip(
            aa_embed,
            smi_embed,
            context_mask = smis_mask
        )

        logits = self.linear_to_logits(interactions)

        # to *-seq prediction

        pred = self.to_pred(logits)

        return self.loss_fn(pred, target)
