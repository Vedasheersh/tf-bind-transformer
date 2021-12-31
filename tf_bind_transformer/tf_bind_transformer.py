import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from functools import wraps

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from contextlib import contextmanager
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import freeze_batchnorms_, freeze_all_but_layernorms_

from tf_bind_transformer.cache_utils import cache_fn
from tf_bind_transformer.protein_utils import get_esm_repr, ESM_EMBED_DIM
from tf_bind_transformer.context_utils import get_text_repr, get_contextual_dim

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def logavgexp(
    t,
    mask = None,
    dim = -1,
    eps = 1e-20,
    temp = 0.01
):
    if exists(mask):
        mask_value = -torch.finfo(t.dtype).max
        t = t.masked_fill(mask, mask_value)
        n = mask.sum(dim = dim)
        norm = torch.log(n)
    else:
        n = t.shape[dim]
        norm = math.log(n)

    t = t / temp
    max_t = t.amax(dim = dim).detach()
    t_exp = (t - max_t.unsqueeze(dim)).exp()
    avg_exp = t_exp.sum(dim = dim).clamp(min = eps) / n
    out = log(avg_exp, eps = eps) + max_t - norm
    return out * temp

# losses and metrics

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, eps = 1e-8):
    x2 = x * x
    y2 = y * y
    xy = x * y
    ex = x.mean(dim = 1)
    ey = y.mean(dim = 1)
    exy = xy.mean(dim = 1)
    ex2 = x2.mean(dim = 1)
    ey2 = y2.mean(dim = 1)
    r = (exy - ex * ey) / (torch.sqrt(ex2 - (ex * ex)) * torch.sqrt(ey2 - (ey * ey)) + eps)
    return r.mean(dim = -1)

# genetic sequence caching enformer forward decorator

def cache_enformer_forward(fn):
    cached_forward = cache_fn(fn, clear = True, path = 'genetic')

    @wraps(fn)
    def inner(seqs, *args, **kwargs):
        if seqs.ndim == 3:
            seqs = seqs.argmax(dim = -1)

        seq_list = seqs.unbind(dim = 0)
        seq_cache_keys = [''.join(list(map(str, one_seq.tolist()))) for one_seq in seq_list]
        outputs = [cached_forward(one_seq, *args, __cache_key = seq_cache_key, **kwargs) for one_seq, seq_cache_key in zip(seq_list, seq_cache_keys)]
        return torch.stack(outputs)

    return inner

# model

class Model(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        enformer_dim = 1536,
        latent_dim = 64,
        latent_heads = 32,
        aa_embed_dim = None,
        contextual_embed_dim = 256,
        use_esm_embeds = False,
        use_free_text_context = False,
        free_text_context_encoder = 'pubmed',
        free_text_embed_method = 'cls',
        dropout = 0.
    ):
        super().__init__()
        assert isinstance(enformer, Enformer), 'enformer must be an instance of Enformer'
        self.enformer = enformer

        # contextual embedding related variables

        assert free_text_embed_method in {'cls', 'mean_pool'}, 'must be either cls or mean_pool'
        self.free_text_embed_method = free_text_embed_method
        self.use_free_text_context = use_free_text_context
        contextual_embed_dim = get_contextual_dim(free_text_context_encoder)

        # protein embedding related variables

        self.use_esm_embeds = use_esm_embeds

        if use_esm_embeds:
            aa_embed_dim = ESM_EMBED_DIM
        else:
            assert exists(aa_embed_dim), 'AA embedding dimensions must be set if not using ESM'

        # latents

        self.latent_heads = latent_heads
        inner_latent_dim = latent_heads * latent_dim

        self.seq_embed_to_latent_w = nn.Parameter(torch.randn(enformer_dim * 2, inner_latent_dim))
        self.seq_embed_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.aa_seq_embed_to_latent_w = nn.Parameter(torch.randn(aa_embed_dim, inner_latent_dim))
        self.aa_seq_embed_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.to_logits_w = nn.Parameter(torch.randn(latent_heads, latent_heads))
        self.contextual_projection = nn.Linear(contextual_embed_dim, latent_heads * latent_heads)

        self.dropout = nn.Dropout(dropout)

        self.to_pred = nn.Sequential(
            nn.Linear(latent_heads, 1),
            Rearrange('... 1 -> ...'),
            nn.Softplus()
        )

    def forward(
        self,
        seq,
        *,
        aa = None,
        aa_embed = None,
        contextual_embed = None,
        contextual_free_text = None,
        aa_mask = None,
        target = None,
        return_corr_coef = False,
        finetune_enformer = False,
        finetune_enformer_ln_only = False
    ):
        latent_heads = self.latent_heads

        # prepare enformer for training
        # - set to eval and no_grad if not fine-tuning
        # - always freeze the batchnorms

        freeze_batchnorms_(self.enformer)
        enformer_forward = self.enformer.forward

        if finetune_enformer:
            enformer_context = null_context()
        elif finetune_enformer_ln_only:
            enformer_context = null_context()
            freeze_all_but_layernorms_(self.enformer)
        else:
            self.enformer.eval()
            enformer_context = torch.no_grad()
            enformer_forward = cache_enformer_forward(enformer_forward)

        # genetic sequence embedding

        with enformer_context:
            seq_embed = enformer_forward(seq, return_only_embeddings = True)

        # protein related embeddings

        if self.use_esm_embeds:
            assert exists(aa), 'aa must be passed in as tensor of integers from 0 - 20 (20 being padding)'
            aa_embed, aa_mask = get_esm_repr(aa, device = seq.device)
        else:
            assert exists(aa_embed), 'protein embeddings must be given as aa_embed'

        # free text embeddings, for cell types and experimental params

        if not exists(contextual_embed):
            assert self.use_free_text_context, 'use_free_text_context must be set to True if one is not passing in contextual_embed tensor'
            assert exists(contextual_free_text), 'context must be supplied as array of strings as contextual_free_text if contextual_embed is not supplied'

            contextual_embed = get_text_repr(
                contextual_free_text,
                return_cls_token = (self.free_text_embed_method == 'cls'),
                device = seq.device
            )

        # project both embeddings into shared latent space

        seq_latent = einsum('b n d, d e -> b n e', seq_embed, self.seq_embed_to_latent_w)
        seq_latent = seq_latent + self.seq_embed_to_latent_b

        seq_latent = rearrange(seq_latent, 'b n (h d) -> b h n d', h = latent_heads)

        aa_latent = einsum('b n d, d e -> b n e', aa_embed, self.aa_seq_embed_to_latent_w)
        aa_latent = aa_latent + self.aa_seq_embed_to_latent_b

        aa_latent = rearrange(aa_latent, 'b n (h d) -> b h n d', h = latent_heads)

        aa_latent, seq_latent = map(l2norm, (aa_latent, seq_latent))

        # fine grained interaction between dna and protein sequences
        # FILIP https://arxiv.org/abs/2111.07783

        if seq_latent.shape[0] == 1:
            # in the case one passes in 1 genomic sequence track
            # but multiple factors + contexts, as in enformer training
            seq_latent = rearrange(seq_latent, '1 ... -> ...')
            einsum_eq = 'h i d, b h j d -> b h i j'
        else:
            einsum_eq = 'b h i d, b h j d -> b h i j'

        interactions = einsum(einsum_eq, seq_latent, aa_latent)

        # dropout

        interactions = self.dropout(interactions)

        # reduction

        if exists(aa_mask):
            aa_mask = rearrange(aa_mask, 'b j -> b 1 1 j')

        interactions = logavgexp(interactions, mask = aa_mask, dim = -1)
        interactions = rearrange(interactions, 'b h i -> b i h')

        # derive contextual gating, from hypergrids paper

        gating = self.contextual_projection(contextual_embed).sigmoid()
        gating = rearrange(gating, 'b (i o) -> b i o', i = int(math.sqrt(gating.shape[-1])))

        # gate interactions projection with context

        to_logits_w = rearrange(self.to_logits_w, 'i o -> 1 i o') * gating
        logits = einsum('b n d, b d e -> b n e', interactions, to_logits_w)

        # to *-seq prediction

        pred = self.to_pred(logits)

        if exists(target):
            if return_corr_coef:
                return pearson_corr_coef(pred, target)

            return poisson_loss(pred, target)

        return pred
