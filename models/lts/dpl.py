import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import math
from torch import Tensor
from typing import Optional, Tuple, List

import os
import torch
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1):
    length = x.size(dim)
    pad_len = (-length) % multiple  
    if pad_len == 0:
        return x, 0
    pad_sizes = [0] * (2 * x.dim())
    pad_sizes[2*(x.dim()-dim)-2] = pad_len
    x_padded = F.pad(x, pad_sizes, mode='constant', value=0.0)
    return x_padded, pad_len
    
class BetaMish(nn.Module):
    def __init__(self, channels: int = 16, init_beta: float = 0.5):
        super().__init__()                                  

        self.log_beta = nn.Parameter(
            torch.full((channels,), torch.log(torch.tensor(init_beta)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = F.softplus(self.log_beta)                     
        
        shape = (1, -1) + (1,) * (x.dim() - 2)                 
        beta = beta.view(shape)
        return x * torch.tanh(F.softplus(beta * x))

class LinearformerBlock(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            heads=4,
            dropout_p=0.1,
            bidirectional=False,
    ):
        super().__init__()
        self.lf = LinearAttentionTransformer(dim = emb_dim,
                    heads = heads,
                    depth = 1,
                    max_seq_len = 8192,
                    n_local_attn_heads = 0,
                    attn_dropout = dropout_p
                ).to(DEVICE)

        self.dense = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, x):
        # x:(b,t,d)
        x = self.lf(x)
        x = self.dense(x)
        return x


class DualPathLF(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm((n_freqs, emb_dim))
        self.f_lf = LinearformerBlock(emb_dim, emb_dim // 2 * 3, 4, dropout_p, bidirectional=True)
        self.t_lf = LinearformerBlock(emb_dim, emb_dim // 2 * 3, 4, dropout_p, bidirectional=False)


    def forward(self, x):
        # x:(b,d,t,f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)  # (b,t,f,d)

        x_res = x
        x = self.norm(x)
        x = x.reshape(B * T, F, D)  # (b*t,f,d)
        x = self.f_lf(x)
        x = x.reshape(B, T, F, D)
        x = x + x_res

        x_res = x
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3)  # (b,f,t,d)
        x = x.reshape(B * F, T, D)
        x = self.t_lf(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3) # (b,t,f,d)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)
        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = BetaMish(channels=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x:(b,d,t,f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x 

class DPL(nn.Module):
    def __init__(
            self,
            emb_dim=16,
            hidden_dim=24,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.lf_block = DualPathLF(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=2, dropout_p=dropout_p)

    def forward(self, x):
        x = self.lf_block(x)
        x = self.conv_glu(x)
        return x