import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl
import math
from einops import rearrange
from typing import Any

"""
Dimension key:

B: batch size
S: sequence length
M: B * S
D: embedding dimension
N: number of experts
Ng: number of gating experts
Ks: (selected) number of shared experts
Kg: selected number of gating experts
K: selected total number of experts
H: dimension of the hidden layer
"""


"""
A basic MoE layer written in PyTorch. Supports tuning the number of
experts (N), the number of selected experts (K), and the
hidden ratio of the fully connected layer (H/D). This implementation
batches everything into a couple of matrix multiplications for each
used expert.

Takes in an input of shape (B, S, D) and outputs a shape (B, S, D)
tensor.
"""
class TorchMoE(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_experts: int,
        num_chosen_experts: int,
        num_shared_experts: int,
        hidden_ratio: int,
        dropout: float = 0.0,
        sigmoid: bool = False,
        bias_update_rate: float = 1e-3,
        dtype: torch.dtype = torch.bfloat16,

        Wg_DN: torch.Tensor | None = None,
        Wl1_ND2H: torch.Tensor | None = None,
        Wl2_NHD: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        super(TorchMoE, self).__init__()
        self.D = hidden_size # embedding dimension
        self.N = num_experts # number of experts
        self.K = num_chosen_experts # number of chosen experts
        self.Ks = num_shared_experts # number of shared experts
        self.H = hidden_ratio * self.D # hidden dimension of the fully connected layer

        # Gating network parameters
        assert Wg_DN is None or Wg_DN.dtype == dtype and Wl1_ND2H is None or Wl1_ND2H.dtype == dtype and Wl2_NHD is None or Wl2_NHD.dtype == dtype

        if Wg_DN is not None:
            self.Wg_DN = nn.Parameter(Wg_DN.clone().to(device))
        else:
            self.Wg_DN = nn.Parameter(torch.randn(self.D, self.N, dtype=dtype, device=device, requires_grad=True) / math.sqrt(self.D))
        if Wl1_ND2H is not None:
            self.Wl1_ND2H = nn.Parameter(Wl1_ND2H.clone().to(device))
        else:
            self.Wl1_ND2H = nn.Parameter(torch.randn(self.N, self.D, 2 * self.H, dtype=dtype, device=device, requires_grad=True) / math.sqrt(self.D))
        if Wl2_NHD is not None:
            self.Wl2_NHD = nn.Parameter(Wl2_NHD.clone().to(device))
        else:
            self.Wl2_NHD = nn.Parameter(torch.randn(self.N, self.H, self.D, dtype=dtype, device=device, requires_grad=True) / math.sqrt(self.D))
        
        # Bias
        self.bias_update_rate = bias_update_rate
        self.biases_N = torch.cat([torch.zeros(self.N - self.Ks, dtype=torch.float32), torch.full((self.Ks,), float('inf'), dtype=torch.float32)]).to(device)

        self.sigmoid = sigmoid
        self.threshold = 0.18 # TODO: make this a parameter
        self.dropout = dropout
        self.DTYPE = dtype

    def do_gating_and_experts_softmax(
        self,
        x_MD: torch.Tensor,
    ) -> torch.Tensor:
        '''
        This function does the gating and the experts calculations with softmax gating.
        It returns the output of the experts and the number of tokens that went to each expert.
        '''

        assert not self.sigmoid
        M, D = x_MD.shape

        # Gating network
        s_raw_MN = x_MD @ self.Wg_DN

        s_MN = torch.empty((M, self.N), device=x_MD.device)
        s_MN[:, :self.N-self.Ks] = torch.softmax(s_raw_MN[:, :self.N-self.Ks], dim=-1, dtype=torch.float)
        s_MN[:, self.N-self.Ks:] = torch.softmax(s_raw_MN[:, self.N-self.Ks:], dim=-1, dtype=torch.float)
        s_MNg, s_MKs = s_MN[:, :self.N-self.Ks], s_MN[:, self.N-self.Ks:]
        values_MKg, indices_MKg = torch.topk(s_MNg + self.biases_N[:self.N-self.Ks][None, :], self.K - self.Ks, dim=-1)

        # Normalize the gating experts and the shared experts separately
        mask_MNg = torch.zeros_like(s_MNg).scatter_(1, indices_MKg, 1).bool()
        s_w_MNg = s_MNg * mask_MNg
        Ng_M = torch.sum(s_w_MNg, dim=-1)
        Ks_M = torch.sum(s_MKs, dim=-1)
        s_MNg = s_MNg / (Ng_M[:, None]) * mask_MNg
        s_MKs = s_MKs / (Ks_M[:, None])
        s_MN = torch.cat([s_MNg, s_MKs], dim=-1)
        mask_NM = torch.ones([self.N, M], device=x_MD.device, dtype=torch.int32)
        mask_NM[:self.N-self.Ks, :] = rearrange(torch.scatter(torch.zeros([M, self.N-self.Ks], device=x_MD.device, dtype=torch.int32), dim=-1, index=indices_MKg, src=torch.ones([M, self.N - self.Ks], device=x_MD.device, dtype=torch.int32)), 'M N -> N M')

        s_MN = s_MN.to(x_MD.dtype)

        # Experts
        y_MD = torch.zeros((M, D), device=x_MD.device, dtype=x_MD.dtype)
        c_N = torch.zeros((self.N,), device=x_MD.device)

        for i in range(self.N):
            expert_indices_Mi = torch.nonzero(mask_NM[i], as_tuple=True)[0]
            x_MiD = x_MD[expert_indices_Mi]

            gate_Mi = s_MN[expert_indices_Mi, i]

            z_Mi2H = x_MiD @ self.Wl1_ND2H[i]
            a_MiH, b_MiH = z_Mi2H.chunk(2, dim=-1)
            h_MiH = F.silu(b_MiH) * a_MiH

            y_MiD = h_MiH @ self.Wl2_NHD[i]

            c_N[i] += gate_Mi.shape[0]

            y_MD[expert_indices_Mi] += y_MiD * rearrange(gate_Mi, '(Mi X) -> Mi X', X=1)

        return y_MD.to(x_MD.dtype), c_N

    def do_gating_and_experts_sigmoid(self, x_MD: torch.Tensor) -> torch.Tensor:
        '''
        This function does the gating and the experts calculations with sigmoid gating.
        It returns the output of the experts and the number of tokens that went to each expert.
        '''
        assert self.sigmoid
        # Gating network
        M, D = x_MD.shape

        s_raw_MN = x_MD @ self.Wg_DN
        s_MN = torch.sigmoid(s_raw_MN)

        # Normalize the gating experts and the shared experts separately
        s_MNg, s_MKs = s_MN.split([self.N - self.Ks, self.Ks], dim=-1)
        Ng_M = torch.sum(s_MNg, dim=-1)
        Ks_M = torch.sum(s_MKs, dim=-1)
        # Ng_M = torch.ones([M], device=x_MD.device, dtype=torch.float32)
        # Ks_M = torch.ones([M], device=x_MD.device, dtype=torch.float32)
        s_MNg = s_MNg / (Ng_M[:, None] + 1e-6)
        s_MKs = s_MKs / (Ks_M[:, None] + 1e-6)
        s_MN = torch.cat([s_MNg, s_MKs], dim=-1)

        s_NM = rearrange(s_MN, 'M N -> N M')
        s_bias_NM = s_NM.to(self.biases_N.dtype) + self.biases_N[:, None]
        N, _ = s_NM.shape

        # Experts
        y_MD = torch.zeros(M, D, device=x_MD.device, dtype=torch.float32)
        c_N = torch.zeros(N, device=x_MD.device)
        for i in range(N): # for each gating expert
            # Mi = M where any of indices_MKg == i
            s_bias_i_M = s_bias_NM[i]
            s_i_M = s_NM[i]

            # get the indices of the tokens that go to expert i 
            expert_indices_Mi = torch.nonzero((s_bias_i_M > self.threshold), as_tuple=True)[0]
            x_MiD = x_MD[expert_indices_Mi]

            gate_Mi = s_i_M[expert_indices_Mi].to(torch.float32)

            z_Mi2H = x_MiD @ self.Wl1_ND2H[i]
            a_MiH, b_MiH = z_Mi2H.chunk(2, dim=-1)
            h_MiH = a_MiH * F.silu(b_MiH)
            y_MiD = (h_MiH @ self.Wl2_NHD[i]).to(torch.float32)

            c_N[i] += gate_Mi.shape[0]

            y_MD = y_MD.index_add(0, expert_indices_Mi, y_MiD * rearrange(gate_Mi, '(Mi X) -> Mi X', X=1))

        return y_MD.to(x_MD.dtype), c_N

    def forward(
        self,
        x_BSD: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        B, S, D = x_BSD.shape
        M = B * S
        x_MD = rearrange(x_BSD, 'B S D -> (B S) D')

        # Gating network and expert calculations
        if not self.sigmoid: # we want to do the softmax and the topk here
            y_MD, c_N = self.do_gating_and_experts_softmax(x_MD)
        else: # sigmoid version without topk
            y_MD, c_N = self.do_gating_and_experts_sigmoid(x_MD)

        # Update biases of the gating experts
        if self.N - self.Ks > 0:
            c_avg = M * (self.K - self.Ks) / (self.N - self.Ks) # average desired number of tokens per expert
            self.biases_N += self.bias_update_rate * torch.sign(c_avg - c_N)

        return rearrange(y_MD, '(B S) D -> B S D', B=B, S=S).to(self.DTYPE) # no RMS or residual here

def torch_moe(
    x_BSD: torch.Tensor,
    Wg_DN: torch.Tensor,
    Wl1_ND2H: torch.Tensor,
    Wl2_NHD: torch.Tensor,
    biases_N: torch.Tensor,
    K: int,
    Ks: int,
    dropout: float = 0.0,
    sigmoid: bool = False,
    bias_update_rate: float = 1e-3,
    threshold: float = 0.2,
    save_percent: float = 0.0,
) -> torch.Tensor:
    B, S, D = x_BSD.shape
    N, H, _ = Wl2_NHD.shape
    device = x_BSD.device
    dtype = x_BSD.dtype
    torchmoe = TorchMoE(D, N, K, Ks, H, 0.0, sigmoid, bias_update_rate, dtype, Wg_DN, Wl1_ND2H, Wl2_NHD, device)

    return torchmoe(x_BSD)