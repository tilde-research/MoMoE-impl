import torch
import torch.nn as nn
from momoe import MoMoE
from topk_router import TopKRouter

"""
A simple end-to-end MoE implementation using MoMoE and TopKRouter.

Made as an example, showing how to use MoMoE together with a top-k router.

This implementation supports:
- Shared experts
- Auxiliary-free load balancing
- Customizable memory saving for backward
"""
class MoE(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        intermediate_dim,
        num_experts,
        num_chosen_experts,
        num_shared_experts,
        save_percent=0,
        bias_update_rate=1e-4,
        Wg_DN: torch.Tensor | None = None,
        Wl1_ND2H: torch.Tensor | None = None,
        Wl2_NHD: torch.Tensor | None = None,
    ):
        super(MoE, self).__init__()
        self.D = embedding_dim
        self.H = intermediate_dim
        self.N = num_experts
        self.K = num_chosen_experts
        self.Ks = num_shared_experts
        self.save_percent = save_percent
        self.bias_update_rate = bias_update_rate

        self.router = TopKRouter(
            embedding_dim=self.D,
            num_experts=self.N,
            num_chosen_experts=self.K,
            num_shared_experts=self.Ks,
            Wg_DN=Wg_DN,
        )

        self.momoe = MoMoE(
            embedding_dim=self.D,
            intermediate_dim=self.H,
            num_experts=self.N,
            num_chosen_experts=self.K,
            num_shared_experts=self.Ks,
            save_percent=self.save_percent,
            Wl1_ND2H=Wl1_ND2H,
            Wl2_NHD=Wl2_NHD,
        )

        self.biases_N = nn.Parameter(torch.cat([torch.zeros((self.N - self.Ks), device="cuda", dtype=torch.float32), torch.full((self.Ks,), float("-inf"), device="cuda", dtype=torch.float32)], dim=-1), requires_grad=False)

    def forward(self, x_BSD):
        x_BSD, mask_NM, s_NM = self.router(x_BSD, self.biases_N)
        y_BSD, sizes_N = self.momoe(x_BSD, mask_NM, s_NM)

        # Update biases using DeepSeek style auxiliary-free load balancing
        B, S, _ = x_BSD.shape
        c_avg = B * S * (self.K - self.Ks) / (self.N - self.Ks)
        self.biases_N += self.bias_update_rate * torch.sign(c_avg - sizes_N.to(self.biases_N.device))

        return y_BSD
        
        
        