import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange

"""
Triton kernel for the backward pass of the MoE, which does a backward over
the normalization and softmax operations
"""
@triton.jit
def normalize_backward_kernel(
        G_s_ptr_NM, s_ptr_MN, s_processed_ptr_NM, Ng_ptr_M, Ks_ptr_M, mask_ptr_NM,
        N: tl.constexpr, Ks: tl.constexpr, pw_Ng: tl.constexpr, pw_Ks: tl.constexpr,
        stride_GsN, stride_GsM,
        stride_sM, stride_sN,
        stride_sprocessedN, stride_sprocessedM,
        stride_NgM, stride_KsM,
        stride_maskN, stride_maskM
):
    pid = tl.program_id(0)

    # offsets and masks
    offs_Ng = tl.arange(0, pw_Ng)
    offs_Ks = tl.arange(pw_Ng // 2, pw_Ng // 2 + pw_Ks)
    mask_Ng = (offs_Ng < N-Ks)
    mask_Ks = (offs_Ks < N) & (offs_Ks >= N-Ks)

    # load row m
    G_s_Ng = tl.load(G_s_ptr_NM + pid * stride_GsM + offs_Ng * stride_GsN, mask=mask_Ng, other=0.0).to(tl.float32)
    msk_Ng = tl.load(mask_ptr_NM + pid * stride_maskM + offs_Ng * stride_maskN, mask=mask_Ng, other=0).to(tl.float32)
    G_s_Ks = tl.load(G_s_ptr_NM + pid * stride_GsM + offs_Ks * stride_GsN, mask=mask_Ks, other=0.0).to(tl.float32)
    s_postsoft_Ng = tl.load(s_ptr_MN + pid * stride_sM + offs_Ng * stride_sN, mask=mask_Ng, other=0.0).to(tl.float32) # the pre-normalized row
    s_postsoft_Ks = tl.load(s_ptr_MN + pid * stride_sM + offs_Ks * stride_sN, mask=mask_Ks, other=0.0).to(tl.float32) # the pre-normalized row
    s_postnorm_Ng = tl.load(s_processed_ptr_NM + pid * stride_sprocessedM + offs_Ng * stride_sprocessedN, mask=mask_Ng, other=0.0).to(tl.float32)
    s_postnorm_Ks = tl.load(s_processed_ptr_NM + pid * stride_sprocessedM + offs_Ks * stride_sprocessedN, mask=mask_Ks, other=0.0).to(tl.float32)

    # compute the two dot-prods
    dot_gz_Ng = tl.sum(G_s_Ng * s_postnorm_Ng * msk_Ng) # mask applies here
    dot_gz_Ks = tl.sum(G_s_Ks * s_postnorm_Ks) # implicit mask of ones

    Ng_val = tl.load(Ng_ptr_M + pid * stride_NgM).to(tl.float32)
    Ks_val = tl.load(Ks_ptr_M + pid * stride_KsM).to(tl.float32)

    out_Ng = (G_s_Ng - dot_gz_Ng) * (1.0 / Ng_val)
    out_Ks = (G_s_Ks - dot_gz_Ks) * (1.0 / Ks_val)

    # compute softmax backward
    Ng_soft = tl.sum(out_Ng * s_postsoft_Ng)
    Ks_soft = tl.sum(out_Ks * s_postsoft_Ks)
    final_Ng = (s_postsoft_Ng * (out_Ng - Ng_soft)).to(tl.bfloat16)
    final_Ks = (s_postsoft_Ks * (out_Ks - Ks_soft)).to(tl.bfloat16)

    # store back
    tl.store(G_s_ptr_NM + pid * stride_GsM + offs_Ng * stride_GsN, final_Ng, mask=mask_Ng)
    tl.store(G_s_ptr_NM + pid * stride_GsM + offs_Ks * stride_GsN, final_Ks, mask=mask_Ks)

"""
Performs the backward pass of the router, which fuses:
1. Normalization backward pass
2. Softmax backward pass
"""
def normalize_backward(G_s_NM, s_MN, s_processed_NM, Ng_M, Ks_M, mask_NM, Ks):
    N, M = G_s_NM.shape

    pw_Ng = 1 << ((N - Ks - 1).bit_length())
    pw_Ks = 1 << ((max(1, N - (pw_Ng >> 1)) - 1).bit_length())

    grid = (M,)
    
    normalize_backward_kernel[grid](
        G_s_NM, s_MN, s_processed_NM, Ng_M, Ks_M, mask_NM,
        N, Ks, pw_Ng, pw_Ks,
        G_s_NM.stride(0), G_s_NM.stride(1),
        s_MN.stride(0), s_MN.stride(1),
        s_processed_NM.stride(0), s_processed_NM.stride(1),
        Ng_M.stride(0), Ks_M.stride(0),
        mask_NM.stride(0), mask_NM.stride(1)
    )

    return G_s_NM.T

class TopKRouterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_BSD, Wg_DN, biases_N, N, K, Ks):
        B, S, _ = x_BSD.shape
        x_MD = rearrange(x_BSD, 'B S D -> (B S) D')
        M = x_MD.shape[0]

        assert Wg_DN.dtype == torch.bfloat16, "Wg_DN must be of type torch.bfloat16"

        x_MD = x_MD.to(device=Wg_DN.device, dtype=torch.bfloat16)
        Wg_DN = Wg_DN.to(torch.bfloat16)

        s_raw_MN = x_MD @ Wg_DN
        s_MN = torch.empty((M, N), device=x_MD.device)
        s_MN[:, :N-Ks] = torch.softmax(s_raw_MN[:, :N-Ks], dim=-1, dtype=torch.float)
        s_MN[:, N-Ks:] = torch.softmax(s_raw_MN[:, N-Ks:], dim=-1, dtype=torch.float)

        s_MNg, s_MKs = s_MN[:, :N-Ks], s_MN[:, N-Ks:]
        values_MKg, indices_MKg = torch.topk(s_MNg + biases_N.to(x_MD.device)[:N-Ks][None, :], K - Ks, dim=-1)
        mask_MNg = torch.zeros_like(s_MNg).scatter_(1, indices_MKg, 1).bool()
        s_w_MNg = s_MNg * mask_MNg
        Ng_M = torch.sum(s_w_MNg, dim=-1)
        Ks_M = torch.sum(s_MKs, dim=-1)
        s_postsoft_MN = torch.cat([s_w_MNg, s_MKs], dim=-1)
        s_MN[:, :N-Ks] = s_w_MNg / (Ng_M[:, None])
        s_MN[:, N-Ks:] = s_MKs / (Ks_M[:, None])

        mask_NM = torch.ones([N, M], device=x_MD.device, dtype=torch.int32)
        mask_NM[:N-Ks, :] = rearrange(
            torch.scatter(
                torch.zeros([M, N-Ks], device=x_MD.device, dtype=torch.int32), 
                dim=-1, 
                index=indices_MKg, 
                src=torch.ones([M, N - Ks], device=x_MD.device, dtype=torch.int32)
            ), 
            'M N -> N M'
        )
        s_NM = rearrange(s_MN, 'M N -> N M')

        ctx.save_for_backward(x_MD, Wg_DN, mask_NM, s_postsoft_MN, s_NM, Ng_M, Ks_M)
        ctx.Ks = Ks
        return x_BSD, mask_NM, s_NM # returning inputs again so that autograd will work properly

    @staticmethod
    def backward(ctx, G_x_BSD, _, G_s_NM):
        x_MD, Wg_DN, mask_NM, s_postsoft_MN, s_NM, Ng_M, Ks_M = ctx.saved_tensors
        Ks = ctx.Ks

        B, S, _ = G_x_BSD.shape
        M, _ = x_MD.shape

        G_x_MD = rearrange(G_x_BSD, 'B S D -> (B S) D')

        G_s_raw_MN = normalize_backward(G_s_NM, s_postsoft_MN, s_NM, Ng_M, Ks_M, mask_NM, Ks).to(torch.bfloat16)

        # Gating network backwards (two matmuls and an add, hard to fuse/speed up)
        G_x_MD = torch.addmm(input=G_x_MD.to(torch.bfloat16), mat1=G_s_raw_MN, mat2=Wg_DN.T.to(torch.bfloat16))

        G_Wg_DN = x_MD.T @ G_s_raw_MN

        G_x_BSD = rearrange(G_x_MD, '(B S) D -> B S D', B=B, S=S)

        return G_x_BSD.to(G_s_NM.device), G_Wg_DN.to(Wg_DN.device), None, None, None, None


"""
A simple top-k router implementation using a combination of triton and torch.
"""
class TopKRouter(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        num_experts, 
        num_chosen_experts, 
        num_shared_experts,
        Wg_DN: torch.Tensor | None = None,
    ):
        super(TopKRouter, self).__init__()
        self.D = embedding_dim
        self.N = num_experts
        self.K = num_chosen_experts
        self.Ks = num_shared_experts

        if Wg_DN is None:
            self.Wg_DN = nn.Parameter(torch.randn((self.D, self.N), device="cuda", dtype=torch.bfloat16) / self.D, requires_grad=True)
        else:
            self.Wg_DN = nn.Parameter(Wg_DN.clone().detach().to(torch.bfloat16), requires_grad=True)

    def forward(self, x_BSD, biases_N):
        x_BSD, mask_NM, s_NM = TopKRouterFunction.apply(x_BSD, self.Wg_DN, biases_N, self.N, self.K, self.Ks)
        return x_BSD, mask_NM, s_NM
