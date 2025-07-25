import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange

"""
Dimension key:

B: batch size
S: sequence length
M: B * S
D: embedding dimension
N: number of experts
Ng: number of gating experts
Kg: selected number of gating experts
K: selected total number of experts
H: dimension of the hidden layer
"""


full_lin_autotune = [
    triton.Config({'BLOCK_SIZE_Mi': bmi, 'BLOCK_SIZE_H': bh, 'BLOCK_SIZE_D': bd, 'GROUP_SIZE_Mi': g}, num_stages=ns, num_warps=nw)
    for bmi in [32, 64, 128, 256]
    for bh in [32, 64, 128, 256]
    for bd in [32, 64, 128, 256]
    for g in [4, 8]
    for ns in [4, 5]
    for nw in [4, 8]
]


"""
Triton configs for linear 1 kernel, which are used in:
lin1_kernel, lin2_kernel, making_nice_scatter_kernel, swiglu_backward_kernel, lin1_backward_kernel
"""
lin_autotune = [
    triton.Config({'BLOCK_SIZE_Mi': 128, 'BLOCK_SIZE_H': 128, 'BLOCK_SIZE_D': 32, 'GROUP_SIZE_Mi': 4}, num_stages=5, num_warps=8),
    triton.Config({'BLOCK_SIZE_Mi': 128, 'BLOCK_SIZE_H': 128, 'BLOCK_SIZE_D': 32, 'GROUP_SIZE_Mi': 8}, num_stages=5, num_warps=8),
    triton.Config({'BLOCK_SIZE_Mi': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_D': 128, 'GROUP_SIZE_Mi': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_Mi': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_D': 128, 'GROUP_SIZE_Mi': 4}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_Mi': 128, 'BLOCK_SIZE_H': 128, 'BLOCK_SIZE_D': 64, 'GROUP_SIZE_Mi': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_Mi': 128, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_D': 128, 'GROUP_SIZE_Mi': 4}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_Mi': 128, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_D': 128, 'GROUP_SIZE_Mi': 8}, num_stages=5, num_warps=4),
    triton.Config({'BLOCK_SIZE_Mi': 128, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_D': 128, 'GROUP_SIZE_Mi': 4}, num_stages=5, num_warps=4),
]

"""
Triton configs for weights update kernel, which are used in:
weights_update_kernel
"""
weights_update_autotune = [
    triton.Config({'BLOCK_SIZE_Mi': 32, 'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 128, 'GROUP_SIZE_X': 8}, num_stages=5, num_warps=8),
    triton.Config({'BLOCK_SIZE_Mi': 64, 'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 128, 'GROUP_SIZE_X': 8}, num_stages=5, num_warps=4),
    triton.Config({'BLOCK_SIZE_Mi': 32, 'BLOCK_SIZE_X': 64, 'BLOCK_SIZE_Y': 128, 'GROUP_SIZE_X': 8}, num_stages=5, num_warps=4),
    triton.Config({'BLOCK_SIZE_Mi': 32, 'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 128, 'GROUP_SIZE_X': 4}, num_stages=5, num_warps=8),
]


"""
Triton kernel for the first part of our MoE forward pass, which fuses:
1. Gather right experts
2. Up projection and gate projection
3. SwiGLU
The output is stored in h_ptr_MsumH, which is an intermediate 
tensor which is used in the second part of our MoE.
"""

@triton.autotune(
    configs=lin_autotune,
    key=['Msum', 'H', 'D'],
)
@triton.jit
def lin1_kernel(
        x_ptr_MD, Wl1_ptr_ND2H, a_MsaveH, b_MsaveH, h_ptr_MsumH, idx_ptr_Msum, cumsums_N,
        M: tl.constexpr, D: tl.constexpr, Msum: tl.constexpr, Msave: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
        stride_xM, stride_xD,
        stride_WN, stride_WD, stride_W2H,
        stride_aMsave, stride_aH,
        stride_bMsave, stride_bH,
        stride_hMsum, stride_hH,
        stride_idxMsum,
        stride_cumsumsN,
        BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
        GROUP_SIZE_Mi: tl.constexpr, BLOCK_SIZE_Mi: tl.constexpr
):
    pid = tl.program_id(0)
    i = tl.program_id(1)

    cumsum = tl.load(cumsums_N + (i + 1) * stride_cumsumsN)
    Mi = cumsum - tl.load(cumsums_N + i * stride_cumsumsN)
    
    blocks_Mi = tl.cdiv(Mi, BLOCK_SIZE_Mi)
    blocks_H = tl.cdiv(H, BLOCK_SIZE_H)

    if pid >= blocks_Mi * blocks_H:
        return

    # Efficient block assignment
    group_id = pid // (GROUP_SIZE_Mi * blocks_H)
    start_block_Mi_in_group = group_id * GROUP_SIZE_Mi
    start_pid_in_group = start_block_Mi_in_group * blocks_H
    num_blocks_Mi_in_group = min(GROUP_SIZE_Mi, blocks_Mi - start_block_Mi_in_group)
    block_row = start_block_Mi_in_group + ((pid - start_pid_in_group) % num_blocks_Mi_in_group)
    block_col = (pid - start_pid_in_group) // num_blocks_Mi_in_group

    # Compute offsets
    offset_og_xMi = block_row * BLOCK_SIZE_Mi + tl.arange(0, BLOCK_SIZE_Mi)
    offset_xMi_mod = offset_og_xMi % Mi
    offset_xMi = offset_xMi_mod + (cumsum - Mi)
    offset_og_a_WH = block_col * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offset_a_WH = offset_og_a_WH % H
    offset_b_WH = offset_a_WH + H
    offset_D = tl.arange(0, BLOCK_SIZE_D)

    idx_ptrs_Mi = idx_ptr_Msum + offset_xMi * stride_idxMsum
    idx_vals_Mi = tl.load(idx_ptrs_Mi, mask=offset_og_xMi < Mi, other=0)
    x_ptrs_bMibD = x_ptr_MD + idx_vals_Mi[:, None] * stride_xM + offset_D[None, :] * stride_xD

    Wl1_ptrs_a_bDbH = Wl1_ptr_ND2H + i * stride_WN + offset_D[:, None] * stride_WD + offset_a_WH[None, :] * stride_W2H
    Wl1_ptrs_b_bDbH = Wl1_ptr_ND2H + i * stride_WN + offset_D[:, None] * stride_WD + offset_b_WH[None, :] * stride_W2H

    # Initialize accumulator
    acc_a_bMibH = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_H), dtype=tl.float32)
    acc_b_bMibH = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_H), dtype=tl.float32)
    for d in range(0, D, BLOCK_SIZE_D):
        x_bMibD = tl.load(x_ptrs_bMibD, mask=offset_D[None, :] < D - d, other=0.0)
        Wl1_a_bDbH = tl.load(Wl1_ptrs_a_bDbH, mask=offset_D[:, None] < D - d, other=0.0, cache_modifier=".cg")
        Wl1_b_bDbH = tl.load(Wl1_ptrs_b_bDbH, mask=offset_D[:, None] < D - d, other=0.0, cache_modifier=".cg")
        
        acc_a_bMibH = tl.dot(x_bMibD, Wl1_a_bDbH, acc_a_bMibH)
        acc_b_bMibH = tl.dot(x_bMibD, Wl1_b_bDbH, acc_b_bMibH)

        x_ptrs_bMibD += BLOCK_SIZE_D * stride_xD
        Wl1_ptrs_a_bDbH += BLOCK_SIZE_D * stride_WD
        Wl1_ptrs_b_bDbH += BLOCK_SIZE_D * stride_WD

    store_mask_bMibH = (offset_og_xMi[:, None] < Mi) & (offset_og_a_WH[None, :] < H)

    # Dynamic saving for backward pass
    store_mask_dynamic_bMibH = store_mask_bMibH & (offset_xMi[:, None] < Msave)
    a_ptrs_bMibH = a_MsaveH + offset_xMi[:, None] * stride_aMsave + offset_a_WH[None, :] * stride_aH
    b_ptrs_bMibH = b_MsaveH + offset_xMi[:, None] * stride_bMsave + offset_a_WH[None, :] * stride_bH
    tl.store(a_ptrs_bMibH, acc_a_bMibH.to(tl.bfloat16), mask=store_mask_dynamic_bMibH)
    tl.store(b_ptrs_bMibH, acc_b_bMibH.to(tl.bfloat16), mask=store_mask_dynamic_bMibH)

    # Apply SwiGLU
    acc_bMibH = (acc_b_bMibH * tl.sigmoid(acc_b_bMibH) * acc_a_bMibH).to(tl.bfloat16)

    # Store results
    h_ptrs_MiH = h_ptr_MsumH + offset_xMi[:, None] * stride_hMsum + offset_a_WH[None, :] * stride_hH
    tl.store(h_ptrs_MiH, acc_bMibH, mask=store_mask_bMibH)

"""
Triton kernel for the second part of our MoE forward pass, which fuses:
1. Down projection
2. Scatter into output
The output is stored in y_ptr_MD, which is expected to be a float32 tensor
(so that we can use the atomic add operation)
"""
@triton.autotune(
    configs=lin_autotune,
    key=['Msum', 'H', 'D'],
)
@triton.jit
def lin2_kernel(
        h_ptr_MsumH, Wl2_ptr_NHD, y_pre_MsaveD, y_ptr_KMD, s_ptr_NM, idx_ptr_Msum, which_ptr_Msum, cumsums_N,
        M: tl.constexpr, D: tl.constexpr, Msum: tl.constexpr, Msave: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
        stride_hMsum, stride_hH,
        stride_WN, stride_WH, stride_WD,
        stride_ypreMsave, stride_ypreD,
        stride_yK, stride_yM, stride_yD,
        stride_sN, stride_sM,
        stride_idxMsum, stride_whichMsum,
        stride_cumsumsN,
        BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
        GROUP_SIZE_Mi: tl.constexpr, BLOCK_SIZE_Mi: tl.constexpr
):
    pid = tl.program_id(0)
    i = tl.program_id(1)

    cumsum = tl.load(cumsums_N + (i + 1) * stride_cumsumsN)
    Mi = cumsum - tl.load(cumsums_N + i * stride_cumsumsN)
    
    blocks_Mi = tl.cdiv(Mi, BLOCK_SIZE_Mi)
    blocks_D = tl.cdiv(D, BLOCK_SIZE_D)

    if pid >= blocks_Mi * blocks_D:
        return

    # Efficient block assignment
    group_id = pid // (GROUP_SIZE_Mi * blocks_D)
    start_block_Mi_in_group = group_id * GROUP_SIZE_Mi
    start_pid_in_group = start_block_Mi_in_group * blocks_D
    num_blocks_Mi_in_group = min(GROUP_SIZE_Mi, blocks_Mi - start_block_Mi_in_group)
    block_row = start_block_Mi_in_group + ((pid - start_pid_in_group) % num_blocks_Mi_in_group)
    block_col = (pid - start_pid_in_group) // num_blocks_Mi_in_group

    # Compute offsets
    offset_og_xMi = block_row * BLOCK_SIZE_Mi + tl.arange(0, BLOCK_SIZE_Mi)
    offset_xMi_mod = offset_og_xMi % Mi
    offset_xMi = offset_xMi_mod + (cumsum - Mi)
    offset_og_WD = block_col * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offset_WD = offset_og_WD % D
    offset_H = tl.arange(0, BLOCK_SIZE_H)

    h_ptrs_bMibH = h_ptr_MsumH + offset_xMi[:, None] * stride_hMsum + offset_H[None, :] * stride_hH
    Wl2_ptrs_bHbD = Wl2_ptr_NHD + i * stride_WN + offset_H[:, None] * stride_WH + offset_WD[None, :] * stride_WD

    # Initialize accumulator
    acc_bMibD = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_D), dtype=tl.float32)
    
    for h in range(0, H, BLOCK_SIZE_H):
        h_bMibH = tl.load(h_ptrs_bMibH, mask=offset_H[None, :] < H - h, other=0.0)
        Wl2_bHbD = tl.load(Wl2_ptrs_bHbD, mask=offset_H[:, None] < H - h, other=0.0)
        
        acc_bMibD = tl.dot(h_bMibH, Wl2_bHbD, acc_bMibD)

        h_ptrs_bMibH += BLOCK_SIZE_H * stride_hH
        Wl2_ptrs_bHbD += BLOCK_SIZE_H * stride_WH

    store_mask_bMibD = (offset_og_xMi[:, None] < Mi) & (offset_og_WD[None, :] < D)

    # Dynamic saving for backward pass
    store_mask_dynamic_bMibD = store_mask_bMibD & (offset_xMi[:, None] < Msave)
    y_pre_ptrs_bMibD = y_pre_MsaveD + offset_xMi[:, None] * stride_ypreMsave + offset_WD[None, :] * stride_ypreD
    tl.store(y_pre_ptrs_bMibD, acc_bMibD.to(tl.bfloat16), mask=store_mask_dynamic_bMibD)

    # Store results
    idx_ptrs_bMi = idx_ptr_Msum + offset_xMi * stride_idxMsum
    which_ptrs_bMi = which_ptr_Msum + offset_xMi * stride_whichMsum
    idx_vals_bMi = tl.load(idx_ptrs_bMi, mask=offset_og_xMi < Mi, other=0)
    which_vals_bMi = tl.load(which_ptrs_bMi, mask=offset_og_xMi < Mi, other=0)
    y_ptrs_bMibD = y_ptr_KMD + (idx_vals_bMi * stride_yM + which_vals_bMi * stride_yK)[:, None] + offset_WD[None, :] * stride_yD
    s_ptrs_bMi = s_ptr_NM + i * stride_sN + idx_vals_bMi * stride_sM
    s_vals_bMi = tl.load(s_ptrs_bMi, mask=offset_og_xMi < Mi, other=0.0)
    tl.store(y_ptrs_bMibD, (acc_bMibD * s_vals_bMi[:, None]).to(tl.bfloat16), mask=store_mask_bMibD)

"""
Apply the experts to the input tensor x_MD, using the gating weights s_NM
Uses the 2 Triton kernels defined above to do the forward pass
Returns the output y_MD
"""
def apply_experts(x_MD, s_NM, idx_Msum, which_Msum, cumsums_N, Wl1_ND2H, Wl2_NHD, K, save_percent):
    
    M, D = x_MD.shape
    Msum = idx_Msum.shape[0]
    N, H, D = Wl2_NHD.shape

    # Dynamic saving for backward pass
    Msave = int(Msum * save_percent / 100.0)
    a_MsaveH = torch.empty((Msave, H), device=x_MD.device, dtype=torch.bfloat16)
    b_MsaveH = torch.empty((Msave, H), device=x_MD.device, dtype=torch.bfloat16)
    y_pre_MsaveD = torch.empty((Msave, D), device=x_MD.device, dtype=torch.bfloat16)

    h_MsumH = torch.empty((Msum, H), device=x_MD.device, dtype=torch.bfloat16) # storage for a * silu(b)
    y_KMD = torch.empty((K, M, D), device=x_MD.device, dtype=torch.bfloat16)

    def grid1(META):
        return (triton.cdiv(H, META['BLOCK_SIZE_H']) * triton.cdiv(M, META['BLOCK_SIZE_Mi']), N)
    def grid2(META):
        return (triton.cdiv(D, META['BLOCK_SIZE_D']) * triton.cdiv(M, META['BLOCK_SIZE_Mi']), N)

    # linear layer 1 + SwiGLU stored in an Msum x H matrix
    lin1_kernel[grid1](
        x_MD, Wl1_ND2H, a_MsaveH, b_MsaveH, h_MsumH, idx_Msum, cumsums_N,
        M, D, Msum, Msave, H, N,
        x_MD.stride(0), x_MD.stride(1),
        Wl1_ND2H.stride(0), Wl1_ND2H.stride(1), Wl1_ND2H.stride(2),
        a_MsaveH.stride(0), a_MsaveH.stride(1),
        b_MsaveH.stride(0), b_MsaveH.stride(1),
        h_MsumH.stride(0), h_MsumH.stride(1),
        idx_Msum.stride(0),
        cumsums_N.stride(0),
    )

    # linear layer 2 + scatter into y_MD, properly multiplied by gating weights
    lin2_kernel[grid2](
        h_MsumH, Wl2_NHD, y_pre_MsaveD, y_KMD, s_NM, idx_Msum, which_Msum, cumsums_N,
        M, D, Msum, Msave, H, N,
        h_MsumH.stride(0), h_MsumH.stride(1),
        Wl2_NHD.stride(0), Wl2_NHD.stride(1), Wl2_NHD.stride(2),
        y_pre_MsaveD.stride(0), y_pre_MsaveD.stride(1),
        y_KMD.stride(0), y_KMD.stride(1), y_KMD.stride(2),
        s_NM.stride(0), s_NM.stride(1),
        idx_Msum.stride(0), which_Msum.stride(0),
        cumsums_N.stride(0),
    )

    return y_KMD.sum(dim=0).to(x_MD.dtype), a_MsaveH, b_MsaveH, y_pre_MsaveD


"""
Triton kernel for the backward pass of the MoE, which just performs necessary gathers
This is convenient because the backward pass can avoid unnecessary gather operations,
which used to be the bottleneck of the backward pass.
"""
@triton.jit
def making_nice_scatter_kernel(
        x_ptr_MD, x_ptr_MsumD, G_y_ptr_MD, G_y_ptr_MsumD, idx_ptr_Msum, cumsums_ptr_N, s_ptr_MN,
        N: tl.constexpr, Msum: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
        stride_xM, stride_xD,
        stride_xMsum, stride_xD2,
        stride_GyM, stride_GyD,
        stride_GyMsum, stride_GyD2,
        stride_idxMsum,
        stride_cumsumsN,
        stride_sM, stride_sN,
        BLOCK_SIZE_H: tl.constexpr = 0, BLOCK_SIZE_D: tl.constexpr = 512,
        GROUP_SIZE_Mi: tl.constexpr = 0, BLOCK_SIZE_Mi: tl.constexpr = 64
):
    block_col = tl.program_id(0)
    block_row = tl.program_id(1)
    i = tl.program_id(2)

    cumsum = tl.load(cumsums_ptr_N + (i + 1) * stride_cumsumsN)
    Mi = cumsum - tl.load(cumsums_ptr_N + i * stride_cumsumsN)

    blocks_Mi = tl.cdiv(Mi, BLOCK_SIZE_Mi)

    if block_row >= blocks_Mi:
        return

    # Compute offsets
    offset_og_Mi = block_row * BLOCK_SIZE_Mi + tl.arange(0, BLOCK_SIZE_Mi)
    offset_Mi = (offset_og_Mi % Mi) + (cumsum - Mi)
    offset_og_D = block_col * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offset_D = offset_og_D % D
    
    # Load G_y and s
    idx_ptrs_bMi = idx_ptr_Msum + offset_Mi * stride_idxMsum
    idx_vals_bMi = tl.load(idx_ptrs_bMi, mask=offset_og_Mi < Mi, other=0)
    G_y_ptrs_bMibD = G_y_ptr_MD + idx_vals_bMi[:, None] * stride_GyM + offset_D[None, :] * stride_GyD
    G_y_bMibD = tl.load(G_y_ptrs_bMibD, mask=offset_og_Mi[:, None] < Mi, other=0.0)

    s_ptrs_bMi = s_ptr_MN + i * stride_sN + idx_vals_bMi * stride_sM
    s_bMi = tl.load(s_ptrs_bMi, mask=offset_og_Mi < Mi, other=0.0)

    # Compute and store
    G_y_bMibD = (G_y_bMibD * s_bMi[:, None]).to(tl.bfloat16)
    G_y_ptrs_bMibD2 = G_y_ptr_MsumD + offset_Mi[:, None] * stride_GyMsum + offset_D[None, :] * stride_GyD2
    tl.store(G_y_ptrs_bMibD2, G_y_bMibD, mask=(offset_og_Mi[:, None] < Mi) & (offset_og_D[None, :] < D))

    # Now for x
    x_ptrs_bMibD = x_ptr_MD + idx_vals_bMi[:, None] * stride_xM + offset_D[None, :] * stride_xD
    x_bMibD = tl.load(x_ptrs_bMibD, mask=offset_og_Mi[:, None] < Mi, other=0.0)

    # Store
    x_ptrs_bMibD2 = x_ptr_MsumD + offset_Mi[:, None] * stride_xMsum + offset_D[None, :] * stride_xD2
    tl.store(x_ptrs_bMibD2, x_bMibD, mask=(offset_og_Mi[:, None] < Mi) & (offset_og_D[None, :] < D))
    
"""
Triton kernel for the backward pass of the MoE, which updates the weights of either
linear 1 or linear 2, depending on the input tensors
"""
@triton.autotune(
    configs=weights_update_autotune,
    key=['Msum', 'X', 'Y']
)
@triton.jit
def weights_update_kernel(
        G_W_ptr_NXY, G_b_ptr_MsumY, a_ptr_MsumX, cumsums_ptr_N,
        N: tl.constexpr, X: tl.constexpr, Y: tl.constexpr, Msum: tl.constexpr,
        stride_WN, stride_WX, stride_WY,
        stride_bMsum, stride_bY,
        stride_aMsum, stride_aX,
        stride_cumsumsN,
        BLOCK_SIZE_Mi: tl.constexpr, BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr,
        GROUP_SIZE_X: tl.constexpr
):
    pid = tl.program_id(0) # block id within expert
    i = tl.program_id(1) # expert id

    blocks_X = tl.cdiv(X, BLOCK_SIZE_X)
    blocks_Y = tl.cdiv(Y, BLOCK_SIZE_Y)

    cumsum = tl.load(cumsums_ptr_N + (i + 1) * stride_cumsumsN)
    Mi = cumsum - tl.load(cumsums_ptr_N + i * stride_cumsumsN)

    # Block assignment
    group_id = pid // (GROUP_SIZE_X * blocks_Y)
    start_block_X_in_group = group_id * GROUP_SIZE_X
    start_pid_in_group = start_block_X_in_group * blocks_Y
    num_blocks_X_in_group = min(GROUP_SIZE_X, blocks_X - start_block_X_in_group)
    block_row = start_block_X_in_group + ((pid - start_pid_in_group) % num_blocks_X_in_group)
    block_col = (pid - start_pid_in_group) // num_blocks_X_in_group

    # Compute offsets
    offset_og_X = block_row * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offset_X = offset_og_X % X
    offset_og_Y = block_col * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offset_Y = offset_og_Y % Y
    offset_Mi = tl.arange(0, BLOCK_SIZE_Mi) + cumsum - Mi

    a_ptrs_bXbMi = a_ptr_MsumX + offset_Mi[None, :] * stride_aMsum + offset_X[:, None] * stride_aX
    G_b_ptrs_bMibY = G_b_ptr_MsumY + offset_Mi[:, None] * stride_bMsum + offset_Y[None, :] * stride_bY

    # Initialize accumulator
    acc_bXbY = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_Y), dtype=tl.float32)
    
    # Computation loop
    for mi in range(0, Mi, BLOCK_SIZE_Mi):
        a_bXbMi = tl.load(a_ptrs_bXbMi, mask=offset_Mi[None, :] < cumsum - mi, other=0.0)
        G_b_bMibY = tl.load(G_b_ptrs_bMibY, mask=offset_Mi[:, None] < cumsum - mi, other=0.0)
        acc_bXbY = tl.dot(a_bXbMi, G_b_bMibY, acc_bXbY)
        a_ptrs_bXbMi += BLOCK_SIZE_Mi * stride_aMsum
        G_b_ptrs_bMibY += BLOCK_SIZE_Mi * stride_bMsum
            
    # Store results
    store_mask_bXbY = (offset_og_X[:, None] < X) & (offset_og_Y[None, :] < Y)
    G_W_ptrs_bXbY = G_W_ptr_NXY + i * stride_WN + offset_X[:, None] * stride_WX + offset_Y[None, :] * stride_WY
    tl.store(G_W_ptrs_bXbY, acc_bXbY, mask=store_mask_bXbY)

"""
Triton kernel for the backward pass of the MoE, which fuses:
1. Linear 2 backward pass
2. SwiGLU backward pass
It also recomputes the linear 1 part of the forward pass, as to avoid
saving the intermediate results of the forward pass.
"""
@triton.autotune(
    configs=lin_autotune,
    key=['Msum', 'H', 'D'],
)
@triton.jit
def swiglu_backward_kernel(
        x_ptr_MsumD, a_MsaveH, b_MsaveH, Wl1_ptr_ND2H, G_z_ptr_Msum2H, G_y_ptr_MsumD, Wl2_ptr_NHD, h_ptr_MsumH, cumsums_N,
        M: tl.constexpr, N: tl.constexpr, Msum: tl.constexpr, Msave: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        stride_xMsum, stride_xD,
        stride_aMsave, stride_aH,
        stride_bMsave, stride_bH,
        stride_W1N, stride_W1D, stride_W12H,
        stride_zMsum, stride_z2H,
        stride_yMsum, stride_yD,
        stride_W2N, stride_W2H, stride_W2D,
        stride_hMsum, stride_hH,
        stride_cumsumsN,
        BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
        GROUP_SIZE_Mi: tl.constexpr, BLOCK_SIZE_Mi: tl.constexpr
):
    pid = tl.program_id(0)
    i = tl.program_id(1)

    cumsum = tl.load(cumsums_N + (i + 1) * stride_cumsumsN)
    Mi = cumsum - tl.load(cumsums_N + i * stride_cumsumsN)

    blocks_Mi = tl.cdiv(Mi, BLOCK_SIZE_Mi)
    blocks_H = tl.cdiv(H, BLOCK_SIZE_H)

    if pid >= blocks_Mi * blocks_H:
        return

    # Efficient block assignment
    group_id = pid // (GROUP_SIZE_Mi * blocks_H)
    start_block_Mi_in_group = group_id * GROUP_SIZE_Mi
    start_pid_in_group = start_block_Mi_in_group * blocks_H
    num_blocks_Mi_in_group = min(GROUP_SIZE_Mi, blocks_Mi - start_block_Mi_in_group)
    block_row = start_block_Mi_in_group + ((pid - start_pid_in_group) % num_blocks_Mi_in_group)
    block_col = (pid - start_pid_in_group) // num_blocks_Mi_in_group

    # Compute offsets
    offset_og_Mi = block_row * BLOCK_SIZE_Mi + tl.arange(0, BLOCK_SIZE_Mi)
    offset_Mi_mod = offset_og_Mi % Mi
    offset_Mi = offset_Mi_mod + (cumsum - Mi)
    offset_og_H = block_col * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offset_H = offset_og_H % H
    offset_D = tl.arange(0, BLOCK_SIZE_D)

    # Pointers for the matmul 
    G_y_ptrs_bMibD = G_y_ptr_MsumD + offset_Mi[:, None] * stride_yMsum + offset_D[None, :] * stride_yD    
    Wl2_ptrs_bDbH = Wl2_ptr_NHD + i * stride_W2N + offset_H[None, :] * stride_W2H + offset_D[:, None] * stride_W2D

    # Initialize accumulators
    G_h_bMibH = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_H), dtype=tl.float32)
    for d in range(0, D, BLOCK_SIZE_D):
        G_y_bMibD = tl.load(G_y_ptrs_bMibD, mask=offset_D[None, :] < D - d, other=0.0)
        Wl2_bDbH = tl.load(Wl2_ptrs_bDbH, mask=offset_D[:, None] < D - d, other=0.0, cache_modifier=".cg")
        G_h_bMibH = tl.dot(G_y_bMibD, Wl2_bDbH, G_h_bMibH)
        G_y_ptrs_bMibD += BLOCK_SIZE_D * stride_yD
        Wl2_ptrs_bDbH += BLOCK_SIZE_D * stride_W2D

    # Now we are ready to do SwiGLU backwards (for which we need to recompute lin1 or read from saved activations)
    a_bMibH = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_H), dtype=tl.float32)
    b_bMibH = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_H), dtype=tl.float32)

    if block_row * BLOCK_SIZE_Mi + (cumsum - Mi) + BLOCK_SIZE_Mi <= Msave:
        read_mask_dynamic_bMibH = (offset_og_Mi[:, None] < Mi) & (offset_og_H[None, :] < H)
        a_ptrs_bMibH = a_MsaveH + offset_Mi[:, None] * stride_aMsave + offset_H[None, :] * stride_aH
        b_ptrs_bMibH = b_MsaveH + offset_Mi[:, None] * stride_bMsave + offset_H[None, :] * stride_bH
        a_bMibH = tl.load(a_ptrs_bMibH, mask=read_mask_dynamic_bMibH, other=0.0).to(tl.float32)
        b_bMibH = tl.load(b_ptrs_bMibH, mask=read_mask_dynamic_bMibH, other=0.0).to(tl.float32)
    else:
        offset_a_WH = offset_H
        offset_b_WH = offset_H + H

        x_ptrs_bMibD = x_ptr_MsumD + offset_Mi[:, None] * stride_xMsum + offset_D[None, :] * stride_xD

        Wl1_ptrs_a_bDbH = Wl1_ptr_ND2H + i * stride_W1N + offset_D[:, None] * stride_W1D + offset_a_WH[None, :] * stride_W12H
        Wl1_ptrs_b_bDbH = Wl1_ptr_ND2H + i * stride_W1N + offset_D[:, None] * stride_W1D + offset_b_WH[None, :] * stride_W12H

        # Initialize accumulator
        a_bMibH = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_H), dtype=tl.float32)
        b_bMibH = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_H), dtype=tl.float32)
        
        for d in range(0, D, BLOCK_SIZE_D):
            x_bMibD = tl.load(x_ptrs_bMibD, mask=offset_D[None, :] < D - d, other=0.0)
            Wl1_a_bDbH = tl.load(Wl1_ptrs_a_bDbH, mask=offset_D[:, None] < D - d, other=0.0, cache_modifier=".cg")
            Wl1_b_bDbH = tl.load(Wl1_ptrs_b_bDbH, mask=offset_D[:, None] < D - d, other=0.0, cache_modifier=".cg")
            
            a_bMibH = tl.dot(x_bMibD, Wl1_a_bDbH, a_bMibH)
            b_bMibH = tl.dot(x_bMibD, Wl1_b_bDbH, b_bMibH)

            x_ptrs_bMibD += BLOCK_SIZE_D * stride_xD
            Wl1_ptrs_a_bDbH += BLOCK_SIZE_D * stride_W1D
            Wl1_ptrs_b_bDbH += BLOCK_SIZE_D * stride_W1D

    mask_bMibH = (offset_og_Mi[:, None] < Mi) & (offset_og_H[None, :] < H)

    # Do the SwiGLU backwards part
    G_z1_ptrs_bMibH = G_z_ptr_Msum2H + offset_Mi[:, None] * stride_zMsum + offset_H[None, :] * stride_z2H
    G_z2_ptrs_bMibH = G_z1_ptrs_bMibH + H * stride_z2H

    # Compute everything
    sig_b_bMibH = tl.sigmoid(b_bMibH)
    z1_bMibH = G_h_bMibH * b_bMibH * sig_b_bMibH
    z2_bMibH = G_h_bMibH * a_bMibH * sig_b_bMibH * (1 + b_bMibH * (1 - sig_b_bMibH))

    # Store results
    tl.store(G_z1_ptrs_bMibH, z1_bMibH, mask=mask_bMibH)
    tl.store(G_z2_ptrs_bMibH, z2_bMibH, mask=mask_bMibH)

    # Apply SwiGLU and store
    h_bMibH = (a_bMibH * b_bMibH * sig_b_bMibH)
    h_ptrs_bMibH = h_ptr_MsumH + offset_Mi[:, None] * stride_hMsum + offset_H[None, :] * stride_hH
    tl.store(h_ptrs_bMibH, h_bMibH, mask=mask_bMibH)


"""
Performs the first part of the backward pass of the MoE, which fuses:
1. Scatter
2. Linear 2 backward pass
3. SwiGLU backward pass
"""
def lin2_and_swiglu_backward(G_y_MD, x_MD, a_MsaveH, b_MsaveH, Wl1_ND2H, Wl2_NHD, idx_Msum, cumsums_N, s_MN, save_percent):
    M, D = G_y_MD.shape
    N, H, D = Wl2_NHD.shape
    Msum = idx_Msum.shape[0]
    Msave = int(Msum * save_percent / 100.0)
    
    G_Wl2_NHD = torch.empty_like(Wl2_NHD)

    G_y_MsumD = torch.empty((Msum, D), device=G_y_MD.device, dtype=torch.bfloat16)
    x_MsumD = torch.empty((Msum, D), device=G_y_MD.device, dtype=torch.bfloat16)

    G_z_Msum2H = torch.empty((Msum, 2*H), device=G_y_MD.device, dtype=torch.bfloat16)
    h_MsumH = torch.empty((Msum, H), device=G_y_MD.device, dtype=torch.bfloat16)

    def grid0(META):
        return (triton.cdiv(D, META['BLOCK_SIZE_D']), triton.cdiv(M, META['BLOCK_SIZE_Mi']), N)
    def grid1(META):
        return (triton.cdiv(H, META['BLOCK_SIZE_X']) * triton.cdiv(D, META['BLOCK_SIZE_Y']), N)
    def grid2(META):
        return (triton.cdiv(H, META['BLOCK_SIZE_H']) * triton.cdiv(M, META['BLOCK_SIZE_Mi']), N)

    making_nice_scatter_kernel[grid0](
        x_MD, x_MsumD, G_y_MD, G_y_MsumD, idx_Msum, cumsums_N, s_MN,
        N, Msum, D, H,
        x_MD.stride(0), x_MD.stride(1),
        x_MsumD.stride(0), x_MsumD.stride(1),
        G_y_MD.stride(0), G_y_MD.stride(1),
        G_y_MsumD.stride(0), G_y_MsumD.stride(1),
        idx_Msum.stride(0),
        cumsums_N.stride(0),
        s_MN.stride(0), s_MN.stride(1),
        num_stages=3, num_warps=8
    )

    swiglu_backward_kernel[grid2](
        x_MsumD, a_MsaveH, b_MsaveH, Wl1_ND2H, G_z_Msum2H, G_y_MsumD, Wl2_NHD, h_MsumH, cumsums_N,
        M, N, Msum, Msave, H, D,
        x_MsumD.stride(0), x_MsumD.stride(1),
        a_MsaveH.stride(0), a_MsaveH.stride(1),
        b_MsaveH.stride(0), b_MsaveH.stride(1),
        Wl1_ND2H.stride(0), Wl1_ND2H.stride(1), Wl1_ND2H.stride(2),
        G_z_Msum2H.stride(0), G_z_Msum2H.stride(1),
        G_y_MsumD.stride(0), G_y_MsumD.stride(1),
        Wl2_NHD.stride(0), Wl2_NHD.stride(1), Wl2_NHD.stride(2),
        h_MsumH.stride(0), h_MsumH.stride(1),
        cumsums_N.stride(0),
    )

    # update weights of linear 2
    weights_update_kernel[grid1](
        G_Wl2_NHD, G_y_MsumD, h_MsumH, cumsums_N,
        N, H, D, Msum,
        Wl2_NHD.stride(0), Wl2_NHD.stride(1), Wl2_NHD.stride(2),
        G_y_MsumD.stride(0), G_y_MsumD.stride(1),
        h_MsumH.stride(0), h_MsumH.stride(1),
        cumsums_N.stride(0),
    )

    return G_Wl2_NHD, G_z_Msum2H, x_MsumD, h_MsumH

"""
Triton kernel for the backward pass of the MoE, which fuses:
1. Linear 1 backward pass
2. Scatter
The outputs are stored in G_x_MD and G_s_NM, which are the gradients of the input
and the gate activations, respectively. These are expected to be float32 tensors
so that we can use the atomic add operation.
"""
@triton.autotune(
    configs=lin_autotune,
    key=['Msum', 'H', 'D'],
    reset_to_zero=['G_s_ptr_NM']
)
@triton.jit
def lin1_backward_kernel(
        G_z_ptr_Msum2H, y_pre_MsaveD, Wl1_ptr_ND2H, Wl2_ptr_NHD, h_ptr_MsumH, G_x_ptr_KMD, G_s_ptr_NM, G_y_ptr_MD, idx_ptr_Msum, which_ptr_Msum, cumsums_N,
        M: tl.constexpr, D: tl.constexpr, Msum: tl.constexpr, Msave: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
        stride_zMsum, stride_z2H,
        stride_ypreMsave, stride_ypreD,
        stride_W1N, stride_W1D, stride_W12H,
        stride_W2N, stride_W2H, stride_W2D,
        stride_hMsum, stride_hH,
        stride_GxK, stride_GxM, stride_GxD,
        stride_GsN, stride_GsM,
        stride_GyM, stride_GyD,
        stride_idxMsum, stride_whichMsum,
        stride_cumsumsN,
        BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
        GROUP_SIZE_Mi: tl.constexpr, BLOCK_SIZE_Mi: tl.constexpr
):
    pid = tl.program_id(0)
    i = tl.program_id(1)
    
    cumsum = tl.load(cumsums_N + (i + 1) * stride_cumsumsN)
    Mi = cumsum - tl.load(cumsums_N + i * stride_cumsumsN)
    
    blocks_Mi = tl.cdiv(Mi, BLOCK_SIZE_Mi)
    blocks_D = tl.cdiv(D, BLOCK_SIZE_D)

    if pid >= blocks_Mi * blocks_D:
        return

    # Efficient block assignment
    group_id = pid // (GROUP_SIZE_Mi * blocks_D)
    start_block_Mi_in_group = group_id * GROUP_SIZE_Mi
    start_pid_in_group = start_block_Mi_in_group * blocks_D
    num_blocks_Mi_in_group = min(GROUP_SIZE_Mi, blocks_Mi - start_block_Mi_in_group)
    block_row = start_block_Mi_in_group + ((pid - start_pid_in_group) % num_blocks_Mi_in_group)
    block_col = (pid - start_pid_in_group) // num_blocks_Mi_in_group

    # Compute offsets
    offset_og_xMi = block_row * BLOCK_SIZE_Mi + tl.arange(0, BLOCK_SIZE_Mi)
    offset_xMi_mod = offset_og_xMi % Mi
    offset_xMi = offset_xMi_mod + (cumsum - Mi)
    offset_og_WD = block_col * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offset_WD = offset_og_WD % D
    offset_H = tl.arange(0, BLOCK_SIZE_H)

    G_z_ptrs_bMibH = G_z_ptr_Msum2H + offset_xMi[:, None] * stride_zMsum + offset_H[None, :] * stride_z2H
    Wl1_ptrs_bHbD = Wl1_ptr_ND2H + i * stride_W1N + offset_H[:, None] * stride_W12H + offset_WD[None, :] * stride_W1D

    # Initialize accumulator
    acc_bMibD = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_D), dtype=tl.float32)
    for h in range(0, 2*H, BLOCK_SIZE_H):
        G_z_bMibH = tl.load(G_z_ptrs_bMibH, mask=offset_H[None, :] < 2*H - h, other=0.0)
        Wl1_bHbD = tl.load(Wl1_ptrs_bHbD, mask=offset_H[:, None] < 2*H - h, other=0.0)
        acc_bMibD = tl.dot(G_z_bMibH, Wl1_bHbD, acc_bMibD)
        G_z_ptrs_bMibH += BLOCK_SIZE_H * stride_z2H
        Wl1_ptrs_bHbD += BLOCK_SIZE_H * stride_W12H

    # Store results
    idx_ptrs_bMi = idx_ptr_Msum + offset_xMi * stride_idxMsum
    which_ptrs_bMi = which_ptr_Msum + offset_xMi * stride_whichMsum
    idx_vals_bMi = tl.load(idx_ptrs_bMi, mask=offset_og_xMi < Mi, other=0)
    which_vals_bMi = tl.load(which_ptrs_bMi, mask=offset_og_xMi < Mi, other=0)
    G_x_ptrs_bMibD = G_x_ptr_KMD + (idx_vals_bMi * stride_GxM + which_vals_bMi * stride_GxK)[:, None] + offset_WD[None, :] * stride_GxD
    mask_bMibD = (offset_og_xMi[:, None] < Mi) & (offset_og_WD[None, :] < D)
    tl.store(G_x_ptrs_bMibD, acc_bMibD, mask=mask_bMibD)

    # Get pre-gate activations either by dynamic loading or by recomputation
    y_bMibD = tl.zeros((BLOCK_SIZE_Mi, BLOCK_SIZE_D), dtype=tl.float32)
    if block_row * BLOCK_SIZE_Mi + (cumsum - Mi) + BLOCK_SIZE_Mi <= Msave:
        read_mask_dynamic_bMibD = (offset_og_xMi[:, None] < Mi) & (offset_og_WD[None, :] < D)
        y_pre_ptrs_bMibD = y_pre_MsaveD + offset_xMi[:, None] * stride_ypreMsave + offset_WD[None, :] * stride_ypreD
        y_bMibD = tl.load(y_pre_ptrs_bMibD, mask=read_mask_dynamic_bMibD, other=0.0).to(tl.float32)
    else:
        h_ptrs_bMibH = h_ptr_MsumH + offset_xMi[:, None] * stride_hMsum + offset_H[None, :] * stride_hH
        Wl2_ptrs_bHbD = Wl2_ptr_NHD + i * stride_W2N + offset_H[:, None] * stride_W2H + offset_WD[None, :] * stride_W2D
        for h in range(0, H, BLOCK_SIZE_H):
            h_bMibH = tl.load(h_ptrs_bMibH, mask=offset_H[None, :] < H - h, other=0.0)
            Wl2_bHbD = tl.load(Wl2_ptrs_bHbD, mask=offset_H[:, None] < H - h, other=0.0)
            y_bMibD = tl.dot(h_bMibH, Wl2_bHbD, y_bMibD)
            h_ptrs_bMibH += BLOCK_SIZE_H * stride_hH
            Wl2_ptrs_bHbD += BLOCK_SIZE_H * stride_W2H

    G_s_ptrs_bMi = G_s_ptr_NM + i * stride_GsN + idx_vals_bMi * stride_GsM
    G_y_ptrs_bMibD = G_y_ptr_MD + idx_vals_bMi[:, None] * stride_GyM + offset_WD[None, :] * stride_GyD
    G_y_bMibD = tl.load(G_y_ptrs_bMibD, mask=mask_bMibD, other=0.0).to(tl.float32)
    G_s_bMi = tl.sum(G_y_bMibD * y_bMibD, axis=1)
    _ = tl.atomic_add(G_s_ptrs_bMi, G_s_bMi, mask=(offset_og_xMi < Mi), sem="relaxed")

"""
Performs the second part of the backward pass of the MoE, which fuses:
1. Linear 1 backward pass
2. Scatter
The outputs are stored in G_x_MD and G_s_NM, which are the gradients of the input
and the gate activations, respectively. These are expected to be float32 tensors
so that we can use the atomic add operation.
"""
def lin1_backward(x_MsumD, y_pre_MsaveD, G_z_Msum2H, Wl1_ND2H, Wl2_NHD, h_MsumH, idx_Msum, which_Msum, cumsums_N, G_y_MD, K, save_percent):
    M, D = G_y_MD.shape
    N, D, H2 = Wl1_ND2H.shape
    H = H2 // 2
    Msum = idx_Msum.shape[0]
    Msave = int(Msum * save_percent / 100.0)

    G_Wl1_ND2H = torch.empty_like(Wl1_ND2H)
    G_s_NM = torch.zeros((N, M), device=G_y_MD.device, dtype=torch.float32)
    G_x_KMD = torch.empty((K, M, D), device=G_y_MD.device, dtype=torch.bfloat16)

    def grid1(META):
        return (triton.cdiv(D, META['BLOCK_SIZE_X']) * triton.cdiv(2 * H, META['BLOCK_SIZE_Y']), N)
    def grid2(META):
        return (triton.cdiv(D, META['BLOCK_SIZE_D']) * triton.cdiv(M, META['BLOCK_SIZE_Mi']), N)

    # update weights of linear 1
    weights_update_kernel[grid1](
        G_Wl1_ND2H, G_z_Msum2H, x_MsumD, cumsums_N,
        N, D, 2*H, Msum,
        G_Wl1_ND2H.stride(0), G_Wl1_ND2H.stride(1), G_Wl1_ND2H.stride(2),
        G_z_Msum2H.stride(0), G_z_Msum2H.stride(1),
        x_MsumD.stride(0), x_MsumD.stride(1),
        cumsums_N.stride(0),
    )

    lin1_backward_kernel[grid2](
        G_z_Msum2H, y_pre_MsaveD, Wl1_ND2H, Wl2_NHD, h_MsumH, G_x_KMD, G_s_NM, G_y_MD, idx_Msum, which_Msum, cumsums_N,
        M, D, Msum, Msave, H, N,
        G_z_Msum2H.stride(0), G_z_Msum2H.stride(1),
        y_pre_MsaveD.stride(0), y_pre_MsaveD.stride(1),
        Wl1_ND2H.stride(0), Wl1_ND2H.stride(1), Wl1_ND2H.stride(2),
        Wl2_NHD.stride(0), Wl2_NHD.stride(1), Wl2_NHD.stride(2),
        h_MsumH.stride(0), h_MsumH.stride(1),
        G_x_KMD.stride(0), G_x_KMD.stride(1), G_x_KMD.stride(2),
        G_s_NM.stride(0), G_s_NM.stride(1),
        G_y_MD.stride(0), G_y_MD.stride(1),
        idx_Msum.stride(0), which_Msum.stride(0),
        cumsums_N.stride(0),
    )

    return G_Wl1_ND2H, G_s_NM, G_x_KMD.sum(dim=0)
 
"""
Triton kernel for the forward pass of the MoE, which computes the per-row counts
of the number of experts used in each row.
"""
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1024}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 2048}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 512}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256}, num_stages=5, num_warps=4),

        triton.Config({'BLOCK_SIZE_M': 1024}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 2048}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 512}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256}, num_stages=5, num_warps=8),

        triton.Config({'BLOCK_SIZE_M': 4096}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 8192}, num_stages=3, num_warps=4),
    ],
    key=['M'],
    reset_to_zero=['counts2_ptr']
)
@triton.jit
def scatter_cols_kernel(
    mask_ptr, cnt_ptr, crow_ptr, col_ptr, which_Msum_ptr, counts2_ptr,
    N: tl.constexpr, M: tl.constexpr, N2: tl.constexpr, M2: tl.constexpr, tot,
    stride_mask_row, stride_mask_col,     
    stride_cnt,  
    stride_crow,          
    stride_col,             
    stride_which_Msum,
    stride_counts2,
    BLOCK_SIZE_M: tl.constexpr,
):
    # for row in range(N):
    row = tl.program_id(0)
    if row == 0:
        tl.store(crow_ptr + N * stride_crow, tot)

    # find prefix sum:
    rng = tl.arange(0, N2)
    cnts = tl.load(cnt_ptr + rng * stride_crow, mask=rng < row, other=0)
    base = tl.sum(cnts)
    tl.store(crow_ptr + row * stride_crow, base)

    # running total of how many trues we’ve seen so far in this row
    running = tl.zeros([1], tl.int32)
    offs = tl.arange(0, BLOCK_SIZE_M)

    # loop over the row in tiles of size BLOCK
    for start in range(0, M, BLOCK_SIZE_M):
        m = tl.load(mask_ptr + row * stride_mask_row + offs * stride_mask_col, mask=offs < M, other=0)
        prefix = tl.cumsum(m, axis=0) - m
        ptrs = col_ptr + (base + running + prefix) * stride_col
        tl.store(ptrs, offs, mask=m != 0)
        safe_offs = offs % M
        a = tl.atomic_add(counts2_ptr + safe_offs * stride_counts2, 1, mask=(offs < M) & (m != 0))
        tl.store(which_Msum_ptr + (base + running + prefix) * stride_which_Msum, a, mask=m != 0)
        running += tl.sum(m, axis=0)
        offs += BLOCK_SIZE_M

"""
Useful function for the forward pass which effectively replaces torch.nonzero
with a couple Triton kernels.
"""
def find_used_experts(mask_NM, K):
    N, M = mask_NM.shape
    # Set M to smallest power of 2 greater than M
    M2 = 2**(M-1).bit_length()
    N2 = 2**(N-1).bit_length()

    # Host-side
    cumsums_N = torch.empty((N+1,), dtype=torch.int32, device=mask_NM.device)

    counts = torch.sum(mask_NM, dim=1)
    tot = M * K
    col_idx = torch.empty((tot,), dtype=torch.int32, device=mask_NM.device)
    which_Msum = torch.empty((tot,), dtype=torch.int32, device=mask_NM.device)
    counts2 = torch.zeros((M,), dtype=torch.int32, device=mask_NM.device)
    scatter_cols_kernel[(N,)](
        mask_NM, counts, cumsums_N, col_idx, which_Msum, counts2,
        N, M, N2, M2, tot,
        mask_NM.stride(0), mask_NM.stride(1),
        counts.stride(0), cumsums_N.stride(0), col_idx.stride(0), which_Msum.stride(0), counts2.stride(0)
    )

    return cumsums_N, col_idx, counts, which_Msum

"""
Main class for the MoE forward and backward pass, implemented as a torch.autograd.Function
"""
class MoEFunction(torch.autograd.Function):
    """
    Performs the forward pass of the MoE, which combines:
    1. Routing + Normalization + Softmax
    2. Apply experts
    3. Update expert biases
    4. Save necessary information for the backward pass
    """
    @staticmethod
    def forward(ctx, x_BSD, Wl1_ND2H, Wl2_NHD, mask_NM, s_NM, K, save_percent):
        B, S, _ = x_BSD.shape

        x_MD = rearrange(x_BSD, 'B S D -> (B S) D')

        assert Wl1_ND2H.device == Wl2_NHD.device, "All weights must be on the same device, but got: Wl1_ND2H.device = {}, Wl2_NHD.device = {}".format(Wl1_ND2H.device, Wl2_NHD.device)

        x_MD = x_MD.to(device=Wl1_ND2H.device, dtype=torch.bfloat16)
        Wl1_ND2H = Wl1_ND2H.to(torch.bfloat16)
        Wl2_NHD = Wl2_NHD.to(torch.bfloat16)

        cumsums_N, idx_Msum, sizes_N, which_Msum = find_used_experts(mask_NM, K)
        y_MD, a_MsaveH, b_MsaveH, y_pre_MsaveD = apply_experts(x_MD, s_NM, idx_Msum, which_Msum, cumsums_N, Wl1_ND2H, Wl2_NHD, K, save_percent)

        ctx.K = K
        ctx.save_percent = save_percent
        ctx.save_for_backward(x_MD, s_NM, idx_Msum, which_Msum, cumsums_N, Wl1_ND2H, Wl2_NHD, a_MsaveH, b_MsaveH, y_pre_MsaveD)

        return rearrange(y_MD, '(B S) D -> B S D', B=B, S=S).to(device=x_BSD.device, dtype=torch.bfloat16), sizes_N

    """
    Performs an optimized backward pass of the MoE, recomputing a few things
    and using the saved information from the forward pass to do the backward pass
    """
    @staticmethod
    def backward(ctx, G_y_BSD, _):
        # Load from ctx
        x_MD, s_NM, idx_Msum, which_Msum, cumsums_N, Wl1_ND2H, Wl2_NHD, a_MsaveH, b_MsaveH, y_pre_MsaveD = ctx.saved_tensors
        K = ctx.K
        save_percent = ctx.save_percent

        B, S, _ = G_y_BSD.shape
        
        G_y_MD = rearrange(G_y_BSD, 'B S D -> (B S) D')

        assert Wl1_ND2H.device == Wl2_NHD.device, "All weights must be on the same device, but got: Wl1_ND2H.device = {}, Wl2_NHD.device = {}".format(Wl1_ND2H.device, Wl2_NHD.device)

        G_y_MD = G_y_MD.to(device=Wl1_ND2H.device, dtype=torch.bfloat16)

        # Second linear backwards and SwiGLU backwards
        G_Wl2_NHD, G_z_Msum2H, x_MsumD, h_MsumH = lin2_and_swiglu_backward(G_y_MD, x_MD, a_MsaveH, b_MsaveH, Wl1_ND2H, Wl2_NHD, idx_Msum, cumsums_N, s_NM.T, save_percent)
        
        # First linear backwards
        G_Wl1_ND2H, G_s_NM, G_x_MD = lin1_backward(x_MsumD, y_pre_MsaveD, G_z_Msum2H, Wl1_ND2H, Wl2_NHD, h_MsumH, idx_Msum, which_Msum, cumsums_N, G_y_MD, K, save_percent)

        G_x_BSD = rearrange(G_x_MD, '(B S) D -> B S D', B=B, S=S)

        return G_x_BSD.to(G_y_BSD.device), G_Wl1_ND2H.to(Wl1_ND2H.device), G_Wl2_NHD.to(Wl2_NHD.device), None, G_s_NM.to(s_NM.device), None, None, None


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Memory optimized Mixture of Experts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An implementation of the MoE forward and backward pass using triton.
Optimized for speed, memory usage, and customization.
"""
class MoMoE(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        intermediate_dim,
        num_experts,
        num_chosen_experts,
        save_percent=0,
        Wl1_ND2H: torch.Tensor | None = None,
        Wl2_NHD: torch.Tensor | None = None,
    ):
        super(MoMoE, self).__init__()
        self.D = embedding_dim
        self.H = intermediate_dim
        self.N = num_experts
        self.K = num_chosen_experts
        self.save_percent = save_percent

        if Wl1_ND2H is None:
            self.Wl1_ND2H = nn.Parameter(torch.randn((self.N, self.D, 2 * self.H), device="cuda", dtype=torch.bfloat16) / self.D, requires_grad=True)
        else:
            self.Wl1_ND2H = nn.Parameter(Wl1_ND2H.clone().detach().to(torch.bfloat16), requires_grad=True)

        if Wl2_NHD is None:
            self.Wl2_NHD = nn.Parameter(torch.randn((self.N, self.H, self.D), device="cuda", dtype=torch.bfloat16) / self.D, requires_grad=True)
        else:
            self.Wl2_NHD = nn.Parameter(Wl2_NHD.clone().detach().to(torch.bfloat16), requires_grad=True)

    def forward(self, x_BSD, mask_NM, s_NM):
        """
        The forward pass of the MoMoE, which takes in:

        x_BSD: The input to the MoMoE, which is a tensor of shape (M, D)
        mask_NM: The mask of tokens per expert (as an int32 tensor of 1s and 0s) of shape (N, M)
        s_NM: The post-zeroing (all non-used vals set to 0) weights of the router of shape (N, M)

        and returns:

        y_BSD: The output of the MoMoE, which is a tensor of shape (B, S, D)
        sizes_N: The number of experts used per token of shape (N,)
        """

        y_BSD, sizes_N = MoEFunction.apply(x_BSD, self.Wl1_ND2H, self.Wl2_NHD, mask_NM, s_NM, self.K, self.save_percent)
        return y_BSD, sizes_N