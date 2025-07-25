import torch
from sample_moe import MoE
from torch_moe import TorchMoE

"""
Simple MoE test which trains for 200 steps with AdamW optimizer
on a dummy MoE layer.
"""
if __name__ == "__main__":
    B, S, D, H = 4, 8192, 1024, 512
    N, K, Ks = 32, 6, 0

    x_BSD = torch.randn((B, S, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    Wg_DN = torch.randn((D, N), device="cuda", dtype=torch.bfloat16, requires_grad=True) / D
    Wl1_ND2H = torch.randn((N, D, 2 * H), device="cuda", dtype=torch.bfloat16, requires_grad=True) / D
    Wl2_NHD = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16, requires_grad=True) / D

    moe = MoE(
        embedding_dim=D,
        intermediate_dim=H,
        num_experts=N,
        num_chosen_experts=K,
        num_shared_experts=Ks,
        save_percent=0,
        bias_update_rate=1e-4,
        Wg_DN=Wg_DN,
        Wl1_ND2H=Wl1_ND2H,
        Wl2_NHD=Wl2_NHD,
    )

    torch_moe = TorchMoE(
        D, N, K, Ks, H/D, 0.0, False, 1e-4,
        dtype=torch.bfloat16, Wg_DN=Wg_DN, Wl1_ND2H=Wl1_ND2H, Wl2_NHD=Wl2_NHD, device=torch.device('cuda')
    )

    # Warmup
    for i in range(10):
        y_BSD = moe(x_BSD)
        y_BSD_torch = torch_moe(x_BSD)
    torch.cuda.synchronize()

    optimizer_triton = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    optimizer_torch = torch.optim.AdamW(torch_moe.parameters(), lr=1e-3)

    for i in range(200):
        x_BSD.grad = torch.zeros_like(x_BSD)
        y_BSD = moe(x_BSD)
        loss = torch.mean(torch.square(y_BSD.to(torch.float32) - x_BSD.to(torch.float32)))
        loss.backward()
        optimizer_triton.step()
        aa = x_BSD.grad.clone()
        # print(x_BSD.grad)
        optimizer_triton.zero_grad()

        x_BSD.grad = torch.zeros_like(x_BSD)
        y_BSD_torch = torch_moe(x_BSD)
        loss_torch = torch.mean(torch.square(y_BSD_torch.to(torch.float32) - x_BSD.to(torch.float32)))
        loss_torch.backward()
        optimizer_torch.step()
        # print(x_BSD.grad)
        optimizer_torch.zero_grad()


        print((aa - x_BSD.grad).abs().max())

        # print(loss.item(), loss_torch.item())
