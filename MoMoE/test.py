import torch
from sample_moe import MoE

"""
Simple MoE test which trains for 200 steps with AdamW optimizer
on a dummy MoE layer.

The loss should start at around 1 and decrease downwards toward 0.
"""
if __name__ == "__main__":
    B, S, D, H = 4, 8192, 1024, 512
    N, K, Ks = 32, 6, 1

    x_BSD = torch.randn((B, S, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    
    moe = MoE(
        embedding_dim=D,
        intermediate_dim=H,
        num_experts=N,
        num_chosen_experts=K,
        num_shared_experts=Ks,
        save_percent=0,
        bias_update_rate=1e-4,
    )

    # Warmup
    for i in range(10):
        y_BSD = moe(x_BSD)
    torch.cuda.synchronize()

    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    for i in range(200):
        x_BSD.grad = torch.zeros_like(x_BSD)
        y_BSD = moe(x_BSD)
        loss = torch.mean(torch.square(y_BSD.to(torch.float32) - x_BSD.to(torch.float32)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {i} loss: {loss.item()}")
