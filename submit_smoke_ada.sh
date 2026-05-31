#!/bin/bash
#SBATCH --job-name=cluster-ml-smoke-ada
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/smoke_ada_%j.out
#SBATCH --error=logs/smoke_ada_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# AdaGN forward-only smoke test: confirm UNet(cond=True, ada=True) builds
# and forwards through at both 64 and 128 px, with a mix of real condition
# and NaN null entries. Forward-only (no DDPM.sample loop) keeps this well
# under a minute even on a CPU.

python - <<'EOF'
import torch, train_diffusion as M
d = M.DDPM(T=10, device='cpu')
for ada in [False, True]:
    for size in [64, 128]:
        m = M.UNet(cond=True, ada=ada)
        nparam = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 1, size, size)
        t = torch.randint(0, 10, (2,))
        c = torch.tensor([0.1, float('nan')])
        y = m(d.q(x, t, torch.randn_like(x)), t, c)
        assert y.shape == (2, 1, size, size), f"bad shape {y.shape}"
        print(f"  ada={ada}  size={size}: forward OK  params {nparam/1e6:.2f}M")
print("AdaGN forward path OK at 64 and 128")
EOF
