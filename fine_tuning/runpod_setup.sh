#!/usr/bin/env bash
# Provision a fresh RunPod pod (RTX 6000 Ada) for depth-encoder fine-tuning.
#   bash fine_tuning/runpod_setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== GPU =="
nvidia-smi || echo "  (no nvidia-smi -- CPU box? smoke test still works on CPU)"

echo "== PyTorch =="
if python -c "import torch" 2>/dev/null; then
  python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())"
else
  echo "  torch not found -> installing CUDA 12.6 build (matches docker/Dockerfile_x86_sim)"
  pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126
fi

echo "== Python deps =="
pip install -r "$HERE/requirements.txt"

echo "== .env =="
if [ ! -f "$HERE/.env" ]; then
  cp "$HERE/.env.example" "$HERE/.env"
  echo "  created fine_tuning/.env from template -- fill in WANDB_API_KEY / RUNPOD_API_KEY"
else
  echo "  fine_tuning/.env already present"
fi

echo "== Preflight =="
python "$HERE/preflight.py" || true

cat <<'EOF'

Setup complete. Next:
  # validate the whole pipeline on synthetic data (no sim data needed):
  python fine_tuning/train.py --smoke-test

  # once the sim emits episodes (see fine_tuning/data/README.md):
  python fine_tuning/train.py --episodes /path/to/episodes --wandb
EOF
