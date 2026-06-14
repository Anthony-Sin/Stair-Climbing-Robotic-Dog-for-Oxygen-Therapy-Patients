from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


TensorLike = Union[np.ndarray, torch.Tensor]


def _pairwise_sqeuclidean(x: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    dist = x2 + x2.t() - 2.0 * (x @ x.t())
    return dist.clamp_min_(0.0)


def _to_tensor(feats: TensorLike, device: torch.device) -> tuple:
    input_is_numpy = isinstance(feats, np.ndarray)
    if isinstance(feats, torch.Tensor):
        tensor = feats.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(feats, dtype=torch.float32, device=device)
    return tensor, input_is_numpy


def nfc(feats: TensorLike, k1: int = 2, k2: int = 2, device: str = "cuda") -> TensorLike:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    target_device = torch.device(device if use_cuda else "cpu")

    x, input_is_numpy = _to_tensor(feats, target_device)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor, got shape={tuple(x.shape)}")

    # NFC neighbor selection must operate on unit-length embeddings so
    # squared Euclidean remains equivalent to cosine ranking.
    x = F.normalize(x, p=2, dim=1)

    n = int(x.shape[0])
    if n <= 1:
        return x.detach().cpu().numpy() if input_is_numpy else x

    k1 = max(1, min(int(k1), n - 1))
    k2 = max(1, min(int(k2), k1))

    dist = _pairwise_sqeuclidean(x)
    dist.fill_diagonal_(float("inf"))
    nn_idx = torch.topk(dist, k=k1, dim=1, largest=False, sorted=True).indices

    frozen = x.clone()
    refined = frozen.clone()

    for i in range(n):
        neighbors = nn_idx[i]
        reverse = (nn_idx[neighbors] == i).any(dim=1)
        mutual = neighbors[reverse]
        if mutual.numel() > 0:
            refined[i] = refined[i] + frozen[mutual].sum(dim=0)

    refined = F.normalize(refined, p=2, dim=1)

    return refined.detach().cpu().numpy() if input_is_numpy else refined
