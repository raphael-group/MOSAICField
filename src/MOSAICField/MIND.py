import torch
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple

# -------------------------------
# Helpers
# -------------------------------

def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Accept (H,W), (C,H,W), or (N,C,H,W) and return (N,C,H,W).
    """
    if x.dim() == 2:        # (H,W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:      # (C,H,W)
        x = x.unsqueeze(0)
    elif x.dim() == 4:      # (N,C,H,W)
        pass
    else:
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
    return x

def _shift2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    Shift (N,C,H,W) by (dy, dx) with 'replicate' padding.
    """
    N, C, H, W = x.shape
    pad_l = max(dx, 0)
    pad_r = max(-dx, 0)
    pad_t = max(dy, 0)
    pad_b = max(-dy, 0)
    xpad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    y0 = pad_b
    x0 = pad_r
    return xpad[:, :, y0:y0+H, x0:x0+W]

def _patch_aggregate(ssd: torch.Tensor, patch_radius: int) -> torch.Tensor:
    """
    Aggregate per-pixel dissimilarities over a (2r+1)^2 window via average pooling.
    """
    k = 2 * patch_radius + 1
    return F.avg_pool2d(ssd, kernel_size=k, stride=1, padding=patch_radius)

# -------------------------------
# MIND descriptor (multi-channel, 2D)
# -------------------------------

def mind_descriptor_2d_multichannel(
    img: torch.Tensor,
    patch_radius: int = 1,
    offsets: Optional[Iterable[Tuple[int, int]]] = None,
    dilation: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-channel MIND descriptor for 2D images.

    Args:
        img: (H,W), (C,H,W), or (N,C,H,W). Any C >= 1.
        patch_radius: r for the (2r+1)x(2r+1) aggregation window.
        offsets: list of (dy,dx) shifts. If None, uses 4-neighborhood with given dilation.
        dilation: pixel step for the neighborhood.
        eps: small constant.

    Returns:
        (N, K, H, W) tensor, where K = number of offsets.
    """
    x = _ensure_nchw(img).float()              # (N,C,H,W)
    N, C, H, W = x.shape

    if offsets is None:
        d = int(dilation)
        offsets = [(d, 0), (-d, 0), (0, d), (0, -d)]

    Ds = []
    for (dy, dx) in offsets:
        xr = _shift2d(x, dy, dx)               # (N,C,H,W)
        # L2 across channels per pixel (use squared L2 for SSD aggregation):
        # per_pixel = ||x - xr||_2^2 = sum_c (x_c - xr_c)^2
        per_pixel_l2sq = (x - xr).pow(2).sum(dim=1, keepdim=True)  # (N,1,H,W)
        D_r = _patch_aggregate(per_pixel_l2sq, patch_radius)       # (N,1,H,W)
        Ds.append(D_r)

    D = torch.cat(Ds, dim=1)                   # (N,K,H,W)

    # MIND-SSC normalization
    V = D.mean(dim=1, keepdim=True)            # local variance proxy
    m = D.min(dim=1, keepdim=True).values
    MIND = torch.exp(-(D - m) / (V + eps))     # (N,K,H,W)
    return MIND

# -------------------------------
# MIND loss (multi-channel, 2D)
# -------------------------------
def zscore_batch(x):
    mean = x.mean(dim=(-2,-1), keepdim=True)
    std  = x.std(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std

def normalize(x):
    img_min = x.min()
    img_max = x.max()
    return (x - img_min) / (img_max - img_min + 1e-8)

def mind_loss_2d_multichannel(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    patch_radius: int = 1,
    offsets: Optional[Iterable[Tuple[int, int]]] = None,
    dilation: int = 1,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L1 distance between multi-channel MIND descriptors of two 2D images.

    Args:
        img_a, img_b: (H,W), (C,H,W), or (N,C,H,W). Same (N,H,W); C may be same or broadcastable.
        patch_radius, offsets, dilation, eps: as above.
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        Scalar if reduction != 'none'; else (N,K,H,W) loss map.
    """
    desc_a = mind_descriptor_2d_multichannel(img_a, patch_radius, offsets, dilation, eps)
    desc_b = mind_descriptor_2d_multichannel(img_b, patch_radius, offsets, dilation, eps)
    l = (desc_a - desc_b).abs()
    if reduction == "mean":
        return l.mean()
    elif reduction == "sum":
        return l.sum()
    elif reduction == "none":
        return l
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
