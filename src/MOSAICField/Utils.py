import numpy as np
# import nibabel as nib
from Loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    Spatial transformer that warps an image according to a dense displacement field.

    Convention:
      flow[:, 0, :, :] = displacement along Y (vertical / rows)
      flow[:, 1, :, :] = displacement along X (horizontal / cols)

    grid_sample expects the last dimension ordering (x, y) normalized to [-1, 1].
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        H, W = size

        # Create base pixel coordinate grid: (1, 2, H, W)
        ys = torch.arange(0, H)
        xs = torch.arange(0, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        # stack as (y, x)
        grid = torch.stack((grid_y, grid_x), dim=0).unsqueeze(0).float()
        self.register_buffer('grid', grid)

    def _normalize_grid(self, grid_yx_pix, H, W):
        """
        Convert pixel-space grid (N, 2, H, W) -> normalized grid (N, H, W, 2)
        expected by grid_sample.
        """
        y = 2 * (grid_yx_pix[:, 0] / (H - 1) - 0.5)
        x = 2 * (grid_yx_pix[:, 1] / (W - 1) - 0.5)
        grid_xy = torch.stack((x, y), dim=1)  # (N,2,H,W)
        return grid_xy.permute(0, 2, 3, 1)    # (N,H,W,2), (x,y) order

    def forward(self, src, flow, return_phi=False):
        """
        src:  (N, C, H, W)
        flow: (N, 2, H, W), displacement in pixels (dy, dx)
        """
        _, _, H, W = src.shape

        # Compute target sampling locations in pixel space
        locs_yx_pix = self.grid + flow  # (N,2,H,W)

        # Normalize for grid_sample (expects x,y ordering)
        new_locs = self._normalize_grid(locs_yx_pix, H, W)

        out = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode='border')
        return (out, new_locs) if return_phi else out

    @torch.no_grad()
    def backward(self, dst, flow, n_iter=10, tol=1e-3, relax=1.0, return_phi=False):
        """
        Approximate inverse warping (fixed-point iteration)
        dst:  (N, C, H, W)
        flow: (N, 2, H, W), (dy, dx)
        """
        device = dst.device
        _, _, H, W = dst.shape

        grid_yx_pix = self.grid.to(device)
        phi_yx_pix = grid_yx_pix.clone()

        for _ in range(n_iter):
            phi_norm_xy = self._normalize_grid(phi_yx_pix, H, W)
            sampled_flow = F.grid_sample(flow, phi_norm_xy, align_corners=True, mode='bilinear')
            phi_new = grid_yx_pix - sampled_flow
            delta = torch.mean(torch.abs(phi_new - phi_yx_pix))
            phi_yx_pix = (1 - relax) * phi_yx_pix + relax * phi_new
            if delta < tol:
                break

        phi_norm_xy = self._normalize_grid(phi_yx_pix, H, W)
        src_rec = F.grid_sample(dst, phi_norm_xy, align_corners=True, mode=self.mode)
        return (src_rec, phi_yx_pix) if return_phi else src_rec


def generate_grid2D_tensor(shape):
    h_grid = torch.linspace(-1., 1., shape[0])
    w_grid = torch.linspace(-1., 1., shape[1])
    h_grid, w_grid = torch.meshgrid(h_grid, w_grid, indexing='ij')

    grid = torch.stack([h_grid, w_grid], dim=0)
    return grid


