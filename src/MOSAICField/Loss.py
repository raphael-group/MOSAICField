import torch
import torch.nn as nn
import torch.nn.functional as F


def JacobianDet(J):
    """
    Compute per-pixel Jacobian determinant for a 2D displacement field.
    Accepts J of shape (B,2,H,W) or (B,H,W,2), normalized in [-1,1].
    Returns a tensor of shape (B,H-1,W-1) giving the determinant at each location.
    """
    # bring channels to last dim if needed
    if J.size(-1) != 2:
        J = J.permute(0, 2, 3, 1)
    # un-normalize: from [-1,1] to [0,1] and scale to pixel units
    J = (J + 1) / 2.0
    scale = torch.tensor([J.size(1), J.size(2)], dtype=J.dtype, device=J.device)
    scale = scale.view(1, 1, 1, 2)
    J = J * scale
    # finite differences
    dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
    dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]
    # 2x2 det: | dx0 dx1 |
    #            | dy0 dy1 |
    det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    return det

def neg_Jdet_loss(J):
    Jdet = JacobianDet(J)
    neg_Jdet = -1.0 * (Jdet - 0.5)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet ** 2)

def magnitude_loss(all_v):
    all_v_x_2 = all_v[:, 0, :, :] * all_v[:, 0, :, :]
    all_v_y_2 = all_v[:, 1, :, :] * all_v[:, 1, :, :]
    all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2)
    return all_v_magnitude