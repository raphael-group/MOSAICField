import torch
import ot
import numpy as np
from scipy.spatial import distance
from scipy.linalg import lstsq


def lstsq_proj_mat(sp1, sp2, pi):
    '''
    Uses Least Squares to calculate a affine transformation.
    The loss function is \sum w_{ij} |Rx_i - y_j|_2^2. 
    The 6 free parameters denote the upper two rows of R.
    The expected transformation is Y = RX.
    '''
    nnz = np.sum(pi > 0)
    A = np.zeros((2 * nnz, 6))
    B = np.zeros(2 * nnz)
    l = 0
    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            if pi[i][j] > 0:
                w = pi[i][j] ** 0.5
                x1, y1 = sp1[i]
                x2, y2 = sp2[j]
                A[l] = [x1 * w, y1 * w, w, 0, 0, 0]
                A[l + 1] = [0, 0, 0, x1 * w, y1 * w, w]
                B[l] = x2 * w
                B[l + 1] = y2 * w
                l += 2
    return A, B


def lstsq_proj(sp1, sp2, pi):
    A, B = lstsq_proj_mat(sp1, sp2, pi)
    T, _, _, _ = lstsq(A, B)
    return np.append(T, [0, 0, 1]).reshape(3, 3)


def affine_transformation(X, Y, P):
    T = lstsq_proj(sp1=X, sp2=Y, pi=P)
    homogeneous_X = np.vstack([X.T, np.ones((1, X.shape[0]))])
    transformed_homogeneous_X = T @ homogeneous_X
    return transformed_homogeneous_X[:2, :].T, T


def FGW_affine(X, Y, FX, FY, max_iter=100, alpha=0.8, device='cpu'):
    # center
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    D_X = distance.cdist(FX, FX)
    D_Y = distance.cdist(FY, FY)
    D_X /= D_X.max()
    D_Y /= D_Y.max()
    D_X = torch.tensor(D_X).to(device)
    D_Y = torch.tensor(D_Y).to(device)

    T_net = np.eye(3)
    for iter in range(max_iter):
        print('Iter:', iter)
        C = distance.cdist(X, Y)
        C /= C[C>0].max()
        C = torch.tensor(C).to(device)

        # FGW
        P = ot.gromov.fused_gromov_wasserstein(C, D_X, D_Y, alpha=alpha).to(torch.float32)
        # Affine Transformation
        X, T = affine_transformation(X=X, Y=Y, P=P.cpu().numpy())

        T_net = T @ T_net

    # Compute final P
    C = distance.cdist(X, Y)
    C /= C[C>0].max()
    C = torch.tensor(C).to(device)
    P = ot.gromov.fused_gromov_wasserstein(C, D_X, D_Y, alpha=alpha).to(torch.float32)
    return X, Y, T_net, P


def affine_align(slice1, slice2, obsm_name=None, max_iter=10, alpha=0.8):
    """
    Computes the optimal affine transformation between two slices, potential from two different modalities.

    param: slice1 - AnnData object of slice 1
    param: slice2 - AnnData object of slice 2, potentially from a different modality than slice 1
    param: obsm_name - The name of the obsm field to use as features for alignment. If None, use slice.X
    param: max_iter - Maximum number of iterations to run the global invariant optimal transport algorithm
    param: alpha - Balance parameter of the two terms. Note: 0 ≤ alpha ≤ 1

    return: slice1, slice2 - The AnnData object of affine aligned slice 1 and slice 2. The spatial coordinates of slice 1 are affined transformed on to slice 2, so
    slice1.obsm['spatial'] is updated, while slice2 is just a copy of the input slice2
    return: T - a 3 by 3 numpy array, indicating the affine transformation matrix that registers slice 1 onto slice 2, in homogeneous coordinates.
    return: P - probabilistic mapping between the locations in slice 1 and the locations in slice 2.
    """
    X = slice1.obsm['spatial'].astype(np.float32)
    Y = slice2.obsm['spatial'].astype(np.float32)
    if obsm_name is None:
        FX = slice1.X.astype(np.float32)
        FY = slice2.X.astype(np.float32)
    else:
        FX = slice1.obsm[obsm_name].astype(np.float32)
        FY = slice2.obsm[obsm_name].astype(np.float32)

    X, Y, T, P = FGW_affine(X, Y, FX, FY, max_iter=max_iter, alpha=alpha)

    slice1_copy = slice1.copy()
    slice2_copy = slice2.copy()
    slice1_copy.obsm['spatial'] = X
    return slice1_copy, slice2_copy, T, P


