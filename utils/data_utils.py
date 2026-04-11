import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse.linalg import eigs

class TrafficDataset(Dataset):
    """
    PyTorch Dataset for sliding window time-series generation.
    """
    def __init__(self, data, n_his):
        """
        :param data: np.ndarray, shape [Time, Node, Channel]
        :param n_his: int, length of historical steps to look back
        """
        self.data = data
        self.n_his = n_his

    def __len__(self):
        # We need (n_his) steps for input and 1 step for target
        return len(self.data) - self.n_his

    def __getitem__(self, idx):
        # Input sequence: [n_his, n_route, channel]
        x = self.data[idx : idx + self.n_his]
        # Target sequence: [1, n_route, channel]
        y = self.data[idx + self.n_his : idx + self.n_his + 1]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

def scaled_laplacian(W):
    """
    Normalised graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    """
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    
    # Calculate the largest eigenvalue
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.asmatrix(2 * L / lambda_max - np.identity(n))

def cheb_poly_approx(L, Ks, n):
    """
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: torch.Tensor, [n_route, Ks*n_route].
    """
    L0, L1 = np.asmatrix(np.identity(n)), np.asmatrix(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.asmatrix(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        
        # Concatenate into shape [n_route, Ks * n_route]
        L_concat = np.concatenate(L_list, axis=-1)
        return torch.FloatTensor(L_concat)
    elif Ks == 1:
        return torch.FloatTensor(L0)
    else:
        raise ValueError(f'ERROR: kernel size Ks must be greater than 1, received {Ks}')

def first_approximation(W, n):
    """
    1st-order approximation function (Modernized for PyTorch).
    """
    # A = W + I (Self-loop)
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    
    # D^(-1/2) calculation
    # Added a small epsilon (1e-12) to prevent division by zero
    sinvD = np.sqrt(1.0 / (d + 1e-12))
    sinvD = np.diag(sinvD)
    
    # Eq. 5: L = I + D^(-1/2) * A * D^(-1/2)
    # Using @ for matrix multiplication
    res = np.identity(n) + (sinvD @ A @ sinvD)
    return torch.FloatTensor(res)

def inverse_z_score(x, mean, std):
    return x * std + mean
    
def split_data(X, args):
    train_len = int(X.shape[0] * args.train_ratio)
    val_len = int(X.shape[0] * args.val_ratio)

    train = X[:train_len]
    val = X[train_len:train_len + val_len]
    test = X[train_len + val_len:]
    return train, val, test

def split_data(X, args):
    steps_per_day = getattr(args, 'steps_per_day', 288)
    split_days = getattr(args, 'split_days', [34, 5, 5])

    train_len = int(split_days[0] * steps_per_day)
    val_len   = int(split_days[1] * steps_per_day)
    test_len  = int(split_days[2] * steps_per_day)
    
    total_needed = train_len + val_len + test_len

    args.actual_split_str = f"{split_days[0]}/{split_days[1]}/{split_days[2]} Days"

    train = X[:train_len]
    val   = X[train_len : train_len + val_len]
    test  = X[train_len + val_len : train_len + val_len + test_len]
    
    return train, val, test

def z_score(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def inverse_z_score(x, mean, std):
    return x * std + mean

def create_dataset(X, n_his, horizon):
    x, y = [], []
    T = X.shape[0]
    for i in range(T - n_his - horizon + 1):
        x.append(X[i:i + n_his])
        y.append(X[i + n_his + horizon - 1])
    return np.array(x), np.array(y)

def to_model_input(x):
    return x.transpose(0, 3, 2, 1)

def to_model_target(y):
    y = y.transpose(0, 2, 1)
    return y[:, :, :, None]

def build_graph_kernel(A, ks):
    A = A.astype(np.float32)
    N = A.shape[0]
    I = np.eye(N, dtype=np.float32)

    A_tilde = A + I
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    supports = [I]
    cur = A_norm
    for _ in range(1, ks):
        supports.append(cur)
        cur = cur @ A_norm

    kernel = np.concatenate(supports, axis=1).astype(np.float32)
    return torch.tensor(kernel, dtype=torch.float32)

def masked_mape(y_true, y_pred, threshold=10.0, eps=1e-8):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mask = np.abs(y_true) >= threshold
    if mask.sum() == 0:
        return np.nan

    denom = np.maximum(np.abs(y_true[mask]), eps)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom)) * 100.0

def compute_metrics(y_true, y_pred, args):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = masked_mape(y_true, y_pred, threshold=args.mape_threshold)
    return mae, rmse, mape

def persistence_baseline(x_window):
    return x_window[:, -1, :, :]   # (B, N, 1)