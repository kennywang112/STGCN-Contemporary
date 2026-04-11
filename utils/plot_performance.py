# import os
# current_dir = os.getcwd()
# parent_dir = os.path.dirname(current_dir)

# os.chdir(parent_dir)

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from Model.model import STGCN, GCGRU
import matplotlib.pyplot as plt
from Model.engin import inverse_z_score
from utils.data_utils import split_data, z_score, create_dataset, to_model_input, scaled_laplacian, cheb_poly_approx, first_approximation

class ReplotArgs:
    n_his = 12
    kt = 3
    blocks = [[1, 16, 64], [64, 16, 64]]
    plot_station = 3
    plot_points = 300
    results_plot = Path("Results/Plots")
    results_model = Path("Results/Model")

def plot_prediction_comparison(y_real, pred_dict, ds_name, H, plot_station, plot_points, results_dir):
    """
    pred_dict: {'STGCN (Cheb)': pred_real, 'STGCN (k=1)': pred_real, 'GCGRU': pred_real}
    """
    s = plot_station
    pts = min(plot_points, y_real.shape[0])

    plt.figure(figsize=(14, 5))
    
    # Plot ground truth
    plt.plot(y_real[:pts, s, 0], label="Ground Truth", linewidth=2.5, color='black', zorder=5)
    
    # Plot predictions from different models
    colors = {'STGCN (Cheb)': '#1f77b4', 'STGCN (k=1)': '#ff7f0e', 'GCGRU': '#2ca02c'}
    for model_name, pred_real in pred_dict.items():
        plt.plot(pred_real[:pts, s, 0], label=model_name, alpha=0.7, linewidth=1.5, color=colors.get(model_name))
    
    plt.title(f"{ds_name} Forecast Comparison (Station {s}) - H={H} hour", fontsize=12, fontweight='bold')
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = results_dir / f"comparison_{ds_name}_H{H}_station{s}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Figure saved to {save_path}")

def replot_multiple_models(ds_name, H, ks_list, v_path, w_path):
    """
    ks_list: [3, 1] for Chebyshev and first approximation
    """
    args = ReplotArgs()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading {ds_name} H={H}...")
    A = pd.read_csv(w_path, header=None).values
    X_raw = pd.read_csv(v_path, header=None).values
    if len(X_raw.shape) == 2:
        X_raw = np.expand_dims(X_raw, axis=-1)

    print(f"Data shape: {X_raw.shape}, Nodes: {A.shape[0]}")

    current_n_his = 30 if "Shanghai" in ds_name else 12

    temp_args = type('obj', (object,), {
        'steps_per_day': 1440 if "Shanghai" in ds_name else 288,
        'split_days': [7, 2, 2] if "Shanghai" in ds_name else [34, 5, 5]
    })
    
    print("Splitting and normalizing data...")
    train_raw, _, test_raw = split_data(X_raw, temp_args)
    mean, std = train_raw.mean(), train_raw.std()
    
    test_norm = z_score(test_raw, mean, std)
    teX_win, teY = create_dataset(test_norm, current_n_his, H)
    teX = to_model_input(teX_win)

    W = A.copy()
    np.fill_diagonal(W, 0)

    pred_dict = {}

    # Prepare graph kernels once
    print("Preparing graph kernels...")
    L = scaled_laplacian(W)
    graph_kernel_cheb = cheb_poly_approx(L, 3, A.shape[0]).to(device)
    if graph_kernel_cheb.dim() == 2:
        graph_kernel_cheb = graph_kernel_cheb.unsqueeze(0)

    graph_kernel_1 = first_approximation(W, A.shape[0]).to(device)
    if graph_kernel_1.dim() == 2:
        graph_kernel_1 = graph_kernel_1.unsqueeze(0)

    teX_torch = torch.tensor(teX).float().to(device)

    # Prepare edge_index and edge_weight for GCGRU (once)
    edge_indices = np.where(A > 0)
    edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long).to(device)
    edge_weight = torch.tensor(A[edge_indices], dtype=torch.float32).to(device)

    # STGCN with ks=3 (Chebyshev)
    print(f"Running STGCN (Cheb) for H={H}...")
    try:
        model_cheb = STGCN(
            Ks=3, Kt=args.kt, blocks=args.blocks, 
            n_his=current_n_his, n_nodes=A.shape[0]
        ).to(device)
        
        model_path = args.results_model / f"stgcn_{ds_name}_H{H}_ks3.pt"
        if model_path.exists():
            model_cheb.load_state_dict(torch.load(model_path, map_location=device))
            model_cheb.eval()
            
            with torch.no_grad():
                pred_norm = model_cheb(teX_torch, graph_kernel_cheb).cpu().numpy()
            
            pred_norm = pred_norm[:, 0, :, 0][:, :, None]
            pred_real = inverse_z_score(pred_norm, mean, std)
            pred_dict["STGCN (Cheb)"] = pred_real
        else:
            print(f"STGCN Cheb model not found at {model_path}")
    except Exception as e:
        print(f"STGCN (Cheb) loading failed: {e}")

    # STGCN with ks=1
    print(f"Running STGCN (k=1) for H={H}...")
    try:
        model_1 = STGCN(
            Ks=1, Kt=args.kt, blocks=args.blocks, 
            n_his=current_n_his, n_nodes=A.shape[0]
        ).to(device)
        
        model_path = args.results_model / f"stgcn_{ds_name}_H{H}_ks1.pt"
        if model_path.exists():
            model_1.load_state_dict(torch.load(model_path, map_location=device))
            model_1.eval()
            
            with torch.no_grad():
                pred_norm = model_1(teX_torch, graph_kernel_1).cpu().numpy()
            
            pred_norm = pred_norm[:, 0, :, 0][:, :, None]
            pred_real = inverse_z_score(pred_norm, mean, std)
            pred_dict["STGCN (k=1)"] = pred_real
            print("STGCN (k=1) completed")
        else:
            print(f"STGCN k=1 model not found at {model_path}")
    except Exception as e:
        print(f"STGCN (k=1) loading failed: {e}")

    # GCGRU model
    print(f"Running GCGRU for H={H}...")
    try:
        model_gcgru = GCGRU(
            in_channels=1, hidden_channels=64, out_channels=1, K=1
        ).to(device)
        
        model_path_gcgru = args.results_model / f"gcgru_{ds_name}_H{H}_ks1.pt"
        if model_path_gcgru.exists():
            model_gcgru.load_state_dict(torch.load(model_path_gcgru, map_location=device))
            model_gcgru.eval()

            with torch.no_grad():
                pred_norm_gcgru = model_gcgru(teX_torch, edge_index, edge_weight).cpu().numpy()

            pred_norm_gcgru = pred_norm_gcgru[:, 0, :, 0][:, :, None]
            pred_real_gcgru = inverse_z_score(pred_norm_gcgru, mean, std)
            pred_dict["GCGRU"] = pred_real_gcgru
            print("GCGRU completed")
        else:
            print(f"GCGRU model not found at {model_path_gcgru}")
    except Exception as e:
        print(f"GCGRU loading failed: {e}")

    y_real = inverse_z_score(teY, mean, std)

    print(f"Generating plots for {ds_name} H={H}...")
    plot_prediction_comparison(
        y_real, pred_dict, ds_name, H, 
        args.plot_station, args.plot_points, args.results_plot
    )
    print(f"{ds_name} H={H} completed!\n")

def plot_learning_curves(ds_name, H, ks_list, results_dir):
    """
    Plot training and validation curves for STGCN and GCGRU models
    ks_list: [3, 1] for Chebyshev and first approximation
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {3: '#1f77b4', 1: '#ff7f0e', 'gcgru': '#2ca02c'}
    
    # Plot STGCN curves for different ks values
    for ks in ks_list:
        csv_path = Path("Results/Performance") / f"learning_curve_{ds_name}_H{H}_ks{ks}_stgcn.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            model_label = f"STGCN (Cheb, ks={ks})" if ks > 1 else f"STGCN (k={ks})"
            color = colors.get(ks, '#2ca02c')
            
            # Plot training MSE
            axes[0].plot(df['epoch'], df['train_mse'], label=model_label, 
                        linewidth=2, color=color, marker='o', markersize=4)
            
            # Plot validation MSE
            axes[1].plot(df['epoch'], df['val_mse'], label=model_label, 
                        linewidth=2, color=color, marker='o', markersize=4)
        else:
            print(f"Learning curve file not found: {csv_path}")
    
    # Plot GCGRU curves
    gcgru_csv_path = Path("Results/Performance") / f"learning_curve_{ds_name}_H{H}_ks1_gcgru.csv"
    if gcgru_csv_path.exists():
        df_gcgru = pd.read_csv(gcgru_csv_path)
        
        # Plot training MSE
        axes[0].plot(df_gcgru['epoch'], df_gcgru['train_mse'], label="GCGRU", 
                    linewidth=2, color=colors['gcgru'], marker='s', markersize=4)
        
        # Plot validation MSE
        axes[1].plot(df_gcgru['epoch'], df_gcgru['val_mse'], label="GCGRU", 
                    linewidth=2, color=colors['gcgru'], marker='s', markersize=4)
    else:
        print(f"GCGRU learning curve file not found: {gcgru_csv_path}")
    
    # Configure left plot (Training MSE)
    axes[0].set_title(f"{ds_name} Training MSE (H={H})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train MSE")
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Configure right plot (Validation MSE)
    axes[1].set_title(f"{ds_name} Validation MSE (H={H})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val MSE")
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = results_dir / f"learning_curve_{ds_name}_H{H}_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Figure saved to {save_path}")

def replot_multiple_models_airquality(ds_name, H, ks_list, v_npy_path, w_npy_path, results_plot, results_model):
    """
    Load AirQuality data from .npy files and plot predictions
    """
    args = ReplotArgs()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load from .npy files
    A = np.load(w_npy_path)
    X_raw = np.load(v_npy_path)
    
    if len(X_raw.shape) == 2:
        X_raw = np.expand_dims(X_raw, axis=-1)

    current_n_his = 12

    temp_args = type('obj', (object,), {
        'steps_per_day': 288,
        'split_days': [34, 5, 5]
    })
    
    train_raw, _, test_raw = split_data(X_raw, temp_args)
    mean, std = train_raw.mean(), train_raw.std()
    
    test_norm = z_score(test_raw, mean, std)
    teX_win, teY = create_dataset(test_norm, current_n_his, H)
    teX = to_model_input(teX_win)

    W = A.copy()
    np.fill_diagonal(W, 0)

    pred_dict = {}

    # Prepare graph kernel for STGCN (ks=3, Chebyshev)
    L = scaled_laplacian(W)
    graph_kernel_cheb = cheb_poly_approx(L, 3, A.shape[0]).to(device)
    if graph_kernel_cheb.dim() == 2:
        graph_kernel_cheb = graph_kernel_cheb.unsqueeze(0)

    # Prepare graph kernel for STGCN (ks=1)
    graph_kernel_1 = first_approximation(W, A.shape[0]).to(device)
    if graph_kernel_1.dim() == 2:
        graph_kernel_1 = graph_kernel_1.unsqueeze(0)

    teX_torch = torch.tensor(teX).float().to(device)

    # STGCN with ks=3 (Chebyshev)
    try:
        model_cheb = STGCN(
            Ks=3, Kt=args.kt, blocks=args.blocks, 
            n_his=current_n_his, n_nodes=A.shape[0]
        ).to(device)
        
        model_path = results_model / f"stgcn_{ds_name}_H{H}_ks3.pt"
        if model_path.exists():
            model_cheb.load_state_dict(torch.load(model_path, map_location=device))
            model_cheb.eval()
            
            with torch.no_grad():
                pred_norm = model_cheb(teX_torch, graph_kernel_cheb).cpu().numpy()
            
            pred_norm = pred_norm[:, 0, :, 0][:, :, None]
            pred_real = inverse_z_score(pred_norm, mean, std)
            pred_dict["STGCN (Cheb)"] = pred_real
        else:
            print(f"STGCN Cheb model not found at {model_path}")
    except Exception as e:
        print(f"STGCN (Cheb) loading failed: {e}")

    # STGCN with ks=1
    try:
        model_1 = STGCN(
            Ks=1, Kt=args.kt, blocks=args.blocks, 
            n_his=current_n_his, n_nodes=A.shape[0]
        ).to(device)
        
        model_path = results_model / f"stgcn_{ds_name}_H{H}_ks1.pt"
        if model_path.exists():
            model_1.load_state_dict(torch.load(model_path, map_location=device))
            model_1.eval()
            
            with torch.no_grad():
                pred_norm = model_1(teX_torch, graph_kernel_1).cpu().numpy()
            
            pred_norm = pred_norm[:, 0, :, 0][:, :, None]
            pred_real = inverse_z_score(pred_norm, mean, std)
            pred_dict["STGCN (k=1)"] = pred_real
        else:
            print(f"STGCN k=1 model not found at {model_path}")
    except Exception as e:
        print(f"STGCN (k=1) loading failed: {e}")

    # GCGRU model
    try:
        model_gcgru = GCGRU(
            in_channels=1, hidden_channels=64, out_channels=1, K=1
        ).to(device)
        
        model_path_gcgru = results_model / f"gcgru_{ds_name}_H{H}_ks1.pt"
        if model_path_gcgru.exists():
            model_gcgru.load_state_dict(torch.load(model_path_gcgru, map_location=device))
            model_gcgru.eval()

            edge_indices = np.where(A > 0)
            edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long).to(device)
            edge_weight = torch.tensor(A[edge_indices], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred_norm_gcgru = model_gcgru(teX_torch, edge_index, edge_weight).cpu().numpy()

            pred_norm_gcgru = pred_norm_gcgru[:, 0, :, 0][:, :, None]
            pred_real_gcgru = inverse_z_score(pred_norm_gcgru, mean, std)
            pred_dict["GCGRU"] = pred_real_gcgru
        else:
            print(f"GCGRU model not found at {model_path_gcgru}")
    except Exception as e:
        print(f"GCGRU loading failed: {e}")

    y_real = inverse_z_score(teY, mean, std)

    plot_prediction_comparison(
        y_real, pred_dict, ds_name, H, 
        args.plot_station, args.plot_points, results_plot
    )
