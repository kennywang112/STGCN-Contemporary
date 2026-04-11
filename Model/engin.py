import time
import torch
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import dense_to_sparse
from Model.model import STGCN, GCGRU
from utils.data_utils import *
from utils.plot import plot_training_curves, plot_prediction_comparison

def run_one_horizon(X_raw, A, H, args, device, ds_name, ks=3):
    print("\n" + "=" * 60)
    print(f"Horizon H={H} (predict t+{H} hour)")

    train_raw, val_raw, test_raw = split_data(X_raw, args)

    mean = train_raw.mean()
    std = train_raw.std()

    train_norm = z_score(train_raw, mean, std)
    val_norm = z_score(val_raw, mean, std)
    test_norm = z_score(test_raw, mean, std)

    trX_win, trY = create_dataset(train_norm, args.n_his, H)
    vaX_win, vaY = create_dataset(val_norm, args.n_his, H)
    teX_win, teY = create_dataset(test_norm, args.n_his, H)

    trX = to_model_input(trX_win)
    vaX = to_model_input(vaX_win)
    teX = to_model_input(teX_win)

    trY_t = to_model_target(trY)
    vaY_t = to_model_target(vaY)
    teY_t = to_model_target(teY)

    print("train:", trX.shape, "->", trY_t.shape)
    print("val  :", vaX.shape, "->", vaY_t.shape)
    print("test :", teX.shape, "->", teY_t.shape)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(trX).float(), torch.tensor(trY_t).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(vaX).float(), torch.tensor(vaY_t).float()),
        batch_size=args.batch_size, shuffle=False
    )

    # Don't remove this ! this is from the oiriginal paper
    W = A.copy()
    np.fill_diagonal(W, 0)
    L = scaled_laplacian(W)

    if ks > 1:
        # Use Chebyshev approximation
        print(f"Using Chebyshev polynomial approximation (Ks={ks})")
        graph_kernel = cheb_poly_approx(L, ks, args.n_route).to(device)
    else:
        # Use 1st-order approximation
        print(f"Using 1st-order approximation (Ks={ks})")
        graph_kernel = first_approximation(W, args.n_route).to(device)

    if graph_kernel.dim() == 2:
        graph_kernel = graph_kernel.unsqueeze(0)

    model = STGCN(
        Ks=ks,
        Kt=args.kt,
        blocks=args.blocks,
        n_his=args.n_his,
        n_nodes=args.n_route
    ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = args.results_model / f"stgcn_{ds_name}_H{H}_ks{ks}.pt"

    epoch_times = []
    val_rmse_list = []
    val_mae_list = []

    learning_curve_data = []
    start_time = time.perf_counter()

    patience = args.patience
    patience_counter = 0
    actual_epochs = args.epoch

    for epoch in range(1, args.epoch + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb, graph_kernel)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb, graph_kernel)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)

                val_preds.append(pred.cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        scheduler.step()

        val_loss /= len(val_loader.dataset)

        val_pred_norm = np.concatenate(val_preds, axis=0)
        val_target_norm = np.concatenate(val_targets, axis=0)

        val_pred_norm = val_pred_norm[:, 0, :, 0][:, :, None]
        val_target_norm = val_target_norm[:, 0, :, 0][:, :, None]

        val_pred_real = inverse_z_score(val_pred_norm, mean, std)
        val_target_real = inverse_z_score(val_target_norm, mean, std)

        val_mae, val_rmse, _ = compute_metrics(val_target_real, val_pred_real, args)

        epoch_times.append(time.perf_counter() - start_time)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)

        learning_curve_data.append({
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae
                })
    
        print(
            f"Epoch {epoch:02d}/{args.epoch} | "
            f"train MSE={train_loss:.5f} | "
            f"val MSE={val_loss:.5f} | "
            f"val RMSE={val_rmse:.2f} | "
            f"val MAE={val_mae:.2f}"
        )

        # This is for early stopping
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'early stopping in epoch {epoch}')
            actual_epochs = epoch
            break

    lc_path = args.results_csv / f"learning_curve_{ds_name}_H{H}_ks{ks}_stgcn.csv"
    pd.DataFrame(learning_curve_data).to_csv(lc_path, index=False)
    print(f"Saved learning curve data to: {lc_path}")

    total_train_time = time.perf_counter() - start_time

    plot_training_curves(epoch_times, val_rmse_list, val_mae_list, H, args.results_plot, ds_name, ks)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    test_start_time = time.perf_counter()

    teX_torch = torch.tensor(teX).float().to(device)
    with torch.no_grad():
        pred_norm = model(teX_torch, graph_kernel).cpu().numpy()

    pred_norm = pred_norm[:, 0, :, 0][:, :, None]
    y_norm = teY

    pred_real = inverse_z_score(pred_norm, mean, std)
    y_real = inverse_z_score(y_norm, mean, std)

    stgcn_mae, stgcn_rmse, stgcn_mape = compute_metrics(y_real, pred_real, args)

    base_pred_norm = persistence_baseline(teX_win)
    base_pred_real = inverse_z_score(base_pred_norm, mean, std)
    base_mae, base_rmse, base_mape = compute_metrics(y_real, base_pred_real, args)

    test_time = time.perf_counter() - test_start_time

    print(f"[Test - STGCN]     MAE={stgcn_mae:.4f} | RMSE={stgcn_rmse:.4f} | MAPE={stgcn_mape:.2f}%")
    print(f"[Test - Baseline]  MAE={base_mae:.4f} | RMSE={base_rmse:.4f} | MAPE={base_mape:.2f}%")

    plot_prediction_comparison(y_real, pred_real, ds_name, H, args.plot_station, args.plot_points, args.results_plot, ks)

    return {
        "H": H,
        "STGCN_MAE": stgcn_mae,
        "STGCN_RMSE": stgcn_rmse,
        "STGCN_MAPE": stgcn_mape,
        "BASE_MAE": base_mae,
        "BASE_RMSE": base_rmse,
        "BASE_MAPE": base_mape,
        "epochs": actual_epochs,
        "split": "70/10/20",
        "Training_Time_s": round(total_train_time, 3),
        "Test_Time_s": round(test_time, 3),
        "Total_Time_s": round(total_train_time + test_time, 3)
    }

def run_one_horizon_gcgru(X_raw, A, H, args, device, ds_name, ks=3):
    print("\n" + "=" * 60)
    print(f"Horizon H={H} (predict t+{H} hour)")

    train_raw, val_raw, test_raw = split_data(X_raw, args)

    mean = train_raw.mean()
    std = train_raw.std()

    train_norm = z_score(train_raw, mean, std)
    val_norm = z_score(val_raw, mean, std)
    test_norm = z_score(test_raw, mean, std)

    trX_win, trY = create_dataset(train_norm, args.n_his, H)
    vaX_win, vaY = create_dataset(val_norm, args.n_his, H)
    teX_win, teY = create_dataset(test_norm, args.n_his, H)

    trX = to_model_input(trX_win)
    vaX = to_model_input(vaX_win)
    teX = to_model_input(teX_win)

    trY_t = to_model_target(trY)
    vaY_t = to_model_target(vaY)
    teY_t = to_model_target(teY)

    print("train:", trX.shape, "->", trY_t.shape)
    print("val  :", vaX.shape, "->", vaY_t.shape)
    print("test :", teX.shape, "->", teY_t.shape)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(trX).float(), torch.tensor(trY_t).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(vaX).float(), torch.tensor(vaY_t).float()),
        batch_size=args.batch_size, shuffle=False
    )

    # Don't remove this ! this is from the oiriginal paper
    W = A.copy()
    np.fill_diagonal(W, 0)
    adj_tensor = torch.tensor(W).float()

    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = GCGRU(in_channels=1, hidden_channels=64, out_channels=1, K=ks).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = args.results_model / f"gcgru_{ds_name}_H{H}_ks{ks}.pt"

    epoch_times = []
    val_rmse_list = []
    val_mae_list = []

    learning_curve_data = []
    start_time = time.perf_counter()

    patience = args.patience
    patience_counter = 0
    actual_epochs = args.epoch

    for epoch in range(1, args.epoch + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb, edge_index, edge_weight)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb, edge_index, edge_weight)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)

                val_preds.append(pred.cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        scheduler.step()

        val_loss /= len(val_loader.dataset)
    
        val_pred_norm = np.concatenate(val_preds, axis=0)
        val_target_norm = np.concatenate(val_targets, axis=0)

        val_pred_norm = val_pred_norm[:, 0, :, 0][:, :, None]
        val_target_norm = val_target_norm[:, 0, :, 0][:, :, None]

        val_pred_real = inverse_z_score(val_pred_norm, mean, std)
        val_target_real = inverse_z_score(val_target_norm, mean, std)

        val_mae, val_rmse, _ = compute_metrics(val_target_real, val_pred_real, args)

        epoch_times.append(time.perf_counter() - start_time)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)

        learning_curve_data.append({
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae
                })
        print(
            f"Epoch {epoch:02d}/{args.epoch} | "
            f"train MSE={train_loss:.5f} | "
            f"val MSE={val_loss:.5f} | "
            f"val RMSE={val_rmse:.2f} | "
            f"val MAE={val_mae:.2f}"
        )

        # This is for early stopping
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1 

        if patience_counter >= patience:
            print(f'early stopping in epoch {epoch}')
            actual_epochs = epoch
            break

    lc_path = args.results_csv / f"learning_curve_{ds_name}_H{H}_ks{ks}_gcgru.csv"
    pd.DataFrame(learning_curve_data).to_csv(lc_path, index=False)
    print(f"Saved learning curve data to: {lc_path}")

    total_train_time = time.perf_counter() - start_time

    plot_training_curves(epoch_times, val_rmse_list, val_mae_list, H, args.results_plot, ds_name, ks, modelname="GCGRU")

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    test_start_time = time.perf_counter()

    teX_torch = torch.tensor(teX).float().to(device)
    with torch.no_grad():
        pred_norm = model(teX_torch, edge_index, edge_weight).cpu().numpy()

    pred_norm = pred_norm[:, 0, :, 0][:, :, None]
    y_norm = teY

    pred_real = inverse_z_score(pred_norm, mean, std)
    y_real = inverse_z_score(y_norm, mean, std)

    stgcn_mae, stgcn_rmse, stgcn_mape = compute_metrics(y_real, pred_real, args)

    base_pred_norm = persistence_baseline(teX_win)
    base_pred_real = inverse_z_score(base_pred_norm, mean, std)
    base_mae, base_rmse, base_mape = compute_metrics(y_real, base_pred_real, args)

    test_time = time.perf_counter() - test_start_time

    print(f"[Test - STGCN]     MAE={stgcn_mae:.4f} | RMSE={stgcn_rmse:.4f} | MAPE={stgcn_mape:.2f}%")
    print(f"[Test - Baseline]  MAE={base_mae:.4f} | RMSE={base_rmse:.4f} | MAPE={base_mape:.2f}%")

    plot_prediction_comparison(y_real, pred_real, ds_name, H, args.plot_station, args.plot_points, args.results_plot, ks, modelname="GCGRU")

    return {
        "H": H,
        "STGCN_MAE": stgcn_mae,
        "STGCN_RMSE": stgcn_rmse,
        "STGCN_MAPE": stgcn_mape,
        "BASE_MAE": base_mae,
        "BASE_RMSE": base_rmse,
        "BASE_MAPE": base_mape,
        "epochs": actual_epochs,
        "split": "70/10/20",
        "Training_Time_s": round(total_train_time, 3),
        "Test_Time_s": round(test_time, 3),
        "Total_Time_s": round(total_train_time + test_time, 3)
    }