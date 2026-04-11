import matplotlib.pyplot as plt

def plot_training_curves(epoch_times, val_rmse_list, val_mae_list, H, results_dir, ds_name, ks, modelname="STGCN"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epoch_times, val_rmse_list)
    axes[0].set_xlabel("Training Time (s)")
    axes[0].set_ylabel("Validation RMSE")
    axes[0].set_title(f"RMSE vs Training Time (H={H})")

    axes[1].plot(range(1, len(val_mae_list) + 1), val_mae_list)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation MAE")
    axes[1].set_title(f"MAE vs Epoch (H={H})")

    plt.tight_layout()
    curve_path = results_dir / f"training_curve_{ds_name}_H{H}_{ks}_{modelname}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print("Saved training curve:", curve_path)

def plot_prediction_comparison(y_real, pred_real, ds_name, H, plot_station, plot_points, results_dir, ks, modelname="STGCN"):
    s = plot_station
    pts = min(plot_points, y_real.shape[0])

    plt.figure(figsize=(10, 4))
    plt.plot(y_real[:pts, s, 0], label="Ground Truth", linewidth=2)
    plt.plot(pred_real[:pts, s, 0], label="STGCN Prediction", alpha=0.8)
    plt.title(f"{ds_name} Forecast (Station {s}) - H={H} hour")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = results_dir / f"prediction_{ds_name}_H{H}_{ks}_{modelname}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("Saved prediction plot:", fig_path)

