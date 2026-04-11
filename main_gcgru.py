import csv

import numpy as np
import pandas as pd

import torch
from pathlib import Path

from Model.engin import run_one_horizon_gcgru

class Args:
    results_csv = Path("Results/Performance")
    results_plot = Path("Results/Plots")
    results_model = Path("Results/Model")

    # Hyperparameters from the original paper
    ks_values = [1]
    batch_size = 50
    epoch = 50
    ks = 3
    kt = 3
    lr = 5e-5

    train_ratio = 0.7
    val_ratio = 0.1
    # added
    patience = 5
    min_delta = 1e-4

    plot_station = 1
    plot_points = 200
    mape_threshold = 10.0


def main(args):
    dct_dt = {
        "AirQuality": ['./dataset/AirQuality/air_quality_adj.npy', './dataset/AirQuality/air_quality_data.npy'],
        "Shanghai": ['./dataset/ShanghaiRailway/W_43.csv', './dataset/ShanghaiRailway/V_43.csv'],
        "PeMSD7_228": ['./dataset/PeMSD7/W_228_P.csv', './dataset/PeMSD7/V_228.csv'],
        "PeMSD7_1026": ['./dataset/PeMSD7/W_1026_P.csv', './dataset/PeMSD7/V_1026.csv']
    }

    DATASET_CONFIGS = {
        "PeMSD7_1026": {"n_his": 12, "horizons": [6], "min_multiplier": 5,  "steps_per_day": 288,  "split_days": [34, 5, 5]},
        "PeMSD7_228":  {"n_his": 12, "horizons": [3, 6, 9], "min_multiplier": 5,  "steps_per_day": 288,  "split_days": [34, 5, 5]},
        "AirQuality":  {"n_his": 12, "horizons": [1, 2, 3], "min_multiplier": 60, "steps_per_day": 24,   "split_days": [1095, 183, 183]},
        "Shanghai":    {"n_his": 30, "horizons": [5, 10],   "min_multiplier": 1,  "steps_per_day": 1440, "split_days": [7, 2, 2]}
    }

    all_datasets_results = []

    for ds_name, paths in dct_dt.items():
        # Training with mps is bad for no reason, switch to cpu if using mac
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        config = DATASET_CONFIGS.get(ds_name, {})
        args.n_his = config.get("n_his", 12)
        args.horizons = config.get("horizons", [3, 6, 9])
        args.steps_per_day = config.get("steps_per_day", 288)
        args.split_days = config.get("split_days", [34, 5, 5])
        multiplier = config.get("min_multiplier", 5)

        print(f"\n" + "="*30)
        print(f"Processing Dataset: {ds_name}")
    
        if paths[0].endswith('.csv'):
            A = pd.read_csv(paths[0], header=None).values
        else:
            A = np.load(paths[0])

        if paths[1].endswith('.csv'):
            X_raw = pd.read_csv(paths[1], header=None).values
        else:
            X_raw = np.load(paths[1])
    
        if len(X_raw.shape) == 2:
            X_raw = np.expand_dims(X_raw, axis=-1)

        args.n_route = A.shape[0]

        dataset_results = []

        dataset_csv_path = args.results_csv / f"metrics_{ds_name}_gcgru.csv"

        for H in args.horizons:
            for ks in args.ks_values:
                print(f"\nTraining with Ks={ks}")
                result = run_one_horizon_gcgru(X_raw, A, H, args, device, ds_name, ks=ks)

                result["Dataset"] = ds_name
                result["Ks"] = ks
                result["H"] = H
                result["Minutes"] = H * multiplier
                file_exists = dataset_csv_path.exists()

                with open(dataset_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result)

                ks_csv_path = args.results_csv / f"metrics_{ds_name}_Ks{ks}_gcgru.csv"
                ks_file_exists = ks_csv_path.exists()
                with open(ks_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                    if not ks_file_exists:
                        writer.writeheader()
                    writer.writerow(result)

                print(f"Result for H={H}, Ks={ks} saved")
                
                dataset_results.append(result)

        all_datasets_results.extend(dataset_results)


if __name__ == "__main__":
    args = Args()
    args.results_csv.mkdir(exist_ok=True)
    args.results_plot.mkdir(exist_ok=True)
    args.results_model.mkdir(exist_ok=True)

    main(args)