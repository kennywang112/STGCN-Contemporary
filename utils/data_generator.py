import numpy as np
import pandas as pd
import os

def generate_dataset(v_raw_path, w_raw_path, ds_name, output_dir='./dataset/PeMSD7/'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dist_matrix = pd.read_csv(w_raw_path, header=None).values

    sigma = 10 
    epsilon = 0.5
    
    # exp(-d^2 / sigma^2)
    w_weighted = np.exp(-(dist_matrix**2) / (sigma**2))
    w_weighted[w_weighted < epsilon] = 0

    np.fill_diagonal(w_weighted, 0)

    pd.DataFrame(w_weighted).to_csv(os.path.join(output_dir, f'W_{ds_name}_P.csv'), header=None, index=None)
    print(f"Saved: {output_dir}W_{ds_name}_P.csv")

    v_data = pd.read_csv(v_raw_path, header=None)

    v_data.to_csv(os.path.join(output_dir, f'V_{ds_name}.csv'), header=None, index=None)
    print(f"Saved: {output_dir}V_{ds_name}.csv")

def process_shanghai_data(v_path, w_path, output_dir="."):

    v_df = pd.read_csv(v_path, header=None)

    v_out_path = os.path.join(output_dir, 'V_43.csv') 
    v_df.to_csv(v_out_path, index=False, header=False)
    print(f"Saved original 1min V data: {v_out_path}")

    w_df = pd.read_csv(w_path, header=None)
    w_values = w_df.values

    if np.max(w_values) > 1.0:
        std = np.std(w_values[w_values > 0])
        w_adj = np.exp(- (w_values / std) ** 2)
        w_adj[w_adj < 0.1] = 0
        np.fill_diagonal(w_adj, 0)
        
        w_out_path = os.path.join(output_dir, 'W_43.csv')
        pd.DataFrame(w_adj).to_csv(w_out_path, index=False, header=False)
        print(f"Saved W matrix: {w_out_path}")


if __name__ == '__main__':
    dct_dt = {
        "228": ['./dataset/PeMSD7/PeMSD7_Full/PeMSD7_V_228.csv', './dataset/PeMSD7/PeMSD7_Full/PeMSD7_W_228.csv'],
        "1026": ['./dataset/PeMSD7/PeMSD7_Full/PeMSD7_V_1026.csv', './dataset/PeMSD7/PeMSD7_Full/PeMSD7_W_1026.csv']
    }

    for key, paths in dct_dt.items():
        print(f"Processing {key}")
        generate_dataset(*paths, key)

    process_shanghai_data(
        v_path='./dataset/ShanghaiRailway/V_43.csv', 
        w_path='./dataset/ShanghaiRailway/W_43_P.csv',
        output_dir='./dataset/ShanghaiRailway/' 
    )
