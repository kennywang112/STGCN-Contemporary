## ST-GCN

The code referenced from [Jam_Forecaster](https://github.com/martin-xia0/Jam_Forecaster) and [VeritasYin](https://github.com/VeritasYin/STGCN_IJCAI-18?tab=readme-ov-file), since it last updated in years, this repo updates from TF 1.x to Pytorch and implement the original data based on the [paper](https://arxiv.org/pdf/1709.04875), but also test with the new dataset.<br/>

## Datasets

In this project, we evaluate the model's performance and cross-domain generalization capability using three distinct datasets, ranging from conventional highway traffic to high-frequency urban road networks and macroscopic environmental data.

| Dataset | Domain | Nodes (Sensors/Stations) | Timesteps | Time Interval |
| :--- | :--- | :--- | :--- | :--- |
| **PeMSD7 (M)** | Highway Traffic | 228 | 12,672 | 5 minutes |
| **PeMSD7 (L)** | Highway Traffic | 1,026 | 12,672 | 5 minutes |
| **Shanghai Railway** | Urban Traffic | 43 | 15,999 | 1 minute |
| **Air Quality** | Environment | 12 | 35,064 | 1 hour |

* **PeMSD7 (Caltrans):** Collected by the California Department of Transportation Performance Measurement System (District 7). We utilize both the Medium (228 nodes) and Large (1026 nodes) graph configurations to reproduce baseline spatial-temporal forecasting results.
* **Shanghai Railway Station:** Represents a highly congested urban hub. It includes 43 road segments with extremely dense, high-frequency data sampling at 1-minute intervals.
* **Air Quality:** A regional environmental dataset comprising 12 monitoring stations. Unlike traffic flow, this dataset captures macro-level, non-periodic spatial diffusion over a long-term span, recorded at 1-hour intervals.

To run the code:
```pshell
# create env
conda env create -f environment.yml
conda activate MLDS

python main.py
python main_gcgru.py
```

## Framework
The main structure implement in this image for the original paper:
1. `/Model/model.py`: STConvBlock class implements the full sandwich structure
2. `/Model/gcgru.py`: This copied directly from [PyTorch repo](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/gconv_gru.py)
3. `data_utils`: Chebychev needs a scaled Laplacian, so the code is in this file


<div style="text-align: center; margin: 20px;">
    <a href="Results/Plots/Architecture.png">
        <img src="Results/Plots/Architecture.png" alt="Image" style="max-width: 100%; height: auto; border: 1px solid #ccc; padding: 10px;">
    </a>
</div>

## Folder

```
├── dataset
├── EDA
├── environment.yml
├── main_gcgru.py
├── main.py
├── Model
│   ├── air_data_preparing.py
│   ├── data_generator.py
│   ├── engin.py
│   ├── gcgru.py
│   └── model.py
├── README.md
├── Results
└── utils
    ├── data_utils.py
    ├── plot_performance.py
    └── plot.py
```