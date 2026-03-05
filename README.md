# AgeMoE
This model is an anti-aging drug scoring model based on the MoE framework and incorporating transcription information..<br>
# Installation
## step 1: Download AgeMoE
```
git clone https://github.com/wzhstat/AgeMoE.git
cd AgeMoE
```
## Step 2: Environment
You can manually add the environment in the following ways. The version about ```pytorch``` and ```cudatoolkit``` should be depended on your machine.<br>
```
conda create -n AgeMoE python=3.10 \
conda activate AgeMoE \
pip3 install torch==2.4.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
pip install numpy==1.24.3 \
pip install pyyaml==6.0.3 \
pip install scanpy \
pip install pandas==2.3.3 \
pip install torch-geometric \
conda install rdkit=2025.9.1 -c rdkit \
```
We also provide the corresponding environment file, you can reproduce the environment directly through the provided .yaml file.<br>
```
conda env create -f environment.yaml
```
## Step 3: Necessary documents
To ensure the accurate input of the model's transcription information, the ```Lincs_L1000.h5ad``` file needs to be placed under the ```lincs_cache/``` folder. This file can be downloaded from https://drive.google.com/file/d/1Z05yalquBrXG4sVitkspbvScFFXee7DJ/view?usp=sharing.<br>

# Training
Training is driven by Hydra configs in ```configs/```. The default entry point is ```train.py```.<br>
**Basic command:**<br>
```
python train.py \
  --train_csv ./Data/train_mean_std.csv \
  --val_csv ./Data/valid_mean_std.csv \
  --label_col Class \
  --save_dir ./checkpoints/experts_6_hidden_128_MultiClass \
```
**Key args to modify:**
- `--train_csv`: Path to training CSV file.
- `--val_csv`: Path to validation CSV file.
- `--label_col`: Name of the label column in CSV.
- `--save_dir`: The storage path of checkpoints.
- `--num_experts`: The number of experts used in the model.
- `--predict_uncertainty`: Whether to predict uncertainty (for regression tasks).
- `--hidden_channels`: Hidden channels for MPNN.
- `--moe_hidden_dim`: Hidden dimension for MoE experts.
- `--lr`: Initial learning rate.
- `--config`: Path to config file.

**Outputs**:
- Checkpoints under `--save_dir`.
- By default, best checkpoints are selected with `training.monitor_metric`

# Evaluation/Interpretability
For regression models, evaluation is performed by `viz_regression.py`. e.g
```
python viz_regression.py \
  --config ./config/model_config_regression.yaml \
  --checkpoint ./checkpoints/experts_3_round_12_mean_sota/best_model.pt \
```
For classification models, evaluation is performed by `viz_classification.py`. e.g
```
python viz_classification.py \
  --config ./config/model_config_classification.yaml \
  --checkpoint ./checkpoints/experts_6_hidden_128_MultiClass/best_model.pt \
```
We also provided an interpretability module for analyzing the expertise of experts.
```
python viz_gene_importance.py \
  --config ./config/model_config_classification.yaml \
  --checkpoint ./checkpoints/experts_6_hidden_128_MultiClass/best_model.pt \
```
**Note:** Please ensure that the parameters of the evaluation model are consistent with in the config.
