import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
from trainer.utils import build_dataset_from_csv, custom_collate_fn
from models.full_model import MoleculeMoEPRnet

def evaluate_and_plot(model, config, output_dir='./plots', file_name='regression_analysis.png'):
    device = torch.device(config['project']['device'])
    model.to(device)
    model.eval()
    
    print("Preparing Datasets for Visualization...")
    _, train_scaler = build_dataset_from_csv(
        config['data']['train_csv'], 
        config, 
        augment=False, 
        scaler=None
    )
    
    val_dataset, _ = build_dataset_from_csv(
        config['data']['val_csv'], 
        config, 
        augment=False, 
        scaler=train_scaler
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )

    print("Running Inference...")
    y_true = []
    y_pred = []
    y_std = [] 
    
    predict_uncertainty = config['task'].get('predict_uncertainty', False)
    
    with torch.no_grad():
        for smiles_list, dose_list, labels, cell_lines, rdkit_feats, weights in val_loader:
            labels = labels.to(device)
            rdkit_feats = rdkit_feats.to(device)
            cell_line_name = cell_lines[0]
            
            outputs = model(smiles_list, dose_list, cell_line_name, rdkit_feats)
            
            y_true.extend(labels.cpu().numpy())
            
            if config['task']['type'] == 'regression':
                if predict_uncertainty:
                    mu = outputs[:, 0]
                    log_var = outputs[:, 1]
                    std = torch.sqrt(torch.exp(torch.clamp(log_var, min=-5, max=5)))
                    
                    y_pred.extend(mu.cpu().numpy())
                    y_std.extend(std.cpu().numpy())
                else:
                    if outputs.dim() == 2:
                        outputs = outputs.squeeze(-1)
                    y_pred.extend(outputs.cpu().numpy())
                    y_std.extend([0.0] * len(outputs))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print(f"Metrics -> R2: {r2:.4f} | RMSE: {rmse:.4f} | Pearson: {pearson_corr:.4f}")


    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=300)
    

    sns.set_theme(style="whitegrid")
    

    if predict_uncertainty:
        scatter = plt.scatter(y_true, y_pred, c=y_std, cmap='viridis_r', alpha=0.7, edgecolors='w', s=60)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Predicted Uncertainty (Std Dev)', rotation=270, labelpad=15)
        print("-> Plotting with Uncertainty color map.")
    else:
        plt.scatter(y_true, y_pred, color='royalblue', alpha=0.6, edgecolors='w', s=60)


    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.1
    plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
             ls='--', c='gray', alpha=0.8, label='Ideal (y=x)')


    textstr = '\n'.join((
        f'$R^2 = {pearson_corr*pearson_corr:.4f}$',
        f'$RMSE = {rmse:.4f}$',
        f'$Pearson = {pearson_corr:.4f}$',
        f'$N = {len(y_true)}$'))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.xlabel('Experimental Values (True)', fontsize=14)
    plt.ylabel('Model Predictions', fontsize=14)
    plt.title(f'Regression Analysis: {config["project"]["name"]}', fontsize=16)
    plt.legend(loc='lower right')
    plt.tight_layout()

    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

    # save predictions for further analysis
    raw_data = pd.read_csv(config['data']['val_csv'])
    raw_data['y_true'] = y_true
    raw_data['y_pred'] = y_pred
    raw_data['y_std'] = y_std
    #save under predictions directory
    if not os.path.exists('./predictions'):
        os.makedirs('./predictions')
    pred_save_path = os.path.join('./predictions', "val_predictions_regression.csv")
    raw_data.to_csv(pred_save_path, index=False)
    print(f"Predictions saved to {pred_save_path}")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Regression Model and Plot Results")
    parser.add_argument('--config', type=str, default='./config/model_config_regression.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/experts_3_round_12_mean_sota/best_model.pt', help='Path to model checkpoint')
    args = parser.parse_args()

    config_path = args.config
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    

    cfg['model']['encoders']['mpnn']['hidden_channels'] = 128
    cfg['model']['encoders']['mpnn']['num_layers'] = 3
    cfg['model']['moe']['num_experts'] = 3
    cfg['model']['moe']['expert_hidden_dim'] = 128
    cfg['task']['predict_uncertainty'] = False
    cfg['training']['batch_size'] = 64 
    model = MoleculeMoEPRnet(config=cfg)
    

    checkpoint_path = args.checkpoint
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Warning: No checkpoint found! Plotting with random weights.")


    evaluate_and_plot(model, cfg)
