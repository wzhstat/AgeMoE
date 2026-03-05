# train.py

import yaml
from models.full_model import MoleculeMoEPRnet
from trainer.core import Trainer
from trainer.utils import build_dataset_from_csv
import torch
import argparse
#torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=None, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None, help='Path to validation CSV file')
    parser.add_argument('--label_col', type=str, default=None, help='Name of the label column in CSV')
    parser.add_argument('--num_experts', type=int, default=None)
    parser.add_argument('--predict_uncertainty', action='store_true', help='Whether to predict uncertainty (for regression tasks)')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--hidden_channels', type=int, default=None, help='Hidden channels for MPNN') 
    parser.add_argument('--moe_hidden_dim', type=int, default=None, help='Hidden dimension for MoE experts')
    parser.add_argument('--lr', type=float, default=None, help='Initial learning rate')
    parser.add_argument('--config', type=str, default='./MoEv2/config/model_config.yaml', help='Path to config file')
    args = parser.parse_args()
    config_path = args.config
    
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if args.num_experts is not None:
        print(f"!!! Overriding num_experts: {args.num_experts}")
        cfg['model']['moe']['num_experts'] = args.num_experts
    
    if args.hidden_channels is not None:
        print(f"!!! Overriding hidden_channels: {args.hidden_channels}")
        cfg['model']['encoders']['mpnn']['hidden_channels'] = args.hidden_channels
    
    if args.save_dir is not None:
        print(f"!!! Overriding save_dir: {args.save_dir}")
        cfg['training']['save_dir'] = args.save_dir
    
    if args.moe_hidden_dim is not None:
        print(f"!!! Overriding moe_hidden_dim: {args.moe_hidden_dim}")
        cfg['model']['moe']['expert_hidden_dim'] = args.moe_hidden_dim
    
    if args.lr is not None:
        print(f"!!! Overriding learning rate: {args.lr}")
        cfg['training']['lr'] = args.lr
    
    if args.predict_uncertainty:
        print("!!! Enabling uncertainty prediction mode.")
        cfg['task']['predict_uncertainty'] = True
    
    if args.train_csv is not None:
        print(f"!!! Overriding train_csv: {args.train_csv}")
        cfg['data']['train_csv'] = args.train_csv
    if args.val_csv is not None:
        print(f"!!! Overriding val_csv: {args.val_csv}")
        cfg['data']['val_csv'] = args.val_csv
    
    if args.label_col is not None:
        print(f"!!! Overriding label_col: {args.label_col}")
        cfg['data']['label_col'] = args.label_col

    print("\n=== Building Datasets ===")
    
    train_dataset, train_scaler = build_dataset_from_csv(
        cfg['data']['train_csv'], 
        cfg, 
        augment=True,
        scaler=None
    )
    

    val_dataset, _ = build_dataset_from_csv(
        cfg['data']['val_csv'], 
        cfg, 
        augment=False,
        scaler=train_scaler
    )

    print(f"\n=== Initializing Model (Experts={cfg['model']['moe']['num_experts']}) ===")
    model = MoleculeMoEPRnet(config=cfg) 


    print("\n=== Starting Trainer ===")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg
    )

    trainer.fit()

if __name__ == "__main__":
    main()