import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer.utils import build_dataset_from_csv, custom_collate_fn
from models.full_model import MoleculeMoEPRnet

def load_gene_names(gene_list_path, top_n):
    if not os.path.exists(gene_list_path):
        return [f"Gene_{i}" for i in range(top_n)]
    
    with open(gene_list_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    return genes[:top_n]

def analyze_gene_importance(config_path, checkpoint_path, output_dir='./plots'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['project']['device'])
    os.makedirs(output_dir, exist_ok=True)

    print("Preparing Datasets...")
    _, train_scaler = build_dataset_from_csv(cfg['data']['train_csv'], cfg, augment=False, scaler=None)
    val_dataset, _ = build_dataset_from_csv(cfg['data']['val_csv'], cfg, augment=False, scaler=train_scaler)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    print("Loading Model...")
    model = MoleculeMoEPRnet(config=cfg)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    top_n = cfg['data']['top_n_genes']
    gene_names = load_gene_names(cfg['data']['gene_list_path'], top_n)
    num_experts = cfg['model']['moe']['num_experts']

    global_importance = np.zeros((num_experts, top_n))
    sample_count = 0

    print("Calculating Gradient-based Importance...")
    
    for smiles_list, dose_list, labels, cell_lines, rdkit_feats, weights in tqdm(val_loader):
        cell_line_name = cell_lines[0]
        
        with torch.no_grad():
            feat_prnet_raw = model.prnet_predictor(
                cell_line_name=cell_line_name,
                smiles_list=smiles_list,
                dose_list=dose_list,
                top_n=top_n,
                output_delta=cfg['data']['prnet_delta']
            )
        
        feat_prnet_raw.requires_grad_(True)
        feat_prnet_raw.retain_grad()
        
        router_weights = model.router(feat_prnet_raw) # [Batch, Num_Experts]
        
        batch_importance = np.zeros((num_experts, top_n))
        
        for expert_idx in range(num_experts):
            target = router_weights[:, expert_idx].sum()
            
            if feat_prnet_raw.grad is not None:
                feat_prnet_raw.grad.zero_()
            
            target.backward(retain_graph=True)
            
            grads = feat_prnet_raw.grad.data
            
            saliency = torch.abs(grads).mean(dim=0).cpu().numpy()
            
            batch_importance[expert_idx] = saliency
            
        global_importance += batch_importance
        sample_count += 1

    global_importance /= sample_count

    print("Plotting Gene Importance Heatmap...")
    plot_data = global_importance / (global_importance.max(axis=1, keepdims=True) + 1e-9)
    
    df_plot = pd.DataFrame(plot_data, columns=gene_names, index=[f"Expert {i}" for i in range(num_experts)])
    
    plt.figure(figsize=(15, 8))
    # 使用 ClusterMap 可以自动聚类，把关注相似基因的专家聚在一起
    g = sns.clustermap(df_plot, cmap="magma", standard_scale=None, 
                       figsize=(14, 8), row_cluster=False, col_cluster=True,
                       dendrogram_ratio=(0.1, 0.2),
                       cbar_pos=(0.02, 0.8, 0.03, 0.15))
    
    g.fig.suptitle(f'Gene Importance Analysis (Top {top_n} Genes)', fontsize=16, y=1.02)
    g.ax_heatmap.set_xlabel("Genes (PRnet Input)")
    g.ax_heatmap.set_ylabel("Experts")
    
    save_path = os.path.join(output_dir, 'gene_importance_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    
    # 输出文本报告 (每个专家的 Top 5 基因)
    print("\n=== Top 5 Genes per Expert ===")
    for i in range(num_experts):
        indices = np.argsort(global_importance[i])[::-1]
        top_genes = [gene_names[idx] for idx in indices[:5]]
        print(f"Expert {i}: {', '.join(top_genes)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Gene Importance for MoE Model")
    parser.add_argument('--config', type=str, default='./config/model_config_classification.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/experts_6_hidden_128_MultiClass/best_model.pt', help='Path to model checkpoint')
    args = parser.parse_args()
    
    analyze_gene_importance(args.config, args.checkpoint)