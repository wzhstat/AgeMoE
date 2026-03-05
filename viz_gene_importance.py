import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入项目模块
from trainer.utils import build_dataset_from_csv, custom_collate_fn
from models.full_model import MoleculeMoEPRnet

def load_gene_names(gene_list_path, top_n):
    """
    读取基因列表文件，获取 Gene Symbol
    """
    if not os.path.exists(gene_list_path):
        return [f"Gene_{i}" for i in range(top_n)]
    
    with open(gene_list_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    # 我们之前的逻辑是取前 Top N
    # 但要注意：PRnetBatchPredictor 内部是根据 list 里的基因去匹配 L1000 的
    # 如果 gene_list_path 里就是按顺序排好的 Top N 基因，那就直接取
    return genes[:top_n]

def analyze_gene_importance(config_path, checkpoint_path, output_dir='./plots'):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['project']['device'])
    os.makedirs(output_dir, exist_ok=True)

    # 2. 准备数据
    print("Preparing Datasets...")
    # 只需要验证集即可
    _, train_scaler = build_dataset_from_csv(cfg['data']['train_csv'], cfg, augment=False, scaler=None)
    val_dataset, _ = build_dataset_from_csv(cfg['data']['val_csv'], cfg, augment=False, scaler=train_scaler)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    # 3. 加载模型
    print("Loading Model...")
    model = MoleculeMoEPRnet(config=cfg)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    # 4. 获取基因名称列表
    top_n = cfg['data']['top_n_genes']
    gene_names = load_gene_names(cfg['data']['gene_list_path'], top_n)
    num_experts = cfg['model']['moe']['num_experts']

    # 初始化重要性矩阵 [Num_Experts, Num_Genes]
    # 我们累加每个 Batch 的梯度绝对值
    global_importance = np.zeros((num_experts, top_n))
    sample_count = 0

    print("Calculating Gradient-based Importance...")
    
    # 5. 循环计算梯度
    for smiles_list, dose_list, labels, cell_lines, rdkit_feats, weights in tqdm(val_loader):
        cell_line_name = cell_lines[0]
        
        # A. 获取 PRnet 的原始输出 (Input to Router)
        # 我们需要先拿到这个 Tensor，然后开启 requires_grad
        with torch.no_grad():
            feat_prnet_raw = model.prnet_predictor(
                cell_line_name=cell_line_name,
                smiles_list=smiles_list,
                dose_list=dose_list,
                top_n=top_n,
                output_delta=cfg['data']['prnet_delta']
            )
        
        # B. 开启梯度追踪 (关键步骤)
        feat_prnet_raw.requires_grad_(True)
        feat_prnet_raw.retain_grad()
        
        # C. 通过 Router
        router_weights = model.router(feat_prnet_raw) # [Batch, Num_Experts]
        
        # D. 对每个专家分别反向传播
        batch_importance = np.zeros((num_experts, top_n))
        
        for expert_idx in range(num_experts):
            # 目标：最大化该专家的权重总和
            target = router_weights[:, expert_idx].sum()
            
            # 清空之前的梯度
            if feat_prnet_raw.grad is not None:
                feat_prnet_raw.grad.zero_()
            
            # 反向传播，计算 d(Expert_Weight) / d(Gene_Input)
            target.backward(retain_graph=True)
            
            # 获取梯度: [Batch, Num_Genes]
            grads = feat_prnet_raw.grad.data
            
            # 计算显著性: |Gradient| (也可以用 Gradient * Input)
            # 这里取平均绝对梯度
            saliency = torch.abs(grads).mean(dim=0).cpu().numpy()
            
            batch_importance[expert_idx] = saliency
            
        # 累加到全局
        global_importance += batch_importance
        sample_count += 1

    # 平均化
    global_importance /= sample_count

    # 6. 绘图 (ClusterMap)
    print("Plotting Gene Importance Heatmap...")
    
    # 归一化以便绘图 (Min-Max 到 0-1 之间，让图更好看)
    # 每一行(专家)单独归一化，看该专家最关注谁
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
    
    # 7. 输出文本报告 (每个专家的 Top 5 基因)
    print("\n=== Top 5 Genes per Expert ===")
    for i in range(num_experts):
        # 获取排序索引 (从大到小)
        indices = np.argsort(global_importance[i])[::-1]
        top_genes = [gene_names[idx] for idx in indices[:5]]
        print(f"Expert {i}: {', '.join(top_genes)}")

if __name__ == "__main__":
    CONFIG_PATH = './MoEv2/config/model_config.yaml'
    CHECKPOINT_PATH = './MoEv2/checkpoints/experts_6_hidden_128_MultiClass/best_model.pt' # 确保这里是你想分析的模型
    
    analyze_gene_importance(CONFIG_PATH, CHECKPOINT_PATH)