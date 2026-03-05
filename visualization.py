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

# 引入项目模块
from trainer.utils import build_dataset_from_csv, custom_collate_fn
from models.full_model import MoleculeMoEPRnet

def evaluate_and_plot(model, config, output_dir='./plots', file_name='regression_analysis.png'):
    """
    输入模型和配置，绘制回归预测结果图
    """
    # 1. 设置设备
    device = torch.device(config['project']['device'])
    model.to(device)
    model.eval()
    
    # 2. 准备数据 (必须重新拟合 Scaler 以保证特征一致性)
    print("Preparing Datasets for Visualization...")
    # 先加载训练集以获取 Scaler (这一步很快，因为是并行计算)
    _, train_scaler = build_dataset_from_csv(
        config['data']['train_csv'], 
        config, 
        augment=False, 
        scaler=None
    )
    
    # 加载验证集 (使用训练集的 Scaler)
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

    # 3. 运行推理
    print("Running Inference...")
    y_true = []
    y_pred = []
    y_std = [] # 用于存储标准差 (不确定性)
    
    predict_uncertainty = config['task'].get('predict_uncertainty', False)
    
    with torch.no_grad():
        for smiles_list, dose_list, labels, cell_lines, rdkit_feats, weights in val_loader:
            # 移动数据到设备
            labels = labels.to(device)
            rdkit_feats = rdkit_feats.to(device)
            cell_line_name = cell_lines[0]
            
            # 模型前向传播
            outputs = model(smiles_list, dose_list, cell_line_name, rdkit_feats)
            
            # 收集真实值
            y_true.extend(labels.cpu().numpy())
            
            # 收集预测值
            if config['task']['type'] == 'regression':
                if predict_uncertainty:
                    # 输出是 [Batch, 2] -> (mu, log_var)
                    mu = outputs[:, 0]
                    log_var = outputs[:, 1]
                    std = torch.sqrt(torch.exp(torch.clamp(log_var, min=-5, max=5)))
                    
                    y_pred.extend(mu.cpu().numpy())
                    y_std.extend(std.cpu().numpy())
                else:
                    # 普通回归 [Batch, 1] -> [Batch]
                    if outputs.dim() == 2:
                        outputs = outputs.squeeze(-1)
                    y_pred.extend(outputs.cpu().numpy())
                    y_std.extend([0.0] * len(outputs)) # 填充0

    # 转换为 Numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    # 4. 计算指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print(f"Metrics -> R2: {r2:.4f} | RMSE: {rmse:.4f} | Pearson: {pearson_corr:.4f}")

    # 5. 绘图
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 设置风格
    sns.set_theme(style="whitegrid")
    
    # 如果有不确定性，将其作为点的颜色或大小
    if predict_uncertainty:
        # 颜色映射：不确定性越高，颜色越暖
        scatter = plt.scatter(y_true, y_pred, c=y_std, cmap='viridis_r', alpha=0.7, edgecolors='w', s=60)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Predicted Uncertainty (Std Dev)', rotation=270, labelpad=15)
        print("-> Plotting with Uncertainty color map.")
    else:
        # 普通散点
        plt.scatter(y_true, y_pred, color='royalblue', alpha=0.6, edgecolors='w', s=60)

    # 绘制理想线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.1
    plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
             ls='--', c='gray', alpha=0.8, label='Ideal (y=x)')

    # 绘制拟合线 (Regression Line)
    #sns.regplot(x=y_true, y=y_pred, scatter=False, color='darkred', 
    #            line_kws={'label': f'Fit: y={np.polyfit(y_true, y_pred, 1)[0]:.2f}x + {np.polyfit(y_true, y_pred, 1)[1]:.2f}'})

    # 添加文本框显示指标
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

# ==========================================
# 使用示例 (Main)
# ==========================================
if __name__ == "__main__":
    # 1. 加载配置
    config_path = './MoEv2/config/model_config.yaml'
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. 初始化模型
    cfg['model']['encoders']['mpnn']['hidden_channels'] = 128
    cfg['model']['encoders']['mpnn']['num_layers'] = 3
    cfg['model']['moe']['num_experts'] = 3
    cfg['model']['moe']['expert_hidden_dim'] = 128
    cfg['task']['predict_uncertainty'] = False
    cfg['training']['batch_size'] = 64 
    cfg['training']['save_dir'] = './MoEv2/checkpoints/experts_3_round_2_mean/' #experts_3_hidden_256_GNLL_1_sota_model
    model = MoleculeMoEPRnet(config=cfg)
    
    # 3. 加载权重 (这里你需要指定你要画哪个权重文件)
    # 默认加载 best_model.pt
    checkpoint_path = os.path.join(cfg['training']['save_dir'], 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Warning: No checkpoint found! Plotting with random weights.")

    # 4. 执行绘图
    evaluate_and_plot(model, cfg)