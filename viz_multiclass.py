import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from trainer.utils import build_dataset_from_csv, custom_collate_fn
from models.full_model import MoleculeMoEPRnet

def plot_multiclass_roc(y_true, y_score, n_classes, output_dir):
    # One-hot encoding
    # y_true=[0, 2] -> [[1,0,0], [0,0,1]]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    plt.figure(figsize=(10, 8))
    lw = 2

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.4f})',
             color='deeppink', linestyle=':', linewidth=4)


    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'][:n_classes]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:0.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC for Ageing Model', fontsize=15)
    plt.legend(loc="lower right")
    
    save_path = os.path.join(output_dir, 'multiclass_roc.png')
    plt.savefig(save_path, dpi=300)
    print(f"ROC curve saved to {save_path}")
    plt.close()

def plot_expert_analysis(router_weights, y_true, n_experts, output_dir):
    # === 图 1: 类别-专家 热力图 ===
    cols = [f"Expert_{i}" for i in range(n_experts)]
    df = pd.DataFrame(router_weights, columns=cols)
    df['Class'] = y_true

    heatmap_data = df.groupby('Class').mean()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Expert Preference by Class (Average Weight)", fontsize=14)
    plt.ylabel("Target Class")
    plt.xlabel("Expert ID")
    
    save_path = os.path.join(output_dir, 'expert_heatmap.png')
    plt.savefig(save_path, dpi=300)
    print(f"Expert Heatmap saved to {save_path}")
    plt.close()

    # === 图 2: Router 权重 t-SNE ===
    print("Running t-SNE on Router Weights...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    weights_embedded = tsne.fit_transform(router_weights)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(weights_embedded[:, 0], weights_embedded[:, 1], 
                          c=y_true, cmap='viridis', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title("t-SNE Visualization of Router Weights", fontsize=14)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    
    save_path = os.path.join(output_dir, 'expert_tsne.png')
    plt.savefig(save_path, dpi=300)
    print(f"Expert t-SNE saved to {save_path}")
    plt.close()


def run_analysis(config_path, checkpoint_path, output_dir='./plots'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['project']['device'])
    os.makedirs(output_dir, exist_ok=True)
    print("Preparing Datasets...")
    _, train_scaler = build_dataset_from_csv(
        cfg['data']['train_csv'], cfg, augment=False, scaler=None
    )
    val_dataset, _ = build_dataset_from_csv(
        cfg['data']['val_csv'], cfg, augment=False, scaler=train_scaler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], 
        shuffle=False, collate_fn=custom_collate_fn
    )
    print("Loading Model...")
    model = MoleculeMoEPRnet(config=cfg)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    print("Running Inference...")
    all_probs = []
    all_labels = []
    all_router_weights = []

    with torch.no_grad():
        for smiles_list, dose_list, labels, cell_lines, rdkit_feats, _ in val_loader:
            labels = labels.to(device)
            rdkit_feats = rdkit_feats.to(device)
            cell_line_name = cell_lines[0]

            probs, weights = model(
                smiles_list, dose_list, cell_line_name, rdkit_feats, 
                return_router_weights=True
            )

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_router_weights.append(weights.cpu().numpy())

    y_score = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    router_weights = np.concatenate(all_router_weights)

    print(f"Generating Plots in {output_dir}...")
    
    # A. ROC Curve
    num_classes = cfg['task']['num_classes']
    plot_multiclass_roc(y_true, y_score, num_classes, output_dir)
    
    # B. Expert Analysis
    num_experts = cfg['model']['moe']['num_experts']
    plot_expert_analysis(router_weights, y_true, num_experts, output_dir)

    #save the predictions results for further analysis
    raw_data = pd.read_csv(cfg['data']['val_csv'])
    raw_data['pred_0'] = y_score[:, 0]
    raw_data['pred_1'] = y_score[:, 1]
    raw_data['pred_2'] = y_score[:, 2]
    raw_data['true_label'] = y_true
    
    # save in predictions folder
    if not os.path.exists('./predictions'):
        os.makedirs('./predictions')
    pred_save_path = os.path.join('./predictions', 'val_predictions_classification.csv')
    raw_data.to_csv(pred_save_path, index=False)
    print(f"Predictions saved to {pred_save_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Gene Importance for MoE Model")
    parser.add_argument('--config', type=str, default='./config/model_config_classification.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/experts_6_hidden_128_MultiClass/best_model.pt', help='Path to model checkpoint')
    args = parser.parse_args()
    
    run_analysis(args.config, args.checkpoint)
