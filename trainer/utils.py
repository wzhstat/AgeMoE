# trainer/utils.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


def _calc_mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except:
        mol = None
        
    desc_list = Descriptors._descList
    
    if mol:
        feats = []
        for _, func in desc_list:
            try:
                val = func(mol)
                if not np.isfinite(val):
                    val = 0.0
                feats.append(val)
            except:
                feats.append(0.0)
    else:
        feats = [0.0] * len(desc_list)
        
    return feats

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, dose_list, labels, cell_lines, rdkit_features, augment=False, weight_config=None):
        self.smiles_list = smiles_list
        self.dose_list = dose_list
        self.labels = labels
        self.rdkit_features = rdkit_features
        self.weights = self._calculate_weights(labels, weight_config)
        
        if isinstance(cell_lines, str):
            self.cell_lines = [cell_lines] * len(smiles_list)
        else:
            self.cell_lines = cell_lines

    def __len__(self):
        return len(self.smiles_list)
    
    def _calculate_weights(self, labels, config):
        weights = np.ones_like(labels, dtype=np.float32)
        
        if config and config.get('enabled', False):
            threshold = config.get('threshold', -2.0)
            scale = config.get('scale_factor', 5.0)
            mask = labels < threshold
            weights[mask] = 1.0 + scale * (threshold - labels[mask])
            
            print(f"-> Sample Weighting Enabled. Max Weight: {weights.max():.2f}, Min Weight: {weights.min():.2f}")
            print(f"-> Focusing on labels < {threshold}")
            
        return torch.tensor(weights, dtype=torch.float32)

    def __getitem__(self, idx):
        return {
            'smiles': self.smiles_list[idx],
            'dose': float(self.dose_list[idx]),
            'label': float(self.labels[idx]),
            'cell_line': self.cell_lines[idx],
            'rdkit_feat': self.rdkit_features[idx],
            'weight': self.weights[idx] 
        }

def custom_collate_fn(batch):
    smiles = [item['smiles'] for item in batch]
    doses = [item['dose'] for item in batch]
    cell_lines = [item['cell_line'] for item in batch]
    rdkit_feats = torch.stack([item['rdkit_feat'] for item in batch])
    weights = torch.stack([item['weight'] for item in batch])
    
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    return smiles, doses, labels, cell_lines, rdkit_feats, weights

def build_dataset_from_csv(csv_path, cfg, augment=False, scaler=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    smiles_col = cfg['data']['smiles_col']
    label_col = cfg['data']['label_col']
    
    if df[label_col].isnull().any():
        df = df.dropna(subset=[label_col])
    
    smiles_list = df[smiles_col].astype(str).tolist()
    labels = df[label_col].values.astype(np.float32)
    
    if cfg['task']['type'] == 'regression':
        if np.max(np.abs(labels)) > 100:
            print("Auto-scaling labels (Log1p).")
            labels = np.log1p(labels)
            
    num_samples = len(smiles_list)
    fixed_dosage = float(cfg['data']['fixed_dosage'])
    dose_list = [fixed_dosage] * num_samples
    fixed_cell_line = str(cfg['data']['fixed_cell_line'])
    cell_lines = [fixed_cell_line] * num_samples

    print(f"Calculating RDKit descriptors for {num_samples} molecules (Parallel)...")
    
    features_list = Parallel(n_jobs=-1)(
        delayed(_calc_mol_features)(smi) for smi in smiles_list
    )
    
    features = np.array(features_list, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
    features = np.clip(features, a_min=-1e4, a_max=1e4)
    features = np.sign(features) * np.log1p(np.abs(features))
    if scaler is None:
        print("Fitting new StandardScaler (Train Mode)...")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        print("Applying existing StandardScaler (Val/Test Mode)...")
        features = scaler.transform(features)
        
    features = np.nan_to_num(features, nan=0.0)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    weight_config = cfg['training'].get('loss_weighting', None)
    return MoleculeDataset(smiles_list, dose_list, labels, cell_lines, features_tensor, augment=augment, weight_config=weight_config), scaler

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode
        
        if self.mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def __call__(self, current_val, model):
        
        is_improvement = False
        
        if self.mode == 'min':
            if current_val < self.best_score - self.delta:
                is_improvement = True
        else:
            if current_val > self.best_score + self.delta:
                is_improvement = True

        if is_improvement:
            self.best_score = current_val
            self.save_checkpoint(current_val, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}. Best: {self.best_score:.4f}, Current: {current_val:.4f}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            self.trace_func(f'Metric improved. Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)

def calculate_metrics(y_true, y_pred, task_type, metric_names=None):
    results = {}
    if not metric_names:
        return results

    y_pred_label = None
    
    if task_type == 'classification':
        y_pred_label = (y_pred > 0.5).astype(int)
    elif task_type == 'multiclass':
        y_pred_label = np.argmax(y_pred, axis=1)
        y_true = y_true.astype(int)

    for metric in metric_names:
        try:
            if task_type == 'regression':
                if metric == 'mse':
                    results['mse'] = mean_squared_error(y_true, y_pred)
                elif metric == 'rmse':
                    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                elif metric == 'mae':
                    results['mae'] = mean_absolute_error(y_true, y_pred)
                elif metric == 'r2':
                    results['r2'] = r2_score(y_true, y_pred)
                elif metric == 'pearson':
                    corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
                    results['pearson'] = corr

            elif task_type == 'classification':
                if metric == 'accuracy':
                    results['accuracy'] = accuracy_score(y_true, y_pred_label)
                elif metric == 'f1':
                    results['f1'] = f1_score(y_true, y_pred_label)
                elif metric == 'precision':
                    results['precision'] = precision_score(y_true, y_pred_label, zero_division=0)
                elif metric == 'recall':
                    results['recall'] = recall_score(y_true, y_pred_label, zero_division=0)
                elif metric == 'auc':
                    try:
                        results['auc'] = roc_auc_score(y_true, y_pred)
                    except ValueError:
                        results['auc'] = 0.0 

            if task_type == 'multiclass':
                if metric == 'accuracy':
                    results['accuracy'] = accuracy_score(y_true, y_pred_label)
                    
                elif metric == 'f1_macro':
                    results['f1_macro'] = f1_score(y_true, y_pred_label, average='macro')
                    
                elif metric == 'f1_micro':
                    results['f1_micro'] = f1_score(y_true, y_pred_label, average='micro')
                
                elif metric == 'auc':
                    try:
                        results['auc'] = roc_auc_score(
                            y_true, 
                            y_pred, 
                            multi_class='ovr', 
                            average='micro'
                        )
                    except ValueError as e:
                        print(f"Warning: AUC calculation failed (e.g., missing class in batch): {e}")
                        results['auc'] = 0.0
        except Exception as e:
            print(f"Warning: Could not calculate metric '{metric}': {e}")
            results[metric] = -1.0

    return results