# trainer/core.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import yaml

from .utils import EarlyStopping, calculate_metrics, custom_collate_fn

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config, logger=print):
        self.model = model
        self.logger = logger
        
        self.cfg = config
            
        self.device = torch.device(self.cfg['project']['device'])
        self.task_type = self.cfg['task']['type']

        self.predict_uncertainty = self.cfg['task'].get('predict_uncertainty', False)

        self.metric_names = self.cfg['task'].get('metrics', [])
        if not self.metric_names:
            if self.task_type == 'regression': self.metric_names = ['mse', 'r2']
            elif self.task_type == 'classification': self.metric_names = ['accuracy', 'auc']
            else: self.metric_names = ['accuracy']
        
        # DataLoader
        self.batch_size = self.cfg.get('training', {}).get('batch_size', 32)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, collate_fn=custom_collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, collate_fn=custom_collate_fn
        )
        
        lr = float(self.cfg.get('training', {}).get('lr', 1e-4))
        weight_decay = float(self.cfg.get('training', {}).get('weight_decay', 1e-5))
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self._setup_criterion()
        
        # Warm-up + Linear Decay
        self.epochs = self.cfg.get('training', {}).get('epochs', 100)
        warmup_epochs = self.cfg.get('training', {}).get('warmup_epochs', 5)
        self.total_steps = len(self.train_loader) * self.epochs
        self.warmup_steps = len(self.train_loader) * warmup_epochs
        
        self.scheduler = self._get_scheduler()
        
        patience = self.cfg.get('training', {}).get('patience', 10)
        save_path = os.path.join(self.cfg.get('training', {}).get('save_dir', './checkpoints'), f'best_model.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.monitor_metric = self.cfg.get('training', {}).get('monitor_metric', 'loss')
        self.monitor_mode = self.cfg.get('training', {}).get('monitor_mode', 'min')
        
        self.early_stopping = EarlyStopping(
            patience=patience, 
            verbose=True, 
            path=save_path, 
            trace_func=self.logger,
            mode=self.monitor_mode
        )

    def _setup_criterion(self):
        if self.task_type == 'regression':
            if self.predict_uncertainty:
                self.criterion = nn.GaussianNLLLoss(reduction='none')
                print("-> Using GaussianNLLLoss (Aleatoric Uncertainty Mode).")
            else:
                self.criterion = nn.MSELoss(reduction='none')
        elif self.task_type == 'classification':
            self.criterion = nn.BCELoss()
        elif self.task_type == 'multiclass':
            self.criterion = nn.NLLLoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _get_scheduler(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0, float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
            )
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training', leave=False)
        
        for batch_idx, (smiles_list, dose_list, labels, cell_lines, rdkit_feats, weights) in loop:
            
            labels = labels.to(self.device)
            rdkit_feats = rdkit_feats.to(self.device)
            weights = weights.to(self.device)
            if self.task_type == 'multiclass':
                labels = labels.long()
            

            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print(f"\n[CRASH DETECTED] Batch {batch_idx}: Found NaN/Inf in LABELS!")
                print("Bad Labels content:", labels)
                bad_idx = torch.where(torch.isnan(labels) | torch.isinf(labels))[0]
                print("Bad Indices:", bad_idx)
                print("Corresponding SMILES:", [smiles_list[i] for i in bad_idx])
                raise ValueError("Training stopped due to bad labels.")

            cell_line_name = cell_lines[0] 
            
            outputs = self.model(smiles_list, dose_list, cell_line_name, rdkit_feats)
            if self.task_type == 'regression' and self.predict_uncertainty:
                mu = outputs[:, 0]
                log_var = outputs[:, 1]
                
                log_var = torch.clamp(log_var, min=-3.0, max=5.0)
                
                var = torch.exp(log_var)
                
                raw_loss = self.criterion(mu, labels, var)
                weighted_loss = raw_loss * weights
                loss = weighted_loss.mean()
            else:
                if outputs.dim() == 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(-1)
                
                if self.task_type == 'classification':
                    outputs = outputs.squeeze(-1)
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\n[CRASH DETECTED] Batch {batch_idx}: Model output contains NaN/Inf!")
                    print("Model Output Min:", outputs.min().item(), "Max:", outputs.max().item())
                    print("Input SMILES causing crash (First 5):", smiles_list[:5])
                    
                    raise ValueError("Training stopped due to model outputting NaN.")

                # Compute Loss
                if self.task_type == 'multiclass':
                    loss = self.criterion(torch.log(outputs + 1e-9), labels)
                else:
                    raw_loss = self.criterion(outputs, labels)
                    loss = (raw_loss * weights).mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[CRASH DETECTED] Batch {batch_idx}: Loss is NaN/Inf!")
                    print("Outputs:", outputs.flatten()[:10])
                    print("Labels:", labels.flatten()[:10])
                    raise ValueError("Training stopped due to NaN Loss.")

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for smiles_list, dose_list, labels, cell_lines, rdkit_feats, weights in loader:
                labels = labels.to(self.device)
                rdkit_feats = rdkit_feats.to(self.device)
                if self.task_type == 'multiclass':
                    labels = labels.long()
                
                cell_line_name = cell_lines[0]
                
                outputs = self.model(smiles_list, dose_list, cell_line_name, rdkit_feats)

                if self.task_type == 'regression' and self.predict_uncertainty:
                    mu = outputs[:, 0]
                    log_var = outputs[:, 1]
                    log_var = torch.clamp(log_var, min=-10, max=10)
                    var = torch.exp(log_var)
                    
                    batch_loss_vector = self.criterion(mu, labels, var)
                    
                    current_preds = mu
                
                else:
                    if self.task_type == 'classification':
                        outputs = outputs.squeeze(-1)
                    
                    if self.task_type == 'multiclass':
                        batch_loss_vector = self.criterion(torch.log(outputs + 1e-9), labels)
                    elif self.task_type == 'multiclass': # 三分类
                        outputs = torch.softmax(outputs, dim=1)
                    else:
                        if outputs.dim() == 2 and outputs.shape[1] == 1:
                            outputs = outputs.squeeze(-1)
                        batch_loss_vector = self.criterion(outputs, labels)
                    current_preds = outputs

                loss = batch_loss_vector.mean()
                total_loss += loss.item()
                
                all_preds.append(current_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        

        metrics = calculate_metrics(
            all_labels, 
            all_preds, 
            self.task_type, 
            metric_names=self.metric_names
        )
        metrics['loss'] = total_loss / len(loader)
        
        return metrics

    def fit(self):
        self.logger(f"Start training for {self.epochs} epochs on {self.device}...")
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val: {metric_str}")
            
            # Early Stopping Check
            if self.monitor_metric not in val_metrics:
                self.logger(f"Warning: Metric '{self.monitor_metric}' not found in validation metrics. Available: {list(val_metrics.keys())}")
                current_val = val_metrics['loss']
            else:
                current_val = val_metrics[self.monitor_metric]
            self.early_stopping(current_val, self.model)
            
            if self.early_stopping.early_stop:
                self.logger(f"Early stopping triggered. Best {self.monitor_metric}: {self.early_stopping.best_score:.4f}")
                break
                
        self.logger("Training complete.")