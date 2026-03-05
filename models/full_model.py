# models/full_model.py

import torch
import torch.nn as nn
import yaml
import sys
from .encoders import RDKitEncoder, MPNNEncoder
from .layers import MoEBlock, ExpertRouter

sys.path.append('..') 
from prnet_module import PRnetBatchPredictor

class MoleculeMoEPRnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.cfg = config
        
        d_model = self.cfg['model']['d_model']
        self.device = torch.device(self.cfg['project']['device'])
        
        # ================= ENCODERS =================
        # 1. Molecule Encoder (MPNN)
        mpnn_cfg = self.cfg['model']['encoders']['mpnn']
        self.mpnn = MPNNEncoder(
            atom_dim=mpnn_cfg['atom_dim'],
            edge_dim=mpnn_cfg['edge_dim'],
            hidden_dim=mpnn_cfg['hidden_channels'],
            out_dim=d_model,
            num_layers=mpnn_cfg['num_layers'],
            dropout=self.cfg['model']['dropout']
        )
        
        
        # 2. Global Encoder Part A (RDKit)
        self.rdkit_encoder = RDKitEncoder(d_model=d_model)
        
        # 3. Global Encoder Part B (PRnet)
        self.prnet_predictor = PRnetBatchPredictor(
            model_dir=self.cfg['data']['prnet_cache_dir'],
            gene_list_path=self.cfg['data']['gene_list_path'],
            device=self.cfg['project']['device']
        )
        self.prnet_projection = nn.Linear(
            self.cfg['data']['top_n_genes'], 
            d_model
        )
        
        # ================= ROUTER =================
        # 只有 PRnet 的信息进入 Router
        self.router = ExpertRouter(
            prnet_dim=self.cfg['data']['top_n_genes'],
            num_experts=self.cfg['model']['moe']['num_experts']
        )
        
        # ================= MoE BLOCK =================
        self.moe_block = MoEBlock(
            d_model=d_model,
            num_experts=self.cfg['model']['moe']['num_experts'],
            expert_hidden_dim=self.cfg['model']['moe']['expert_hidden_dim'],
            num_heads=self.cfg['model']['moe']['num_heads'],
            dropout=self.cfg['model']['dropout']
        )
        
        # ================= TASK HEAD =================
        task_type = self.cfg['task']['type']
        num_classes = self.cfg['task']['num_classes']
        d_model = self.cfg['model']['d_model']
        
        self.predict_uncertainty = self.cfg['task'].get('predict_uncertainty', False)
        output_dim = num_classes * 2 if self.predict_uncertainty else num_classes
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim)
        )

        self.to(self.device)

    def forward(self, smiles_list, dose_list, cell_line_name, rdkit_features, return_router_weights=False):
        
        # A. MPNN -> [Batch, d_model]
        feat_mpnn = self.mpnn(smiles_list, self.device)
        
        # B. RDKit -> [Batch, d_model]
        feat_rdkit = self.rdkit_encoder(rdkit_features, self.device)
        
        # C. PRnet (Raw) -> [Batch, Top_N]
        with torch.no_grad():
            feat_prnet_raw = self.prnet_predictor(
                cell_line_name=cell_line_name,
                smiles_list=smiles_list,
                dose_list=dose_list,
                top_n=self.cfg['data']['top_n_genes'],
                output_delta=self.cfg['data']['prnet_delta']
            )
        
        # PRnet Proj -> [Batch, d_model]
        feat_prnet_proj = self.prnet_projection(feat_prnet_raw)
        
        # Stack -> [Batch, 3, d_model]
        # Sequence order: [MPNN, RDKit, PRnet]
        fused_features = torch.stack([feat_mpnn, feat_rdkit, feat_prnet_proj], dim=1)
        
        # Weights -> [Batch, Num_Experts]
        router_weights = self.router(feat_prnet_raw)
        
        # Output -> [Batch, d_model]
        moe_output = self.moe_block(fused_features, router_weights)
        
        logits = self.output_head(moe_output)
        
        task_type = self.cfg['task']['type']
        
        task_type = self.cfg['task']['type']
        if task_type == 'classification':
            output = torch.sigmoid(logits)
        elif task_type == 'multiclass':
            output = torch.softmax(logits, dim=1)
        else: 
            output = logits
        if return_router_weights:
            return output, router_weights
        else:
            return output