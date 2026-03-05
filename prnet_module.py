import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc

from PRnet.models.PRnet import PRnet
from PRnet.data._utils import Drug_dose_encoder
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class PRnetBatchPredictor(nn.Module):
    def __init__(self, 
                 model_dir='./lincs_cache', 
                 gene_list_path=None, 
                 device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Init] Running on device: {self.device}")

        self._load_resources(model_dir)
        self._build_model(model_dir)

        self.sorted_indices = None
        if gene_list_path and os.path.exists(gene_list_path):
            self._load_gene_sorting(gene_list_path)
        else:
            print("[Warning] No gene list provided. Output will use default L1000 order.")

    def _load_resources(self, model_dir):
        adata_path = os.path.join(model_dir, 'Lincs_L1000.h5ad')
        print(f"[Init] Loading reference data from {adata_path}...")
        self.ref_adata = sc.read(adata_path)
        sc.pp.normalize_total(self.ref_adata)
        sc.pp.log1p(self.ref_adata)
        
        self.model_genes = np.array(self.ref_adata.var_names.tolist())
        self.n_vars = len(self.model_genes)
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.model_genes)}

    def _build_model(self, model_dir):
        self.prnet = PRnet(
            self.ref_adata,
            x_dimension=self.n_vars,
            hidden_layer_sizes=[128],
            z_dimension=64,
            adaptor_layer_sizes=[128],
            comb_dimension=64,
            comb_num=1,
            drug_dimension=1024,
            dr_rate=0.05
        )
        
        weight_path = os.path.join(model_dir, 'lincs_best_epoch_all.pt')
        state_dict = torch.load(weight_path, map_location=self.device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.prnet.PGM.load_state_dict(new_state_dict)
        self.prnet.PGM.to(self.device)
        self.prnet.PGM.eval()

    def _load_gene_sorting(self, gene_list_path):
        print(f"[Init] Loading user gene list from {gene_list_path}...")
        with open(gene_list_path, 'r') as f:
            user_genes = [line.strip() for line in f if line.strip()]
        
        valid_indices = []
        valid_genes = []
        
        for gene in user_genes:
            if gene in self.gene_to_idx:
                valid_indices.append(self.gene_to_idx[gene])
                valid_genes.append(gene)
        
        if len(valid_indices) == 0:
            raise ValueError("None of the genes in your list match the L1000 model genes!")
            
        print(f"[Init] {len(valid_indices)} genes matched out of {len(user_genes)} in list.")
        
        self.sorted_indices = torch.tensor(valid_indices, dtype=torch.long).to(self.device)
        self.sorted_gene_names = valid_genes

    def get_control_tensor(self, cell_line_name):
        ctrl_idx = self.ref_adata.obs[
            (self.ref_adata.obs['cell_id'] == cell_line_name) & 
            (self.ref_adata.obs['dose'] == 0.0)
        ].index
        if len(ctrl_idx) == 0:
            ctrl_idx = self.ref_adata.obs[
                (self.ref_adata.obs['cell_type'] == cell_line_name) & 
                (self.ref_adata.obs['dose'] == 0.0)
            ].index

        if len(ctrl_idx) == 0:
            print(set(self.ref_adata.obs['cell_id'].unique()))
            raise ValueError(f"Control sample not found for {cell_line_name}")

        control_vec = self.ref_adata[ctrl_idx[0]].X
        if hasattr(control_vec, "toarray"):
            control_vec = control_vec.toarray()
            
        return torch.tensor(control_vec, dtype=torch.float32).to(self.device)

    def forward(self, cell_line_name, smiles_list, dose_list, top_n=None, output_delta=True):
        batch_size = len(smiles_list)
        
        control_base = self.get_control_tensor(cell_line_name)
        control_tensor = control_base.repeat(batch_size, 1)

        drug_matrix = Drug_dose_encoder(smiles_list, dose_list, num_Bits=1024, comb_num=1)
        drug_tensor = torch.tensor(drug_matrix, dtype=torch.float32).to(self.device)

        noise = torch.zeros((batch_size, 10)).to(self.device)

        with torch.no_grad():
            full_output = self.prnet.PGM(control_tensor, drug_tensor, noise)
            dim = full_output.size(1) // 2
            pred_expression = full_output[:, :dim]

        if output_delta:
            # 预测值 - 对照值 = Log Fold Change
            final_output = pred_expression - control_tensor
        else:
            final_output = pred_expression

        if self.sorted_indices is not None:
            limit = len(self.sorted_indices)
            if top_n is not None:
                limit = min(top_n, limit)
            
            target_indices = self.sorted_indices[:limit]
            
            filtered_output = torch.index_select(final_output, 1, target_indices)
            
            return filtered_output
        else:
            if top_n:
                return final_output[:, :top_n]
            return final_output