# models/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ExpertRouter(nn.Module):
    def __init__(self, prnet_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(prnet_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, prnet_features):
        logits = self.gate(prnet_features)
        weights = self.softmax(logits)
        return weights

class MoEBlock(nn.Module):
    def __init__(self, d_model, num_experts, expert_hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # 2. Experts
        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden_dim, d_model, dropout) 
            for _ in range(num_experts)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, fused_features, router_weights):
        attn_out, _ = self.attention(fused_features, fused_features, fused_features)
        x = self.norm(fused_features + attn_out)
        context_vector = x.mean(dim=1) 
        expert_outputs = torch.stack([exp(context_vector) for exp in self.experts], dim=1)
        router_weights = router_weights.unsqueeze(-1)
        final_output = torch.sum(expert_outputs * router_weights, dim=1)
        
        return final_output