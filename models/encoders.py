# models/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


ATOM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 
    'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 
    'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 
    'Pt', 'Hg', 'Pb'
] # 43 elements + 1 unknown

MAX_DEGREE = 10
MAX_NUM_HS = 8
MAX_VALENCE = 6
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
]

def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def get_atom_features(atom):
    features = []
    
    # 1. 原子类型 (Symbol) - One Hot
    features += one_hot_encoding(atom.GetSymbol(), ATOM_LIST)
    
    # 2. 度 (Degree) - One Hot
    features += one_hot_encoding(atom.GetTotalDegree(), list(range(MAX_DEGREE + 1)))
    
    # 3. 形式电荷 (Formal Charge) - One Hot
    features += one_hot_encoding(atom.GetFormalCharge(), FORMAL_CHARGES)
    
    # 4. 手性 (Chirality) - One Hot
    features += one_hot_encoding(atom.GetChiralTag(), CHIRALITY)
    
    # 5. 氢原子数 (Num Hs) - One Hot
    features += one_hot_encoding(atom.GetTotalNumHs(), list(range(MAX_NUM_HS + 1)))
    
    # 6. 杂化方式 (Hybridization) - One Hot
    features += one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION)
    
    # 7. 芳香性 (Aromaticity) - Boolean (0/1)
    features.append(1 if atom.GetIsAromatic() else 0)
    
    # 8. 原子质量 (Mass) - Float (Scaled by 0.01 to keep range small)
    features.append(atom.GetMass() * 0.01)
    
    return features

def get_bond_features(bond):
    features = []
    
    # 1. 键类型 (Bond Type) - One Hot
    bt = bond.GetBondType()
    features += one_hot_encoding(bt, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ])
    
    # 2. 共轭 (Conjugated) - Boolean
    features.append(1 if bond.GetIsConjugated() else 0)
    
    # 3. 在环中 (In Ring) - Boolean
    features.append(1 if bond.IsInRing() else 0)
    
    # 4. 立体化学 (Stereo) - One Hot
    features += one_hot_encoding(bond.GetStereo(), [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ])
    
    return features

_dummy_mol = Chem.MolFromSmiles("C")
ATOM_FEATURE_DIM = len(get_atom_features(_dummy_mol.GetAtoms()[0]))
BOND_FEATURE_DIM = len(get_bond_features(_dummy_mol.GetBonds()[0])) if _dummy_mol.GetBonds() else 14

print(f"DEBUG: Calculated Atom Feature Dim: {ATOM_FEATURE_DIM}")
print(f"DEBUG: Calculated Edge Feature Dim: {BOND_FEATURE_DIM}")


def smiles_to_graph_data(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    # 1. Atom Features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # 2. Edge Features
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = get_bond_features(bond)

            # Undirected graph: add both (i, j) and (j, i)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class RDKitEncoder(nn.Module):
    def __init__(self, d_model, input_dim=None):
        super().__init__()
        
        if input_dim is None:
            input_dim = len(Descriptors._descList)
        
        self.input_dim = input_dim
        
        self.ln = nn.LayerNorm(self.input_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model)
        )

    def forward(self, precomputed_features, device):
        """
        input: precomputed_features [Batch, input_dim]
        """
        x = precomputed_features.to(device)
        
        # Norm & Project
        x = self.ln(x)
        out = self.projection(x)
        return out

class MPNNEncoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim, out_dim, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # 2. GNN Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_impl = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn_impl, train_eps=True)) # train_eps=True allows the model to learn how much to weigh the central node's own features vs. its neighbors during message passing.

        # 3. Readout & Output
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, smiles_list, device):
        data_list = []
        for smi in smiles_list:
            data = smiles_to_graph_data(str(smi))
            if data is not None:
                data_list.append(data)
            else:
                dummy_x = torch.zeros((1, self.atom_encoder.in_features)) 
                dummy_edge_index = torch.empty((2, 0), dtype=torch.long)
                dummy_edge_attr = torch.empty((0, self.edge_encoder.in_features))
                data_list.append(Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr))
        
        batch_data = Batch.from_data_list(data_list).to(device)
        
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        
        # === 1. Embedding / Projection ===
        x = self.atom_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # === 2. Message Passing ===
        for conv in self.convs:
            h = conv(x, edge_index, edge_attr)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h # Residual connection
            
        # === 3. Readout (Global Pooling) ===
        x_graph = global_mean_pool(x, batch)
        
        # === 4. Final Projection ===
        out = self.output_layer(x_graph)
        
        return out