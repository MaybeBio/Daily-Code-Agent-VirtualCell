"""Trainable readouts shared by frozen representation adapters."""
import torch
import torch.nn.functional as F
from torch import nn

class Pooler_classify(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.proj_1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.proj_2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.proj_3 = nn.Linear(256, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, cell_emb: torch.Tensor) -> torch.Tensor:
        cell_emb = self.dropout(self.relu(self.bn1(self.proj_1(cell_emb))))
        cell_emb = self.dropout(self.relu(self.bn2(self.proj_2(cell_emb))))
        pooled_output = F.normalize(self.proj_3(cell_emb), dim=-1)
        return pooled_output

class Decoder(nn.Module):
    def __init__(self, emb_dim, gene_dim):
        super().__init__()
        self.proj_1 = nn.Linear(emb_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.proj_2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.proj_3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.proj_out = nn.Linear(512, gene_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.final_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.gelu(self.bn1(self.proj_1(x))))
        x = self.dropout(self.gelu(self.bn2(self.proj_2(x))))
        x = self.dropout(self.gelu(self.bn3(self.proj_3(x))))
        x = self.proj_out(x)
        return self.final_activation(x)

class PerturbDecoder(nn.Module):
    def __init__(self, emb_dim, pert_num, gene_num, pert_emb_dim=128):
        super().__init__()

        self.pert_emb = nn.Embedding(pert_num, pert_emb_dim, padding_idx=0)
        self.proj_1 = nn.Linear(emb_dim + pert_emb_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.proj_2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.proj_3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.proj_out = nn.Linear(512, gene_num)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.final_activation = nn.Identity()
        self.register_buffer('gene_index', None)

    def bind_genes(self, table):
        """Map each condition id to perturbation-gene embedding ids (0 = pad)."""
        tensor = table if torch.is_tensor(table) else torch.as_tensor(table, dtype=torch.long)
        self.gene_index = tensor.to(self.pert_emb.weight.device)

    def forward(self, ctrl_emb, pert_id):
        if self.gene_index is None:
            pert_vec = self.pert_emb(pert_id)
        else:
            idx = self.gene_index[pert_id]
            mask = idx > 0
            vecs = self.pert_emb(idx)
            weight = mask.unsqueeze(-1).to(vecs.dtype)
            pert_vec = vecs.mul(weight).sum(1) / weight.sum(1).clamp(min=1)
        x = torch.cat([ctrl_emb, pert_vec], dim=1)
        x = self.dropout(self.gelu(self.bn1(self.proj_1(x))))
        x = self.dropout(self.gelu(self.bn2(self.proj_2(x))))
        x = self.dropout(self.gelu(self.bn3(self.proj_3(x))))
        x = self.proj_out(x)
        return self.final_activation(x)
