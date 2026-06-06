"""
Self-Supervised Teacher GNN Module


- DGI (Deep Graph Infomax)
- GraphCL (Graph Contrastive Learning)
- BGRL (Bootstrapped Graph Representation Learning)
- GCA (Graph Contrastive learning with Adaptive augmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge, add_self_loops, remove_self_loops
import numpy as np
from abc import ABC, abstractmethod


class GNNEncoder(nn.Module):
    """Base GNN Encoder for Self-Supervised Learning"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 dropout=0.5, backbone='gcn', heads=8):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = backbone.lower()
        self.heads = heads
        self.output_dim = output_dim
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        if self.backbone == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif self.backbone == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        elif self.backbone == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            if self.backbone == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.backbone == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif self.backbone == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer (output embedding)
        if num_layers > 1:
            if self.backbone == 'gcn':
                self.convs.append(GCNConv(hidden_dim, output_dim))
            elif self.backbone == 'sage':
                self.convs.append(SAGEConv(hidden_dim, output_dim))
            elif self.backbone == 'gat':
                self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
    
    def forward(self, x, edge_index):
        """Forward pass returning node embeddings"""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.num_layers > 1:
            x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get normalized embeddings"""
        embeddings = self.forward(x, edge_index)
        return F.normalize(embeddings, p=2, dim=-1)


class SelfSupervisedTeacher(ABC, nn.Module):
    """Abstract base class for Self-Supervised Teacher GNN"""
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=2,
                 dropout=0.5, backbone='gcn'):
        super(SelfSupervisedTeacher, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = backbone
    
    @abstractmethod
    def forward(self, data):
        """Forward pass for training"""
        pass
    
    @abstractmethod
    def get_embeddings(self, data):
        """Get node embeddings for knowledge transfer"""
        pass
    
    @abstractmethod
    def compute_loss(self, data):
        """Compute self-supervised loss"""
        pass


class DGITeacher(SelfSupervisedTeacher):
    """
    Deep Graph Infomax (DGI) Teacher
    
    """
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=2,
                 dropout=0.5, backbone='gcn'):
        super(DGITeacher, self).__init__(
            input_dim, hidden_dim, embedding_dim, num_layers, dropout, backbone
        )
        
        # GNN Encoder
        self.encoder = GNNEncoder(
            input_dim, hidden_dim, embedding_dim, 
            num_layers, dropout, backbone
        )
        
        # Readout function (graph-level summary)
        self.readout = lambda x: torch.sigmoid(x.mean(dim=0))
        
        # Discriminator (memory-efficient bilinear score)
        # nn.Bilinear can trigger very large intermediates for big N; use an explicit
        # projection + dot-product to keep memory O(N*D) instead of O(N*D^2).
        self.disc_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # For corruption
        self.corruption_ratio = 0.5
    
    def corrupt(self, x, edge_index):
        """Corruption function: shuffle node features"""
        perm = torch.randperm(x.size(0), device=x.device)
        return x[perm], edge_index
    
    def forward(self, data):
        """
        Returns:
        """
        x, edge_index = data.x, data.edge_index
        
        pos_embeddings = self.encoder(x, edge_index)
        
        # Graph summary
        summary = self.readout(pos_embeddings)
        
        x_corrupted, _ = self.corrupt(x, edge_index)
        neg_embeddings = self.encoder(x_corrupted, edge_index)
        
        return pos_embeddings, neg_embeddings, summary
    
    def compute_loss(self, data):
        """DGI Loss: Binary Cross Entropy for Discriminator"""
        pos_emb, neg_emb, summary = self.forward(data)
        
        # Discriminator scores: s(h, c) = (W h) · c
        summary_expanded = summary.unsqueeze(0).expand(pos_emb.size(0), -1)
        pos_scores = (self.disc_proj(pos_emb) * summary_expanded).sum(dim=-1)
        neg_scores = (self.disc_proj(neg_emb) * summary_expanded).sum(dim=-1)
        
        # Labels
        pos_labels = torch.ones(pos_emb.size(0), device=pos_emb.device)
        neg_labels = torch.zeros(neg_emb.size(0), device=neg_emb.device)
        
        # BCE Loss
        loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels) + \
               F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
        
        return loss
    
    def get_embeddings(self, data):
        """Get node embeddings for knowledge transfer"""
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            embeddings = self.encoder(x, edge_index)
            return F.normalize(embeddings, p=2, dim=-1)


class GraphCLTeacher(SelfSupervisedTeacher):
    """
    Graph Contrastive Learning (GraphCL) Teacher
    
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_layers=2,
        dropout=0.5,
        backbone='gcn',
        aug_ratio=0.2,
        temperature=0.5,
        max_nodes_for_loss: int = 4096,
    ):
        super(GraphCLTeacher, self).__init__(
            input_dim, hidden_dim, embedding_dim, num_layers, dropout, backbone
        )
        
        self.aug_ratio = aug_ratio
        self.temperature = temperature
        self.max_nodes_for_loss = int(max_nodes_for_loss) if max_nodes_for_loss is not None else 0
        
        # Shared GNN Encoder
        self.encoder = GNNEncoder(
            input_dim, hidden_dim, embedding_dim,
            num_layers, dropout, backbone
        )
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def augment_graph(self, x, edge_index):
        """Graph augmentation: edge dropout + feature masking"""
        # Edge dropout
        edge_index_aug, _ = dropout_edge(edge_index, p=self.aug_ratio, training=True)
        
        # Feature masking
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.aug_ratio))
        x_aug = x * mask
        
        return x_aug, edge_index_aug
    
    def forward(self, data):
        """
        Returns:
        """
        x, edge_index = data.x, data.edge_index
        
        # View 1
        x1, edge_index1 = self.augment_graph(x, edge_index)
        h1 = self.encoder(x1, edge_index1)
        z1 = self.projector(h1)
        
        # View 2  
        x2, edge_index2 = self.augment_graph(x, edge_index)
        h2 = self.encoder(x2, edge_index2)
        z2 = self.projector(h2)
        
        return z1, z2
    
    def compute_loss(self, data):
        """InfoNCE Contrastive Loss"""
        z1, z2 = self.forward(data)
        
        # Normalize
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        
        # Similarity matrix is O(N^2) memory; cap nodes for large graphs
        batch_size = z1.size(0)
        if self.max_nodes_for_loss and batch_size > self.max_nodes_for_loss:
            idx = torch.randperm(batch_size, device=z1.device)[: self.max_nodes_for_loss]
            z1 = z1[idx]
            z2 = z2[idx]
            batch_size = z1.size(0)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Positive pairs: diagonal elements
        labels = torch.arange(batch_size, device=z1.device)
        
        # InfoNCE loss (symmetric)
        loss = (F.cross_entropy(sim_matrix, labels) + 
                F.cross_entropy(sim_matrix.t(), labels)) / 2
        
        return loss
    
    def get_embeddings(self, data):
        """Get node embeddings for knowledge transfer"""
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            embeddings = self.encoder(x, edge_index)
            return F.normalize(embeddings, p=2, dim=-1)


class BGRLTeacher(SelfSupervisedTeacher):
    """
    Bootstrapped Graph Representation Learning (BGRL) Teacher
    
    - Online encoder + Target encoder (EMA update)
    """
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=2,
                 dropout=0.5, backbone='gcn', aug_ratio=0.2, momentum=0.99):
        super(BGRLTeacher, self).__init__(
            input_dim, hidden_dim, embedding_dim, num_layers, dropout, backbone
        )
        
        self.aug_ratio = aug_ratio
        self.momentum = momentum
        
        # Online encoder
        self.online_encoder = GNNEncoder(
            input_dim, hidden_dim, embedding_dim,
            num_layers, dropout, backbone
        )
        
        # Target encoder (EMA of online encoder)
        self.target_encoder = GNNEncoder(
            input_dim, hidden_dim, embedding_dim,
            num_layers, dropout, backbone
        )
        
        # Initialize target encoder with online encoder weights
        self._init_target_encoder()
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def _init_target_encoder(self):
        """Initialize target encoder with online encoder weights"""
        for param_o, param_t in zip(self.online_encoder.parameters(), 
                                     self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder"""
        for param_o, param_t in zip(self.online_encoder.parameters(),
                                     self.target_encoder.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_o.data
    
    def augment_graph(self, x, edge_index):
        """Graph augmentation"""
        edge_index_aug, _ = dropout_edge(edge_index, p=self.aug_ratio, training=True)
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.aug_ratio))
        x_aug = x * mask
        return x_aug, edge_index_aug
    
    def forward(self, data):
        """
        Returns:
            online_pred: Online encoder + predictor output
            target_proj: Target encoder output (detached)
        """
        x, edge_index = data.x, data.edge_index
        
        # View 1 (online)
        x1, edge_index1 = self.augment_graph(x, edge_index)
        online_emb = self.online_encoder(x1, edge_index1)
        online_pred = self.predictor(online_emb)
        
        # View 2 (target)
        x2, edge_index2 = self.augment_graph(x, edge_index)
        with torch.no_grad():
            target_emb = self.target_encoder(x2, edge_index2)
        
        return online_pred, target_emb.detach()
    
    def compute_loss(self, data):
        """BGRL Loss: Cosine similarity"""
        online_pred, target_proj = self.forward(data)
        
        # Normalize
        online_pred = F.normalize(online_pred, p=2, dim=-1)
        target_proj = F.normalize(target_proj, p=2, dim=-1)
        
        # Negative cosine similarity
        loss = 2 - 2 * (online_pred * target_proj).sum(dim=-1).mean()
        
        return loss
    
    def get_embeddings(self, data):
        """Get node embeddings from online encoder"""
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            embeddings = self.online_encoder(x, edge_index)
            return F.normalize(embeddings, p=2, dim=-1)


class GCATeacher(SelfSupervisedTeacher):
    """
    Graph Contrastive learning with Adaptive augmentation (GCA) Teacher
    
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_layers=2,
        dropout=0.5,
        backbone='gcn',
        temperature=0.5,
        max_nodes_for_loss: int = 4096,
    ):
        super(GCATeacher, self).__init__(
            input_dim, hidden_dim, embedding_dim, num_layers, dropout, backbone
        )
        
        self.temperature = temperature
        self.max_nodes_for_loss = int(max_nodes_for_loss) if max_nodes_for_loss is not None else 0
        
        self.encoder = GNNEncoder(
            input_dim, hidden_dim, embedding_dim,
            num_layers, dropout, backbone
        )
        
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def compute_node_centrality(self, edge_index, num_nodes):
        """Compute degree centrality"""
        row, col = edge_index
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))
        
        # Normalize to [0, 1]
        centrality = (degree - degree.min()) / (degree.max() - degree.min() + 1e-8)
        return centrality
    
    def adaptive_augment(self, x, edge_index, drop_prob=0.3):
        """Adaptive graph augmentation based on centrality"""
        num_nodes = x.size(0)
        centrality = self.compute_node_centrality(edge_index, num_nodes)
        
        # Edge drop probability inversely proportional to endpoint centrality
        row, col = edge_index
        edge_centrality = (centrality[row] + centrality[col]) / 2
        edge_drop_prob = drop_prob * (1 - edge_centrality)
        
        # Drop edges
        mask = torch.bernoulli(1 - edge_drop_prob).bool()
        edge_index_aug = edge_index[:, mask]
        
        # Feature masking (less masking for central nodes)
        feature_keep_prob = 1 - drop_prob * (1 - centrality.unsqueeze(-1))
        mask = torch.bernoulli(feature_keep_prob)
        x_aug = x * mask
        
        return x_aug, edge_index_aug
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Two augmented views
        x1, edge_index1 = self.adaptive_augment(x, edge_index)
        x2, edge_index2 = self.adaptive_augment(x, edge_index)
        
        h1 = self.encoder(x1, edge_index1)
        h2 = self.encoder(x2, edge_index2)
        
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        return z1, z2
    
    def compute_loss(self, data):
        """InfoNCE loss"""
        z1, z2 = self.forward(data)
        
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        
        batch_size = z1.size(0)
        if self.max_nodes_for_loss and batch_size > self.max_nodes_for_loss:
            idx = torch.randperm(batch_size, device=z1.device)[: self.max_nodes_for_loss]
            z1 = z1[idx]
            z2 = z2[idx]
            batch_size = z1.size(0)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(batch_size, device=z1.device)
        
        loss = (F.cross_entropy(sim_matrix, labels) + 
                F.cross_entropy(sim_matrix.t(), labels)) / 2
        
        return loss
    
    def get_embeddings(self, data):
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            embeddings = self.encoder(x, edge_index)
            return F.normalize(embeddings, p=2, dim=-1)


def create_ssl_teacher(method: str, input_dim: int, hidden_dim: int, 
                       embedding_dim: int, **kwargs) -> SelfSupervisedTeacher:
    """
    Factory function to create Self-Supervised Teacher
    
    Args:
        method: 'dgi', 'graphcl', 'bgrl', 'gca'
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        embedding_dim: Output embedding dimension
        **kwargs: Additional arguments for specific methods
    
    Returns:
        SelfSupervisedTeacher instance
    """
    method = method.lower()
    if method == 'grace':
        method = 'graphcl'
    
    if method == 'dgi':
        return DGITeacher(input_dim, hidden_dim, embedding_dim, **kwargs)
    elif method == 'graphcl':
        return GraphCLTeacher(input_dim, hidden_dim, embedding_dim, **kwargs)
    elif method == 'bgrl':
        return BGRLTeacher(input_dim, hidden_dim, embedding_dim, **kwargs)
    elif method == 'gca':
        return GCATeacher(input_dim, hidden_dim, embedding_dim, **kwargs)
    else:
        raise ValueError(f"Unknown SSL method: {method}. "
                        f"Available: dgi, graphcl, bgrl, gca")


class SSLTeacherTrainer:
    """Trainer for Self-Supervised Teacher GNN"""
    
    def __init__(self, teacher: SelfSupervisedTeacher, lr=0.001, weight_decay=1e-5):
        self.teacher = teacher
        self.optimizer = torch.optim.Adam(
            teacher.parameters(), lr=lr, weight_decay=weight_decay
        )
    
    def train_epoch(self, data):
        """Train for one epoch"""
        self.teacher.train()
        self.optimizer.zero_grad()
        
        loss = self.teacher.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        
        # Update target encoder for BGRL
        if isinstance(self.teacher, BGRLTeacher):
            self.teacher.update_target_encoder()
        
        return loss.item()
    
    def train(self, data, epochs=200, verbose=True):
        """Full training loop"""
        losses = []
        
        for epoch in range(epochs):
            loss = self.train_epoch(data)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"[SSL Teacher] Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
