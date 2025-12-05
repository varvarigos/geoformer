"""
GeoFormer: Geometry-Aware Graph Transformer Framework
Includes HyGT, SpGT, and GeoFormer-Mix variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, List, Tuple
import math

from .manifolds import Manifold, get_manifold, HyperbolicManifold, SphericalManifold, EuclideanManifold


class GeometricLinear(nn.Module):
    """Linear layer that operates in curved space via exponential and logarithmic maps."""
    
    def __init__(self, in_features: int, out_features: int, manifold: Manifold, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        
        # Weight matrix in tangent space (Euclidean)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Use smaller initialization for manifold operations to avoid numerical issues
        nn.init.xavier_uniform_(self.weight, gain=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform point x on manifold:
        1. Map to tangent space at origin
        2. Apply linear transformation
        3. Map back to manifold
        """
        # For spherical manifold, use origin at north pole for input space
        if isinstance(self.manifold, SphericalManifold):
            origin_in = torch.zeros_like(x)
            origin_in[..., -1] = self.manifold.radius
        else:
            origin_in = torch.zeros_like(x)
        
        # Map to tangent space
        v = self.manifold.log_map(origin_in, x)
        
        # Apply linear transformation
        v_transformed = F.linear(v, self.weight, self.bias)
        
        # Create origin for output space (matching transformed dimension)
        if isinstance(self.manifold, SphericalManifold):
            origin_out = torch.zeros_like(v_transformed)
            origin_out[..., -1] = self.manifold.radius
        else:
            origin_out = torch.zeros_like(v_transformed)
        
        # Map back to manifold
        y = self.manifold.exp_map(origin_out, v_transformed)
        
        return self.manifold.proj(y)


class GeometricAttention(MessagePassing):
    """
    Geometric attention mechanism for graph transformers.
    Performs attention computation in curved space.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.0,
        manifold: Manifold = None,
        add_self_loops: bool = False,
        bias: bool = True,
    ):
        super().__init__(node_dim=0, aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        if manifold is None:
            self.manifold = EuclideanManifold()
        else:
            self.manifold = manifold
        
        # Query, Key, Value projections
        self.lin_query = GeometricLinear(in_channels, heads * out_channels, self.manifold, bias=bias)
        self.lin_key = GeometricLinear(in_channels, heads * out_channels, self.manifold, bias=bias)
        self.lin_value = GeometricLinear(in_channels, heads * out_channels, self.manifold, bias=bias)
        
        # Output projection
        if concat:
            self.lin_out = GeometricLinear(heads * out_channels, heads * out_channels, self.manifold, bias=bias)
        else:
            self.lin_out = GeometricLinear(out_channels, out_channels, self.manifold, bias=bias)
        
        # Attention scaling
        self.scale = 1.0 / math.sqrt(out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_out.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features (optional)
            return_attention_weights: Whether to return attention weights
        """
        # Compute Q, K, V
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        
        # Message passing
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            size=None,
        )
        
        # Reshape and project output
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        out = self.lin_out(out)
        
        if return_attention_weights:
            # Note: This is simplified - full implementation would store attention
            return out, None
        
        return out
    
    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ):
        """
        Compute attention-weighted messages using geodesic distances.
        """
        # Compute attention scores using manifold distance
        # In tangent space: similarity via dot product after log map
        alpha = torch.sum(query_i * key_j, dim=-1) * self.scale
        
        # For non-Euclidean manifolds, incorporate geodesic distance penalty
        if not isinstance(self.manifold, EuclideanManifold):
            # Distance-based attention modulation
            # This encourages attention to nearby nodes in curved space
            dist = self.manifold.distance(query_i, key_j).squeeze(-1)
            alpha = alpha - 0.1 * dist  # Hyperparameter 0.1 can be tuned
        
        # Softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weighted values
        out = value_j * alpha.unsqueeze(-1)
        
        return out


class GeometricFeedForward(nn.Module):
    """Feed-forward network that operates in curved space."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        manifold: Manifold,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.manifold = manifold
        
        self.fc1 = GeometricLinear(in_features, hidden_features, manifold)
        self.fc2 = GeometricLinear(hidden_features, out_features, manifold)
        self.dropout = nn.Dropout(dropout)
        
        # Activation in tangent space
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation:
        1. Map to tangent space
        2. Apply transformations with activation
        3. Map back to manifold
        """
        # First transformation
        x = self.fc1(x)
        
        # Non-linearity in tangent space
        if isinstance(self.manifold, SphericalManifold):
            origin = torch.zeros_like(x)
            origin[..., -1] = self.manifold.radius
        else:
            origin = torch.zeros_like(x)
        
        v = self.manifold.log_map(origin, x)
        v = self.activation(v)
        v = self.dropout(v)
        x = self.manifold.exp_map(origin, v)
        
        # Second transformation
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class GeoTransformerLayer(nn.Module):
    """Single layer of geometric graph transformer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        manifold: Manifold,
        dropout: float = 0.1,
        ff_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.manifold = manifold
        
        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * hidden_dim
        
        # Attention layer
        self.attention = GeometricAttention(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            manifold=manifold,
        )
        
        # Feed-forward layer
        self.ff = GeometricFeedForward(
            in_features=hidden_dim,
            hidden_features=ff_hidden_dim,
            out_features=hidden_dim,
            manifold=manifold,
            dropout=dropout,
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        Note: Residuals are computed in tangent space for non-Euclidean geometries.
        """
        # Attention block with residual
        x_att = self.attention(x, edge_index)
        x = self._residual_connection(x, x_att)
        
        # Normalize in Euclidean space (map to tangent, normalize, map back)
        x_norm = self._geometric_norm(x, self.norm1)
        
        # Feed-forward block with residual
        x_ff = self.ff(x_norm)
        x = self._residual_connection(x_norm, x_ff)
        
        # Final normalization
        x = self._geometric_norm(x, self.norm2)
        
        return x
    
    def _residual_connection(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Compute residual connection in tangent space."""
        if isinstance(self.manifold, EuclideanManifold):
            return x + residual
        
        # For curved spaces: x ⊕ residual via exponential map
        if isinstance(self.manifold, SphericalManifold):
            origin = torch.zeros_like(x)
            origin[..., -1] = self.manifold.radius
        else:
            origin = torch.zeros_like(x)
        
        # Map residual to tangent space at x
        v = self.manifold.log_map(x, residual)
        
        # Small step along geodesic
        return self.manifold.exp_map(x, v)
    
    def _geometric_norm(self, x: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        """Apply normalization in tangent space."""
        if isinstance(self.manifold, EuclideanManifold):
            return norm_layer(x)
        
        if isinstance(self.manifold, SphericalManifold):
            origin = torch.zeros_like(x)
            origin[..., -1] = self.manifold.radius
        else:
            origin = torch.zeros_like(x)
        
        # Map to tangent space, normalize, map back
        v = self.manifold.log_map(origin, x)
        v_norm = norm_layer(v)
        return self.manifold.exp_map(origin, v_norm)


class HyGT(nn.Module):
    """
    Hyperbolic Graph Transformer for tree-like and hierarchical datasets.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        curvature: float = -1.0,
    ):
        super().__init__()
        
        self.manifold = HyperbolicManifold(curvature=curvature)
        
        # Input embedding
        self.embedding = GeometricLinear(in_channels, hidden_channels, self.manifold)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GeoTransformerLayer(
                hidden_dim=hidden_channels,
                num_heads=num_heads,
                manifold=self.manifold,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output = GeometricLinear(hidden_channels, out_channels, self.manifold)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through hyperbolic graph transformer."""
        # Embed to manifold
        x = self.embedding(x)
        x = self.manifold.proj(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Output projection
        x = self.output(x)
        
        return x


class SpGT(nn.Module):
    """
    Spherical Graph Transformer for cyclic or densely clustered datasets.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        curvature: float = 1.0,
    ):
        super().__init__()
        
        self.manifold = SphericalManifold(curvature=curvature)
        
        # Input embedding
        self.embedding = GeometricLinear(in_channels, hidden_channels, self.manifold)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GeoTransformerLayer(
                hidden_dim=hidden_channels,
                num_heads=num_heads,
                manifold=self.manifold,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output = GeometricLinear(hidden_channels, out_channels, self.manifold)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through spherical graph transformer."""
        # Embed to manifold
        x = self.embedding(x)
        x = self.manifold.proj(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Output projection
        x = self.output(x)
        
        return x


class MixedGeometricAttention(MessagePassing):
    """
    Multi-geometry attention where different heads operate in different geometries.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 9,  # Should be divisible by 3 for equal split
        concat: bool = True,
        dropout: float = 0.0,
        curvatures: List[float] = None,  # One per head
        bias: bool = True,
    ):
        super().__init__(node_dim=0, aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Default: split heads equally among geometries
        if curvatures is None:
            heads_per_geometry = heads // 3
            curvatures = (
                [-1.0] * heads_per_geometry +  # Hyperbolic
                [0.0] * heads_per_geometry +   # Euclidean
                [1.0] * (heads - 2 * heads_per_geometry)  # Spherical
            )
        
        assert len(curvatures) == heads, "Must provide one curvature per head"
        
        # Create manifold for each head
        self.manifolds = nn.ModuleList([
            get_manifold(curv) for curv in curvatures
        ])
        
        # Separate Q, K, V projections for each head
        self.head_projections = nn.ModuleList([
            nn.ModuleDict({
                'query': GeometricLinear(in_channels, out_channels, manifold, bias=bias),
                'key': GeometricLinear(in_channels, out_channels, manifold, bias=bias),
                'value': GeometricLinear(in_channels, out_channels, manifold, bias=bias),
            })
            for manifold in self.manifolds
        ])
        
        # Output projection
        if concat:
            self.lin_out = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        else:
            self.lin_out = nn.Linear(out_channels, out_channels, bias=bias)
        
        self.scale = 1.0 / math.sqrt(out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for proj_dict in self.head_projections:
            proj_dict['query'].reset_parameters()
            proj_dict['key'].reset_parameters()
            proj_dict['value'].reset_parameters()
        self.lin_out.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Forward pass with multi-geometry attention.
        Each head operates in its own geometric space.
        """
        head_outputs = []
        
        # Process each head in its respective geometry
        for h, (manifold, projections) in enumerate(zip(self.manifolds, self.head_projections)):
            # Compute Q, K, V for this head
            query = projections['query'](x)
            key = projections['key'](x)
            value = projections['value'](x)
            
            # Perform message passing for this head
            out_h = self._propagate_single_head(
                edge_index, query, key, value, manifold
            )
            
            head_outputs.append(out_h)
        
        # Concatenate or average head outputs
        if self.concat:
            out = torch.cat(head_outputs, dim=-1)
        else:
            out = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        # Final projection (in Euclidean space for simplicity)
        out = self.lin_out(out)
        
        if return_attention_weights:
            return out, None
        
        return out
    
    def _propagate_single_head(
        self,
        edge_index: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        manifold: Manifold,
    ):
        """Helper to propagate messages for a single head."""
        # Simple aggregation for mixed geometry
        row, col = edge_index
        
        # Compute attention scores
        alpha = torch.sum(query[row] * key[col], dim=-1) * self.scale
        
        # Distance-based modulation for non-Euclidean spaces
        if not isinstance(manifold, EuclideanManifold):
            dist = manifold.distance(query[row], key[col]).squeeze(-1)
            alpha = alpha - 0.1 * dist
        
        # Softmax normalization
        alpha = softmax(alpha, row, num_nodes=query.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate values
        out = torch.zeros_like(value)
        out.index_add_(0, row, value[col] * alpha.unsqueeze(-1))
        
        return out


class GeoFormerMix(nn.Module):
    """
    GeoFormer-Mix: Multi-Geometry Attention for Mixed-Curvature Data.
    Different attention heads operate in different geometric spaces.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 9,  # Divisible by 3 for equal geometry split
        dropout: float = 0.1,
        curvatures: List[float] = None,
    ):
        super().__init__()
        
        # Default curvature assignment if not provided
        if curvatures is None:
            heads_per_geometry = num_heads // 3
            curvatures = (
                [-1.0] * heads_per_geometry +  # Hyperbolic heads
                [0.0] * heads_per_geometry +   # Euclidean heads
                [1.0] * (num_heads - 2 * heads_per_geometry)  # Spherical heads
            )
        
        # Input embedding (Euclidean)
        self.embedding = nn.Linear(in_channels, hidden_channels)
        
        # Mixed-geometry transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attention = MixedGeometricAttention(
                in_channels=hidden_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                curvatures=curvatures,
            )
            
            # Projection to ensure dimension matches
            # (handle cases where hidden_channels is not divisible by num_heads)
            att_output_dim = (hidden_channels // num_heads) * num_heads
            att_proj = nn.Linear(att_output_dim, hidden_channels) if att_output_dim != hidden_channels else nn.Identity()
            
            ff = nn.Sequential(
                nn.Linear(hidden_channels, 4 * hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_channels, hidden_channels),
                nn.Dropout(dropout),
            )
            
            self.layers.append(nn.ModuleDict({
                'attention': attention,
                'att_proj': att_proj,
                'ff': ff,
                'norm1': nn.LayerNorm(hidden_channels),
                'norm2': nn.LayerNorm(hidden_channels),
            }))
        
        # Output projection
        self.output = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through mixed-geometry transformer."""
        # Input embedding
        x = self.embedding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            # Attention block with residual
            x_att = layer['attention'](x, edge_index)
            x_att = layer['att_proj'](x_att)  # Project to match hidden_channels if needed
            x = layer['norm1'](x + x_att)
            
            # Feed-forward block with residual
            x_ff = layer['ff'](x)
            x = layer['norm2'](x + x_ff)
        
        # Output projection
        x = self.output(x)
        
        return x


def build_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to build GeoFormer models.
    
    Args:
        model_type: One of ['hygt', 'spgt', 'geoformer_mix']
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional model-specific arguments
    """
    model_type = model_type.lower()
    
    if model_type == 'hygt':
        return HyGT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            curvature=kwargs.get('curvature', -1.0),
        )
    elif model_type == 'spgt':
        return SpGT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            curvature=kwargs.get('curvature', 1.0),
        )
    elif model_type == 'geoformer_mix':
        return GeoFormerMix(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            curvatures=kwargs.get('curvatures', None),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from ['hygt', 'spgt', 'geoformer_mix']")
