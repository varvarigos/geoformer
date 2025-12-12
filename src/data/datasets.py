"""
Dataset loaders for GeoFormer benchmarking.
Supports PROTEINS, AIDS, MUTAG, Cora, and AIRPORT datasets.
"""

import torch
import numpy as np
from torch_geometric.datasets import TUDataset, Planetoid, Airports
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from typing import Tuple, Optional, List
import os


class DatasetLoader:
    """Unified interface for loading graph datasets."""
    
    def __init__(self, root: str = './data'):
        self.root = root
        os.makedirs(root, exist_ok=True)
    
    def load_dataset(self, name: str, task: str = 'classification'):
        """
        Load a dataset by name.
        
        Args:
            name: Dataset name ('PROTEINS', 'AIDS', 'MUTAG', 'Cora', 'AIRPORT')
            task: Task type ('classification', 'node_classification', 'graph_classification')
        
        Returns:
            dataset: PyTorch Geometric dataset
            num_features: Number of input features
            num_classes: Number of output classes
        """
        name = name.upper()
        
        if name in ['PROTEINS', 'AIDS', 'MUTAG']:
            return self._load_tu_dataset(name)
        elif name == 'CORA':
            return self._load_cora()
        elif name == 'AIRPORT':
            return self._load_airport()
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    def _load_tu_dataset(self, name: str):
        """Load TUDataset (PROTEINS, AIDS, MUTAG)."""
        dataset = TUDataset(root=os.path.join(self.root, name), name=name)
        
        # Add node features if not present
        if dataset[0].x is None:
            # Use one-hot degree encoding as features
            max_degree = 0
            for data in dataset:
                degrees = torch.bincount(data.edge_index[0])
                max_degree = max(max_degree, degrees.max().item())
            
            for data in dataset:
                degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
                # One-hot encode degrees
                one_hot = torch.zeros(data.num_nodes, max_degree + 1)
                one_hot[torch.arange(data.num_nodes), degrees] = 1
                data.x = one_hot
        
        num_features = dataset[0].x.shape[1]
        num_classes = dataset.num_classes
        
        return dataset, num_features, num_classes
    
    def _load_cora(self):
        """Load Cora citation network."""
        dataset = Planetoid(root=os.path.join(self.root, 'Cora'), name='Cora')
        
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        
        return dataset, num_features, num_classes
    
    def _load_airport(self):
        """Load AIRPORT dataset from PyTorch Geometric (USA variant)."""
        dataset = Airports(root=os.path.join(self.root, 'Airports'), name='USA')
        
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        
        return dataset, num_features, num_classes


def split_dataset(
    dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: PyTorch Geometric dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_graphs = len(dataset)
    indices = torch.randperm(num_graphs).tolist()
    
    train_size = int(train_ratio * num_graphs)
    val_size = int(val_ratio * num_graphs)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    return train_dataset, val_dataset, test_dataset


def get_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Create a DataLoader for graph datasets.
    
    Args:
        dataset: PyTorch Geometric dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader
    """
    from torch_geometric.loader import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def prepare_data(
    dataset_name: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    root: str = './data',
):
    """
    Convenience function to load and prepare dataset with dataloaders.
    
    Args:
        dataset_name: Name of dataset to load
        batch_size: Batch size for dataloaders
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        root: Root directory for datasets
    
    Returns:
        Dictionary with train/val/test loaders and dataset info
    """
    loader = DatasetLoader(root=root)
    dataset, num_features, num_classes = loader.load_dataset(dataset_name)
    
    # For node classification datasets (like Cora and AIRPORT), use built-in masks
    if dataset_name.upper() in ['CORA', 'AIRPORT']:
        data = dataset[0]
        
        # For AIRPORT, create train/val/test masks if not present
        if dataset_name.upper() == 'AIRPORT' and not hasattr(data, 'train_mask'):
            num_nodes = data.num_nodes
            indices = torch.randperm(num_nodes)
            
            train_size = int(train_ratio * num_nodes)
            val_size = int(val_ratio * num_nodes)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
            
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        
        return {
            'data': data,
            'num_features': num_features,
            'num_classes': num_classes,
            'task': 'node_classification',
        }
    
    # For graph classification datasets, split and create loaders
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )
    
    train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_features': num_features,
        'num_classes': num_classes,
        'task': 'graph_classification',
    }


# Dataset characteristics for reference
DATASET_INFO = {
    'PROTEINS': {
        'type': 'graph_classification',
        'domain': 'molecular',
        'characteristics': 'varying densities and biochemical connectivity',
        'suggested_geometry': 'mixed',  # Can use GeoFormer-Mix
    },
    'AIDS': {
        'type': 'graph_classification',
        'domain': 'molecular',
        'characteristics': 'molecular graphs with varying structures',
        'suggested_geometry': 'mixed',
    },
    'MUTAG': {
        'type': 'graph_classification',
        'domain': 'molecular',
        'characteristics': 'small molecular compounds',
        'suggested_geometry': 'mixed',
    },
    'CORA': {
        'type': 'node_classification',
        'domain': 'citation',
        'characteristics': 'citation network with hierarchical structure',
        'suggested_geometry': 'hyperbolic',  # Hierarchical -> HyGT
    },
    'AIRPORT': {
        'type': 'node_classification',
        'domain': 'transportation',
        'characteristics': 'transportation network with cyclic patterns',
        'suggested_geometry': 'spherical',  # Cyclic/clustered -> SpGT
    },
}


def get_suggested_model(dataset_name: str) -> str:
    """Get suggested model type based on dataset characteristics."""
    dataset_name = dataset_name.upper()
    if dataset_name in DATASET_INFO:
        geom = DATASET_INFO[dataset_name]['suggested_geometry']
        if geom == 'hyperbolic':
            return 'hygt'
        elif geom == 'spherical':
            return 'spgt'
        else:
            return 'geoformer_mix'
    return 'geoformer_mix'  # Default to mixed


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loading...")
    
    for dataset_name in ['MUTAG', 'CORA']:
        print(f"\nLoading {dataset_name}...")
        data_dict = prepare_data(dataset_name, batch_size=32)
        print(f"  Features: {data_dict['num_features']}")
        print(f"  Classes: {data_dict['num_classes']}")
        print(f"  Task: {data_dict['task']}")
        print(f"  Suggested model: {get_suggested_model(dataset_name)}")
