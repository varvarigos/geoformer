# GeoFormer: Geometry-Aware Graph Transformer Framework

A unified non-Euclidean Graph Transformer framework that learns directly in curved manifolds. GeoFormer extends self-attention and feed-forward computation to non-Euclidean manifolds of learnable curvature κ, preserving geodesic distances and angular relations between nodes.

## 🌟 Key Features

- **Three Model Variants:**
  - **HyGT (Hyperbolic Graph Transformer)**: For tree-like and hierarchical datasets
  - **SpGT (Spherical Graph Transformer)**: For cyclic or densely clustered datasets  
  - **GeoFormer-Mix**: Multi-geometry attention where different heads operate in different geometric spaces

- **Manifold Operations:**
  - Exponential and logarithmic maps for Hyperbolic, Spherical, and Euclidean spaces
  - Geodesic distance computations
  - Manifold-aware linear projections and attention mechanisms

- **Comprehensive Benchmarking:**
  - Support for PROTEINS, AIDS, MUTAG, Cora, and AIRPORT datasets
  - Automated benchmarking across models and datasets
  - Visualization of results with publication-quality plots

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- PyTorch Geometric
- NumPy, SciPy, scikit-learn
- matplotlib, seaborn, pandas
- geoopt

## 🚀 Quick Start

### 1. Train a Single Model

Train GeoFormer-Mix on MUTAG dataset:

```bash
python3 src/main.py --mode train --model geoformer_mix --dataset MUTAG --epochs 100
```

Train HyGT on Cora (hierarchical citation network):

```bash
python3 src/main.py --mode train --model hygt --dataset CORA --epochs 200
```

Train SpGT on AIRPORT (cyclic transportation network):

```bash
python3 src/main.py --mode train --model spgt --dataset AIRPORT --epochs 200
```

### 2. Auto-Select Best Model

Let GeoFormer automatically suggest the best model based on dataset characteristics:

```bash
python3 src/main.py --mode train --model auto --dataset CORA
```

### 3. Run Comprehensive Benchmark

Benchmark all models across all datasets:

```bash
python3 src/main.py --mode benchmark
```

## 📁 Project Structure

```
geoformer/
├── src/
│   ├── model/
│   │   ├── manifolds.py        # Manifold operations (exp/log maps)
│   │   └── geoformer.py        # HyGT, SpGT, GeoFormer-Mix implementations
│   ├── data/
│   │   └── datasets.py         # Dataset loaders and utilities
│   ├── utils/
│   │   └── benchmark.py        # Benchmarking and visualization
│   ├── configs/
│   │   └── conf.yaml           # Configuration file
│   ├── train.py                # Training pipeline
│   └── main.py                 # Main entry point
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `src/configs/conf.yaml` to customize:

```yaml
model:
  type: 'geoformer_mix'  # 'hygt', 'spgt', or 'geoformer_mix'
  hidden_channels: 128
  num_layers: 4
  num_heads: 9
  dropout: 0.1
  hyperbolic_curvature: -1.0
  spherical_curvature: 1.0

dataset:
  name: 'MUTAG'
  batch_size: 32
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  optimizer: 'adam'
  scheduler: 'cosine'
  early_stopping_patience: 20
  device: 'cuda'

logging:
  use_wandb: false
  save_dir: './checkpoints'
```

## 🎯 Command Line Arguments

```bash
python src/main.py [OPTIONS]

Options:
  --config PATH           Path to config file (default: src/configs/conf.yaml)
  --mode {train,benchmark}  Operation mode
  --model {hygt,spgt,geoformer_mix,auto}  Model type
  --dataset NAME          Dataset name (PROTEINS, AIDS, MUTAG, CORA)
  --device {cuda,cpu}     Device to use
  --epochs INT            Number of training epochs
  --seed INT              Random seed
  --no-wandb              Disable WandB logging
```

## 📊 Supported Datasets

| Dataset | Type | Domain | Suggested Model | Characteristics |
|---------|------|--------|----------------|-----------------|
| **PROTEINS** | Graph Classification | Molecular | GeoFormer-Mix | Varying densities and biochemical connectivity |
| **AIDS** | Graph Classification | Molecular | GeoFormer-Mix | Molecular graphs with varying structures |
| **MUTAG** | Graph Classification | Molecular | GeoFormer-Mix | Small molecular compounds |
| **Cora** | Node Classification | Citation | HyGT | Citation network with hierarchical structure |

## 🏗️ Model Architecture

### Geometric Linear Layer
Operates in curved space via exponential and logarithmic maps:
1. Map point to tangent space at origin (log map)
2. Apply linear transformation
3. Map back to manifold (exp map)

### Geometric Attention
Attention mechanism adapted to curved spaces:
- Query, Key, Value projections in manifold
- Geodesic distance-based attention weighting
- Manifold-aware message passing

### GeoFormer-Mix Architecture
Multi-geometry attention with head-level mixing:
- Hyperbolic heads (κ < 0): Capture hierarchical patterns
- Euclidean heads (κ = 0): Model local neighborhoods  
- Spherical heads (κ > 0): Represent cyclic structures

## 📈 Benchmarking

Run benchmarks to compare all models:

```python
from src.utils.benchmark import Benchmark
from src.configs.conf import load_config

config = load_config('src/configs/conf.yaml')
benchmark = Benchmark(config, results_dir='./results')

benchmark.run_benchmark(
    datasets=['MUTAG', 'PROTEINS', 'CORA'],
    models=['hygt', 'spgt', 'geoformer_mix'],
    num_runs=3
)
```

Results are saved to `./results/` including:
- `results.csv`: Raw results
- `results.json`: Detailed metrics
- `summary.txt`: Statistical summary
- `benchmark_results.png`: Visualization plots
- `accuracy_comparison.png`: Accuracy comparison

## 🔬 Research Background

This implementation is based on the concept of geometry-aware graph transformers that:

1. **Preserve Geometric Structure**: By operating directly on curved manifolds, GeoFormer preserves the inherent geometric properties of graphs

2. **Adaptive Curvature**: Different graph structures benefit from different geometries:
   - **Hierarchical graphs** (trees, citation networks) → Hyperbolic space (κ < 0)
   - **Cyclic/clustered graphs** (social networks, transportation) → Spherical space (κ > 0)
   - **Mixed structure graphs** → Multi-geometry attention (GeoFormer-Mix)

3. **Manifold Operations**: All computations use proper Riemannian geometry:
   - Exponential map: exp_x(v) maps tangent vector to manifold
   - Logarithmic map: log_x(y) maps manifold point to tangent space
   - Geodesic distance: Shortest path in curved space
