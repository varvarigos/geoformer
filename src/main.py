"""
GeoFormer: Geometry-Aware Graph Transformer Framework
Main entry point for training and evaluation.
"""

import argparse
import yaml
import torch
import numpy as np
import os
import sys

from data.datasets import prepare_data, get_suggested_model
from model.geoformer import build_model
from train import train_model
from utils.benchmark import Benchmark


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_single_model(config: dict):
    """Train a single model on a single dataset."""
    print("\n" + "="*80)
    print("GEOFORMER: GEOMETRY-AWARE GRAPH TRANSFORMER")
    print("="*80)
    print(f"Model: {config['model']['type'].upper()}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Device: {config['training']['device']}")
    print("="*80 + "\n")
    
    # Set seeds (handle both int and list)
    seeds_config = config['experiment'].get('seeds', 42)
    if isinstance(seeds_config, int):
        seed = seeds_config
    else:
        # For single train mode, use first seed from list
        seed = seeds_config[0]
    set_seeds(seed)
    
    # Prepare data
    print("Loading dataset...")
    data_dict = prepare_data(
        dataset_name=config['dataset']['name'],
        batch_size=config['dataset']['batch_size'],
        train_ratio=config['dataset']['train_ratio'],
        val_ratio=config['dataset']['val_ratio'],
        test_ratio=config['dataset']['test_ratio'],
        seed=config['dataset']['seed'],
        root=config['dataset']['root'],
    )
    
    print(f"  Number of features: {data_dict['num_features']}")
    print(f"  Number of classes: {data_dict['num_classes']}")
    print(f"  Task: {data_dict['task']}\n")
    
    # Build model
    print("Building model...")
    model_type = config['model']['type']
    
    # Get appropriate curvature
    if model_type == 'hygt':
        curvature = config['model']['hyperbolic_curvature']
    elif model_type == 'spgt':
        curvature = config['model']['spherical_curvature']
    else:
        curvature = None
    
    model = build_model(
        model_type=model_type,
        in_channels=data_dict['num_features'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=data_dict['num_classes'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        curvature=curvature,
        curvatures=config['model'].get('curvatures', None),
        learnable_curvature=config['model'].get('learnable_curvature', False),
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}\n")
    
    # Setup WandB if enabled
    if config['logging']['use_wandb']:
        try:
            import wandb
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                name=f"{model_type}_{config['dataset']['name']}",
                config=config,
            )
            print("WandB logging enabled\n")
        except:
            print("WandB not available, skipping logging\n")
    
    # Train model
    print("Starting training...\n")
    train_config = {
        'device': config['training']['device'],
        'learning_rate': config['training']['learning_rate'],
        'curvature_learning_rate': config['training'].get('curvature_learning_rate', None),
        'weight_decay': config['training']['weight_decay'],
        'optimizer': config['training']['optimizer'],
        'scheduler': config['training']['scheduler'],
        'num_epochs': config['training']['num_epochs'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'use_wandb': config['logging']['use_wandb'],
        'model_type': model_type,
        'dataset': config['dataset']['name'],
    }
    
    results = train_model(
        model=model,
        data_dict=data_dict,
        config=train_config,
        save_dir=config['logging']['save_dir'],
    )
    
    # Print final results
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print(f"Test F1 Score: {results['test_f1']:.4f}")
    print("="*80 + "\n")
    
    if config['logging']['use_wandb']:
        try:
            import wandb
            wandb.finish()
        except:
            pass
    
    return results


def run_benchmark(config: dict):
    """Run comprehensive benchmark across datasets and models."""
    print("\n" + "="*80)
    print("GEOFORMER BENCHMARK SUITE")
    print("="*80 + "\n")
    
    # Get first seed for initialization (handle both int and list)
    seeds_config = config['experiment'].get('seeds', 42)
    if isinstance(seeds_config, int):
        first_seed = seeds_config
    else:
        first_seed = seeds_config[0]
    set_seeds(first_seed)
    
    # Initialize benchmark
    benchmark = Benchmark(
        config=config,
        results_dir=config['benchmark']['results_dir'],
    )
    
    # Run benchmark (seeds are now read from config inside benchmark)
    benchmark.run_benchmark(
        datasets=config['benchmark']['datasets'],
        models=config['benchmark']['models'],
    )
    
    print("\nBenchmark results saved to:", config['benchmark']['results_dir'])


def auto_select_model(config: dict):
    """Automatically select best model based on dataset characteristics."""
    dataset_name = config['dataset']['name']
    suggested_model = get_suggested_model(dataset_name)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Suggested model based on graph characteristics: {suggested_model.upper()}")
    
    response = input(f"Use suggested model '{suggested_model}'? (y/n): ").lower()
    
    if response == 'y':
        config['model']['type'] = suggested_model
        print(f"Using {suggested_model.upper()} for {dataset_name}\n")
    else:
        print(f"Keeping configured model: {config['model']['type'].upper()}\n")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='GeoFormer: Geometry-Aware Graph Transformer Framework'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/conf.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'benchmark'],
        default='train',
        help='Operation mode: train a single model or run benchmark'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['hygt', 'spgt', 'geoformer_mix', 'auto'],
        help='Model type (overrides config)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model and args.model != 'auto':
        config['model']['type'] = args.model
    
    if args.dataset:
        config['dataset']['name'] = args.dataset
    
    if args.device:
        config['training']['device'] = args.device
    
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    if args.seed:
        config['experiment']['seed'] = args.seed
        config['dataset']['seed'] = args.seed
    
    if args.no_wandb:
        config['logging']['use_wandb'] = False
    
    # Auto-select model if requested
    if args.model == 'auto':
        config = auto_select_model(config)
    
    # Run appropriate mode
    if args.mode == 'train':
        train_single_model(config)
    elif args.mode == 'benchmark':
        run_benchmark(config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
