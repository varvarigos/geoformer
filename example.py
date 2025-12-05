"""
Quick example demonstrating GeoFormer usage
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import build_model, HyGT, SpGT, GeoFormerMix
from data import prepare_data
from train import Trainer


def example_train_hygt():
    """Example: Train HyGT on Cora dataset"""
    print("\n" + "="*80)
    print("Example 1: Training HyGT on Cora (Hierarchical Citation Network)")
    print("="*80 + "\n")
    
    # Prepare data
    data_dict = prepare_data(
        dataset_name='CORA',
        batch_size=32,
        root='./data'
    )
    
    print(f"Dataset loaded:")
    print(f"  Features: {data_dict['num_features']}")
    print(f"  Classes: {data_dict['num_classes']}")
    print(f"  Task: {data_dict['task']}\n")
    
    # Build HyGT model
    model = HyGT(
        in_channels=data_dict['num_features'],
        hidden_channels=64,
        out_channels=data_dict['num_classes'],
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        curvature=-1.0,  # Hyperbolic space
    )
    
    print(f"Model: HyGT")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    trainer = Trainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
    )
    
    print("Training for 20 epochs (quick demo)...\n")
    results = trainer.train_node_classification(
        data=data_dict['data'],
        num_epochs=20,
        early_stopping_patience=10,
    )
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {results['test_acc']:.4f}")
    print(f"  Test F1: {results['test_f1']:.4f}")


def example_train_geoformer_mix():
    """Example: Train GeoFormer-Mix on MUTAG dataset"""
    print("\n" + "="*80)
    print("Example 2: Training GeoFormer-Mix on MUTAG (Molecular Graphs)")
    print("="*80 + "\n")
    
    # Prepare data
    data_dict = prepare_data(
        dataset_name='MUTAG',
        batch_size=32,
        root='./data'
    )
    
    print(f"Dataset loaded:")
    print(f"  Features: {data_dict['num_features']}")
    print(f"  Classes: {data_dict['num_classes']}")
    print(f"  Task: {data_dict['task']}\n")
    
    # Build GeoFormer-Mix model
    model = GeoFormerMix(
        in_channels=data_dict['num_features'],
        hidden_channels=64,
        out_channels=data_dict['num_classes'],
        num_layers=2,
        num_heads=6,  # 2 hyperbolic, 2 euclidean, 2 spherical
        dropout=0.1,
    )
    
    print(f"Model: GeoFormer-Mix (Multi-Geometry Attention)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Geometry split: 2 hyperbolic + 2 euclidean + 2 spherical heads\n")
    
    # Train
    trainer = Trainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
    )
    
    print("Training for 20 epochs (quick demo)...\n")
    results = trainer.train_graph_classification(
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        test_loader=data_dict['test_loader'],
        num_epochs=20,
        early_stopping_patience=10,
    )
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {results['test_acc']:.4f}")
    print(f"  Test F1: {results['test_f1']:.4f}")


def example_model_comparison():
    """Example: Compare models on same dataset"""
    print("\n" + "="*80)
    print("Example 3: Comparing All Models on MUTAG")
    print("="*80 + "\n")
    
    # Prepare data once
    data_dict = prepare_data(
        dataset_name='MUTAG',
        batch_size=32,
        root='./data'
    )
    
    models = {
        'HyGT': HyGT,
        'SpGT': SpGT,
        'GeoFormer-Mix': GeoFormerMix,
    }
    
    results_summary = {}
    
    for model_name, ModelClass in models.items():
        print(f"\nTraining {model_name}...")
        
        # Build model
        if model_name == 'HyGT':
            model = ModelClass(
                in_channels=data_dict['num_features'],
                hidden_channels=64,
                out_channels=data_dict['num_classes'],
                num_layers=2,
                num_heads=4,
                curvature=-1.0,
            )
        elif model_name == 'SpGT':
            model = ModelClass(
                in_channels=data_dict['num_features'],
                hidden_channels=64,
                out_channels=data_dict['num_classes'],
                num_layers=2,
                num_heads=4,
                curvature=1.0,
            )
        else:  # GeoFormer-Mix
            model = ModelClass(
                in_channels=data_dict['num_features'],
                hidden_channels=64,
                out_channels=data_dict['num_classes'],
                num_layers=2,
                num_heads=6,
            )
        
        # Train
        trainer = Trainer(model=model)
        results = trainer.train_graph_classification(
            train_loader=data_dict['train_loader'],
            val_loader=data_dict['val_loader'],
            test_loader=data_dict['test_loader'],
            num_epochs=20,
            early_stopping_patience=10,
        )
        
        results_summary[model_name] = results['test_acc']
    
    # Print comparison
    print("\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    for model_name, acc in results_summary.items():
        print(f"{model_name:20s}: {acc:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GeoFormer Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='Which example to run (1: HyGT on Cora, 2: GeoFormer-Mix on MUTAG, 3: Compare all models)'
    )
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_train_hygt()
    elif args.example == 2:
        example_train_geoformer_mix()
    elif args.example == 3:
        example_model_comparison()
    
    print("\n✅ Example completed successfully!\n")
