"""
Benchmarking utilities for comparing GeoFormer variants across datasets.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
import json
import time
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import prepare_data, get_suggested_model
from model.geoformer import build_model
from train import train_model


class Benchmark:
    """Benchmark GeoFormer models across multiple datasets."""
    
    def __init__(self, config: Dict, results_dir: str = './results'):
        self.config = config
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.results = []
    
    def run_experiment(
        self,
        dataset_name: str,
        model_type: str,
        seed: int = 42,
    ) -> Dict:
        """
        Run a single experiment: train and evaluate one model on one dataset.
        
        Args:
            dataset_name: Name of dataset
            model_type: Type of model ('hygt', 'spgt', 'geoformer_mix')
            seed: Random seed
        
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*80}")
        print(f"Running Experiment: {model_type.upper()} on {dataset_name}")
        print(f"{'='*80}\n")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Prepare data
        data_dict = prepare_data(
            dataset_name=dataset_name,
            batch_size=self.config['dataset']['batch_size'],
            train_ratio=self.config['dataset']['train_ratio'],
            val_ratio=self.config['dataset']['val_ratio'],
            test_ratio=self.config['dataset']['test_ratio'],
            seed=seed,
            root=self.config['dataset']['root'],
        )
        
        # Build model
        model = build_model(
            model_type=model_type,
            in_channels=data_dict['num_features'],
            hidden_channels=self.config['model']['hidden_channels'],
            out_channels=data_dict['num_classes'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            dropout=self.config['model']['dropout'],
            curvature=self.config['model'].get('hyperbolic_curvature', -1.0) 
                      if model_type == 'hygt' else 
                      self.config['model'].get('spherical_curvature', 1.0),
            learnable_curvature=self.config['model'].get('learnable_curvature', False),
        )
        
        # Train model
        train_config = {
            'device': self.config['training']['device'],
            'learning_rate': self.config['training']['learning_rate'],
            'curvature_learning_rate': self.config['training'].get('curvature_learning_rate', None),
            'weight_decay': self.config['training']['weight_decay'],
            'optimizer': self.config['training']['optimizer'],
            'scheduler': self.config['training']['scheduler'],
            'num_epochs': self.config['training']['num_epochs'],
            'early_stopping_patience': self.config['training']['early_stopping_patience'],
            'use_wandb': self.config['logging']['use_wandb'],
            'model_type': model_type,
            'dataset': dataset_name,
        }
        
        start_time = time.time()
        results = train_model(
            model=model,
            data_dict=data_dict,
            config=train_config,
            save_dir=self.config['logging']['save_dir'],
        )
        training_time = time.time() - start_time
        
        # Compile results
        experiment_results = {
            'dataset': dataset_name,
            'model': model_type,
            'seed': seed,
            'test_accuracy': results['test_acc'],
            'test_f1': results['test_f1'],
            'best_val_accuracy': results['best_val_acc'],
            'training_time': training_time,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_features': data_dict['num_features'],
            'num_classes': data_dict['num_classes'],
        }
        
        return experiment_results
    
    def run_benchmark(
        self,
        datasets: List[str],
        models: List[str],
        num_runs: int = None,  # Deprecated, kept for backward compatibility
    ):
        """
        Run comprehensive benchmark across datasets and models.
        
        Args:
            datasets: List of dataset names
            models: List of model types
            num_runs: Deprecated (use config['experiment']['seeds'] instead)
        """
        # Get seeds from config (can be int or list)
        seeds_config = self.config['experiment'].get('seeds', 42)
        if isinstance(seeds_config, int):
            seeds_list = [seeds_config]
        else:
            seeds_list = seeds_config
        
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE BENCHMARK")
        print("="*80)
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"Seeds: {seeds_list}")
        print(f"Total runs per model: {len(seeds_list)}")
        print("="*80 + "\n")
        
        for dataset in datasets:
            for model in models:
                for run_idx, seed in enumerate(seeds_list):
                    try:
                        results = self.run_experiment(
                            dataset_name=dataset,
                            model_type=model,
                            seed=seed,
                        )
                        results['run'] = run_idx
                        results['seed'] = seed
                        self.results.append(results)
                        
                        # Save intermediate results
                        self.save_results()
                        
                    except Exception as e:
                        print(f"Error in experiment {model} on {dataset} (seed {seed}): {e}")
                        continue
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETED")
        print("="*80 + "\n")
        
        # Generate summary
        self.print_summary()
        self.plot_results()
    
    def save_results(self):
        """Save results to JSON and CSV."""
        # Save as JSON
        json_path = os.path.join(self.results_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.results_dir, 'results.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {self.results_dir}")
    
    def print_summary(self):
        """Print summary statistics of benchmark results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80 + "\n")
        
        # Group by dataset and model
        summary = df.groupby(['dataset', 'model']).agg({
            'test_accuracy': ['mean', 'std'],
            'test_f1': ['mean', 'std'],
            'training_time': 'mean',
            'num_parameters': 'first',
        }).round(4)
        
        print(summary)
        print("\n")
        
        # Best model per dataset
        print("Best Model per Dataset (by test accuracy):")
        print("-" * 80)
        best_models = df.loc[df.groupby('dataset')['test_accuracy'].idxmax()]
        for _, row in best_models.iterrows():
            print(f"{row['dataset']}: {row['model']} "
                  f"(Acc: {row['test_accuracy']:.4f}, F1: {row['test_f1']:.4f})")
        print("\n")
        
        # Save summary
        summary_path = os.path.join(self.results_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BENCHMARK SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(str(summary))
            f.write("\n\n")
            f.write("Best Model per Dataset:\n")
            f.write("-" * 80 + "\n")
            for _, row in best_models.iterrows():
                f.write(f"{row['dataset']}: {row['model']} "
                       f"(Acc: {row['test_accuracy']:.4f}, F1: {row['test_f1']:.4f})\n")
    
    def plot_results(self):
        """Generate visualization plots of benchmark results."""
        if not self.results:
            print("No results to plot.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Test Accuracy by Dataset and Model
        ax1 = axes[0, 0]
        summary = df.groupby(['dataset', 'model'])['test_accuracy'].mean().reset_index()
        pivot_data = summary.pivot(index='dataset', columns='model', values='test_accuracy')
        pivot_data.plot(kind='bar', ax=ax1)
        ax1.set_title('Test Accuracy by Dataset and Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_xlabel('Dataset')
        ax1.legend(title='Model')
        ax1.grid(True, alpha=0.3)
        
        # 2. Test F1 Score by Dataset and Model
        ax2 = axes[0, 1]
        summary_f1 = df.groupby(['dataset', 'model'])['test_f1'].mean().reset_index()
        pivot_f1 = summary_f1.pivot(index='dataset', columns='model', values='test_f1')
        pivot_f1.plot(kind='bar', ax=ax2)
        ax2.set_title('Test F1 Score by Dataset and Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Test F1 Score')
        ax2.set_xlabel('Dataset')
        ax2.legend(title='Model')
        ax2.grid(True, alpha=0.3)
        
        # 3. Training Time Comparison
        ax3 = axes[1, 0]
        time_summary = df.groupby(['dataset', 'model'])['training_time'].mean().reset_index()
        pivot_time = time_summary.pivot(index='dataset', columns='model', values='training_time')
        pivot_time.plot(kind='bar', ax=ax3)
        ax3.set_title('Training Time by Dataset and Model', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_xlabel('Dataset')
        ax3.legend(title='Model')
        ax3.grid(True, alpha=0.3)
        
        # 4. Model Size (Number of Parameters)
        ax4 = axes[1, 1]
        param_summary = df.groupby('model')['num_parameters'].first()
        param_summary.plot(kind='bar', ax=ax4, color='steelblue')
        ax4.set_title('Model Size (Number of Parameters)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Parameters')
        ax4.set_xlabel('Model')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'benchmark_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
        
        # Also save individual plots for clarity
        self._save_individual_plots(df)
    
    def _save_individual_plots(self, df: pd.DataFrame):
        """Save individual plots for better clarity."""
        # Accuracy comparison
        plt.figure(figsize=(12, 6))
        summary = df.groupby(['dataset', 'model'])['test_accuracy'].agg(['mean', 'std']).reset_index()
        
        for model in df['model'].unique():
            model_data = summary[summary['model'] == model]
            plt.errorbar(
                model_data['dataset'],
                model_data['mean'],
                yerr=model_data['std'],
                marker='o',
                label=model,
                capsize=5,
                linewidth=2,
            )
        
        plt.title('Test Accuracy Comparison Across Datasets', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.legend(title='Model', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_dir, 'accuracy_comparison.png'), dpi=300)
        plt.close()


def run_quick_benchmark(config_path: str = None):
    """Quick benchmark function for testing."""
    if config_path is None:
        # Default minimal config
        config = {
            'model': {
                'hidden_channels': 64,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
                'hyperbolic_curvature': -1.0,
                'spherical_curvature': 1.0,
            },
            'dataset': {
                'batch_size': 32,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'root': './data',
            },
            'training': {
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.00001,
                'optimizer': 'adam',
                'scheduler': 'cosine',
                'early_stopping_patience': 10,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            },
            'logging': {
                'use_wandb': False,
                'save_dir': './checkpoints',
            },
            'experiment': {
                'seed': 42,
            },
        }
    else:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    benchmark = Benchmark(config, results_dir='./results')
    
    # Quick test on MUTAG with all models
    benchmark.run_benchmark(
        datasets=['MUTAG'],
        models=['hygt', 'spgt', 'geoformer_mix'],
        num_runs=1,
    )


if __name__ == '__main__':
    run_quick_benchmark()
