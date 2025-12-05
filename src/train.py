"""
Training and evaluation pipeline for GeoFormer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
import os
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Trainer:
    """Unified trainer for graph classification and node classification tasks."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer_type: str = 'adam',
        scheduler_type: str = 'cosine',
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Initialize scheduler
        self.scheduler_type = scheduler_type
        self.scheduler = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def _setup_scheduler(self, num_epochs: int):
        """Setup learning rate scheduler."""
        if self.scheduler_type.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-6,
            )
        elif self.scheduler_type.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True,
            )
    
    def train_graph_classification(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_epochs: int = 100,
        early_stopping_patience: int = 20,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Train model for graph classification task.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
        
        Returns:
            Dictionary with training history and final metrics
        """
        self._setup_scheduler(num_epochs)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch_graph(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1 = self._evaluate_graph(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if self.scheduler_type.lower() == 'plateau':
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | Best Val Acc: {best_val_acc:.4f}")
            
            # Log to wandb if enabled
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                    })
                except:
                    pass
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model and evaluate on test set
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        test_loss, test_acc, test_f1 = self._evaluate_graph(test_loader)
        print(f"\nFinal Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_loss': test_loss,
        }
    
    def _train_epoch_graph(self, loader) -> Tuple[float, float]:
        """Train one epoch for graph classification."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index)
            
            # For graph classification, we need to pool node representations
            # Use global mean pooling
            from torch_geometric.nn import global_mean_pool
            out = global_mean_pool(out, batch.batch)
            
            # Compute loss
            loss = F.cross_entropy(out, batch.y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.num_graphs
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _evaluate_graph(self, loader) -> Tuple[float, float, float]:
        """Evaluate model on graph classification."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        total_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Forward pass
                out = self.model(batch.x, batch.edge_index)
                
                # Pool node representations
                from torch_geometric.nn import global_mean_pool
                out = global_mean_pool(out, batch.batch)
                
                # Compute loss
                loss = F.cross_entropy(out, batch.y)
                
                # Track metrics
                total_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                total_samples += batch.num_graphs
        
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train_node_classification(
        self,
        data,
        num_epochs: int = 200,
        early_stopping_patience: int = 20,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Train model for node classification task.
        
        Args:
            data: PyG Data object with train/val/test masks
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
        
        Returns:
            Dictionary with training history and final metrics
        """
        self._setup_scheduler(num_epochs)
        
        data = data.to(self.device)
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch_node(data)
            
            # Validate
            val_loss, val_acc, val_f1 = self._evaluate_node(data, 'val')
            
            # Update learning rate
            if self.scheduler is not None:
                if self.scheduler_type.lower() == 'plateau':
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Val F1: {val_f1:.4f}")
            
            # Log to wandb if enabled
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                    })
                except:
                    pass
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model and evaluate on test set
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        test_loss, test_acc, test_f1 = self._evaluate_node(data, 'test')
        print(f"\nFinal Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_loss': test_loss,
        }
    
    def _train_epoch_node(self, data) -> Tuple[float, float]:
        """Train one epoch for node classification."""
        self.model.train()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        
        # Compute loss on training nodes only
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracy on training nodes
        pred = out[data.train_mask].argmax(dim=1)
        correct = (pred == data.y[data.train_mask]).sum().item()
        accuracy = correct / data.train_mask.sum().item()
        
        return loss.item(), accuracy
    
    def _evaluate_node(self, data, split: str = 'val') -> Tuple[float, float, float]:
        """Evaluate model on node classification."""
        self.model.eval()
        
        # Get appropriate mask
        if split == 'val':
            mask = data.val_mask
        elif split == 'test':
            mask = data.test_mask
        else:
            mask = data.train_mask
        
        with torch.no_grad():
            # Forward pass
            out = self.model(data.x, data.edge_index)
            
            # Compute loss
            loss = F.cross_entropy(out[mask], data.y[mask])
            
            # Compute metrics
            pred = out[mask].argmax(dim=1)
            labels = data.y[mask].cpu().numpy()
            predictions = pred.cpu().numpy()
            
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
        
        return loss.item(), accuracy, f1


def train_model(
    model,
    data_dict,
    config: Dict,
    save_dir: str = './checkpoints',
) -> Dict:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        data_dict: Dictionary with data loaders or data
        config: Training configuration
        save_dir: Directory to save checkpoints
    
    Returns:
        Training results dictionary
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5),
        optimizer_type=config.get('optimizer', 'adam'),
        scheduler_type=config.get('scheduler', 'cosine'),
        use_wandb=config.get('use_wandb', False),
    )
    
    # Save path
    model_name = config.get('model_type', 'model')
    dataset_name = config.get('dataset', 'dataset')
    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_best.pt")
    
    # Train based on task type
    if data_dict['task'] == 'graph_classification':
        results = trainer.train_graph_classification(
            train_loader=data_dict['train_loader'],
            val_loader=data_dict['val_loader'],
            test_loader=data_dict['test_loader'],
            num_epochs=config.get('num_epochs', 100),
            early_stopping_patience=config.get('early_stopping_patience', 20),
            save_path=save_path,
        )
    elif data_dict['task'] == 'node_classification':
        results = trainer.train_node_classification(
            data=data_dict['data'],
            num_epochs=config.get('num_epochs', 200),
            early_stopping_patience=config.get('early_stopping_patience', 20),
            save_path=save_path,
        )
    else:
        raise ValueError(f"Unknown task type: {data_dict['task']}")
    
    return results
