import torch
from pathlib import Path
from localizing.probe.heuristic_vec_gather import get_or_serialize_heuristic_localizations
from localizing.probe.probe_data_gather import get_or_serialize_localizations_embedded, localizations_to_single_vec_to_label_dataset
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.multi_data_gathering import DatasetName
from lmwrapper.openai_wrapper import OpenAiModelNames
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
# Remove unused sklearn calibration imports if CalibratedProbe is gone
# from sklearn.calibration import calibration_curve, CalibratedClassifierCV 
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from localizing.probe.probe_data_gather import SingleVecToLabelDatasetMultiFold, SingleVecToLabelDataset

# Directly import calipy - assumes it's available
from calipy.experiment_results import ExperimentResults, BinStrategy


# Model definition
class LogisticRegressionProbe(nn.Module):
    """A logistic regression model for token-level prediction."""
    
    def __init__(self, input_dim: int, dtype: torch.dtype = torch.float32, dropout_rate: float = 0.2):
        """
        Initialize the logistic regression probe.
        
        Args:
            input_dim: Dimension of input features
            dtype: Data type for the model parameters (default: float32)
            dropout_rate: Dropout probability for regularization (default: 0.2)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(input_dim, 1, dtype=dtype)
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Raise error if input tensor contains NaNs/Infs
        if torch.isinf(x).any() or torch.isnan(x).any():
            raise ValueError("Input tensor x to forward() contains NaNs or Infs.")
            # x = torch.nan_to_num(x)
            
        # Ensure input has the same dtype as the model
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        # Ensure input is on the same device as the model parameters
        if x.device != self.linear.weight.device:
            x = x.to(self.linear.weight.device)
        
        # Apply dropout for regularization
        x = self.dropout(x)
            
        # Calculate logits
        logits = self.linear(x).squeeze(-1)
        
        # Raise error if output logits contain NaNs/Infs
        if torch.isinf(logits).any() or torch.isnan(logits).any():
            raise ValueError("Output logits from linear layer contain NaNs or Infs.")
            # logits = torch.nan_to_num(logits)
            
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities."""
        with torch.no_grad():
            self.eval()  # Set model to evaluation mode to disable dropout
            probs = torch.sigmoid(self.forward(x))
            self.train()  # Reset to training mode
            return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict binary labels."""
        return (self.predict_proba(x) >= 0.5).float()


# DataLoader with dtype conversion
class TypeConvertingDataLoader(DataLoader):
    """DataLoader that automatically converts data to the specified dtype."""
    
    def __init__(
        self, 
        dataset, 
        dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the data loader with type conversion.
        
        Args:
            dataset: Dataset to load from
            dtype: Data type to convert tensors to
            device: Device to load data to
            **kwargs: Additional arguments for DataLoader
        """
        super().__init__(dataset, **kwargs)
        self.dtype = dtype
        self.device = device
        
        # Create a collate function that converts types
        original_collate_fn = self.collate_fn
        
        def collate_with_conversion(batch):
            # Use the original collate function first
            batch = original_collate_fn(batch)
            
            # Convert the batch to the desired dtype
            if isinstance(batch, tuple) and len(batch) == 2:
                # Convert inputs to specified dtype
                inputs, targets = batch
                inputs = inputs.to(dtype=self.dtype)
                
                # Only move to device if specified
                if self.device is not None:
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)
                
                return inputs, targets
            return batch
        
        self.collate_fn = collate_with_conversion


# Trainer class
class Trainer:
    """Base trainer for model training and evaluation."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        weight_decay: float = 0.01,
        batch_size: int = 64,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the trainer.
        
        Args:
            learning_rate: Learning rate for the optimizer
            num_epochs: Number of training epochs
            weight_decay: L2 regularization strength
            batch_size: Batch size for training (use None for full batch)
            device: Device to use for training ('cuda', 'cpu', or None for auto-detection)
            dtype: Data type to use for model and tensors (default: float32)
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dtype = dtype
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
    
    def _get_dataloader(self, dataset: SingleVecToLabelDataset, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader for the dataset with automatic type conversion."""
        # Use full batch if batch_size is None or if it's larger than dataset
        effective_batch_size = len(dataset) if self.batch_size is None else min(self.batch_size, len(dataset))
        
        return TypeConvertingDataLoader(
            dataset=dataset, 
            batch_size=effective_batch_size, 
            shuffle=shuffle,
            dtype=self.dtype,
            device=self.device
        )
    
    def train(self, model: nn.Module, train_data: SingleVecToLabelDataset) -> nn.Module:
        """
        Train the model on the given data.
        
        Args:
            model: The model to train
            train_data: Training data
            
        Returns:
            Trained model
        """
        # Move model to the correct device and dtype first
        model = model.to(device=self.device, dtype=self.dtype)
        model.train()
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        train_loader = self._get_dataloader(train_data)
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                # Explicitly ensure inputs and targets are on the correct device
                inputs = inputs.to(device=self.device, dtype=self.dtype)
                targets = targets.to(device=self.device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
            
            avg_loss = total_loss / len(train_data)
            # Uncomment for logging if needed
            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}/{self.num_epochs}, Loss: {avg_loss:.6f}")
        
        return model
    
    def evaluate(self, model: nn.Module, test_data: SingleVecToLabelDataset) -> Dict[str, float]:
        """
        Evaluate the model on test data, including calibration metrics using calipy.
        
        Args:
            model: The model to evaluate
            test_data: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Move model to the correct device and dtype first
        model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        
        test_loader = self._get_dataloader(test_data, shuffle=False)
        criterion = nn.BCEWithLogitsLoss()
        
        all_outputs = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Explicitly ensure inputs and targets are on the correct device
                inputs = inputs.to(device=self.device, dtype=self.dtype)
                targets = targets.to(device=self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                
                # Move outputs back to CPU for metric calculation
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Combine batches
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        
        # Calculate metrics
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        
        # Convert to numpy for sklearn metrics
        y_true = targets.numpy()
        y_pred = preds.numpy()
        y_prob = probs.numpy()
        
        # Clip probabilities before passing to metrics/calibration to avoid log(0) etc.
        epsilon = 1e-6
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
            'test_loss': total_loss / len(test_data),
        }
        
        # Calculate calibration metrics using calipy (no fallback)
        # print(f"DEBUG: y_prob min={y_prob.min()}, max={y_prob.max()}, dtype={y_prob.dtype}")
        # print(f"DEBUG: y_true min={y_true.min()}, max={y_true.max()}, dtype={y_true.dtype}")
        exp_results = ExperimentResults(
            predicted_probabilities=y_prob,
            true_labels=y_true,
            bin_strategy=BinStrategy.Uniform,
            num_bins=10
        )
        # Get raw calibration metrics
        metrics.update({
            'ece': exp_results.ece,
            'brier_score': exp_results.brier_score,
            'brier_skill_score': exp_results.skill_score,
        })
        
        # Get Platt-scaled calibration metrics
        platt_results = exp_results.to_platt_scaled()
        metrics.update({
            'platt_ece': platt_results.ece,
            'platt_brier_score': platt_results.brier_score,
            'platt_brier_skill_score': platt_results.skill_score,
        })
        
        # Save the experiment results for plotting later
        self.last_calibration_results = {
            'raw': exp_results,
            'platt': platt_results
        }
            
        return metrics
    
    def plot_reliability_diagram(self, save_path=None, scaled=False):
        """
        Plot a reliability diagram for the last evaluated model using calipy.
        
        Args:
            save_path: Path to save the plot to (optional)
        
        Returns:
            The figure and axes from calipy's plot.
        """
        if not hasattr(self, 'last_calibration_results') or self.last_calibration_results is None:
            print("No calibration results available. Run evaluate() first.")
            return None
        
        # Directly use calipy plotting (no fallback)
        fig, ax = plt.subplots(figsize=(8, 8))
        kind = "Scaled" if scaled else "Unscaled"
        
        # Plot raw results using calipy
        self.last_calibration_results['raw'].reliability_plot(
            ax, 
            show_unscaled=not scaled, 
            show_scaled=scaled,
            show_counts=kind,
            show_quantiles=kind,
            annotate=kind # Or customize annotation
        )
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved reliability diagram to {save_path}")
        
        return fig, ax

    def train_and_evaluate(
        self, 
        model: nn.Module, 
        train_data: SingleVecToLabelDataset, 
        test_data: SingleVecToLabelDataset
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Train and evaluate a model.
        
        Args:
            model: The model to train
            train_data: Training data
            test_data: Test data
            
        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        model = self.train(model, train_data)
        metrics = self.evaluate(model, test_data)
        return model, metrics


def train_model_on_folds(
    dataset: SingleVecToLabelDatasetMultiFold,
    model_class=LogisticRegressionProbe,
    learning_rate: float = 0.01,
    num_epochs: int = 100,
    weight_decay: float = 0.01,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,  # Default to float32 for stability
) -> Dict[str, Union[Dict[str, float], List[nn.Module]]]:
    """
    Train models on each fold of the dataset and report average metrics.
    
    Args:
        dataset: Dataset with multiple folds for cross-validation
        model_class: Model class to use (defaults to LogisticRegressionProbe)
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        weight_decay: L2 regularization strength
        batch_size: Batch size for training (None for full batch)
        device: Device to use for training
        dtype: Data type to use for model and tensors (default: float32)
        
    Returns:
        Dictionary with average metrics across all folds and list of trained models
    """
    if len(dataset.folds_train_test) == 0:
        raise ValueError("Dataset contains no folds")
    
    # Initialize trainer
    trainer = Trainer(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    
    # Initialize storage for metrics and models across folds
    all_metrics = []
    trained_models = []
    
    # Train and evaluate on each fold
    for fold_idx, (train_data, test_data) in enumerate(dataset.folds_train_test):
        # Display data info for this fold
        print(f"Fold {fold_idx+1} - Training data: {len(train_data)} samples, Test data: {len(test_data)} samples")
        print(f"Original dtype - Embeddings: {train_data.embeddings.dtype}, Labels: {train_data.labels.dtype}")
        print(f"Using model and training dtype: {dtype}")
        
        # Create model for this fold with the correct dtype
        input_dim = train_data.embeddings.shape[1]
        model = model_class(input_dim, dtype=dtype)
        
        # Train and evaluate
        model, metrics = trainer.train_and_evaluate(model, train_data, test_data)
        
        all_metrics.append(metrics)
        trained_models.append(model)
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold[metric] for fold in all_metrics])
        avg_metrics[f'{metric}_std'] = np.std([fold[metric] for fold in all_metrics])
    
    return {
        'metrics': avg_metrics,
        'models': trained_models,
        'trainer': trainer  # Save the trainer for visualization
    }


def print_fold_results(results: Dict[str, Any]) -> None:
    """
    Print formatted results from fold training.
    
    Args:
        results: Results dictionary from train_model_on_folds
    """
    metrics = results['metrics']
    models = results['models']
    
    print("\nLogistic Regression Performance (averaged across folds):")
    print(f"Accuracy: {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"Recall: {metrics['recall']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"AUC: {metrics['auc']:.4f} ± {metrics['auc_std']:.4f}")
    
    print("\nCalibration Metrics (Raw):")
    print(f"Expected Calibration Error: {metrics['ece']:.4f} ± {metrics['ece_std']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f} ± {metrics['brier_score_std']:.4f}")
    print(f"Brier Skill Score: {metrics['brier_skill_score']:.4f} ± {metrics['brier_skill_score_std']:.4f}")
    
    print("\nCalibration Metrics (Platt Scaling):")
    print(f"Expected Calibration Error: {metrics['platt_ece']:.4f} ± {metrics['platt_ece_std']:.4f}")
    print(f"Brier Score: {metrics['platt_brier_score']:.4f} ± {metrics['platt_brier_score_std']:.4f}")
    print(f"Brier Skill Score: {metrics['platt_brier_skill_score']:.4f} ± {metrics['platt_brier_skill_score_std']:.4f}")
    
    print(f"\nTest Loss: {metrics['test_loss']:.4f} ± {metrics['test_loss_std']:.4f}")
    print(f"Number of trained models: {len(models)}")


def train_and_evaluate_style(
    style: EmbeddingStyle,
    device: str,
    learning_rate: float = 0.01,
    num_epochs: int = 20,
    weight_decay: float = 0.01,
    batch_size: Optional[int] = None,
    datasets: list[DatasetName] = [DatasetName.livecodebench],
    gen_model_name: str = OpenAiModelNames.gpt_4o_mini,
    embed_lm_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    fix_reference: str = OpenAiModelNames.o3_mini,
) -> Dict[str, float]:
    """
    Train and evaluate a model using a specific embedding style.
    
    Args:
        style: The embedding style to use
        device: Device to use for training ('cuda' or 'cpu')
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        weight_decay: L2 regularization strength
        batch_size: Batch size for training (None for full batch)
        datasets: List of datasets to use for training (default: [DatasetName.livecodebench])
        gen_model_name: Model name to use for generating code (default: OpenAiModelNames.gpt_4o_mini)
        embed_lm_name: Model name to use for embeddings (default: "Qwen/Qwen2.5-Coder-0.5B")
        fix_reference: Model name to use for reference (default: OpenAiModelNames.o3_mini)

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Training with {style.value} embedding style")
    datasets_str = ", ".join(str(ds) for ds in datasets)
    print(f"Using datasets: {datasets_str}")
    print(f"Using generation model: {gen_model_name}")
    print(f"Using embedding model: {embed_lm_name}")
    print(f"{'='*80}")
    
    # Get dataset and train
    use_heuristic = False
    if not use_heuristic:
        # Create ProbeConfig from parameters
        from localizing.probe.agg_models.agg_config import ProbeConfig
        config = ProbeConfig(
            datasets=datasets,
            gen_model_name=gen_model_name,
            embed_lm_name=embed_lm_name,
            fix_reference=fix_reference,
            embedding_style=style,
        )
        localizations = get_or_serialize_localizations_embedded(config)
    else:
        localizations = get_or_serialize_heuristic_localizations(
            datasets=datasets,
            gen_model_name=gen_model_name,
            top_k_tokens=20,
            tokenizer_key=embed_lm_name,
        )
    print(f"Loaded {len(localizations)} localizations")
    print(f"Datasets present: {', '.join(str(ds) for ds in localizations.get_dataset_name_set())}")
    dataset = localizations_to_single_vec_to_label_dataset(localizations, n_folds=5)
    print(dataset)

    # Train models on folds using float32 for stability
    results = train_model_on_folds(
        dataset,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
    )
    
    # Print individual results
    print(f"\nResults for {style.value}:")
    print_fold_results(results)
    
    # Plot reliability diagram if trainer is available
    save_dir = Path("plots/probe/reliability_diagrams")
    save_dir = save_dir / f"{datasets_str.replace(', ', '_')}__{gen_model_name.replace('/', '_')}__{embed_lm_name.replace('/', '_')}__{fix_reference.replace('/', '_')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    if 'trainer' in results:
        results['trainer'].plot_reliability_diagram(save_path=save_dir / f"reliability_diagram_{style.value}.png", scaled=False)
        results['trainer'].plot_reliability_diagram(save_path=save_dir / f"reliability_diagram_{style.value}_SCALED.png", scaled=True)
        print(f"Plotted reliability diagram - see reliability_diagram_{style.value}.png")
    
    # Store metrics and clear memory
    metrics = results['metrics']
    del localizations
    del dataset
    del results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics


def create_comparison_table(all_results: Dict[str, Dict[str, float]]) -> None:
    """
    Create and print a comparison table of results across all embedding styles.
    
    Args:
        all_results: Dictionary mapping embedding style names to their metrics
    """
    print("\n\nComparison Table (best values in bold):")
    print("="*120)
    
    # Define metrics to compare (excluding std metrics)
    metrics_to_compare = [
        'accuracy', 'precision', 'recall', 'f1', 'auc',
        'ece', 'brier_score', 'brier_skill_score',
        'platt_ece', 'platt_brier_score', 'platt_brier_skill_score',
        'test_loss'
    ]
    
    # Print header
    header = f"{'Metric':<20} " + " ".join(f"{style:<10}" for style in all_results.keys())
    print(header)
    print("-"*120)
    
    # For each metric, find the best value and print the row
    for metric in metrics_to_compare:
        # Get all values for this metric
        values = {style: results[metric] for style, results in all_results.items()}
        
        # Determine if higher or lower is better
        is_higher_better = metric not in ['ece', 'brier_score', 'platt_ece', 'platt_brier_score', 'test_loss']
        
        # Find best value
        best_value = max(values.values()) if is_higher_better else min(values.values())
        
        # Format row with best value in bold
        row = f"{metric:<20} "
        for style in all_results.keys():
            value = values[style]
            if value == best_value:
                row += f"\033[1m{value:10.4f}\033[0m "
            else:
                row += f"{value:10.4f} "
        print(row)
    
    print("="*120)
    print("Note: Bold values indicate the best performance for each metric")
    print("      Higher is better for all metrics except ECE, Brier Score, and Test Loss")


def main():
    """Main execution function for training and comparing different embedding styles."""
    print("--- Running Probe Training Script ---")
    
    # Check available devices
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
        device = 'cuda'
    else:
        print("CUDA is not available. Using CPU.")
        device = 'cpu'
    
    # Store results for each embedding style
    all_results = {}
    
    # Define datasets to use
    datasets_to_use = [
        DatasetName.livecodebench, 
        DatasetName.humaneval_plus,
        DatasetName.mbpp_plus,
    ]
    
    # Default models to use
    gen_model = OpenAiModelNames.gpt_4o_mini
    fix_reference = OpenAiModelNames.o4_mini_2025_04_16
    #fix_reference = OpenAiModelNames.o3_mini
    embed_model = "Qwen/Qwen2.5-Coder-0.5B"
    #embed_model = "Qwen/Qwen2.5-Coder-3B"
    
    # Train and evaluate each embedding style
    for style in EmbeddingStyle:
        metrics = train_and_evaluate_style(
            style=style,
            device=device,
            learning_rate=0.001,
            num_epochs=20,
            weight_decay=0.005,
            batch_size=256,
            datasets=datasets_to_use,
            gen_model_name=gen_model,
            embed_lm_name=embed_model,
            fix_reference=fix_reference,
        )
        all_results[style.value] = metrics
    
    # Create and print comparison table
    create_comparison_table(all_results)


if __name__ == "__main__":
    main()
