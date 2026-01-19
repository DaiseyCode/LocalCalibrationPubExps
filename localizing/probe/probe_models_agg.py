from unittest.mock import DEFAULT
import torch
import torch.nn.functional as F
import torchvision.ops.focal_loss
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from localizing.filter_helpers import debug_str_filterables
from localizing.predictions_repr import ProblemPredictions
from localizing.probe.probe_data_gather import EmbeddedTokenization, GroupedVecLabelDataset, GroupedVecLabelDatasetMultiFold
from localizing.localizing_structs import TokenizedLocalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from localizing.probe.probe_data_gather import get_or_serialize_localizations_embedded, localizations_to_grouped_vec_label_dataset
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.multi_data_gathering import DatasetName
from lmwrapper.openai_wrapper import OpenAiModelNames
from calipy.experiment_results import ExperimentResults, BinStrategy
import matplotlib.pyplot as plt
from pathlib import Path
from plot_manager import plot_and_save
from localizing.probe.agg_models.agg_config import ProbeAggregator, ProbeConfig, ProbeLoss
from localizing.probe.agg_models.agg_data_structures import AggregatedBatch
from localizing.probe.agg_models.agg_dataloader import AggQueriesDataLoader


class AggLineProbe(nn.Module):
    """
    A model that processes code at the line level.
    
    This model:
    1. Projects embeddings to a lower dimension
    2. Applies a non-linearity
    3. Max pools over line spans
    4. Projects to predict line-level probabilities
    """
    
    def __init__(
        self, 
        input_dim: int, 
        config: ProbeConfig,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the aggregated line-level probe.
        
        Args:
            input_dim: Dimension of input token embeddings
            hidden_dim: Dimension of hidden layer after down projection
            dropout_rate: Dropout probability for regularization
            dtype: Data type for the model parameters
        """
        super().__init__()
        self.dtype = dtype
        
        # Define layers with specified dtype
        self.dropout = nn.Dropout(p=config.dropout_rate)
        
        # Initial down projection from input_dim to hidden_dim
        self.down_projection = nn.Linear(input_dim, config.hidden_dim, dtype=dtype)
        
        # Final projection from hidden_dim to 1 (probability)
        if config.agg_style == ProbeAggregator.MHA_POOL_4_HEADS:
            down_dim_size = config.hidden_dim - 4
        else:
            down_dim_size = config.hidden_dim
        self.final_projection = nn.Linear(down_dim_size, 1, dtype=dtype)

        self.agg_style = config.agg_style

        # Activation function
        self.activation = nn.GELU()

    def _agg_function(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features using the specified aggregation style.
        """
        if self.agg_style == ProbeAggregator.MAX_POOL:
            return torch.max(features, dim=0)[0]
        elif self.agg_style == ProbeAggregator.AVG_POOL:
            return torch.mean(features, dim=0)
        elif self.agg_style == ProbeAggregator.MHA_POOL_4_HEADS:
            return mha_pool(features, num_heads=4)
        else:
            raise ValueError(f"Invalid agg style: {self.agg_style}")


    def forward(self, batch: AggregatedBatch) -> Dict[str, torch.Tensor]:
        """
        Forward pass that processes embeddings at line level.
        
        Args:
            batch: AggregatedBatch containing embeddings and span information
            
        Returns:
            Dictionary containing:
                - token_logits: Logits for each token
                - line_logits: Logits for each line
                - line_labels: Labels for each line (AND of token labels)
        """
        # Get the inputs and ensure right type
        embeddings = batch.embeddings
        if embeddings.dtype != self.dtype:
            embeddings = embeddings.to(self.dtype)
        
        # Apply dropout for regularization
        embeddings = self.dropout(embeddings)
        
        # Step 1: Down project embeddings
        projected = self.down_projection(embeddings)
        
        # Step 2: Apply non-linearity - but handle attention logits specially for MHA
        if self.agg_style == ProbeAggregator.MHA_POOL_4_HEADS:
            # For MHA: keep attention logits (first 4 dims) linear, apply GELU to values
            attention_logits = projected[:, :4]  # Keep linear for proper attention computation
            value_features = projected[:, 4:]    # Apply activation to values only
            activated_values = self.activation(value_features)
            activated = torch.cat([attention_logits, activated_values], dim=1)
        else:
            # For other aggregation methods: apply activation to all features
            activated = self.activation(projected)
        
        # Prepare to store line-level info
        all_line_features = []
        all_line_labels = []
        all_line_indices = []
        
        # Prepare to store problem-level info
        all_problem_features = []
        all_problem_labels = []
        all_problem_indices = []
        
        # Process each problem and its lines
        for problem_idx in range(batch.get_num_problems()):
            # Get problem span for problem-level pooling
            problem_start, problem_end = batch.problem_spans[problem_idx]
            
            # Extract features for this entire problem
            problem_features = activated[problem_start:problem_end]
            
            # Step 3a: Max pooling over the entire problem
            # Shape: [hidden_dim]
            problem_pooled = self._agg_function(problem_features)
            
            # Store the pooled problem features
            all_problem_features.append(problem_pooled)
            
            # Get labels for this entire problem
            problem_labels = batch.labels[problem_start:problem_end]
            
            # Step 4a: AND operation - problem is good if all tokens are good
            # If any token should be removed (label=0), the problem is bad
            problem_label = torch.min(problem_labels)
            all_problem_labels.append(problem_label)
            
            # Track indices for this problem
            all_problem_indices.append((problem_idx, problem_start, problem_end))
            
            # Skip line processing if this problem has no line spans
            if problem_idx >= len(batch.line_spans):
                continue
                
            problem_lines = batch.line_spans[problem_idx]
            
            # Process each line in this problem
            for line_idx, (start_idx, end_idx) in enumerate(problem_lines):
                if start_idx >= end_idx:
                    continue  # Skip empty lines
                
                # Extract the features for this line
                line_features = activated[start_idx:end_idx]
                
                # Step 3b: Max pooling over the line
                # Shape: [hidden_dim]
                line_pooled = self._agg_function(line_features)
                
                # Store the pooled features
                all_line_features.append(line_pooled)
                
                # Get labels for this line
                line_labels = batch.labels[start_idx:end_idx]
                
                # Step 4b: AND operation - line is good if all tokens are good
                # If any token should be removed (label=0), the line is bad
                line_label = torch.min(line_labels)
                all_line_labels.append(line_label)
                
                # Track indices for this line
                all_line_indices.append((problem_idx, line_idx, start_idx, end_idx))
        
        # If no problems were processed, return empty results
        if not all_problem_features:
            return {
                "token_logits": torch.tensor([], device=embeddings.device, dtype=self.dtype),
                "line_logits": torch.tensor([], device=embeddings.device, dtype=self.dtype),
                "line_labels": torch.tensor([], device=embeddings.device, dtype=batch.labels.dtype),
                "line_indices": [],
                "problem_logits": torch.tensor([], device=embeddings.device, dtype=self.dtype),
                "problem_labels": torch.tensor([], device=embeddings.device, dtype=batch.labels.dtype),
                "problem_indices": []
            }
        
        # Stack all problem features
        problem_features = torch.stack(all_problem_features)
        
        # Final projection to get problem logits
        problem_logits = self.final_projection(problem_features).squeeze(-1)
        
        # Stack problem labels
        problem_labels = torch.stack(all_problem_labels)
        
        # If no lines were processed, create empty line results
        if not all_line_features:
            line_logits = torch.tensor([], device=embeddings.device, dtype=self.dtype)
            line_labels = torch.tensor([], device=embeddings.device, dtype=batch.labels.dtype)
        else:
            # Stack all line features
            line_features = torch.stack(all_line_features)
            
            # Final projection to get line logits
            line_logits = self.final_projection(line_features).squeeze(-1)
            
            # Stack line labels
            line_labels = torch.stack(all_line_labels)
        
        # Also get token-level logits
        if self.agg_style == ProbeAggregator.MHA_POOL_4_HEADS:
            token_projected = self.final_projection(activated[:, 4:]).squeeze(-1)
        else:
            token_projected = self.final_projection(activated).squeeze(-1)
        
        return {
            "token_logits": token_projected,
            "line_logits": line_logits,
            "line_labels": line_labels,
            "line_indices": all_line_indices,
            "problem_logits": problem_logits,
            "problem_labels": problem_labels,
            "problem_indices": all_problem_indices
        }
    
    def predict_line_probs(self, batch: AggregatedBatch) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
        """
        Predict probabilities for each line.
        
        Args:
            batch: AggregatedBatch containing embeddings and span information
            
        Returns:
            Tuple of (line probabilities, line indices)
            where line indices are (problem_idx, line_idx, start_idx, end_idx)
        """
        with torch.no_grad():
            self.eval()  # Set model to evaluation mode
            outputs = self.forward(batch)
            probs = torch.sigmoid(outputs["line_logits"])
            self.train()  # Reset to training mode
            return probs, outputs["line_indices"]
    
    def predict_token_probs(self, batch: AggregatedBatch) -> torch.Tensor:
        """
        Predict probabilities for each token.
        
        Args:
            batch: AggregatedBatch containing embeddings and span information
            
        Returns:
            Tensor of probability predictions for each token
        """
        with torch.no_grad():
            self.eval()  # Set model to evaluation mode
            outputs = self.forward(batch)
            probs = torch.sigmoid(outputs["token_logits"])
            self.train()  # Reset to training mode
            return probs
    
    def predict_problem_probs(self, batch: AggregatedBatch) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        """
        Predict probabilities for each problem.
        
        Args:
            batch: AggregatedBatch containing embeddings and span information
            
        Returns:
            Tuple of (problem probabilities, problem indices)
            where problem indices are (problem_idx, start_idx, end_idx)
        """
        with torch.no_grad():
            self.eval()  # Set model to evaluation mode
            outputs = self.forward(batch)
            probs = torch.sigmoid(outputs["problem_logits"])
            self.train()  # Reset to training mode
            return probs, outputs["problem_indices"]


def mha_pool(features: torch.Tensor, num_heads: int = 4) -> torch.Tensor:
    """
    Pool features using a multi-head attention mechanism.
    
    Args:
        features: [seq_len, hidden_dim] - Input features where first num_heads 
                 dimensions are attention logits, rest are values to pool
        num_heads: Number of attention heads
    
    Returns:
        Pooled features of shape [hidden_dim - num_heads]
        
    Algorithm:
        1. Split input into attention logits and values
        2. Softmax attention logits across sequence dimension  
        3. Reshape values into heads
        4. Apply attention-weighted pooling per head
        5. Flatten result
    """
    seq_len, hidden_dim = features.shape
    
    if hidden_dim < num_heads:
        raise ValueError(f"hidden_dim ({hidden_dim}) must be >= num_heads ({num_heads})")
    
    # Step 1: Split features into attention logits and values
    attention_weights = features[:, :num_heads]  # [seq_len, num_heads]
    value_features = features[:, num_heads:]     # [seq_len, remaining_dim] where remaining_dim = hidden_dim - num_heads
    
    # Step 2: Convert attention logits to probabilities via softmax over sequence
    attention_probs = torch.softmax(attention_weights, dim=0)  # [seq_len, num_heads]
    
    # Step 3: Prepare for head-wise processing
    remaining_dim = hidden_dim - num_heads
    if remaining_dim % num_heads != 0:
        raise ValueError(f"remaining_dim ({remaining_dim}) must be divisible by num_heads ({num_heads})")
    
    head_dim = remaining_dim // num_heads
    # Reshape values into separate heads: [seq_len, num_heads, head_dim]
    value_heads = value_features.view(seq_len, num_heads, head_dim)
    
    # Step 4: Apply attention-weighted pooling
    # Need unsqueeze(-1) to broadcast [seq_len, num_heads] -> [seq_len, num_heads, 1]
    # so it can multiply with [seq_len, num_heads, head_dim]
    attention_probs_expanded = attention_probs.unsqueeze(-1)  # [seq_len, num_heads, 1]
    weighted_heads = value_heads * attention_probs_expanded   # [seq_len, num_heads, head_dim]
    
    # Pool by summing across sequence dimension
    pooled_heads = torch.sum(weighted_heads, dim=0)  # [num_heads, head_dim]
    
    # Step 5: Flatten back to 1D feature vector
    pooled_features = pooled_heads.view(-1)  # [num_heads * head_dim] = [remaining_dim]
    
    return pooled_features


class AggLineBasedLoss(nn.Module):
    """
    Custom loss function for training AggLineProbe models.
    
    Combines token-level, line-level, and problem-level BCE losses.
    """
    
    def __init__(
        self, 
        config: Optional[ProbeConfig],
    ):
        """
        Initialize the loss function.
        
        Args:
            token_weight: Weight for token-level loss
            line_weight: Weight for line-level loss
            problem_weight: Weight for problem-level loss
        """
        super().__init__()
        self.config = config
        if config.loss_style == ProbeLoss.BCE:
            self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        elif config.loss_style == ProbeLoss.FL2:
            self.loss_func = lambda inputs, targets: torchvision.ops.focal_loss.sigmoid_focal_loss(
                inputs, targets, reduction='mean', gamma=2, alpha=-1)
        elif config.loss_style == ProbeLoss.BRIER:
            self.loss_func = lambda inputs, targets: (
                F.mse_loss(inputs.sigmoid(), targets.to(dtype=inputs.dtype), reduction='mean')
                * 2  # Multiply by 2 to get closer to the range of BCE loss
            )
        else:
            raise ValueError("Unknown loss style")
    
    def forward(
        self, 
        model_outputs: Dict[str, torch.Tensor], 
        batch: AggregatedBatch
    ) -> torch.Tensor:
        """
        Compute the combined loss.
        
        Args:
            model_outputs: Dictionary from model's forward pass
            batch: Original batch data
            
        Returns:
            Combined loss value
        """
        # Token-level loss
        token_loss = self.loss_func(model_outputs["token_logits"], batch.labels)
        
        # Line-level loss
        line_loss = self.loss_func(model_outputs["line_logits"], model_outputs["line_labels"])
        
        # Problem-level loss
        problem_loss = self.loss_func(model_outputs["problem_logits"], model_outputs["problem_labels"])
        
        # Combined loss
        total_loss = (
            self.config.token_weight * token_loss
            + self.config.line_weight * line_loss
            + self.config.problem_weight * problem_loss
        )

        return total_loss


class AggLineTrainer:
    """
    Trainer for AggLineProbe models using the AggregatedBatch format.
    """
    
    @staticmethod
    def _stack_to_numpy(tensors: List[torch.Tensor]) -> np.ndarray:
        """Cat a list of tensors (might be empty) and return numpy array."""
        if not tensors:  # empty batch corner-case
            return np.empty((0,))
        return torch.cat(tensors).numpy()

    @staticmethod
    def _basic_cls_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
        """Return accuracy/precision/recall/F1/AUC for one level."""
        return dict(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            auc=float(roc_auc_score(y_true, y_prob)
                      if len(np.unique(y_true)) > 1 else 0.5),
            positive_rate=float(np.mean(y_true)),
        )
    
    def __init__(
        self,
        config: Optional[ProbeConfig] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: ProbeConfig instance containing training parameters (creates default if None)
            device: Device to use for training ('cuda', 'cpu', or None for auto-detection)
            dtype: Data type to use for model and tensors
            verbose: Whether to print progress during training
        """
        # Create default config if none provided
        if config is None:
            config = ProbeConfig()
        
        self.config = config
        self.dtype = dtype
        self.verbose = verbose
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.verbose:
            print(f"Using device: {self.device}")
    
    def _get_dataloader(
        self, 
        dataset: GroupedVecLabelDataset, 
        batch_size: int = 8,
        shuffle: bool = True
    ) -> AggQueriesDataLoader:
        """
        Create an AggQueriesDataLoader for the dataset.
        
        Args:
            dataset: GroupedVecLabelDataset to load from
            batch_size: Number of problems per batch
            shuffle: Whether to shuffle the dataset
            
        Returns:
            AggQueriesDataLoader instance
        """
        return AggQueriesDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            dtype=self.dtype,
            device=self.device,
            include_line_spans=True,
        )
    
    def train(
        self, 
        model: AggLineProbe, 
        train_data: GroupedVecLabelDataset,
        val_data: Optional[GroupedVecLabelDataset] = None,
        batch_size: int = 8
    ) -> AggLineProbe:
        """
        Train the model on the given data.
        
        Args:
            model: The model to train
            train_data: Training data
            val_data: Validation data for early stopping (optional)
            batch_size: Number of problems per batch
            
        Returns:
            Trained model
        """
        # Move model to the correct device and dtype
        model = model.to(device=self.device, dtype=self.dtype)
        model.train()
        
        # Initialize loss function
        criterion = AggLineBasedLoss(self.config)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # Create data loaders
        train_loader = self._get_dataloader(train_data, batch_size=batch_size)
        if val_data is not None:
            val_loader = self._get_dataloader(val_data, batch_size=batch_size, shuffle=False)
            
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                if batch is None:
                    continue
                    
                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average training loss
            avg_train_loss = total_loss / max(num_batches, 1)
            
            # Validation phase
            if val_data is not None:
                val_loss = self._evaluate_loss(model, val_loader, criterion)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.verbose and epoch % 1 == 0:
                    print(f"Epoch {epoch}/{self.config.num_epochs}, Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, Patience: {patience_counter}/{self.config.early_stopping_patience}")
                
                # Stop training if no improvement for several epochs
                if patience_counter >= self.config.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if self.verbose and epoch % 1 == 0:
                    print(f"Epoch {epoch}/{self.config.num_epochs}, Train Loss: {avg_train_loss:.6f}")
        
        # Load best model if we did validation
        if val_data is not None and best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def _evaluate_loss(
        self, 
        model: AggLineProbe, 
        dataloader: AggQueriesDataLoader,
        criterion: AggLineBasedLoss
    ) -> float:
        """
        Evaluate model loss on a dataset.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader for the dataset
            criterion: Loss function
            
        Returns:
            Average loss on the dataset
        """
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                    
                outputs = model(batch)
                loss = criterion(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)

    def make_problem_predictions(
        self,
        model: AggLineProbe,
        locs: list[EmbeddedTokenization],
    ) -> list[ProblemPredictions]:
        test_dataset = GroupedVecLabelDataset(
            localizations=locs
        )
        test_loader = self._get_dataloader(
            test_dataset, batch_size=1, shuffle=False)
        num_batches = 0
        preds = []
        with torch.no_grad():
            for batch, loc in zip(test_loader, locs):
                if batch is None:
                    continue
                
                outputs = model(batch)
                num_batches += 1

                def proc_logits(logits):
                    if logits.numel() == 0:
                        return np.array([])
                    probs = torch.sigmoid(logits)
                    return probs.cpu().numpy()

                def proc_labels(labels):
                    if labels.numel() == 0:
                        return np.array([])
                    return labels.cpu().numpy()

                try:
                    pred = ProblemPredictions(
                        problem_id=loc.base_solve.problem.problem_id if loc.base_solve is not None else str(len(preds)),
                        token_prediction_raw_probs=proc_logits(outputs["token_logits"]),
                        line_prediction_raw_probs=proc_logits(outputs["line_logits"]),
                        prob_level_pred=proc_logits(outputs["problem_logits"]).item(),
                        token_labels=proc_labels(batch.labels),
                        line_labels=proc_labels(outputs["line_labels"]),
                        problem_label=proc_labels(outputs["problem_labels"]).item(),
                    )
                except Exception as e:
                    print(f"Error processing batch")
                    #print(f"Problem ID: {loc.base_solve.problem.problem_id}")
                    print(f"Token outputs shape: {outputs['token_logits'].shape}")
                    print(f"Line outputs shape: {outputs['line_logits'].shape}")
                    print(f"Problem outputs shape: {outputs['problem_logits'].shape}")
                    print(f"Batch labels shape: {batch.labels.shape}")
                    print(f"Line labels shape: {outputs['line_labels'].shape}")
                    print(f"Problem labels shape: {outputs['problem_labels'].shape}")
                    raise e
                preds.append(pred)
        return preds

    
    def evaluate(
        self, 
        model: AggLineProbe, 
        test_data: GroupedVecLabelDataset,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            model: The model to evaluate
            test_data: Test data
            batch_size: Number of problems per batch
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Move model to the correct device and dtype
        model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        
        test_loader = self._get_dataloader(test_data, batch_size=batch_size, shuffle=False)
        criterion = AggLineBasedLoss(self.config)
        
        # Per-level aggregation containers
        store = {
            "token": dict(preds=[], probs=[], labels=[]),
            "line": dict(preds=[], probs=[], labels=[]),
            "problem": dict(preds=[], probs=[], labels=[]),
        }
        
        total_loss, num_batches = 0.0, 0
        
        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue
                
                outputs = model(batch)
                total_loss += criterion(outputs, batch).item()
                num_batches += 1
                
                specs = {
                    "token": (outputs["token_logits"], batch.labels),
                    "line": (outputs["line_logits"], outputs["line_labels"]),
                    "problem": (outputs["problem_logits"], outputs["problem_labels"]),
                }
                
                for lvl, (logits, lbls) in specs.items():
                    if logits.numel() == 0:  # may happen for line/problem
                        continue
                    probs = torch.sigmoid(logits)
                    store[lvl]["preds"].append((probs >= 0.5).float().cpu())
                    store[lvl]["probs"].append(probs.cpu())
                    store[lvl]["labels"].append(lbls.cpu())
        
        # Short-circuit if nothing collected
        if not store["token"]["preds"]:
            return {"error": "No valid test batches"}
        
        # Stack -> numpy & compute metrics
        metrics, cal_results = {}, {}
        epsilon = 1e-6
        
        for lvl in ("token", "line", "problem"):
            preds_np = self._stack_to_numpy(store[lvl]["preds"])
            probs_np = np.clip(self._stack_to_numpy(store[lvl]["probs"]),
                               epsilon, 1 - epsilon)
            labels_np = self._stack_to_numpy(store[lvl]["labels"])
            
            # Basic classification stats
            base = self._basic_cls_metrics(labels_np, preds_np, probs_np)
            metrics.update({f"{lvl}_{k}": v for k, v in base.items()})
            
            # Calibration
            raw = ExperimentResults(
                predicted_probabilities=probs_np,
                true_labels=labels_np,
                bin_strategy=BinStrategy.Uniform,
                num_bins=10,
            )
            platt = raw.to_platt_scaled()
            metrics.update({
                f"{lvl}_ece": float(raw.ece),
                f"{lvl}_brier_score": float(raw.brier_score),
                f"{lvl}_brier_skill_score": float(raw.skill_score),
                f"{lvl}_platt_ece": float(platt.ece),
                f"{lvl}_platt_brier_score": float(platt.brier_score),
                f"{lvl}_platt_brier_skill_score": float(platt.skill_score),
            })
            cal_results[lvl] = {"raw": raw, "platt": platt}
        
        self.last_calibration_results = cal_results
        metrics["test_loss"] = float(total_loss / max(num_batches, 1))
        
        return metrics
    
    def plot_reliability_diagram(self, level: str, save_path=None, scaled=False):
        """
        Plot a reliability diagram for the last evaluated model using calipy.
        
        Args:
            level: Which level to plot ('token', 'line', or 'problem')
            save_path: Path to save the plot to (optional)
            scaled: Whether to plot Platt-scaled results
        
        Returns:
            The figure and axes from calipy's plot, or None if no data.
        """
        if not hasattr(self, 'last_calibration_results') or self.last_calibration_results is None:
            print("No calibration results available. Run evaluate() first.")
            return None
            
        if level not in self.last_calibration_results:
            print(f"No calibration results available for level: {level}")
            return None
        
        results_to_plot = self.last_calibration_results[level]

        # Directly use calipy plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_type_label = 'Scaled' if scaled else 'Unscaled'
        
        # Plot results using calipy, passing the correct string label
        results_to_plot['raw'].reliability_plot(
            ax,
            show_unscaled=not scaled,
            show_scaled=scaled,
            show_counts=plot_type_label,    # Pass 'Scaled' or 'Unscaled'
            show_quantiles=plot_type_label, # Pass 'Scaled' or 'Unscaled'
            annotate=plot_type_label        # Pass 'Scaled' or 'Unscaled'
        )
        
        plot_title = f"Reliability Diagram - {level.capitalize()} {plot_type_label}"
        plt.title(plot_title)
        
        if save_path:
            # Ensure parent directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved {level.capitalize()} {plot_type_label} reliability diagram to {save_path}")
        
        return fig, ax
    
    def train_and_evaluate(
        self, 
        model: AggLineProbe, 
        train_data: GroupedVecLabelDataset, 
        test_data: GroupedVecLabelDataset,
        val_data: Optional[GroupedVecLabelDataset] = None,
        batch_size: int = 8
    ) -> Tuple[AggLineProbe, Dict[str, float]]:
        """
        Train and evaluate a model.
        
        Args:
            model: The model to train
            train_data: Training data
            test_data: Test data
            val_data: Validation data for early stopping (optional)
            batch_size: Number of problems per batch
            
        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        model = self.train(model, train_data, val_data, batch_size=batch_size)
        metrics = self.evaluate(model, test_data, batch_size=batch_size)
        return model, metrics


def _print_metrics_for_level(metrics: Dict[str, float], level: str) -> None:
    """Helper function to print metrics for a specific level (token or line)."""
    prefix = f"{level}_"
    print(f"\n{level.capitalize()}-level metrics:")
    
    # Check for keys before accessing
    has_rate = f'{prefix}positive_rate' in metrics
    has_cal = f'{prefix}ece' in metrics
    has_platt_cal = f'{prefix}platt_ece' in metrics

    if has_rate:
        print(f"Positive class rate: {metrics[f'{prefix}positive_rate']:.4f} ± {metrics.get(f'{prefix}positive_rate_std', 0.0):.4f}")
    
    print(f"Accuracy:  {metrics[f'{prefix}accuracy']:.4f} ± {metrics.get(f'{prefix}accuracy_std', 0.0):.4f}")
    print(f"Precision: {metrics[f'{prefix}precision']:.4f} ± {metrics.get(f'{prefix}precision_std', 0.0):.4f}")
    print(f"Recall:    {metrics[f'{prefix}recall']:.4f} ± {metrics.get(f'{prefix}recall_std', 0.0):.4f}")
    print(f"F1 Score:  {metrics[f'{prefix}f1']:.4f} ± {metrics.get(f'{prefix}f1_std', 0.0):.4f}")
    print(f"AUC:       {metrics[f'{prefix}auc']:.4f} ± {metrics.get(f'{prefix}auc_std', 0.0):.4f}")
    
    if has_cal:
        print(f"\n{level.capitalize()}-level Calibration Metrics (Raw):")
        print(f"ECE:           {metrics[f'{prefix}ece']:.4f} ± {metrics.get(f'{prefix}ece_std', 0.0):.4f}")
        print(f"Brier Score:   {metrics[f'{prefix}brier_score']:.4f} ± {metrics.get(f'{prefix}brier_score_std', 0.0):.4f}")
        print(f"Brier Skill:   {metrics[f'{prefix}brier_skill_score']:.4f} ± {metrics.get(f'{prefix}brier_skill_score_std', 0.0):.4f}")
    
    if has_platt_cal:
        print(f"\n{level.capitalize()}-level Calibration Metrics (Platt Scaling):")
        print(f"Platt ECE:     {metrics[f'{prefix}platt_ece']:.4f} ± {metrics.get(f'{prefix}platt_ece_std', 0.0):.4f}")
        print(f"Platt Brier:   {metrics[f'{prefix}platt_brier_score']:.4f} ± {metrics.get(f'{prefix}platt_brier_score_std', 0.0):.4f}")
        print(f"Platt Skill:   {metrics[f'{prefix}platt_brier_skill_score']:.4f} ± {metrics.get(f'{prefix}platt_brier_skill_score_std', 0.0):.4f}")


def print_fold_results(results: Dict[str, Any], type_label: str = "Line") -> None:
    """
    Print formatted results from fold training.
    
    Args:
        results: Results dictionary from train_agg_line_model_on_folds
        type_label: Label for the model type (e.g., "Line", "Token")
    """
    metrics = results['metrics']
    models = results['models']
    
    print(f"\n{type_label} Probe Performance (averaged across folds):")
    # print(f"Available metrics: {list(metrics.keys())}")
    
    # Print metrics for token level
    _print_metrics_for_level(metrics, 'token')
    
    # Print metrics for line level
    _print_metrics_for_level(metrics, 'line')
    
    # Print metrics for problem level
    _print_metrics_for_level(metrics, 'problem')
    
    print(f"\nTest Loss: {metrics['test_loss']:.4f} ± {metrics.get('test_loss_std', 0.0):.4f}")
    print(f"Number of trained models: {len(models)}")
    
    # Print explanation about base rate if we have them
    has_token_rate = 'token_positive_rate' in metrics
    has_line_rate = 'line_positive_rate' in metrics
    has_problem_rate = 'problem_positive_rate' in metrics
    if has_token_rate or has_line_rate or has_problem_rate:
        print("\nNote: 'Positive class rate' is the proportion of labels that are positive (1).")
        print("      This represents the accuracy you would get by always predicting the majority class.")


def train_agg_line_model_on_folds(
    dataset: GroupedVecLabelDatasetMultiFold,
    input_dim: int,
    config: Optional[ProbeConfig],
    trainer: Optional[AggLineTrainer],
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> Dict[str, Union[Dict[str, float], List[nn.Module]]]:
    """
    Train line-level models on each fold of the dataset and report average metrics.
    
    Args:
        dataset: Dataset with multiple folds for cross-validation
        input_dim: Dimension of token embeddings
        config: ProbeConfig instance containing training parameters (creates default if None)
        trainer: AggLineTrainer instance to use (will create a default one if None)
        hidden_dim: Dimension of hidden layer in the model (overrides config if specified)
        batch_size: Number of problems per batch (overrides config if specified)
        dtype: Data type to use for model and tensors
        verbose: Whether to print progress during training
        
    Returns:
        Dictionary with average metrics across all folds and list of trained models
    """
    if len(dataset.folds_train_test) == 0:
        raise ValueError("Dataset contains no folds")
    
    # Initialize storage for metrics and models across folds
    all_metrics = []
    trained_models = []
    
    # Train and evaluate on each fold
    for fold_idx, (train_data, test_data) in enumerate(dataset.folds_train_test):
        # Display data info for this fold
        if verbose:
            print(f"Fold {fold_idx+1} - Training data: {len(train_data.localizations)} samples, "
                  f"Test data: {len(test_data.localizations)} samples")
        
        # Create model for this fold
        model = AggLineProbe(
            input_dim=input_dim,
            config=config,
            dtype=dtype
        )

        # Train and evaluate
        model, metrics = trainer.train_and_evaluate(
            model, train_data, test_data, batch_size=config.batch_size
        )
        
        # Debug: Check if positive rate metrics are in the fold metrics
        if 'token_positive_rate' not in metrics or 'line_positive_rate' not in metrics:
            print(f"WARNING: Fold {fold_idx+1} metrics missing positive rate keys!")
            print(f"Available metrics: {list(metrics.keys())}")
        
        all_metrics.append(metrics)
        trained_models.append(model)
        
        if verbose:
            print(f"Fold {fold_idx+1} metrics: {metrics}")
    
    # Calculate average metrics across folds
    avg_metrics = {}
    
    # First, collect all possible metric keys from all folds
    all_metric_keys = set()
    for metrics in all_metrics:
        all_metric_keys.update(metrics.keys())
    
    print(f"DEBUG: All available metric keys across folds: {all_metric_keys}")
    
    # Calculate mean and std for each metric 
    for metric in all_metric_keys:
        # Check if this metric exists in all folds and is a float value
        values = [fold_metrics.get(metric) for fold_metrics in all_metrics 
                  if metric in fold_metrics and isinstance(fold_metrics.get(metric), float)]
        
        if values:  # Only calculate if we have valid values
            avg_metrics[metric] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
        else:
            print(f"DEBUG: No valid values for metric '{metric}' across folds")
    
    # Debug: Check final averaged metrics
    print(f"DEBUG: Final averaged metrics keys: {list(avg_metrics.keys())}")
    
    return {
        'metrics': avg_metrics,
        'models': trained_models,
        'trainer': trainer
    }


def train_and_evaluate_with_agg_line_probe(
    device: str,
    config: Optional[ProbeConfig] = None,
) -> Dict[str, float]:
    """
    Train and evaluate a line-level probe model using the embedding style from config.
    
    Args:
        device: Device to use for training ('cuda' or 'cpu')
        config: ProbeConfig instance containing training parameters (creates default if None)

    Returns:
        Dictionary of evaluation metrics
    """
    # Create or update config with any overrides
    if config is None:
        config = ProbeConfig()
    
    print(f"\n{'='*80}")
    print(f"Training with {config.embedding_style.value} embedding style - Aggregated Line Probe")
    datasets_str = ", ".join(str(ds) for ds in config.datasets)
    print(f"Using datasets: {datasets_str}")
    print(f"Using generation model: {config.gen_model_name}")
    print(f"Using embedding model: {config.embed_lm_name}")
    print(f"Loss weights: token={config.token_weight}, line={config.line_weight}, problem={config.problem_weight}")
    print(f"{'='*80}")
    
    # Get localizations dataset
    localizations = get_or_serialize_localizations_embedded(config)
    print(f"Loaded {len(localizations)} localizations")
    print(f"Datasets present: {', '.join(str(ds) for ds in localizations.get_dataset_name_set())}")
    
    print(debug_str_filterables(localizations.iter_all()))

    # Convert to grouped dataset
    dataset = localizations_to_grouped_vec_label_dataset(localizations, n_folds=5)
    print(dataset)
    
    # Determine input dimension from the first available localization
    input_dim = None
    for fold_idx, (train_data, _) in enumerate(dataset.folds_train_test):
        for loc in train_data.localizations:
            if loc.base_tokens_embedding is not None:
                input_dim = loc.base_tokens_embedding.shape[1]
                break
        if input_dim is not None:
            break
    
    if input_dim is None:
        raise ValueError("Could not determine input dimension from localizations")
    
    print(f"Determined input dimension: {input_dim}")
    
    # Create trainer with config
    trainer = AggLineTrainer(
        config=config,
        device=device,
        dtype=torch.float32,
    )
    
    # Train models on folds
    results = train_agg_line_model_on_folds(
        dataset,
        input_dim=input_dim,
        config=config,
        trainer=trainer,
    )
    
    # Print results
    print(f"\nResults for {config.embedding_style.value} with Aggregated Line Probe:")
    print_fold_results(results)
    
    # Plot reliability diagram for the last fold if trainer is available and plotting enabled
    if config.save_plots and 'trainer' in results and hasattr(results['trainer'], 'last_calibration_results'):
        # Prepare properties for plot metadata and filename
        plot_properties = {
            'style': config.embedding_style.value,
            'datasets': datasets_str,
            'gen_model': config.gen_model_name,
            'embed_model': config.embed_lm_name,
            'fix_ref': config.fix_style,
        }
        dir_components = [
            "probe_agg", 
            f"datasets_{datasets_str.replace(', ', '_')}",
            f"gen_{config.gen_model_name.replace('/', '_')}",
            f"embed_{config.embed_lm_name.replace('/', '_')}",
            f"fix_{config.fix_style.replace('/', '_')}",
        ]

        # Loop through levels and scaling options
        for level in ['token', 'line']:
            for scaled in [False, True]:
                fig, ax = results['trainer'].plot_reliability_diagram(
                    level=level, 
                    scaled=scaled
                )
                if fig is not None:
                    scale_label = "scaled" if scaled else "raw"
                    plot_title = f"Reliability Diagram - {config.embedding_style.value} {level.capitalize()} {scale_label}"
                    # Use plot_and_save - it handles directory creation, metadata, and closing
                    plot_and_save(
                        dir_components=dir_components,
                        title=plot_title, 
                        properties={**plot_properties, 'level': level, 'scale': scale_label, 'type': 'reliability'},
                        show=False, # Don't show during multi-plot saving
                        format=['png', 'svg']
                    )
        
        print(f"Saved reliability diagrams - see plots/{'/'.join(dir_components)}")
    
    # Store metrics and clear memory
    metrics = results['metrics']
    del localizations
    del dataset
    del results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    config = ProbeConfig(
        datasets=[
            DatasetName.humaneval_plus,
            DatasetName.livecodebench,
            DatasetName.mbpp_plus,
        ],
        gen_model_name = OpenAiModelNames.gpt_4o,
        max_problems=700,
        learning_rate=0.005,
        num_epochs=25,
        batch_size=4,
        hidden_dim=32,
        embedding_style=EmbeddingStyle.MIDDLE_LAYER,
        line_weight=0.3,
        token_weight=0.4,
        problem_weight=0.3,
    )
    
    # Train using the config
    metrics = train_and_evaluate_with_agg_line_probe(
        device=device,
        config=config,
    )
    
    #print("\nFinal metrics:", metrics)
    print("\nDone!")
