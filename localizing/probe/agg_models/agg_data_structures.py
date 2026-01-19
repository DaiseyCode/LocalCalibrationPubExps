"""Data structures for aggregated probe models."""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any


@dataclass
class AggregatedBatch:
    """Container for an aggregated batch of problem data."""
    # Concatenated embeddings from all problems in the batch
    embeddings: torch.Tensor  # shape: [total_tokens, embedding_dim]
    
    # Concatenated labels from all problems in the batch
    labels: torch.Tensor  # shape: [total_tokens]
    
    # List of (start_idx, end_idx) for each problem
    # These indices refer to positions in the concatenated embeddings/labels
    problem_spans: List[Tuple[int, int]]
    
    # List of line boundary spans for each problem
    # Each problem has a list of (start_idx, end_idx) tuples for each line
    # Indices are adjusted to account for concatenation
    line_spans: List[List[Tuple[int, int]]]
    
    # Metadata about the problems (optional)
    problem_metadata: Optional[List[Dict[str, Any]]] = None

    
    # Helper methods for debugging and information
    def get_num_problems(self) -> int:
        """Get the number of problems in this batch."""
        return len(self.problem_spans)
    
    def get_num_total_tokens(self) -> int:
        """Get the total number of tokens across all problems."""
        return self.embeddings.shape[0]
    
    def get_problem_embedding(self, problem_idx: int) -> torch.Tensor:
        """Get embeddings for a specific problem."""
        start, end = self.problem_spans[problem_idx]
        return self.embeddings[start:end]
    
    def get_problem_labels(self, problem_idx: int) -> torch.Tensor:
        """Get labels for a specific problem."""
        start, end = self.problem_spans[problem_idx]
        return self.labels[start:end]
    
    def get_line_embedding(self, problem_idx: int, line_idx: int) -> torch.Tensor:
        """Get embeddings for a specific line in a problem."""
        if problem_idx >= len(self.line_spans) or line_idx >= len(self.line_spans[problem_idx]):
            raise IndexError(f"Invalid indices: problem_idx={problem_idx}, line_idx={line_idx}")
        start, end = self.line_spans[problem_idx][line_idx]
        return self.embeddings[start:end]
    
    def get_line_labels(self, problem_idx: int, line_idx: int) -> torch.Tensor:
        """Get labels for a specific line in a problem."""
        if problem_idx >= len(self.line_spans) or line_idx >= len(self.line_spans[problem_idx]):
            raise IndexError(f"Invalid indices: problem_idx={problem_idx}, line_idx={line_idx}")
        start, end = self.line_spans[problem_idx][line_idx]
        return self.labels[start:end]
