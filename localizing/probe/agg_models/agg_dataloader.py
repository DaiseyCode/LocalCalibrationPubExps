"""DataLoader implementation for aggregating problem data into batches."""

import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
from localizing.probe.probe_data_gather import GroupedVecLabelDataset
from .agg_data_structures import AggregatedBatch


class AggQueriesDataLoader(DataLoader):
    """DataLoader that aggregates problem data into batches with line boundary tracking."""
    
    def __init__(
        self, 
        dataset: GroupedVecLabelDataset, 
        dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        include_line_spans: bool = True,
        **kwargs
    ):
        """
        Initialize the aggregated queries data loader.
        
        Args:
            dataset: GroupedVecLabelDataset to load from
            dtype: Data type to convert tensors to
            device: Device to load data to
            include_line_spans: Whether to compute and include line spans in the batch
            **kwargs: Additional arguments for DataLoader
        """
        # Set default collate_fn to None so we can override it
        kwargs['collate_fn'] = None
        super().__init__(dataset, **kwargs)
        
        self.dtype = dtype
        self.device = device
        self.include_line_spans = include_line_spans
        
        # Custom collate function that aggregates problem data
        def collate_fn(batch):
            # Check if batch has valid items
            if not batch:
                return None
            
            # Get all localizations from the batch
            localizations = batch  # In GroupedVecLabelDataset, __getitem__ returns the localization directly
            
            all_embeddings = []
            all_labels = []
            problem_spans = []
            line_spans = []
            problem_metadata = []
            
            current_offset = 0
            
            for i, loc in enumerate(localizations):
                # Skip if no embeddings or labels
                if loc.base_tokens_embedding is None:
                    continue
                
                # Get embeddings and labels for this problem
                problem_embeddings = loc.base_tokens_embedding
                problem_labels = torch.tensor(loc.gt_base_token_keeps, dtype=torch.float)
                
                # Verify shapes match
                if problem_embeddings.shape[0] != len(problem_labels):
                    raise ValueError(
                        f"Mismatch in problem {i}: embeddings shape {problem_embeddings.shape[0]} "
                        f"vs labels length {len(problem_labels)}"
                    )
                
                # Add this problem's data
                all_embeddings.append(problem_embeddings)
                all_labels.append(problem_labels)
                
                # Calculate problem span
                problem_length = problem_embeddings.shape[0]
                problem_spans.append((current_offset, current_offset + problem_length))
                
                # Compute line boundaries if requested
                if self.include_line_spans:
                    problem_line_spans = self._compute_line_spans(loc, current_offset)
                    line_spans.append(problem_line_spans)
                
                # Track problem metadata
                metadata = {
                    'problem_id': getattr(loc, 'problem_id', f"problem_{i}"),
                    'token_count': problem_length,
                    'base_tokens': loc.base_tokens,
                }
                problem_metadata.append(metadata)
                
                # Update offset for next problem
                current_offset += problem_length
            
            # If no valid problems were found, return None
            if not all_embeddings:
                return None
            
            # Concatenate all embeddings and labels
            concatenated_embeddings = torch.cat(all_embeddings, dim=0).to(dtype=self.dtype)
            concatenated_labels = torch.cat(all_labels, dim=0)
            
            # Move to device if specified
            if self.device is not None:
                concatenated_embeddings = concatenated_embeddings.to(device=self.device)
                concatenated_labels = concatenated_labels.to(device=self.device)
            
            # Create and return the aggregated batch
            return AggregatedBatch(
                embeddings=concatenated_embeddings,
                labels=concatenated_labels,
                problem_spans=problem_spans,
                line_spans=line_spans if self.include_line_spans else [],
                problem_metadata=problem_metadata
            )
        
        # Set our custom collate function
        self.collate_fn = collate_fn
    

    def _compute_line_spans(self, localization, offset: int) -> List[Tuple[int, int]]:
        """
        Compute line spans for a localization with adjusted offsets.
        
        Args:
            localization: The localization object
            offset: The offset to add to all indices
            
        Returns:
            List of (start_idx, end_idx) spans for each line
        """
        line_spans = localization.get_line_spans(ignore_empty_lines=True)
        line_spans = [(span[0] + offset, span[1] + offset) for span in line_spans]
        return line_spans
