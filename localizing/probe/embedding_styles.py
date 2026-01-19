from enum import Enum
from typing import List, Tuple


class EmbeddingStyle(Enum):
    LAST_LAYER = "last_layer"
    MIDDLE_LAYER = "middle_layer"
    THREE_QUARTERS_LAYER = "three_quarters_layer"
    THREE_LAYER = "three_layer"
    SHIFTED_THREE_QUARTERS = "shifted_three_quarters"
    SHIFTED_THREE_QUARTERS_AND_MIDDLE = "shifted_three_quarters_and_middle"
    
    @property
    def layer_token_combinations(self) -> List[Tuple[int, float]]:
        """Return pairs of (token_offset, layer_fraction) to use for this embedding style.
        
        This is the primary definition for each embedding style.
        
        A token_offset of 0 means the current token, -1 means the previous token, etc.
        A layer_fraction of 1.0 means the final layer, 0.5 is the middle layer, etc.
        
        Returns:
            List of (token_offset, layer_fraction) tuples.
        """
        if self == EmbeddingStyle.LAST_LAYER:
            return [(0, 1.0)]
        elif self == EmbeddingStyle.MIDDLE_LAYER:
            return [(0, 0.5)]
        elif self == EmbeddingStyle.THREE_QUARTERS_LAYER:
            return [(0, 0.75)]
        elif self == EmbeddingStyle.THREE_LAYER:
            return [(0, 0.5), (0, 0.75), (0, 1.0)]
        elif self == EmbeddingStyle.SHIFTED_THREE_QUARTERS:
            return [(0, 0.75), (-1, 0.75)]
        elif self == EmbeddingStyle.SHIFTED_THREE_QUARTERS_AND_MIDDLE:
            return [(0, 0.5), (-1, 0.75)]
        else:
            raise ValueError(f"Unknown embedding style: {self}")
    
    @property
    def hidden_layer_fractions(self) -> List[float]:
        """Return the layer fractions to extract for this embedding style.
        
        Derived from layer_token_combinations.
        
        Returns:
            List of layer fractions to extract.
        """
        return sorted(set(frac for _, frac in self.layer_token_combinations))
    
    @property
    def shift_offsets(self) -> List[int]:
        """Return the token offsets to include in the embedding.
        
        Derived from layer_token_combinations.
        
        A shift offset of 0 means the current token.
        A shift offset of -1 means the previous token.
        A shift offset of 1 would mean the next token (though not currently used).
        
        Returns:
            List of integer offsets representing which relative token positions to include.
        """
        return sorted(set(offset for offset, _ in self.layer_token_combinations))
    
    @property
    def includes_shifted_embeddings(self) -> bool:
        """Whether this embedding style includes shifted (non-current) token embeddings.
        
        Derived from shift_offsets.
        """
        return any(offset != 0 for offset in self.shift_offsets) 