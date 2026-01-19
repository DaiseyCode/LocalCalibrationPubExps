import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numba import jit


@jit(nopython=True)
def compute_token_keep_correlations(multi_token_keeps):
    """
    Compute correlation matrix between token keeps across multiple samples.
    Args:
        multi_token_keeps: numpy array of shape (num_samples, num_tokens) containing
            boolean values (0 or 1) indicating whether each token was kept in each sample

    Returns:
        joint_probs: numpy array of shape (num_tokens, num_tokens) where entry [i,j]
            represents how often tokens i and j are kept together
        marginal_probs: numpy array of shape (num_tokens,) containing probability
            of each token being kept
        conditional_probs: numpy array of shape (num_tokens, num_tokens) where entry [i,j]
            represents P(token i is kept | token j is kept)
            Note: entry [i,j] will be 0 if token j is never kept (marginal_prob[j] = 0)
    """
    if not (multi_token_keeps.ndim == 2):
        raise ValueError("Input must be 2-dimensional")

    num_samples, num_tokens = multi_token_keeps.shape
    if num_samples == 0 or num_tokens == 0:
        raise ValueError("Input cannot be empty")

    # Single pass through the data to both validate and compute joint probabilities
    joint_probs = np.zeros((num_tokens, num_tokens))
    for sample in range(num_samples):
        for i in range(num_tokens):
            val_i = multi_token_keeps[sample, i]
            if not (val_i == 0 or val_i == 1):
                raise ValueError("Input must contain only 0s and 1s")
            if val_i == 1:
                for j in range(num_tokens):
                    val_j = multi_token_keeps[sample, j]
                    if not (val_j == 0 or val_j == 1):  # validate j values too
                        raise ValueError("Input must contain only 0s and 1s")
                    if val_j == 1:
                        joint_probs[i, j] += 1

    joint_probs = joint_probs / num_samples

    # Marginal probs are just the diagonal of joint_probs
    marginal_probs = np.zeros(num_tokens)
    for i in range(num_tokens):
        marginal_probs[i] = joint_probs[i, i]

    # Calculate conditional probabilities
    conditional_probs = np.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        for j in range(num_tokens):
            if marginal_probs[j] > 0:
                conditional_probs[i, j] = joint_probs[i, j] / marginal_probs[j]

    return joint_probs, marginal_probs, conditional_probs


def visualize_token_correlations(
        conditional_probs: np.ndarray,
        marginal_probs: np.ndarray,
        figsize: tuple[int, int] = (10, 10),
        token_labels: list[str] | None = None,
        true_labels: list[bool] | None = None,
        code_context: bool = True
) -> Figure:
    """
    Visualize token correlations with conditional probabilities as a heatmap,
    marginal probabilities along the axes, and optionally a code context preview.

    Args:
        conditional_probs: numpy array of shape (num_tokens, num_tokens) containing
            P(token i kept | token j kept)
        marginal_probs: numpy array of shape (num_tokens,) containing P(token i kept)
        figsize: tuple of (width, height) for the figure
        token_labels: optional list of labels for the tokens. If None, uses indices
        true_labels: optional list of booleans indicating if each token should be kept
        code_context: if True, adds a code-like preview of tokens above the heatmap

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization.
    """
    num_tokens = len(marginal_probs)
    if token_labels is None:
        token_labels = [str(i) for i in range(num_tokens)]

    # Create figure with space for code preview
    fig = plt.figure(figsize=figsize)

    # Adjust height ratios to make room for true labels
    if code_context:
        gs = plt.GridSpec(3, 2, width_ratios=[20, 1],
                          height_ratios=[4, 20, 2],  # Made bottom section bigger
                          hspace=0.3)

        # Add code preview at top
        ax_code = fig.add_subplot(gs[0, :])
        code_text = "Code: " + "".join(token_labels)
        ax_code.text(0.05, 0.5, code_text,
                     fontfamily='monospace',
                     backgroundcolor='whitesmoke',
                     wrap=True)
        ax_code.axis('off')

        # Main heatmap
        ax_heat = fig.add_subplot(gs[1, 0])
    else:
        gs = plt.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[20, 2])
        ax_heat = fig.add_subplot(gs[0, 0])

    im = ax_heat.imshow(conditional_probs, cmap='Blues', aspect='auto')
    ax_heat.set_xticks(range(num_tokens))
    ax_heat.set_yticks(range(num_tokens))
    ax_heat.set_xticklabels(token_labels, rotation=45, ha='right')
    ax_heat.set_yticklabels(token_labels)
    ax_heat.set_xlabel('Token j (condition)')
    ax_heat.set_ylabel('Token i (outcome)')
    ax_heat.set_title('P(token i kept | token j kept)')

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[1 if code_context else 0, 1])
    plt.colorbar(im, cax=ax_cbar)

    # Bottom marginal probabilities with true labels
    ax_right = fig.add_subplot(gs[-1, 0])

    # Color bars based on true labels if provided
    if true_labels is not None:
        colors = ['#2ecc71' if keep else '#e74c3c' for keep in true_labels]  # Green for keep, red for drop
        ax_right.bar(range(num_tokens), marginal_probs, color=colors, alpha=0.6)

        # Add true label indicators below the bars
        ax_right.set_xticks(range(num_tokens))
        ax_right.set_xticklabels(['✓' if keep else '✗' for keep in true_labels],
                                 rotation=0, ha='center')
    else:
        ax_right.bar(range(num_tokens), marginal_probs, alpha=0.5)
        ax_right.set_xticks([])

    ax_right.set_xlim(-0.5, num_tokens - 0.5)
    ax_right.set_ylim(0, max(marginal_probs) * 1.1)
    ax_right.set_ylabel('P(token)')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    pass