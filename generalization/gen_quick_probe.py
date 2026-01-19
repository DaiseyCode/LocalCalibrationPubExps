from calipy.experiment_results import ExperimentResults
import gc
import matplotlib.pyplot as plt
import numpy as np
from generalization.hello_halu import get_embedded_halu_eval, get_halu_eval_base_locs
import pickle
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from itertools import islice
from localizing.filter_helpers import debug_str_filterables
import torch
from localizing.multi_data_gathering import create_tokenized_localizations
from localizing.probe.agg_models.agg_config import ProbeConfig
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.probe.probe_data_gather import GroupedVecLabelDataset, add_embedding_to_localization_list, get_or_serialize_localizations_embedded, localizations_to_grouped_vec_label_dataset
from localizing.probe.probe_models_agg import AggLineProbe, AggLineTrainer, train_agg_line_model_on_folds
from pape.configs import BASE_PAPER_CONFIG, gen_model_name_qwen
import dataclasses
import solve_helpers

# Figure size configuration (in inches)
FIG_SIZE = (4, 4)


def run_quick_gen_probe(
    config: ProbeConfig = dataclasses.replace(
        BASE_PAPER_CONFIG,
        #num_epochs=30,
        #embedding_style=EmbeddingStyle.MIDDLE_LAYER,
        #embed_lm_name=gen_model_name_qwen,
        #num_epochs=3,
        #embedding_style=EmbeddingStyle.SHIFTED_THREE_QUARTERS_AND_MIDDLE,
        #line_weight=0.33,
        #token_weight=0.34,
        #problem_weight=0.33,
    )
    #ProbeConfig(
    #    num_epochs=30,
    #    embedding_style=EmbeddingStyle.MIDDLE_LAYER,
    #    #line_weight=0.2,
    #    #token_weight=0.5,
    #    #problem_weight=0.3,
    #),
):
    print(f"\n{'='*80}")
    print(f"Training with {config.embedding_style.value} embedding style - Aggregated Line Probe")
    datasets_str = ", ".join(str(ds) for ds in config.datasets)
    print(f"Using datasets: {datasets_str}")
    print(f"Using generation model: {config.gen_model_name}")
    print(f"Using embedding model: {config.embed_lm_name}")
    print(f"Loss weights: token={config.token_weight}, line={config.line_weight}, problem={config.problem_weight}")
    print(f"{'='*80}")
    
    # Get localizations dataset
    dev_mode = False
    localizations = get_or_serialize_localizations_embedded(config, dev_mode=dev_mode)
    print(f"Loaded {len(localizations)} localizations")
    print(f"Datasets present: {', '.join(str(ds) for ds in localizations.get_dataset_name_set())}")
    
    print(debug_str_filterables(localizations.iter_all()))

    # Convert to grouped dataset
    dataset = localizations_to_grouped_vec_label_dataset(
        localizations, 
        n_folds=config.n_folds,
    )
    print(dataset)
    #train_data, test_data = localizations_to_grouped_vec_label_dataset_custom_split(
    #    localizations,
    #)
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = AggLineTrainer(
        config=config,
        device=device,
        dtype=torch.float32,
    )
    
    verbose = True

    for fold_idx, (train_data, test_data) in enumerate(dataset.folds_train_test):
        # Display data info for this fold
        if verbose:
            print(f"Fold {fold_idx+1} - Training data: {len(train_data.localizations)} samples, "
                  f"Test data: {len(test_data.localizations)} samples")
        
        # Create model for this fold
        model = AggLineProbe(
            input_dim=input_dim,
            config=config,
            dtype=trainer.dtype
        )

        # Train and evaluate
        model, metrics = trainer.train_and_evaluate(
            model, train_data, test_data, batch_size=config.batch_size
        )
        print(f"Fold {fold_idx+1} metrics:")
        print(metrics)
        break
    

    print("code preds")
    code_preds = trainer.make_problem_predictions(model, list(localizations.iter_passed_filtered()))
    code_result = ExperimentResults(
        predicted_probabilities=np.array([p.prob_level_pred for p in code_preds]),
        true_labels=np.array([p.problem_label for p in code_preds]),
    ).to_platt_scaled()

    # Clear out memory before loading halu
    del dataset
    del localizations
    gc.collect()
    torch.cuda.empty_cache()
    print("cleared out memory")

    print("Evaluating on HaluEval")
    print("loading")
    halu = get_embedded_halu_eval(
        max_problems=1200, embed_lm_name=config.embed_lm_name, embedding_style=config.embedding_style)
    for loc in islice(halu.iter_passed_filtered(), 10):
        print("base text")
        print(loc.get_base_text())
        print("base tokens")
        print(loc.base_tokens)
        print("gt base token keeps")
        print(loc.gt_base_token_keeps)
        break
    print("got dataset convert")
    halu_dataset = GroupedVecLabelDataset(localizations=list(halu.iter_passed_filtered()))
    print("evaluating..")

    print("dumping model")
    pickle.dump(model, open("halumodel.pkl", "wb"))

    eval_metrics = trainer.evaluate(model, halu_dataset, batch_size=config.batch_size)
    print(f"HaluEval metrics: {eval_metrics}")
    preds = trainer.make_problem_predictions(model, list(halu.iter_passed_filtered()))
    print("dumping preds")
    pickle.dump(preds, open("halupreds.pkl", "wb"))


    result = ExperimentResults(
        predicted_probabilities=np.array([p.prob_level_pred for p in preds]),
        true_labels=np.array([p.problem_label for p in preds]),
    )

    pickle.dump(result, open("haluresult.pkl", "wb"))

    print(result)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    result.reliability_plot(
        ax,
        show_quantiles='Scaled',
        show_scaled=True,
        show_unscaled=True,
    )
    ax.set_xlabel('Estimated Probability')
    ax.set_ylabel('Actual Probability')
    plt.savefig("halu_reliability_diagram_both.png")
    plt.savefig("halu_reliability_diagram_both.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    result.reliability_plot(
        ax,
        show_quantiles='Scaled',
        show_scaled=True,
        show_unscaled=False,
        annotate='Scaled',
        show_counts='Scaled',
    )
    ax.set_xlabel('Estimated Probability')
    ax.set_ylabel('Actual Probability')
    plt.savefig("halu_reliability_diagram_scaled.png")
    plt.savefig("halu_reliability_diagram_scaled.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    result.reliability_plot(
        ax,
        show_quantiles='Unscaled',
        show_scaled=False,
        show_unscaled=True,
        annotate='Unscaled',
        show_counts='Unscaled',
    )
    ax.set_xlabel('Estimated Probability')
    ax.set_ylabel('Actual Probability')
    plt.savefig("halu_reliability_diagram_unscaled.png")
    plt.savefig("halu_reliability_diagram_unscaled.svg")
    plt.close()


    scalled_data = code_result._rescaler_all_data.predict(np.array([p.prob_level_pred for p in preds]))
    result = ExperimentResults(
        predicted_probabilities=scalled_data,
        true_labels=np.array([p.problem_label for p in preds]),
    )

    pickle.dump(result, open("haluresult_code_scaled.pkl", "wb"))

    print(result)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    result.reliability_plot(
        ax,
        show_quantiles='Scaled',
        show_scaled=True,
        show_unscaled=True,
    )
    ax.set_xlabel('Estimated Probability')
    ax.set_ylabel('Actual Probability')
    plt.savefig("halu_reliability_diagram_code_scaled_both.png")
    plt.savefig("halu_reliability_diagram_code_scaled_both.svg")
    plt.show()
    

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    result.reliability_plot(
        ax,
        show_quantiles='Scaled',
        show_scaled=True,
        show_unscaled=False,
        annotate='Scaled',
        show_counts='Scaled',
    )
    ax.set_xlabel('Estimated Probability')
    ax.set_ylabel('Actual Probability')
    plt.savefig("halu_reliability_diagram_code_scaled_scaled.png")
    plt.savefig("halu_reliability_diagram_code_scaled_scaled.svg")
    plt.show()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    result.reliability_plot(
        ax,
        show_quantiles='Unscaled',
        show_scaled=False,
        show_unscaled=True,
        annotate='Unscaled',
        show_counts='Unscaled',
    )
    # label the axis. x is Estimated Probability, y is Actual Probability
    ax.set_xlabel('Estimated Probability')
    ax.set_ylabel('Actual Probability')
    plt.savefig("halu_reliability_diagram_code_scaled_unscaled.png")
    plt.savefig("halu_reliability_diagram_code_scaled_unscaled.svg")
    plt.show()




def plot_reliability_diagram_halu(
    level: str, 
    save_path=None, 
    scaled=False
):
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
    fig, ax = plt.subplots(figsize=FIG_SIZE)
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


if __name__ == "__main__":
    run_quick_gen_probe()
    
        
