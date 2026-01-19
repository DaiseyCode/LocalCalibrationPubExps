import dataclasses
import math
import pandas as pd
from pprint import pprint
import hashlib
from calipy.experiment_results import BinStrategy, ExperimentResults
import numpy as np
import torch
from localizing.localizing_structs import TokenizedLocalization
from localizing.predictions_repr import FoldResultsPreds, deserialize_fold_results_preds, flatten_dict, get_combined_vector_from_problem_preds, make_metrics_from_fold_results_multilevel, serialize_fold_results_preds, calc_exp_results_per_fold
from localizing.probe.agg_models.agg_config import FoldMode, ProbeAggregator, ProbeConfig, ProbeLoss
from localizing.probe.embedding_styles import EmbeddingStyle
from localizing.probe.probe_data_gather import EmbeddedTokenization, GroupedVecLabelDataset, get_or_serialize_localizations_embedded, make_basic_serialize_key_args_embedded, default_save_path
from localizing.probe.probe_models_agg import AggLineProbe, AggLineTrainer
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN, gen_model_name_qwen
from localizing.cross_fold import localizations_to_folds
from pape.texrendering import TABLE_DIR, generate_latex_table


def get_fold_results_preds_probe(
    config: ProbeConfig,
    dev_mode = True,
    preloaded_embedded_locs: list[EmbeddedTokenization] = None,
) -> tuple[FoldResultsPreds, list[EmbeddedTokenization]]:
    """
    Creates results for a probe model. Will also return the embedded
    localizations in case they are reused in memory for different configs.
    """
    args = make_basic_serialize_key_args_embedded(config, dev_mode)
    args = (
        *args, 
        config.n_folds,
        config.learning_rate,
        config.num_epochs,
        config.weight_decay,
        config.batch_size,
        config.hidden_dim,
        config.dropout_rate,
        config.token_weight,
        config.line_weight,
    )
    if config.agg_style == ProbeAggregator.MAX_POOL:
        # this is the legacy. Don't modify the args
        pass
    else:
        args = (*args, config.agg_style.value)
    if config.loss_style != ProbeLoss.BCE:
        args = (*args, config.loss_style.value)
    if config.fold_mode != FoldMode.cross_fold:
        args = (*args, config.fold_mode.value)
    arg_combo = "_".join(map(str, args))
    hash_combo = hashlib.sha256(arg_combo.encode()).hexdigest()[:16]

    fn = default_save_path / "fold_results_preds" / (hash_combo + ".pkl.lz4")
    print("FOLD RESULTS PRED NEEDED")
    print(fn)
    if fn.exists():
        return deserialize_fold_results_preds(fn), None
    if preloaded_embedded_locs is None:
        locs = get_or_serialize_localizations_embedded(
            config=config,
            dev_mode=dev_mode,
            save_serialize_embeddings=dev_mode,
        )
        locs = list(locs.iter_passed_filtered())
    else:
        locs = preloaded_embedded_locs
    folds = localizations_to_folds(
        locs, 
        config=config,
    )
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = AggLineTrainer(
        config=config,
        device=device,
        dtype=dtype,
    )
    all_train_preds = []
    all_test_preds = []
    fold_names = []
    for i, (train, test, fold_name) in enumerate(folds):
        print(f"Fold {fold_name}")
        print(f"Train: {len(train)}")
        print(f"Test: {len(test)}")
        train_dataset = GroupedVecLabelDataset(
            localizations=train
        )
        input_dim = find_input_dim(train_dataset)
        test_dataset = GroupedVecLabelDataset(
            localizations=test
        )
        model = AggLineProbe(
            input_dim=input_dim,
            config=config,
            dtype=dtype
        )
        model, metrics = trainer.train_and_evaluate(
            model, 
            train_dataset, 
            test_dataset, 
            batch_size=config.batch_size,
        )
        all_test_preds.append(
            trainer.make_problem_predictions(
                model,
                test,
            )
        )
        all_train_preds.append(
            trainer.make_problem_predictions(
                model,
                train,
            )
        )
        fold_names.append(fold_name)
        #print(preds[0].token_prediction_raw_probs)
        #print(preds[0].line_prediction_raw_probs)
        #print(preds[0].prob_level_pred)
        #raw = ExperimentResults(
        #    predicted_probabilities=get_combined_vector_from_problem_preds(
        #        preds, "line", use_gt=False
        #    ),
        #    true_labels=get_combined_vector_from_problem_preds(
        #        preds, "line", use_gt=True
        #    ),
        #    bin_strategy=BinStrategy.Uniform,
        #    num_bins=10,
        #)
        #print("ECE")
        #print(raw.ece)
        #break
    fold_results_preds = FoldResultsPreds(
        config=config,
        test_preds_each_fold=all_test_preds,
        train_preds_each_fold=all_train_preds,
        fold_names=fold_names,
    )
    serialize_fold_results_preds(fold_results_preds, fn)
    return fold_results_preds, locs


def run_all_da_probs(base: ProbeConfig):
    dev_mode = False
    vals = []
    for embed_lm_name in [
        "Qwen/Qwen2.5-Coder-0.5B",
        gen_model_name_qwen,
    ]:
        for embedding_style in [
            EmbeddingStyle.MIDDLE_LAYER,
            EmbeddingStyle.THREE_QUARTERS_LAYER,
            EmbeddingStyle.LAST_LAYER,
            EmbeddingStyle.SHIFTED_THREE_QUARTERS,
            EmbeddingStyle.SHIFTED_THREE_QUARTERS_AND_MIDDLE,
        ]:
            cached_locs = None # Used to keep embeddings in memory for different configs
            # All of the below params are tryining settings that can reuse embeddings
            for hidden_dim in [
                32,
                64,
            ]:
                for agg_style in [
                    ProbeAggregator.MAX_POOL,
                    ProbeAggregator.AVG_POOL,
                    ProbeAggregator.MHA_POOL_4_HEADS,
                ]:
                    for loss_style in [
                        ProbeLoss.BCE,
                        ProbeLoss.FL2,
                        ProbeLoss.BRIER,
                    ]:
                        print(embed_lm_name, agg_style, embedding_style, loss_style)
                        config = dataclasses.replace(
                            base, 
                            embedding_style=embedding_style,
                            embed_lm_name=embed_lm_name,
                            agg_style=agg_style,
                            num_epochs=base.num_epochs if not dev_mode else base.num_epochs // 3,
                            hidden_dim=hidden_dim,
                            loss_style=loss_style,
                        )
                        fold_results, cached_locs = get_fold_results_preds_probe(
                            config, 
                            dev_mode=dev_mode,
                            preloaded_embedded_locs=cached_locs,
                        )
                        print("Line AUC")
                        print(
                            np.mean([
                                r.roc_auc 
                                for r in calc_exp_results_per_fold(
                                    fold_results, level="line")
                            ])
                        )
                        vals.append({
                            "embed\\_lm\\_name": embed_lm_name.replace("_", "\\_"),
                            "embedding\\_style": embedding_style.value.replace("_", "\\_"),
                            "hidden\\_dim": str(hidden_dim),
                            "agg\\_style": agg_style.value.replace("_", "\\_"),
                            "loss\\_style": loss_style.value.replace("_", "\\_"),
                            **make_metrics_from_fold_results_multilevel(
                                fold_results, include_problem_level=True),
                        })
    pprint(vals)
    table = generate_latex_table(
        vals,
        caption_content=f"Large Probe Comparison {config.tidy_gen_name_human}",
        label=f"tab:large_probe_compare_{config.tidy_gen_name_code}",
        caption_at_top=False,
        resize_to_fit=True,
        full_width=True,
    )
    (TABLE_DIR / f"large_probe_compare_{config.tidy_gen_name_code}.tex").write_text(table)
    pd.DataFrame([flatten_dict(v) for v in vals]).to_csv(TABLE_DIR / f"large_probe_compare_{config.tidy_gen_name_code}.csv", index=False)
    pd.DataFrame([flatten_dict(v) for v in vals]).to_parquet(TABLE_DIR / f"large_probe_compare_{config.tidy_gen_name_code}.parquet", index=False)


def main():
    config = BASE_PAPER_CONFIG
    dev_mode = True
    config, _ = dataclasses.replace(
        config, 
        num_epochs=3,
        embedding_style=EmbeddingStyle.SHIFTED_THREE_QUARTERS_AND_MIDDLE,
        embed_lm_name=gen_model_name_qwen,
    )
    fold_results = get_fold_results_preds_probe(config, dev_mode=dev_mode)
    print(type(fold_results))
    print("Line AUC")
    print(
        np.mean([
            r.ece 
            for r in calc_exp_results_per_fold(fold_results, level="line")
        ])
    )


def find_input_dim(dataset: GroupedVecLabelDataset) -> int:
    for loc in dataset.localizations:
        if loc.base_tokens_embedding is not None:
            return loc.base_tokens_embedding.shape[1]
    raise ValueError("No embeddings found in dataset")


def explore_all_da_probe():
    df = pd.read_parquet(TABLE_DIR / f"large_probe_compare_{BASE_PAPER_CONFIG.tidy_gen_name_code}.parquet")
    df_qwen = pd.read_parquet(TABLE_DIR / f"large_probe_compare_{BASE_PAPER_CONFIG_QWEN.tidy_gen_name_code}.parquet")
    
    # Add gen_model column to track which model each row came from
    df['gen_model'] = BASE_PAPER_CONFIG.tidy_gen_name_code
    df_qwen['gen_model'] = BASE_PAPER_CONFIG_QWEN.tidy_gen_name_code
    
    # Concatenate the dataframes
    df = pd.concat([df, df_qwen], ignore_index=True)

    
    # Clean up column names by removing Unicode arrows and backslashes
    df.columns = df.columns.str.replace(r' \$\\uparrow\$', '', regex=True)
    df.columns = df.columns.str.replace(r' \$\\downarrow\$', '', regex=True)
    df.columns = df.columns.str.replace(r'\\', '', regex=True)
    
    print("Cleaned columns:")
    print(df.columns.tolist())
    """['embed_lm_name', 'embedding_style', 'hidden_dim', 'agg_style', 'loss_style', 'Line-Level__Unscaled__BSS', 'Line-Level__Unscaled__ECE', 'Line-Level__Unscaled__AUC', 'Line-Level__Scaled__BSS', 'Line-Level__Scaled__ECE', 'Token-Level__Unscaled__BSS', 'Token-Level__Unscaled__ECE', 'Token-Level__Unscaled__AUC', 'Token-Level__Scaled__BSS', 'Token-Level__Scaled__ECE', 'Problem-Level__Unscaled__BSS', 'Problem-Level__Unscaled__ECE', 'Problem-Level__Unscaled__AUC', 'Problem-Level__Scaled__BSS', 'Problem-Level__Scaled__ECE']"""
    
    # Step 1: Create composite metrics - BSS and ECE
    bss_unscaled_columns = [
        'Token-Level__Unscaled__BSS',
        'Line-Level__Unscaled__BSS',
        'Problem-Level__Unscaled__BSS'
    ]
    
    bss_scaled_columns = [
        'Token-Level__Scaled__BSS', 
        'Line-Level__Scaled__BSS',
        'Problem-Level__Scaled__BSS'
    ]
    
    ece_unscaled_columns = [
        'Token-Level__Unscaled__ECE',
        'Line-Level__Unscaled__ECE', 
        'Problem-Level__Unscaled__ECE'
    ]
    
    all_columns = bss_unscaled_columns + bss_scaled_columns + ece_unscaled_columns
    
    # Convert all columns to numeric
    for col in all_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create composite metrics
    df['mean_bss_unscaled'] = df[bss_unscaled_columns].mean(axis=1)
    df['mean_bss_scaled'] = df[bss_scaled_columns].mean(axis=1)
    df['mean_bss_all'] = df[bss_unscaled_columns + bss_scaled_columns].mean(axis=1)
    df['mean_ece_unscaled'] = df[ece_unscaled_columns].mean(axis=1)  # Lower is better
    
    print(f"\nComposite metrics created:")
    print(f"  Unscaled BSS (3 levels): {df['mean_bss_unscaled'].min():.4f} to {df['mean_bss_unscaled'].max():.4f}, avg: {df['mean_bss_unscaled'].mean():.4f}")
    print(f"  Scaled BSS (3 levels):   {df['mean_bss_scaled'].min():.4f} to {df['mean_bss_scaled'].max():.4f}, avg: {df['mean_bss_scaled'].mean():.4f}")
    print(f"  All BSS (6 levels):      {df['mean_bss_all'].min():.4f} to {df['mean_bss_all'].max():.4f}, avg: {df['mean_bss_all'].mean():.4f}")
    print(f"  Unscaled ECE (3 levels): {df['mean_ece_unscaled'].min():.4f} to {df['mean_ece_unscaled'].max():.4f}, avg: {df['mean_ece_unscaled'].mean():.4f} (lower=better)")
    
    # Check correlations
    bss_ece_corr = df['mean_bss_unscaled'].corr(df['mean_ece_unscaled'])
    bss_scaled_unscaled_corr = df['mean_bss_unscaled'].corr(df['mean_bss_scaled'])
    print(f"\nCorrelations:")
    print(f"  BSS Scaled/Unscaled:  {bss_scaled_unscaled_corr:.4f}")
    print(f"  BSS/ECE (unscaled):   {bss_ece_corr:.4f} (negative = good BSS → good calibration)")
    
    # Quick check for missing values
    for metric in ['mean_bss_unscaled', 'mean_bss_scaled', 'mean_bss_all', 'mean_ece_unscaled']:
        if df[metric].isna().any():
            print(f"Warning: {df[metric].isna().sum()} rows have missing {metric} values")
    
    print("\nDataset shape:", df.shape)
    
    # Step 2: Factor analysis for all metrics including ECE
    factors = ['gen_model', 'embed_lm_name', 'embedding_style', 'hidden_dim', 'agg_style', 'loss_style']
    metrics = ['mean_bss_unscaled', 'mean_bss_scaled', 'mean_bss_all', 'mean_ece_unscaled']
    metric_names = ['Unscaled BSS', 'Scaled BSS', 'Combined BSS', 'Unscaled ECE']
    metric_directions = ['higher', 'higher', 'higher', 'lower']  # Track direction for sorting
    
    all_results = {}
    
    for metric, metric_name, direction in zip(metrics, metric_names, metric_directions):
        print("\n" + "="*80)
        print(f"ANALYSIS FOR {metric_name.upper()}")
        print("="*80)
        
        print(f"\n{metric_name} Summary Stats by Factor Level:")
        print("-" * 60)
        
        factor_summaries = {}
        
        for factor in factors:
            grouped = df.groupby(factor)[metric]
            summary = grouped.agg(['count', 'mean', 'median', 'std']).round(4)
            
            # Sort by median - ascending for ECE (lower better), descending for BSS (higher better)
            ascending = (direction == 'lower')
            summary = summary.sort_values('median', ascending=ascending)
            factor_summaries[factor] = summary
            
            print(f"\n{factor}:")
            print(summary.head(3))  # Show top 3 for brevity
            
            best_level = summary.index[0]
            best_median = summary.loc[best_level, 'median']
            direction_text = "lowest" if direction == 'lower' else "highest"
            print(f"  → Best: {best_level} ({direction_text} median: {best_median:.4f})")
        
        # Calculate eta-squared for this metric
        print(f"\n{metric_name} - Factor Importance (η²):")
        print("-" * 40)
        
        eta_squared_results = {}
        overall_mean = df[metric].mean()
        total_ss = ((df[metric] - overall_mean) ** 2).sum()
        
        for factor in factors:
            grouped = df.groupby(factor)[metric]
            group_means = grouped.mean()
            group_counts = grouped.count()
            
            between_ss = ((group_means - overall_mean) ** 2 * group_counts).sum()
            eta_squared = between_ss / total_ss
            eta_squared_results[factor] = eta_squared
            
            if eta_squared < 0.01:
                effect_size = "negligible"
            elif eta_squared < 0.06:
                effect_size = "small"  
            elif eta_squared < 0.14:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            print(f"{factor:20s}: η² = {eta_squared:.4f} ({effect_size})")
        
        # Store results for comparison
        all_results[metric] = {
            'factor_summaries': factor_summaries,
            'eta_squared': eta_squared_results,
            'direction': direction
        }
    
    # Step 3: Compare factor importance across all metrics
    print("\n" + "="*80)
    print("FACTOR IMPORTANCE COMPARISON ACROSS ALL METRICS")
    print("="*80)
    
    print(f"\n{'Factor':<20} {'Unscaled BSS':<13} {'Scaled BSS':<12} {'Combined BSS':<13} {'Unscaled ECE':<13}")
    print("-" * 80)
    
    for factor in factors:
        unscaled_bss_eta = all_results['mean_bss_unscaled']['eta_squared'][factor]
        scaled_bss_eta = all_results['mean_bss_scaled']['eta_squared'][factor]
        combined_bss_eta = all_results['mean_bss_all']['eta_squared'][factor]
        ece_eta = all_results['mean_ece_unscaled']['eta_squared'][factor]
        
        print(f"{factor:<20} {unscaled_bss_eta:<13.4f} {scaled_bss_eta:<12.4f} {combined_bss_eta:<13.4f} {ece_eta:<13.4f}")
    
    # Performance vs Calibration comparison
    print(f"\nPERFORMANCE vs CALIBRATION FACTOR IMPORTANCE:")
    print(f"{'Factor':<20} {'Performance':<12} {'Calibration':<12} {'Alignment'}")
    print("-" * 60)
    
    for factor in factors:
        # Use combined BSS as performance measure
        perf_eta = all_results['mean_bss_all']['eta_squared'][factor]
        cal_eta = all_results['mean_ece_unscaled']['eta_squared'][factor]
        
        # Determine alignment
        if perf_eta >= 0.06 and cal_eta >= 0.06:
            alignment = "Both important 🔥"
        elif perf_eta >= 0.06 and cal_eta < 0.01:
            alignment = "Performance only 📈"
        elif perf_eta < 0.01 and cal_eta >= 0.06:
            alignment = "Calibration only 📊"
        elif abs(perf_eta - cal_eta) < 0.02:
            alignment = "Similar effect 🔄"
        else:
            alignment = "Mixed effects ➖"
            
        print(f"{factor:<20} {perf_eta:<12.4f} {cal_eta:<12.4f} {alignment}")
    
    # Check for performance-calibration trade-offs
    print(f"\nBEST FACTOR LEVELS - Performance vs Calibration:")
    print("-" * 60)
    
    for factor in factors:
        bss_best = all_results['mean_bss_all']['factor_summaries'][factor].index[0]
        ece_best = all_results['mean_ece_unscaled']['factor_summaries'][factor].index[0]
        
        if bss_best == ece_best:
            agreement = "✅ Same choice"
        else:
            agreement = "⚠️ Different choices"
        
        print(f"{factor:20s}: BSS→{bss_best}, ECE→{ece_best} ({agreement})")
    
    # Step 4: Best practices for combined metric (most comprehensive)
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATION - Based on Combined BSS")
    print("="*80)
    
    recommended_config = {}
    combined_summaries = all_results['mean_bss_all']['factor_summaries']
    combined_eta = all_results['mean_bss_all']['eta_squared']
    
    for factor in factors:
        best_level = combined_summaries[factor].index[0]
        best_median = combined_summaries[factor].loc[best_level, 'median']
        eta_sq = combined_eta[factor]
        
        recommended_config[factor] = best_level
        
        importance = "🔥" if eta_sq >= 0.06 else "📊" if eta_sq >= 0.01 else "➖"
        print(f"{importance} {factor:20s}: {best_level}")
        print(f"   ↳ Median Combined BSS: {best_median:.4f}, Importance: η² = {eta_sq:.4f}")
        print()
    
    # Test the recommended configuration
    config_mask = True
    for factor, value in recommended_config.items():
        config_mask = config_mask & (df[factor] == value)
    
    if config_mask.sum() > 0:
        for metric, metric_name in zip(metrics, metric_names):
            rec_perf = df[config_mask][metric].mean()
            overall_avg = df[metric].mean()
            improvement = ((rec_perf - overall_avg) / overall_avg) * 100
            print(f"🎯 {metric_name}: {rec_perf:.4f} (vs avg: {overall_avg:.4f}, +{improvement:.1f}%)")
    else:
        print("⚠️  Exact recommended configuration not found in data")
    
    # Generate eta-squared table for paper
    print("\n" + "="*80)
    print("GENERATING ETA-SQUARED TABLE FOR PAPER")
    print("="*80)
    
    # Create table data
    eta_table_rows = []
    
    # Order factors by average importance (BSS + ECE)
    factor_avg_importance = {}
    for factor in factors:
        bss_eta = all_results['mean_bss_all']['eta_squared'][factor]
        ece_eta = all_results['mean_ece_unscaled']['eta_squared'][factor]
        factor_avg_importance[factor] = (bss_eta + ece_eta) / 2
    
    sorted_factors = sorted(factor_avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    for factor, avg_importance in sorted_factors:
        bss_eta = all_results['mean_bss_all']['eta_squared'][factor]
        ece_eta = all_results['mean_ece_unscaled']['eta_squared'][factor]
        
        # Bold if η² >= 0.06 (medium effect size)
        bss_str = f"\\textbf{{{bss_eta:.3f}}}" if bss_eta >= 0.06 else f"{bss_eta:.3f}"
        ece_str = f"\\textbf{{{ece_eta:.3f}}}" if ece_eta >= 0.06 else f"{ece_eta:.3f}"
        
        # Clean factor name for display
        clean_factor_name = factor.replace('_', ' ').title().replace('Lm', 'LM')
        clean_factor_name = {
            "Embed LM Name": "Embedding LM",
            "Embedding Style": "Aggregator",
        }.get(clean_factor_name, clean_factor_name)
        
        eta_table_rows.append({
            'Factor': clean_factor_name,
            r'Mean BSS $\eta^2$': bss_str,
            r'Mean ECE $\eta^2$': ece_str
        })
    
    # Generate LaTeX table
    table_latex = generate_latex_table(
        eta_table_rows,
        caption_content=(
            "A comparison of probe model design choices. Factor importance analysis showing proportion of variance "
            "explained ($\\eta^2$) for Mean BSS and Mean ECE metrics. The mean BSS is averaging 6 values, the "
            "token/line/problem level BSS for both scaled and unscaled. The Mean ECE is the mean of 3 values, the "
            "unscaled ECE (we exclude scaled ECE as it almost always is near 0). Higher $\\eta^2$ "
            "indicates greater factor importance."
        ),
        label="tab:factor_importance_eta_squared",
        caption_at_top=False,
        resize_to_fit=False,
        full_width=False,
    )
    
    # Save table
    table_file = TABLE_DIR / "factor_importance_eta_squared.tex"
    table_file.write_text(table_latex)
    print(f"✅ Eta-squared table saved to: {table_file}")
    
    # Also print for review
    print("\nTable preview:")
    for row in eta_table_rows:
        print(row['Factor'].ljust(20) + " | " + row[r'Mean BSS $\eta^2$'].rjust(15) + " | " + row[r'Mean ECE $\eta^2$'].rjust(15))
    
    return df, all_results  # Return for potential further analysis


def create_factor_breakdown_figure(df, all_results, save_path=None):
    """
    Create a tall stacked figure showing factor level breakdowns for BSS and ECE.
    Each subplot shows factor levels with side-by-side boxplots for Mean BSS and Mean ECE.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    factors = ['gen_model', 'embed_lm_name', 'embedding_style', 'hidden_dim', 'agg_style', 'loss_style']
    
    # Order factors by average importance (same as table)
    factor_avg_importance = {}
    for factor in factors:
        bss_eta = all_results['mean_bss_all']['eta_squared'][factor]
        ece_eta = all_results['mean_ece_unscaled']['eta_squared'][factor]
        factor_avg_importance[factor] = (bss_eta + ece_eta) / 2
    
    sorted_factors = sorted(factor_avg_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_factor_names = [f[0] for f in sorted_factors]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(sorted_factor_names), 1, figsize=(10, 2.5 * len(sorted_factor_names)))
    if len(sorted_factor_names) == 1:
        axes = [axes]
    
    plt.style.use('default')
    
    for i, factor in enumerate(sorted_factor_names):
        ax = axes[i]
        
        # Get factor levels sorted by median BSS performance
        factor_summary = all_results['mean_bss_all']['factor_summaries'][factor]
        sorted_levels = factor_summary.index.tolist()
        
        # Prepare data for plotting
        bss_data = []
        ece_data = []
        level_labels = []
        
        for level in sorted_levels:
            level_mask = df[factor] == level
            bss_values = df[level_mask]['mean_bss_all'].values
            ece_values = df[level_mask]['mean_ece_unscaled'].values
            
            bss_data.append(bss_values)
            ece_data.append(ece_values)
            
            # Clean up level names for display
            level = str(level)
            level = level.replace('_', ' ').replace('\\', '')
            level = {
                "last layer": "LAST",
                "middle layer": "MIDDLE",
                "shifted three quarters": "Shifted 3/4",
                "three quarters layer": "3/4",
                "shifted three quarters and middle": "COMBINED",
                "Qwen/Qwen2.5-Coder-0.5B": "Qwen2.5Coder 0.5B",
                "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5CoderInst 7B",
                "fl2": "FL2",
                "bce": "BCE",
                "brier": "Brier",
                "mha pool 4 heads": "4Head Attention",
            }.get(level, level)
            if len(level) > 30:  # Truncate very long names
                level = level[:27] + '...'
            level_labels.append(level)
        
        # Create horizontal boxplots with levels on y-axis
        y_positions = np.arange(len(sorted_levels))
        
        # Plot horizontal boxplots - BSS on left half, ECE on right half
        bp1 = ax.boxplot(
            bss_data, positions=y_positions, widths=0.6, 
            patch_artist=True, vert=False, 
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            # NOTE: going need to adjust wis if split up the
            # individual folds.
            whis=(5, 95),
            showfliers=False, 
        )
        
        # Get x-range for BSS and set ECE to right half
        round_factor = 10
        bss_min = min([min(data) for data in bss_data])
        bss_min = math.floor(bss_min * round_factor) / round_factor
        bss_max = max([max(data) for data in bss_data])
        bss_max = math.ceil(bss_max * round_factor) / round_factor
        bss_range = bss_max - bss_min
        
        ece_min = min([min(data) for data in ece_data])
        ece_min = math.floor(ece_min * round_factor) / round_factor
        ece_max = max([max(data) for data in ece_data])
        ece_max = math.ceil(ece_max * round_factor) / round_factor
        ece_range = ece_max - ece_min
        
        # Normalize ECE data to right half of plot
        ece_data_normalized = []
        for data in ece_data:
            # Normalize to 0-1, then scale to right half starting after BSS range
            normalized = (np.array(data) - ece_min) / ece_range if ece_range > 0 else np.array(data) * 0
            scaled = normalized * bss_range * 0.8 + bss_max + bss_range * 0.1
            ece_data_normalized.append(scaled)
        
        bp2 = ax.boxplot(
            ece_data_normalized, positions=y_positions, widths=0.6,
            patch_artist=True, vert=False,
            boxprops=dict(facecolor='lightcoral', alpha=0.7)
        )
        
        # Set y-axis with factor levels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(level_labels)
        
        # Create dual x-axis labels
        # Left side for BSS
        num_ticks = 5
        bss_ticks = np.linspace(bss_min, bss_max, num_ticks)
        # Right side for ECE (using original scale)
        ece_ticks_norm = np.linspace(0, 1, num_ticks)
        ece_ticks_pos = ece_ticks_norm * bss_range * 0.8 + bss_max + bss_range * 0.1
        ece_ticks_labels = np.linspace(ece_min, ece_max, num_ticks)
        
        # Combine ticks
        all_ticks = np.concatenate([bss_ticks, ece_ticks_pos])
        all_labels = [f'{x:.3f}' for x in bss_ticks] + [f'{x:.3f}' for x in ece_ticks_labels]
        
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        
        # Add vertical line to separate BSS and ECE sections
        ax.axvline(x=bss_max + bss_range * 0.05, color='gray', linestyle='--', alpha=0.5)
        
        # Add section labels
        bss_center = (bss_min + bss_max) / 2
        ece_center = bss_max + bss_range * 0.5
        ax.text(bss_center, len(sorted_levels) - 0.5, 
                'Mean BSS (All Level) →', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
        ax.text(ece_center, len(sorted_levels) - 0.5, 
                'Mean ECE (All Level Unscaled) ←', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
        
        # Add factor name and eta-squared values
        bss_eta = all_results['mean_bss_all']['eta_squared'][factor]
        ece_eta = all_results['mean_ece_unscaled']['eta_squared'][factor]
        
        clean_factor_name = factor.replace('_', ' ').title().replace('Lm', 'LM')
        title = f"{clean_factor_name} (BSS η²={bss_eta:.3f}, ECE η²={ece_eta:.3f})"
        ax.set_title(title, fontsize=12, pad=20)
        
        # Add grid for readability
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Save as PDF
        pdf_path = save_path.with_suffix('.pdf') if hasattr(save_path, 'with_suffix') else str(save_path).replace('.pdf', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✅ Factor breakdown figure saved as PDF: {pdf_path}")
        
        # Save as PNG
        png_path = save_path.with_suffix('.png') if hasattr(save_path, 'with_suffix') else str(save_path).replace('.pdf', '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        print(f"✅ Factor breakdown figure saved as PNG: {png_path}")
    
    return fig


if __name__ == "__main__":
    df, all_results = explore_all_da_probe()
    
    # Create the factor breakdown figure
    fig_path = TABLE_DIR / "factor_breakdown_figure.pdf"
    create_factor_breakdown_figure(df, all_results, save_path=fig_path)
    exit()
    for base in [
        BASE_PAPER_CONFIG,
        BASE_PAPER_CONFIG_QWEN,
    ]:
        run_all_da_probs(base)
    #main()