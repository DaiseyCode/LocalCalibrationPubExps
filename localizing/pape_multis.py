from collections import Counter
import dataclasses
from pprint import pprint
from typing import Callable, Literal
from calipy import experiment_results
from scipy.stats import gmean
import numpy as np
from localizing.cross_fold import localizations_to_folds
from localizing.filter_helpers import debug_str_filterables
from localizing.localizing_structs import MultiSamplingConfig, LocalizationList, MultisampleTokenEqualsLocalization, TokenizedLocalization
from localizing.predictions_repr import FoldResultsPreds, ProblemPredictions, calc_exp_results_per_fold, make_metrics_from_fold_results_multilevel
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN
from pape.configs import DEFAULT_LINE_AGGREGATOR_INTRINSIC
from localizing.multi_data_gathering import create_tokenized_localizations_from_scratch, create_multis_version_pretokenized, create_token_equals_localizations
from localizing.multi_data_gathering import MultiSampleMode
from synthegrator.synthdatasets import DatasetName
from lmwrapper.openai_wrapper import OpenAiModelNames

from localizing.probe.agg_models.agg_config import ProbeConfig
from localizing.probe.probe_data_gather import deserialize_localizations, get_or_serialize_tokenized_localizations, make_basic_serialize_key_args, make_basic_serialize_key_args_embedded, serialize_localizations
from localizing.probe.probe_data_gather import default_save_path


def pape_multis_equals_from_scratch(
    probe_config: ProbeConfig,
    multi_config: MultiSamplingConfig = MultiSamplingConfig(
        multi_temperature=1.0,
        target_num_samples=20,
        mode=MultiSampleMode.from_prompt,
    ),
    dev_mode: bool = False,
) -> LocalizationList[MultisampleTokenEqualsLocalization]:
    temp_weird_mode = multi_config.mode == MultiSampleMode.repair
    args = make_basic_serialize_key_args(probe_config, dev_mode)
    args = (
        *args,
        multi_config.multi_temperature,
        multi_config.target_num_samples,
        multi_config.mode,
    )
    args_str = "_".join(str(arg) for arg in args)
    fn = default_save_path / "multis" / (args_str + ".pkl.lz4")
    print("pape_multis_equals_from_scratch: fn", fn)
    #fn = default_save_path / "multis" / "HACK_RERUN.pkl.lz4"
    #print(fn)
    localizations = deserialize_localizations(fn)
    print("GOT LOCALIZATIONS", localizations is not None, type(localizations))
    if localizations is not None and not temp_weird_mode:
        print("pape_multis_equals_from_scratch: cache hit")
        return localizations
    
    # We need this to run faster so going to grab these values to see what we can salvage from them
    backup_localizations = None
    backup_localizations = deserialize_localizations(
        default_save_path / "multis_bk_before_repocod" / (args_str + ".pkl.lz4")
    )
    
    localizations = get_or_serialize_tokenized_localizations(
        config=probe_config,
        dev_mode=dev_mode,
    )
    print(len(list(localizations.iter_passed_filtered())))
    print("Name set", localizations.get_dataset_name_set())
    print("Passing name set", set((
            loc.dataset_name
            for loc in localizations.iter_passed_filtered()
        ))
    )
    ###
    if temp_weird_mode:
        # REPOCOD NEEDS ADJUSTMENT to use repair mode
        raise ValueError("RepoCod needs adjustment to use repair mode")
        for loc in localizations.iter_all():
            if loc.dataset_name == DatasetName.repocod:
                loc.annotate_failed_filter("Skip repocod for now")
    ###
    print(f"after tokenized {localizations}")
    localizations = create_multis_version_pretokenized(
        localizations,
        multi_config,
        hacky_backup_localizations=backup_localizations,
    )
    print("Passing name set after tokenization", set((
            loc.dataset_name
            for loc in localizations.iter_passed_filtered()
        ))
    )

    print(f"after add multis {localizations}")

    print("NUM passing before multis", len(list(localizations.iter_passed_filtered())))

    assert isinstance(localizations, LocalizationList)
    localizations = create_token_equals_localizations(
        localizations,
    )

    print("NUM passing after multis", len(list(localizations.iter_passed_filtered())))
    print("Passing name set after multis", set((
            loc.dataset_name
            for loc in localizations.iter_passed_filtered()
        ))
    )
    if not temp_weird_mode:
        serialize_localizations(localizations, fn)
    return localizations


def multis_to_fold_pred(
    localizations: LocalizationList[MultisampleTokenEqualsLocalization],
    n_folds,
    probe_config: ProbeConfig,
    multi_config: MultiSamplingConfig,
    line_aggregator: Literal["mean", "gmean", "min"] = "mean",
) -> FoldResultsPreds:
    all_train_preds = []
    all_test_preds = []
    fold_names = []
    #print("MULTIS TO FOLD")
    #print("Num localizations", len(list(localizations.iter_passed_filtered())))
    repocod_locs = [loc for loc in localizations.iter_all() if loc.dataset_name == DatasetName.repocod]
    #print("num repocod locs", len(repocod_locs))
    #print("REPOCODE Debug")
    #print(debug_str_filterables(repocod_locs))
    for train, test, fold_name in localizations_to_folds(list(localizations.iter_passed_filtered()), probe_config):
        all_train_preds.append([multiloc_to_pred(ex, line_aggregator=line_aggregator) for ex in train])
        all_test_preds.append([multiloc_to_pred(ex, line_aggregator=line_aggregator) for ex in test])
        fold_names.append(fold_name)
    return FoldResultsPreds(
        train_preds_each_fold=all_train_preds,
        test_preds_each_fold=all_test_preds,
        config=[probe_config, multi_config],
        fold_names=fold_names,
    )


def calc_pre_agg_pred(
    localization: MultisampleTokenEqualsLocalization,
    line_aggregator: Literal["mean", "gmean", "min"] = "mean",
) -> np.ndarray:
    """The preagg is applied to the keep tallies themselves within a given sample
    rather than at the token level"""
    keep_tallys = np.array(localization.keep_tallys)
    # Keep talies will be of dim (num_samples, num_tokens)
    agg = name_to_agg(line_aggregator)
    # We want to gather up the lines applying the agg to each line
    line_spans = localization.get_line_spans()
    #print("KEEP TALLYS")
    #print(keep_tallys)
    #print(keep_tallys.shape)
    #print("LINE SPANS")
    #print(line_spans)
    by_line = np.stack([
        agg(keep_tallys[:, start:end], axis=1)
        for start, end in localization.get_line_spans()
    ])
    #print("BY LINE")
    #print(by_line)
    means = np.mean(by_line, axis=1)
    #print("MEANS")
    #print(means)
    return means


def name_to_agg(
    name: Literal["mean", "gmean", "min", "pre_min", "pre_mean", "pre_gmean"]
) -> Callable[[np.ndarray], float]:
    if name == "mean":
        return np.mean
    elif name == "gmean":
        return gmean
    elif name == "min":
        return np.min
    raise ValueError(f"Invalid line_aggregator: {name}")


def loc_token_level_to_agg_pred(
    localization: TokenizedLocalization,
    token_level_pred: np.ndarray,
    line_aggregator: Literal["mean", "gmean", "min", "pre_min"] = DEFAULT_LINE_AGGREGATOR_INTRINSIC,
    problem_aggregator: Literal["mean", "gmean", "min"] = DEFAULT_LINE_AGGREGATOR_INTRINSIC,
) -> ProblemPredictions:
    if line_aggregator.startswith("pre_"):
        line_preds = calc_pre_agg_pred(
            localization, 
            line_aggregator[len("pre_"):],
        )
    else:
        line_agg = name_to_agg(line_aggregator)
        line_preds = np.array([
            line_agg(token_level_pred[span[0]:span[1]])
            for span in localization.get_line_spans()
        ])
    problem_agg = name_to_agg(problem_aggregator)

    return ProblemPredictions(
        problem_id=localization.base_solve.problem.problem_id,
        token_prediction_raw_probs=token_level_pred,
        line_prediction_raw_probs=line_preds,
        prob_level_pred=float(problem_agg(token_level_pred)),
        token_labels=np.array(localization.gt_base_token_keeps, dtype=np.int32),
        line_labels=np.array([
            int(np.min(localization.gt_base_token_keeps[span[0]:span[1]]))
            for span in localization.get_line_spans()
        ], dtype=np.int32),
        problem_label=int(localization.is_base_success),
    )


def loc_token_level_deagg_pred(
    localization: TokenizedLocalization,
    line_level_pred: np.ndarray,
    #line_aggregator: Literal["mean", "gmean", "min"] = DEFAULT_LINE_AGGREGATOR,
    problem_aggregator: Literal["mean", "gmean", "min"] = "gmean",
) -> ProblemPredictions:
    def name_to_agg(name: Literal["mean", "gmean", "min"]) -> Callable[[np.ndarray], float]:
        if name == "mean":
            return np.mean
        elif name == "gmean":
            return gmean
        elif name == "min":
            return np.min
        raise ValueError(f"Invalid line_aggregator: {name}")
    problem_agg = name_to_agg(problem_aggregator)
    # deaggregate the line to tokens
    token_level_pred = np.array([
        float(np.mean(line_level_pred))
    ] * len(localization.base_tokens))
    for span, line_pred in zip(localization.get_line_spans(), line_level_pred, strict=True):
        token_level_pred[span[0]:span[1]] = line_pred


    return ProblemPredictions(
        problem_id=localization.base_solve.problem.problem_id,
        token_prediction_raw_probs=token_level_pred,
        line_prediction_raw_probs=line_level_pred,
        prob_level_pred=float(problem_agg(line_level_pred)),
        token_labels=np.array(localization.gt_base_token_keeps, dtype=np.int32),
        line_labels=np.array([
            int(np.min(localization.gt_base_token_keeps[span[0]:span[1]]))
            for span in localization.get_line_spans()
        ], dtype=np.int32),
        problem_label=int(localization.is_base_success),
    )



def multiloc_to_pred(
    localization: MultisampleTokenEqualsLocalization,
    line_aggregator: Literal["mean", "gmean", "min"] = "mean",
) -> ProblemPredictions:
    return loc_token_level_to_agg_pred(
        localization,
        np.array(localization.estimated_keeps),
        line_aggregator=line_aggregator,
    )


def make_fold_preds_for_multis(
    probe_config: ProbeConfig,
    multi_config: MultiSamplingConfig,
    line_aggregator=DEFAULT_LINE_AGGREGATOR_INTRINSIC,
    dev_mode: bool = False,
) -> FoldResultsPreds:
    print("make_fold_preds_for_multis")
    locs = pape_multis_equals_from_scratch(
        probe_config=probe_config,
        multi_config=multi_config,
        dev_mode=dev_mode,
    )
    print("make_fold_preds_for_multis: got locs")
    results = multis_to_fold_pred(
        locs, 
        n_folds=probe_config.n_folds, 
        probe_config=probe_config, 
        multi_config=multi_config,
        line_aggregator=line_aggregator,
    )
    return results


if __name__ == "__main__":
    dev_mode = True
    probe_config = dataclasses.replace(
        BASE_PAPER_CONFIG,
    )
    multi_config = MultiSamplingConfig(
        mode=MultiSampleMode.repair,
        multi_temperature=0.8,
        target_num_samples=5,
    )
    locs = pape_multis_equals_from_scratch(
        probe_config=probe_config,
        multi_config=multi_config,
        dev_mode=dev_mode,
    )
    for line_aggregator in ["mean", "gmean", "min"]:
        results = multis_to_fold_pred(
            locs, 
            n_folds=probe_config.n_folds, 
            probe_config=probe_config, 
            multi_config=multi_config,
            line_aggregator=line_aggregator,
        )
        print(f"Line aggregator: {line_aggregator}")
        pprint(make_metrics_from_fold_results_multilevel(results))
    exit()
    
    print("TEMPERATURE 1.0")
    locs = pape_multis_equals_from_scratch(
        probe_config=dataclasses.replace(
            BASE_PAPER_CONFIG,
        ),
        multi_config=MultiSamplingConfig(
            mode=MultiSampleMode.from_prompt,
            multi_temperature=1.0,
            target_num_samples=5,
        ),
        dev_mode=True,
    )
    print(locs)
    one_loc = next(locs.iter_all())
    print("estimated_keeps")
    print(one_loc.estimated_keeps)

    exit()
    print("TEMPERATURE 0.8 QWEN")
    locs = pape_multis_equals_from_scratch(
        probe_config=dataclasses.replace(
            BASE_PAPER_CONFIG_QWEN,
        ),
        multi_config=MultiSamplingConfig(
            mode=MultiSampleMode.from_prompt,
            multi_temperature=0.8,
            target_num_samples=5,
        ),
        dev_mode=True,
    )
    print(locs)
    one_loc = next(locs.iter_all())
    print("estimated_keeps")
    print(one_loc.estimated_keeps)