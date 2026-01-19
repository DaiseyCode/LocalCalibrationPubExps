from localizing.localizing_structs import TokenizedLocalization
from typing import TypeVar
from localizing.probe.agg_models.agg_config import FoldMode
from localizing.probe.probe_data_gather import localizations_to_grouped_vec_label_dataset
from localizing.probe.probe_data_gather import GroupedVecLabelDatasetMultiFold
from synthegrator.synthdatasets import CodeProblem
import hashlib
from localizing.probe.agg_models.agg_config import ProbeConfig


T_l = TypeVar('T_l', bound=TokenizedLocalization)


def localizations_to_folds(
    localizations: list[T_l],
    config: ProbeConfig,
) -> list[tuple[list[T_l], list[T_l], str]]:
    assert all(loc.check_passed_all_filters() for loc in localizations)
    assert isinstance(config, ProbeConfig)
    if config.fold_mode == FoldMode.cross_fold:
        fold_num = [
            problem_to_fold_hash(loc) % config.n_folds
            for loc in localizations
        ]
        def name_func(i: int) -> str:
            return f"Fold {i}"
    elif config.fold_mode == FoldMode.dataset_fold:
        dataset_names = set(
            loc.dataset_name for loc in localizations
        )
        datsetname_to_fold_index = {
            str(dataset_name): i 
            for i, dataset_name in enumerate(dataset_names)
        }
        print(datsetname_to_fold_index)
        fold_index_to_dataset_name = {
            v: k for k, v in datsetname_to_fold_index.items()
        }
        print(fold_index_to_dataset_name)
        fold_num = [
            datsetname_to_fold_index[str(loc.dataset_name)]
            for loc in localizations
        ]
        def name_func(i: int) -> str:
            return f"EvalDataset__{fold_index_to_dataset_name[i]}"
    num_folds = max(fold_num) + 1
    return [
        (
            [loc for loc, fold in zip(localizations, fold_num) if fold != i],
            [loc for loc, fold in zip(localizations, fold_num) if fold == i],
            name_func(i)
        )
        for i in range(num_folds)
    ]


def problem_to_fold_hash(loc: TokenizedLocalization) -> int:
    problem_id = loc.base_solve.problem.problem_id
    assert isinstance(problem_id, str)
    hash_bytes = hashlib.sha256(problem_id.encode()).digest()
    return int.from_bytes(hash_bytes[:8], 'big')