from collections import defaultdict, Counter
import scipy.stats
from localizing.probe.probe_data_gather import default_save_path, deserialize_localizations
from pprint import pprint
from localizing.cross_fold import localizations_to_folds
import numpy as np
from localizing.localizing_structs import TokenizedLocalization
import re
from tqdm import tqdm
import random
import ast
from lmwrapper.structs import LmPrompt, LmPrediction
from lmwrapper.abstract_predictor import LmPredictor
from localizing.pape_multis import loc_token_level_deagg_pred
from localizing.predictions_repr import FoldResultsPreds, deserialize_fold_results_preds, make_metrics_from_fold_results_multilevel, serialize_fold_results_preds
from localizing.probe.agg_models.agg_config import FoldMode, ProbeConfig
from localizing.probe.probe_data_gather import get_or_serialize_tokenized_localizations, make_basic_serialize_key_args
from pape.configs import BASE_PAPER_CONFIG, BASE_PAPER_CONFIG_QWEN
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from synthegrator.lang_specs.lang_spec_python import PythonLangSpec


def pull_last_md_block_from_completion(
    text: str,
) -> str | None:
    """
    Pulls the last markdown block from a completion.
    Markdown blocks are ```...```.
    """

    # Find the end of the last code block (working backwards)
    end = text.rfind("```")
    if end == -1:
        return None

    # Find the start of this code block by working backwards from the end
    start = text.rfind("```", 0, end)
    if start == -1:
        return None

    # Find the end of the line after the starting ```
    # This is where the markdown language name is usually located
    end_of_language_name = text.find("\n", start + 3)
    if end_of_language_name == -1:
        # if there's no newline after the starting ```, then it's not a valid block
        return None

    # Make sure the end we found is actually after the language name line
    if end <= end_of_language_name:
        # if the closing ``` is before the end of the language name line, it's not a valid block
        return None

    # Return the text inside the block, excluding the markdown language name
    return text[end_of_language_name + 1 : end]


def estimate_line_level_base_rate(
    locs: list[TokenizedLocalization],
):
    all_labels = []
    for loc in locs:
        keeps = loc.gt_base_token_keeps
        labels = [
            np.min(keeps[i:j])
            for i, j in loc.get_line_spans()
        ]
        all_labels.extend(labels)
    return np.mean(all_labels)


def get_direct_prompt_results(
    config: ProbeConfig,
    dev_mode: bool,
    cache: bool = True,
) -> FoldResultsPreds:
    args = make_basic_serialize_key_args(config, dev_mode)
    if config.fold_mode != FoldMode.cross_fold:
        args = (*args, config.fold_mode.value)
    print("ARGS", args)
    args_str = "_".join(str(arg) for arg in args)
    print("ARGS STR", args_str)
    fn = default_save_path / "direct_prompt_results" / (args_str + ".pkl.lz4")
    if cache:
        results = deserialize_fold_results_preds(fn)
        if results is not None:
            return results

    locs = get_or_serialize_tokenized_localizations(
        config, dev_mode=dev_mode)

    base_rate = estimate_line_level_base_rate(list(locs.iter_passed_filtered()))

    locs = list(locs.iter_passed_filtered())
    lm = get_open_ai_lm(OpenAiModelNames.gpt_4o)
    rng = random.Random(42)
    rng.shuffle(locs)
    locs = locs[:min(750, len(locs))]
    problem_id_estimate = {}
    num_parse_success = 0
    num_parse_fail = 0
    num_parse_fail_reasons = Counter()
    num_respse_tokens = []
    for loc in tqdm(locs):
        vals, parse_fail_reason, response = get_direct_prompt_vals(loc, lm)
        num_respse_tokens.append(len(response.completion_tokens))
        if parse_fail_reason is None:
            num_parse_success += 1
        else:
            num_parse_fail += 1
            num_parse_fail_reasons[parse_fail_reason] += 1
            vals = [base_rate] * len(loc.get_line_spans())
        problem_id_estimate[loc.base_solve.problem.problem_id] = vals
    #print(problem_id_estimate)
    print(f"num_parse_success: {num_parse_success}")
    print(f"num_parse_fail: {num_parse_fail}")
    print(num_parse_fail_reasons)

    # Print dist of num_respse_tokens
    print("Num Response Tokens")
    print(scipy.stats.describe(num_respse_tokens))

    folds = localizations_to_folds(locs, config)
    all_train_preds = []
    all_test_preds = []
    fold_names = []
    for fold_num, (train_locs, test_locs, fold_name) in enumerate(folds):
        print(f"Fold '{fold_name}'")
        pred = lambda loc_list: [
            loc_token_level_deagg_pred(
                loc,
                np.array(problem_id_estimate[loc.base_solve.problem.problem_id]),
            )
            for loc in loc_list
        ]
        all_train_preds.append(pred(train_locs))
        all_test_preds.append(pred(test_locs))
        fold_names.append(fold_name)
    results = FoldResultsPreds(
        train_preds_each_fold=all_train_preds,
        test_preds_each_fold=all_test_preds,
        config=config,
        fold_names=fold_names,
    )
    if cache:
        serialize_fold_results_preds(results, fn)
    return results


def main():
    results = get_direct_prompt_results(
        BASE_PAPER_CONFIG_QWEN, dev_mode=False, cache=False)
    pprint(make_metrics_from_fold_results_multilevel(results))


def get_direct_prompt_vals(
    loc: TokenizedLocalization,
    lm: LmPredictor,
) -> tuple[list[float], bool, LmPrediction]:
    prompt = get_direct_prompt(loc)
    response = lm.predict(prompt)
    text = pull_last_md_block_from_completion(response.completion_text)
    parse_fail_reason = None
    if text is None:
        text = response.completion_text
    vals, parse_fail_reason = _parse_completion_text(text, loc)
    if parse_fail_reason is not None:
        vals = [0.5] * len(loc.base_tokens)
    line_spans = loc.get_line_spans()
    if len(vals) < len(line_spans):
        print(text)
        # Pad with the average
        avg = sum(vals) / len(vals)
        vals = vals + [avg] * (len(loc.base_tokens) - len(vals))
        parse_fail_reason = "TooFewVals"
        #raise ValueError(f"vals is not the same length as the base tokens: {len(vals)} != {len(loc.base_tokens)}")
    if len(vals) > len(line_spans):
        parse_fail_reason = "TooManyVals"
        vals = vals[:len(line_spans)]

    assert len(vals) == len(line_spans)
    return vals, parse_fail_reason, response


def _parse_completion_text(text: str, loc: TokenizedLocalization) -> tuple[list[float], str | None]:
    try:
        vals = ast.literal_eval(text)
    except SyntaxError as e:
        return None, "SyntaxOnAst"
    except ValueError as e:
        return None, "ValueErrorOnAst"
    if not isinstance(vals, list):
        return None, "NotAList"
    vals = [float(max(0.0, min(1.0, val))) for val in vals]
    return vals, None


def strip_all_indentation(text: str) -> str:
    stripped = (line.strip() for line in text.split("\n"))
    return "\n".join(line for line in stripped if line)


def find_method_of_body(source: str, body: str) -> str | None:
    """
    Returns the full method (including the signature and docstring) that contains the body.
    """
    funcs = PythonLangSpec().find_functions(source)
    stripped_body = strip_all_indentation(body)
    for func in funcs:
        if (
            strip_all_indentation(func.get_body_src(include_prefix_comment=False)) == stripped_body
            or strip_all_indentation(func.get_body_src(include_prefix_comment=True)) == stripped_body
        ):
            return func.get_full_function_src()
    return None


def get_direct_prompt(
    loc: TokenizedLocalization,
) -> LmPrompt:
    content = loc.base_solve.apply().get_only_file().content_str
    base_text = find_method_of_body(content, loc.get_base_text())
    if base_text is None:
        #print("DID NOT FIND FUNC")
        base_text = loc.get_base_text()
    else:
        #print("FOUND FUNC")
        pass
    base_tokens = loc.gt_fix_tokens
    lines = [
        "".join(base_tokens[i:j])
        for i, j in loc.get_line_spans()
    ]
    lines_str = [
        "[", 
        ",\n".join(f"    {line!r}" for line in lines),
        "]"
    ]
    prompt = [
        "We are attempting estimated calibrated probabilities that lines of code are correct or if they will need to be edited.",
        "Consider the following code:",
        "```python",
        content,
        "```",
        "We are attempting to estimate the probability that each line of code is correct.",
        "Please provide your estimate as a list[float] where each element is between 0 and 1 representing the probability that the line will be correct and unedited. One or two digits of precision is fine.",
        "This is the individual line probabilities so the sum is not expected to be 1.",
        "These are the line splitting we are using:",
        *lines_str,
        "Create a calibrated estimate of the probability that each line is correct. You can consider any potential issues with the code, but then place your final answer in a markdown code block with only a list[float] of length " + str(len(lines)) + f". Do not end early, and do not stop until you list all {str(len(lines))} probabilities corresponding to the given splits.",
    ]
    prompt_str = "\n".join(prompt)
    return LmPrompt(
        text=prompt_str,
        max_tokens=5000,
        temperature=0.0,
        cache=True,
    )


if __name__ == "__main__":
    main()