from collections import Counter
from synthegrator.synthdatasets import yield_problems_from_name

from tqdm import tqdm
from localizing.probe.probe_data_gather import get_or_serialize_tokenized_localizations
from localizing.problem_processing import solve_to_text
from pape.configs import BASE_PAPER_CONFIG
from synthegrator.synthdatasets import DatasetName


def why_weird_base_text():
    dev_mode = True
    config = BASE_PAPER_CONFIG
    locs = get_or_serialize_tokenized_localizations(
        config, dev_mode=dev_mode)
    weird_locs = []
    weird_loc_dataset_count = Counter()
    for loc in locs.iter_passed_filtered():
        text = loc.get_base_text()
        text_start = text[:min(len(text), 50)]
        if '"""' in text_start:
            print(loc.get_base_text())
            weird_locs.append(loc)
            weird_loc_dataset_count[loc.dataset_name] += 1
    print(len(weird_locs))
    print(weird_loc_dataset_count)
    #exit()
    for loc in weird_locs:
        if loc.dataset_name != DatasetName.humaneval_plus:
            continue
        print("--------------------------------")
        print("-- The base text:")
        print(loc.get_base_text())
        #print("-- The solve_to_text:")
        #print(solve_to_text(loc.base_solve, loc.dataset_name))
        #exit()


def check_problem_ids_repocod():
    dev_mode = False
    config = BASE_PAPER_CONFIG
    locs = get_or_serialize_tokenized_localizations(
        config, dev_mode=dev_mode)
    problem_ids = set()
    dataset = DatasetName.repocod
    for loc in locs.iter_all():
        if loc.dataset_name != dataset:
            continue
        prob_id = loc.base_solve.problem.problem_id
        prob_id = prob_id.replace("_File-level", "")
        prob_id = prob_id.replace("_file-level", "")
        prob_id = prob_id.replace("_Repository-level", "")
        prob_id = prob_id.replace("_repository-level", "")
        prob_id = prob_id.replace("_Self-contained", "")
        prob_id = prob_id.replace("_self-contained", "")
        problem_ids.add(prob_id)
    print(len(problem_ids))
    all_probs = list(
        tqdm(yield_problems_from_name(dataset, max_problems=380), 
            desc=f"Getting problems from {dataset}"))
    prob_ids_new = set()
    for prob in all_probs:
        prob_id = prob.problem_id
        prob_id = prob_id.replace("_File-level", "_filelevel")
        prob_id = prob_id.replace("_file-level", "_filelevel")
        prob_id = prob_id.replace("_Repository-level", "_repositorylevel")
        prob_id = prob_id.replace("_repository-level", "_repositorylevel")
        prob_id = prob_id.replace("_Self-contained", "_selfcontained")
        prob_id = prob_id.replace("_self-contained", "_selfcontained")
        prob_ids_new.add(prob_id)
    print("loaded prob ids: ", len(prob_ids_new))
    print("all prob ids: ", len(all_probs))
    print("prob ids in loaded but not new: ", len(problem_ids - prob_ids_new))
    print("prob ids in new but not loaded: ", len(prob_ids_new - problem_ids))
    print("prob ids in loaded and new: ", len(problem_ids & prob_ids_new))
    print(problem_ids - prob_ids_new)
    print(prob_ids_new)


def main():
    why_weird_base_text()
    #check_problem_ids_repocod()


if __name__ == "__main__":
    main()