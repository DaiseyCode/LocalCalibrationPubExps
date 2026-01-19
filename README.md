# Code for "Localized Calibrated Uncertainty in Code Language Models"

## Installation:

The code was developed with python 3.10 on Ubuntu.

It is recommended you use a environment management system like miniconda.
```sh
conda create -n localcalib python=3.10
conda activate localcalib
pip install -r requirements.txt
pip install -e calipy
```

## Key Files

### Paper figure and table generation
The following scripts are the main entry points
for the figures and tables in the paper.

All scripts expect to be run from project root. eg,
`python -m pape.dataset_stats_table`

CAUTION: some these scripts make a lot of OpenAI/Claude API calls if don't have the cache files loaded. See notes.


- `/pape/dataset_stats_table.py`
    - This makes the Table 1 and 2 for how many examples are each dataset
- `/pape/main_metrics_table.py`
    - The aggregate files  
- `/pape/gen_tex_vars.py`
    - Generates variables used in parts of the paper prose (though some values ended hardcoded in the tex)
- `/localizing/cross_se_robustness.py`
    - Used to make the dataset level tables
- `generalizing/hello_halu.py`
    - used to make the halueval figure

### Other interesting scripts
- `localizing/localizing_structs.py`
    - The abstract classes that lay out how we format and track problems, fixes, and estimates
- `localizing/multi_data_gather.py`
    - localizing.multi_data_gather.create_tokenized_localizations_from_scratch
        - A root on how could the localizations with repairs are created
        - Note letting the full from scratch pipeline run could be a lot of API calls 
          (to remain safe from unexpected calls, unset OPENAI_API_KEY env and don't have a ~/oai_key.txt file)
    - Fully running from scratch requires docker setup
    - The current cache files aren't in this github. I still need to work out a better
      solution for this. Ideally like converting this to a HF parquet with must the key data.
      Otherwise it is admitedly difficult to use. I have
      some ideas for revisiting this work, and will try
      to clean things up then (email me if something comes up sooner).
- `pape/configs.py`
    - Used to setup details for many of the experiment.
- `localizing/fix_adders.py`
    - Some of the code for repairing the programs 
- `localizing/probe/probe_models_agg.py`
    - The modeling and training code for the probes
- `localizing/direct_prompt.py`
    - Reflective prompting scripts
- `pape/run_all_probe_exps`
    - Used to do the grid search to study the hyperparameter influence (like the eta-squared and appendix fig)
- `localizing/intrinsic.py`
    - The logprob confidence estimation
