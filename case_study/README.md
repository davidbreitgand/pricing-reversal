# Case studies: same task, very different bills

`case_study.py` extracts price-variability case studies from the
[price-reversal dataset](https://huggingface.co/datasets/price-reversal/price-reversal):
pick two models, find the tasks where the cheaper-on-paper model actually
billed far more, and export the full story — prompt, token breakdown, cost,
answers, and (for agentic benchmarks) the turn-by-turn transcript.

## Setup

```bash
pip install -r ../requirements.txt   # just pandas + huggingface_hub
```

The main data downloads from Hugging Face automatically.
`case_study_extras.zip` (task texts, transcripts, answers) ships in this
folder and is picked up automatically.

## Workflow

```bash
# 0. what data exists (12 benchmarks x 8 models)
python case_study.py list

# 1. rank tasks by price reversal between two models
#    reversal = (actual cost A / actual cost B) / (listed price A / listed price B)
python case_study.py find --model-a gemini-3-flash --model-b gpt-5.4 \
    --category livecodebench-test --correct both --top-k 5

# 2. full story of the top task (text): prompt, bills, answers, transcripts
python case_study.py show                 # --rank 2 for the runner-up
python case_study.py show --out case.txt

# 3. the same story as a self-contained HTML comparison page
python case_study.py visualize --out case.html
```

`find` options: `--category <benchmark...>` (default: all), `--query email`
(search task text), `--correct both|neither|a|b` (filter by which model
solved it), `--top-k N`. It saves its selection so `show` / `visualize`
need no arguments.

`--out file.csv` on `find` exports the table plus a `*_README.txt`
explaining every field (token definitions, cost basis).

## Notes

- **cost** is the platform-billed USD at the time of the run (early-2026
  list prices).
- **completion tokens = thinking + visible**; multi-turn rows also report
  fresh vs cached input.
- Multi-turn transcripts (tb2 / gaia / cybench) are verbatim per turn, with
  each message capped at 1,500 characters.
- Python API: `from case_study import PriceReversalData` — see the module
  docstring.
