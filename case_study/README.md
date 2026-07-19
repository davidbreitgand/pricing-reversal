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

## Repeated trials (same task, same model, N runs)

Some benchmarks ship 5 independent runs of every task per model (see the
`repeated_runs` column in `list`). `spread` shows how unstable the bill is
run-to-run — identical input, very different cost, sometimes a flipped answer:

```bash
# rank aime tasks by how much gpt-5.4's bill swings across its 5 runs
python case_study.py spread --benchmark aime-hybrid --model gpt-5.4

# the same as a self-contained HTML strip plot
# (each dot = one run, green/red = right/wrong, bar = min–max, dashed = mean)
python case_study.py spread --benchmark aime-hybrid --model gpt-5.4 --out spread.html

# thinking-token variance instead of cost, and only tasks whose answer flipped
python case_study.py spread --benchmark aime-hybrid --model gpt-5.4 \
    --dimension thinking --flips-only
```

Options: `--dimension cost|thinking`, `--metric maxmin|cv|std` (how to rank),
`--flips-only`, `--task-id ID`, `--top-k N`. `--out FILE.html` renders the
plot, `FILE.csv` writes the per-task stats (plus `*_README.txt`), anything else
prints text. Repeated trials are available for `aime-hybrid`, `gpqa-test`, and
`tb2-terminus2` (tb2 records cost only, no thinking split).

Pre-rendered `spread` pages for the six frontier models on AIME and
TerminalBench-2 live in [`spread_examples/`](spread_examples/).

## Notes

- **cost** is the platform-billed USD at the time of the run (early-2026
  list prices).
- **completion tokens = thinking + visible**; multi-turn rows also report
  fresh vs cached input.
- Multi-turn transcripts (tb2 / gaia / cybench) are verbatim per turn, with
  each message capped at 1,500 characters.
- Python API: `from case_study import PriceReversalData` — see the module
  docstring.
