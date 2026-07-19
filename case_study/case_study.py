"""Case-study extraction API for the price-reversal dataset.

Loads the published data from Hugging Face (price-reversal/price-reversal).
Three commands, one workflow:

  find       rank tasks by the price reversal between two models
             (actual cost ratio vs listed price ratio); saves the selection
  show       full story of a found task, as text: task, bills, answers,
             agent transcripts
  visualize  the same story as a self-contained HTML comparison page

Column semantics (also written next to every CSV export by `to_csv`):

  task_text          single-turn: the exact prompt sent to the model;
                     multi-turn (tb2/gaia/cybench): the task instruction given
                     to the agent
  origin_query       the underlying benchmark question without prompt scaffold
                     (single-turn only)
  prompt_tokens      fresh input tokens (excludes prompt-cache reads)
  cached_tokens      input tokens read from the prompt cache
  thinking_tokens    reasoning tokens, hidden from the user
  visible_tokens     generated answer tokens shown to the user
  completion_tokens  thinking_tokens + visible_tokens (total output)
  cost               platform-billed USD at the time of the run

Usage:
    from case_study import PriceReversalData, to_csv
    data = PriceReversalData()
    df = data.find_examples(category="arenahard-test", model_a="claude-haiku-4.5",
                            model_b="gpt-5.4", query="email", top_k=5)

CLI:
    python case_study.py find --model-a claude-haiku-4.5 --model-b gpt-5.4 --top-k 5
    python case_study.py show --out case.txt
    python case_study.py visualize --out case.html
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

HF_REPO_ID = "price-reversal/price-reversal"
MAIN_ZIP = "price_reversal_unified.zip"
EXTRAS_ZIP = "case_study_extras.zip"  # task texts, transcripts, answers
# Pin to a dataset commit for reproducibility (None = latest on main).
DEFAULT_REVISION = None

CORE_COLUMNS = [
    "benchmark", "model", "task_id", "task_text", "origin_query",
    "prompt_tokens", "cached_tokens", "thinking_tokens", "visible_tokens",
    "completion_tokens", "cost", "correct", "n_turns",
    "prediction", "ground_truth",
]

FIELD_NOTES = """\
Field definitions (price-reversal case-study export)
-----------------------------------------------------
task_text          single-turn: the exact prompt sent to the model;
                   multi-turn (tb2/gaia/cybench): the task instruction given to the agent
origin_query       the underlying benchmark question without prompt scaffold (single-turn only)
prompt_tokens      fresh input tokens (excludes prompt-cache reads)
cached_tokens      input tokens read from the prompt cache
thinking_tokens    reasoning tokens, hidden from the user
visible_tokens     generated answer tokens shown to the user
completion_tokens  thinking_tokens + visible_tokens (total output tokens)
cost               platform-billed USD at the time of the run (early 2026 list
                   prices; see the paper for the exact pricing table)
prediction         the model's parsed final answer (single-turn benchmarks only;
                   for tb2/gaia/cybench the outcome lives in the transcript
                   panels of `show` / `visualize`)
ground_truth       the benchmark's expected answer, where defined

find_examples-only task-level columns (comparing model A vs model B on the
same task, one run each):
cost_a / cost_b    what model A / model B actually billed for this task
actual_ratio       cost_a / cost_b
listed_ratio       listed price of A divided by that of B, where a model's
                   listed price = input + output USD per-token price
reversal           actual_ratio / listed_ratio — 1.0 means spending matched
                   the price sheet; larger means A overshot its listed price
                   relative to B by that factor

Data: https://huggingface.co/datasets/price-reversal/price-reversal
"""


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


class PriceReversalData:
    """Entry point holding the (cached) HF zip handles.

    Parameters
    ----------
    revision : dataset commit hash or branch (None = main).
    main_zip, extras_zip : optional local paths overriding the HF download —
        useful for offline work or pre-publication testing.
    """

    # repeated-trial spread: what to measure, and how to rank tasks by its swing
    _DIMENSIONS = {"cost": "cost", "thinking": "thinking_tokens"}
    _METRICS = {"maxmin": "ratio_maxmin", "cv": "cv", "std": "std"}

    def __init__(self, revision=DEFAULT_REVISION, main_zip=None, extras_zip=None):
        self.revision = revision
        self._main_path = main_zip
        self._extras_path = extras_zip
        self._main = None
        self._extras = None
        self._extras_missing = False
        self._task_texts = {}

    # ---------------- data layer ----------------

    def _main_zip(self):
        if self._main is None:
            path = self._main_path or hf_hub_download(
                repo_id=HF_REPO_ID, repo_type="dataset",
                filename=MAIN_ZIP, revision=self.revision)
            self._main = zipfile.ZipFile(path)
        return self._main

    def _extras_zip(self):
        if self._extras is None and not self._extras_missing:
            path = self._extras_path
            if path is None:
                try:
                    path = hf_hub_download(
                        repo_id=HF_REPO_ID, repo_type="dataset",
                        filename=EXTRAS_ZIP, revision=self.revision)
                except EntryNotFoundError:
                    # not published yet — fall back to a local copy
                    candidates = [os.environ.get("PRICE_REVERSAL_EXTRAS"),
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), EXTRAS_ZIP),
                                  EXTRAS_ZIP]
                    path = next((c for c in candidates if c and os.path.exists(c)), None)
            if path is None:
                self._extras_missing = True
                print(f"warning: {EXTRAS_ZIP} is not on the HF dataset and no local "
                      f"copy was found — task texts (prompts), transcripts, and "
                      f"answers will be empty. Fix: upload it to {HF_REPO_ID}, or pass "
                      f"--extras-zip PATH, or set $PRICE_REVERSAL_EXTRAS.",
                      file=sys.stderr)
            else:
                self._extras = zipfile.ZipFile(path)
        return self._extras

    def texts(self, benchmark):
        """{task_id: {task_text, origin_query?}} for one benchmark."""
        if benchmark not in self._task_texts:
            z = self._extras_zip()
            table = {}
            if z is not None:
                try:
                    table = json.loads(z.read(f"task_texts/{benchmark}.json"))
                except KeyError:
                    pass
            self._task_texts[benchmark] = table
        return self._task_texts[benchmark]

    def benchmarks(self):
        return sorted({n.split("/")[0] for n in self._main_zip().namelist()
                       if n.endswith(".json")})

    def models(self, benchmark):
        return sorted(n.split("/")[1][:-len(".json")]
                      for n in self._main_zip().namelist()
                      if n.startswith(benchmark + "/") and n.endswith(".json"))

    def _repeated_models(self, benchmark):
        z = self._extras_zip()
        if z is None:
            return []
        prefix = f"repeated_trials/{benchmark}/"
        return sorted(n[len(prefix):-len(".json")] for n in z.namelist()
                      if n.startswith(prefix) and n.endswith(".json"))

    # ---------------- normalisation ----------------

    def _rows(self, benchmark, payload):
        texts = self.texts(benchmark)
        rows = []
        for t in payload["trials"]:
            tok = t["tokens"]
            reasoning, output = tok.get("reasoning"), tok.get("output", 0)
            if reasoning is None:
                thinking = visible = float("nan")
                completion = output
            else:
                thinking, visible = reasoning, output
                completion = output + reasoning
            text = texts.get(str(t["task_id"]), {})
            rows.append({
                "benchmark": benchmark,
                "model": payload["model"],
                "task_id": t["task_id"],
                "task_text": text.get("task_text"),
                "origin_query": text.get("origin_query"),
                "prompt_tokens": tok.get("input"),
                "cached_tokens": tok.get("cache"),
                "thinking_tokens": thinking,
                "visible_tokens": visible,
                "completion_tokens": completion,
                "cost": t.get("cost_usd"),
                "correct": t.get("correct"),
                "n_turns": t.get("n_turns"),
                "prediction": (t.get("metadata") or {}).get("prediction"),
                "ground_truth": (t.get("metadata") or {}).get("ground_truth")
                                if (t.get("metadata") or {}).get("ground_truth") is not None
                                else text.get("ground_truth"),
            })
        return rows

    @staticmethod
    def _check(value, options, kind):
        """Raise a helpful error (with a did-you-mean hint) for bad names."""
        if value not in options:
            hint = difflib.get_close_matches(str(value), options, n=2, cutoff=0.3)
            msg = f"unknown {kind}: {value!r}."
            if hint:
                msg += f" Did you mean {' or '.join(repr(h) for h in hint)}?"
            msg += f" Available: {sorted(options)}"
            raise KeyError(msg)

    def _read_main(self, benchmark, model):
        self._check(benchmark, self.benchmarks(), "benchmark")
        self._check(model, self.models(benchmark), f"model (on {benchmark})")
        payload = json.loads(self._main_zip().read(f"{benchmark}/{model}.json"))
        return self._rows(benchmark, payload)

    # ---------------- public API ----------------

    def list_available(self):
        """One row per (benchmark, model): task count and repeated-run count."""
        rows = []
        for b in self.benchmarks():
            repeated = {}
            for m in self._repeated_models(b):
                z = self._extras_zip()
                payload = json.loads(z.read(f"repeated_trials/{b}/{m}.json"))
                repeated[m] = payload["n_runs"]
            for m in self.models(b):
                d = json.loads(self._main_zip().read(f"{b}/{m}.json"))
                rows.append({"benchmark": b, "model": m, "n_tasks": d["n_trials"],
                             "repeated_runs": repeated.pop(m, 0)})
            for m, n_runs in repeated.items():  # repeated-trial-only models
                rows.append({"benchmark": b, "model": m, "n_tasks": 0,
                             "repeated_runs": n_runs})
        return pd.DataFrame(rows)

    def load_records(self, benchmarks=None, models=None):
        """Long table of all (task, model) records; None selects everything."""
        benchmarks = _as_list(benchmarks) or self.benchmarks()
        rows = []
        for b in benchmarks:
            for m in _as_list(models) or self.models(b):
                rows.extend(self._read_main(b, m))
        return pd.DataFrame(rows, columns=CORE_COLUMNS)

    def cross_model(self, benchmark, task_id, models=None):
        """One task, one row per model."""
        df = self.load_records(benchmark, models)
        df = df[df["task_id"].astype(str) == str(task_id)]
        if df.empty:
            raise KeyError(f"task_id {task_id!r} not found in {benchmark}")
        return df.sort_values("cost", ascending=False).reset_index(drop=True)

    def transcript(self, benchmark, model, task_id):
        """What the agent actually did on a multi-turn task, turn by turn.

        Returns a DataFrame with columns (turn, role, text): the agent's
        analysis/plan/commands and the environment's responses, each message
        capped at ~1500 chars in the published data. A final row with
        role='final_answer' carries the model's closing answer where one
        exists (gaia/cybench; tb2 is graded on the terminal end state).
        Only tb2-terminus2, gaia-react, and cybench-react have transcripts.
        """
        z = self._extras_zip()
        if z is None:
            raise RuntimeError(f"transcripts live in {EXTRAS_ZIP}; see the warning above")
        self._check(benchmark, self.benchmarks(), "benchmark")
        prefix = "transcripts/"
        have = sorted({n[len(prefix):].split("/")[0] for n in z.namelist()
                       if n.startswith(prefix)})
        self._check(benchmark, have, "benchmark (with transcripts)")
        models = sorted(n.split("/")[2][:-len(".json")] for n in z.namelist()
                        if n.startswith(f"{prefix}{benchmark}/"))
        self._check(model, models, f"model (transcripts on {benchmark})")
        tasks = json.loads(z.read(f"{prefix}{benchmark}/{model}.json"))
        self._check(str(task_id), tasks, f"task_id (transcripts on {benchmark})")
        entry = tasks[str(task_id)]
        rows = [{"turn": i, "role": t.get("role"), "text": t.get("text")}
                for i, t in enumerate(entry["turns"])]
        if entry.get("final_answer") is not None:
            rows.append({"turn": len(rows), "role": "final_answer",
                         "text": entry["final_answer"]})
        return pd.DataFrame(rows, columns=["turn", "role", "text"])

    def _listed_price(self, benchmark, model):
        """Listed price of a model: input + output USD per-token price,
        as recorded in the run data."""
        payload = json.loads(self._main_zip().read(f"{benchmark}/{model}.json"))
        for t in payload["trials"]:
            f = t.get("factors") or {}
            if f.get("p_in") and f.get("p_out"):
                return f["p_in"] + f["p_out"]
        raise KeyError(f"no listed prices recorded for {model} on {benchmark}")

    def find_examples(self, category=None, model_a=None, model_b=None,
                      query=None, top_k=10, correct=None):
        """Rank tasks by the price reversal between two models.

        For each task (one run per model):
            actual_ratio = cost_a / cost_b        (what A vs B actually billed)
            listed_ratio = listed price of A / listed price of B, where a
                           model's listed price = input + output per-token price
            reversal     = actual_ratio / listed_ratio
        reversal == 1 means spending matched what the price sheet predicts;
        the larger it is, the more A overshot its listed price relative to B.
        Swap A and B to look for reversals in the other direction.

        category: benchmark name(s) to search (as in `list_available`);
        None = all. correct filters tasks by outcome: "both" (A and B solved
        it), "neither" (both failed), "a" / "b" (only that model solved it);
        None = no filter. Returns both models' full records for the top_k
        tasks, sorted by reversal, descending.
        """
        if not model_a or not model_b:
            raise ValueError("find_examples requires model_a and model_b")
        df = self.load_records(_as_list(category), [model_a, model_b])
        if query:
            hay = df["origin_query"].fillna(df["task_text"]).fillna("")
            df = df[hay.str.contains(query, case=False, regex=False)]
        if df.empty:
            return df
        wide = (df.pivot_table(index=["benchmark", "task_id"], columns="model",
                               values="cost", aggfunc="first")
                  .dropna().reset_index())
        wide.columns.name = None
        stats = wide.rename(columns={model_a: "cost_a", model_b: "cost_b"})
        stats = stats[stats["cost_b"] > 0]
        if correct is not None:
            ok = df.pivot_table(index=["benchmark", "task_id"], columns="model",
                                values="correct", aggfunc="first")
            a_ok, b_ok = ok[model_a].fillna(0) > 0, ok[model_b].fillna(0) > 0
            masks = {"both": a_ok & b_ok, "neither": ~a_ok & ~b_ok,
                     "a": a_ok & ~b_ok, "b": b_ok & ~a_ok}
            if correct not in masks:
                raise KeyError(f"correct must be one of {list(masks)}")
            keep = ok.index[masks[correct]]
            stats = stats[stats.set_index(["benchmark", "task_id"]).index.isin(keep)]
            if stats.empty:
                return pd.DataFrame(columns=CORE_COLUMNS)
        listed = {b: (self._listed_price(b, model_a)
                      / self._listed_price(b, model_b))
                  for b in stats["benchmark"].unique()}
        stats["actual_ratio"] = stats["cost_a"] / stats["cost_b"]
        stats["listed_ratio"] = stats["benchmark"].map(listed)
        stats["reversal"] = stats["actual_ratio"] / stats["listed_ratio"]
        stats = stats.sort_values("reversal", ascending=False).head(top_k)
        out = stats.merge(df, on=["benchmark", "task_id"], how="left")
        out = out.sort_values(["reversal", "benchmark", "task_id", "cost"],
                              ascending=[False, True, True, False])
        cols = CORE_COLUMNS + ["cost_a", "cost_b", "actual_ratio",
                               "listed_ratio", "reversal"]
        return out[cols].reset_index(drop=True)

    # ---------------- repeated trials ----------------

    def repeated_benchmarks(self):
        """Benchmarks that ship repeated-trial runs in the extras zip."""
        z = self._extras_zip()
        if z is None:
            return []
        return sorted({n.split("/")[1] for n in z.namelist()
                       if n.startswith("repeated_trials/") and n.endswith(".json")})

    def repeated_records(self, benchmark, model):
        """Long table of a model's repeated runs: one row per (task_id, run_index)
        with cost, token split, and outcome. Reads the `unified-v1-repeated`
        payloads (repeated_trials/<benchmark>/<model>.json) from the extras zip."""
        z = self._extras_zip()
        if z is None:
            raise RuntimeError(f"repeated trials live in {EXTRAS_ZIP}; see the warning above")
        self._check(benchmark, self.repeated_benchmarks(), "benchmark (with repeated trials)")
        self._check(model, self._repeated_models(benchmark),
                    f"model (repeated trials on {benchmark})")
        payload = json.loads(z.read(f"repeated_trials/{benchmark}/{model}.json"))
        texts = self.texts(benchmark)
        rows = []
        for t in payload["trials"]:
            tok = t.get("tokens") or {}
            reasoning = tok.get("reasoning")
            output = tok.get("output") or 0
            meta = t.get("metadata") or {}
            text = texts.get(str(t["task_id"]), {})
            gt = meta.get("ground_truth")
            rows.append({
                "task_id": str(t["task_id"]),
                "run_index": t.get("run_index"),
                "cost": t.get("cost_usd"),
                "input_tokens": tok.get("input"),
                "cached_tokens": tok.get("cache"),
                "visible_tokens": output,
                "thinking_tokens": reasoning if reasoning is not None else float("nan"),
                "completion_tokens": output + (reasoning or 0),
                "correct": t.get("correct"),
                "prediction": meta.get("prediction"),
                "ground_truth": gt if gt is not None else text.get("ground_truth"),
                "task_text": text.get("task_text"),
                "origin_query": text.get("origin_query"),
            })
        return (pd.DataFrame(rows)
                  .sort_values(["task_id", "run_index"]).reset_index(drop=True))

    def repeated_stats(self, benchmark, model, dimension="cost"):
        """Per-task summary of run-to-run spread for `dimension` in {cost, thinking}:
        min/median/max/mean/std, CV (std/mean), max/min ratio, and whether the
        outcome flipped across runs."""
        if dimension not in self._DIMENSIONS:
            raise KeyError(f"dimension must be one of {list(self._DIMENSIONS)}")
        col = self._DIMENSIONS[dimension]
        df = self.repeated_records(benchmark, model)
        if df[col].isna().all():
            raise KeyError(f"{dimension!r} is not recorded for {model} on {benchmark} "
                           f"(e.g. tb2 has no thinking-token split); try --dimension cost")
        rows = []
        for tid, g in df.groupby("task_id", sort=False):
            vals = g[col].dropna()
            if vals.empty:
                continue
            n = len(vals)
            vmin, vmax, mean = float(vals.min()), float(vals.max()), float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            corr = g["correct"].dropna()
            n_pass = int((corr > 0).sum())
            rows.append({
                "task_id": tid, "n_runs": n,
                "n_correct": (n_pass if len(corr) else None),
                "n_graded": len(corr),
                "flip": bool(0 < n_pass < len(corr)) if len(corr) else False,
                f"{dimension}_mean": mean,
                f"{dimension}_median": float(vals.median()),
                f"{dimension}_min": vmin,
                f"{dimension}_max": vmax,
                f"{dimension}_std": std,
                f"{dimension}_cv": (std / mean if mean else float("nan")),
                f"{dimension}_ratio_maxmin": (vmax / vmin if vmin > 0 else float("nan")),
            })
        return pd.DataFrame(rows)

    def repeated_summary(self, benchmark, model, dimension="cost"):
        """Model-level rollup of `repeated_stats` (median CV, worst ratio, flip rate)."""
        st = self.repeated_stats(benchmark, model, dimension)
        cv = st[f"{dimension}_cv"].dropna()
        ratio = st[f"{dimension}_ratio_maxmin"].dropna()
        graded = int((st["n_graded"] > 0).sum())
        worst_task = st.loc[ratio.idxmax(), "task_id"] if not ratio.empty else None
        return {
            "benchmark": benchmark, "model": model, "dimension": dimension,
            "n_tasks": int(len(st)),
            "n_runs": int(st["n_runs"].max()) if len(st) else 0,
            "median_cv": float(cv.median()) if not cv.empty else float("nan"),
            "pct_ratio_gt_2x": float((ratio > 2).mean()) if not ratio.empty else float("nan"),
            "n_flipped": int(st["flip"].sum()),
            "pct_flipped": (float(st["flip"].sum() / graded) if graded else float("nan")),
            "worst_ratio": float(ratio.max()) if not ratio.empty else float("nan"),
            "worst_task": worst_task,
        }

    def find_variable(self, benchmark, model, dimension="cost", metric="maxmin",
                      flips_only=False, top_k=10):
        """Rank a model's tasks by run-to-run variability of `dimension`.
        metric: maxmin (max/min ratio), cv (coeff. of variation), or std."""
        if metric not in self._METRICS:
            raise KeyError(f"metric must be one of {list(self._METRICS)}")
        st = self.repeated_stats(benchmark, model, dimension)
        if flips_only:
            st = st[st["flip"]]
        key = f"{dimension}_{self._METRICS[metric]}"
        return (st.sort_values(key, ascending=False, na_position="last")
                  .head(top_k).reset_index(drop=True))


LAST_FIND_FILE = ".case_study_last_find.json"


def _load_selection(from_file, rank):
    """Read the selection saved by `find`; return (benchmark, task_id, models,
    one-line header describing the pick)."""
    try:
        sel = pd.read_json(from_file, orient="records")
    except (FileNotFoundError, ValueError):
        sys.exit(f"no selection at {from_file!r} — run `find` first "
                 "(it saves its result there), or pass --from FILE")
    tasks = sel[["benchmark", "task_id"]].drop_duplicates().reset_index(drop=True)
    if not 1 <= rank <= len(tasks):
        sys.exit(f"--rank must be 1..{len(tasks)} (the last find kept "
                 f"{len(tasks)} tasks)")
    b, t = tasks.iloc[rank - 1]
    pick = sel[(sel["benchmark"] == b) & (sel["task_id"].astype(str) == str(t))]
    models = pick["model"].dropna().unique().tolist()
    header = (f"rank {rank} of last find | reversal {pick['reversal'].iloc[0]:.1f}"
              f" | actual cost ratio {pick['actual_ratio'].iloc[0]:.2f}"
              f" | listed price ratio {pick['listed_ratio'].iloc[0]:.2f}"
              if "reversal" in pick.columns else f"rank {rank} of last find")
    return b, t, models, header


def render_case(data, benchmark, task_id, models=None, header=""):
    """Compose the readable end-to-end story of one task: the task text, every
    model's bill, final answers where they exist, and — for multi-turn
    benchmarks — the turn-by-turn transcript of what each model did."""
    cm = data.cross_model(benchmark, task_id, models)
    first = cm.iloc[0]
    lines = [f"===== case study: {benchmark} / {task_id} ====="]
    if header:
        lines.append(header)
    if isinstance(first["task_text"], str) and first["task_text"]:
        lines += ["", "---- task given to the model ----", first["task_text"]]
    if isinstance(first["origin_query"], str) and first["origin_query"]:
        lines += ["", "---- underlying benchmark question ----", first["origin_query"]]
    cols = ["model", "prompt_tokens", "thinking_tokens", "visible_tokens",
            "completion_tokens", "cost", "correct", "n_turns"]
    lines += ["", "---- the bill ----", cm[cols].to_string(index=False)]
    if isinstance(first["ground_truth"], str) and first["ground_truth"]:
        lines += ["", f"ground truth: {first['ground_truth']}"]
    for _, r in cm.iterrows():
        if isinstance(r["prediction"], str) and r["prediction"]:
            lines += ["", f"---- {r['model']}'s final answer ----", r["prediction"]]
    for m in cm["model"]:
        try:
            tr = data.transcript(benchmark, m, task_id)
        except (KeyError, RuntimeError):
            break  # single-turn benchmark (or extras missing): no transcripts
        lines += ["", f"---- what {m} did ({len(tr)} turns) ----"]
        lines += [f"\n[turn {r.turn} | {r.role}]\n{r.text}" for r in tr.itertuples()]
    return "\n".join(lines)


_VIZ_CSS = """
.viz-root { --surface-1:#fcfcfb; --page:#f9f9f7; --ink-1:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,0.10);
  --c-prompt:#2a78d6; --c-cache:#86b6ef; --c-think:#4a3aa7; --c-visible:#1baf7a;
  --good:#0ca30c; --bad:#d03b3b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink-1); background: var(--page); margin:0; padding:24px; }
@media (prefers-color-scheme: dark) {
  .viz-root { --surface-1:#1a1a19; --page:#0d0d0d; --ink-1:#ffffff; --ink-2:#c3c2b7;
    --grid:#2c2c2a; --border:rgba(255,255,255,0.10);
    --c-prompt:#3987e5; --c-cache:#184f95; --c-think:#9085e9; --c-visible:#199e70; } }
.viz-wrap { max-width: 920px; margin: 0 auto; }
.viz-root h1 { font-size:20px; margin:0 0 4px; }
.viz-root .sub { color:var(--ink-2); font-size:13px; margin-bottom:20px; }
.panel { background:var(--surface-1); border:1px solid var(--border);
  border-radius:8px; padding:16px 20px; margin-bottom:16px; }
.panel h2 { font-size:13px; font-weight:600; color:var(--ink-2);
  text-transform:none; margin:0 0 12px; }
.tiles { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }
.tile { background:var(--surface-1); border:1px solid var(--border);
  border-radius:8px; padding:12px 16px; flex:1; min-width:130px; }
.tile .lbl { font-size:12px; color:var(--ink-2); margin-bottom:4px; }
.tile .val { font-size:26px; font-weight:600; }
.row { display:grid; grid-template-columns:150px 1fr 90px; align-items:center;
  gap:10px; margin:8px 0; }
.row .name { font-size:13px; color:var(--ink-1); text-align:right;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.row .val { font-size:13px; color:var(--ink-2); font-variant-numeric:tabular-nums; }
.track { position:relative; height:20px; }
.bar { position:absolute; top:0; height:20px; border-radius:0 4px 4px 0; }
.seg { display:inline-block; height:20px; margin-right:2px; vertical-align:top; }
.seg:last-child { border-radius:0 4px 4px 0; margin-right:0; }
.legend { display:flex; gap:16px; font-size:12px; color:var(--ink-2);
  margin:10px 0 2px; flex-wrap:wrap; }
.key { display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:5px; vertical-align:baseline; }
.badge { font-size:12px; font-weight:600; }
.badge.good { color:var(--good); } .badge.bad { color:var(--bad); }
.viz-root table { border-collapse:collapse; font-size:13px; width:100%; }
.viz-root th { text-align:right; color:var(--ink-2); font-weight:600;
  border-bottom:1px solid var(--grid); padding:6px 10px; }
.viz-root td { text-align:right; padding:6px 10px;
  border-bottom:1px solid var(--grid); font-variant-numeric:tabular-nums; }
.viz-root th:first-child, .viz-root td:first-child { text-align:left; }
.viz-root pre { white-space:pre-wrap; word-break:break-word; font-size:12.5px;
  line-height:1.5; background:var(--page); border:1px solid var(--grid);
  border-radius:6px; padding:10px 12px; margin:8px 0; }
.viz-root details > summary { cursor:pointer; font-size:13px; color:var(--ink-2);
  margin:4px 0; }
.turn-role { font-size:11px; font-weight:600; color:var(--muted);
  margin:10px 0 2px; text-transform:uppercase; letter-spacing:.04em; }
.foot { color:var(--muted); font-size:12px; margin-top:20px; line-height:1.5; }
#tip { position:fixed; opacity:0; pointer-events:none; background:var(--ink-1);
  color:var(--surface-1); font-size:12px; padding:4px 8px; border-radius:4px;
  transition:opacity .1s; z-index:10; }
"""

_VIZ_JS = """
var tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(function (el) {
  el.addEventListener('mousemove', function (e) {
    tip.textContent = el.dataset.tip;
    tip.style.left = (e.clientX + 12) + 'px';
    tip.style.top = (e.clientY + 12) + 'px';
    tip.style.opacity = 1;
  });
  el.addEventListener('mouseleave', function () { tip.style.opacity = 0; });
});
"""


def render_case_html(data, benchmark, task_id, models=None, title_note=""):
    """Self-contained HTML page comparing the models on one task: cost bars,
    token breakdown, answers/transcripts. Light & dark mode, no external deps."""
    from html import escape as esc

    cm = data.cross_model(benchmark, task_id, models)
    first = cm.iloc[0]

    def fcost(c):
        return f"${c:,.4f}" if c >= 0.01 else f"${c:.5f}"

    def ftok(v):
        return "—" if pd.isna(v) else f"{int(v):,}"

    # headline stats (two-model comparison; cross_model sorts cost desc)
    tiles = ""
    for _, r in cm.iterrows():
        tiles += (f'<div class="tile"><div class="lbl">{esc(r["model"])} billed</div>'
                  f'<div class="val">{fcost(r["cost"])}</div></div>')
    if len(cm) == 2 and cm["cost"].iloc[1] > 0:
        ratio = cm["cost"].iloc[0] / cm["cost"].iloc[1]
        tiles += (f'<div class="tile"><div class="lbl">actual cost ratio</div>'
                  f'<div class="val">{ratio:,.1f}&times;</div></div>')
        try:
            lr = (data._listed_price(benchmark, cm["model"].iloc[0])
                  / data._listed_price(benchmark, cm["model"].iloc[1]))
            tiles += (f'<div class="tile"><div class="lbl">listed price ratio</div>'
                      f'<div class="val">{lr:,.2f}&times;</div></div>'
                      f'<div class="tile"><div class="lbl">price reversal</div>'
                      f'<div class="val">{ratio / lr:,.1f}&times;</div></div>')
        except KeyError:
            pass

    # cost bars — one measure, rows labeled by model, single hue
    cmax = cm["cost"].max()
    cost_rows = ""
    for _, r in cm.iterrows():
        w = 100 * r["cost"] / cmax if cmax else 0
        cost_rows += (
            f'<div class="row"><div class="name">{esc(r["model"])}</div>'
            f'<div class="track"><div class="bar" data-tip="{esc(r["model"])}: '
            f'{fcost(r["cost"])}" style="width:{w:.2f}%;'
            f'background:var(--c-prompt)"></div></div>'
            f'<div class="val">{fcost(r["cost"])}</div></div>')

    # token breakdown — stacked prompt / thinking / visible per model
    seg_defs = [("prompt_tokens", "fresh input", "--c-prompt"),
                ("cached_tokens", "cached input", "--c-cache"),
                ("thinking_tokens", "thinking", "--c-think"),
                ("visible_tokens", "visible", "--c-visible")]
    totals = cm[[c for c, _, _ in seg_defs]].fillna(0).sum(axis=1)
    tmax = totals.max()
    tok_rows = ""
    for (_, r), tot in zip(cm.iterrows(), totals):
        segs = ""
        for col, name, var in seg_defs:
            v = 0 if pd.isna(r[col]) else r[col]
            if v <= 0:
                continue
            w = 100 * v / tmax if tmax else 0
            segs += (f'<div class="seg" data-tip="{esc(r["model"])} {name}: '
                     f'{ftok(v)} tokens" style="width:calc({w:.2f}% - 2px);'
                     f'background:var({var})"></div>')
        tok_rows += (f'<div class="row"><div class="name">{esc(r["model"])}</div>'
                     f'<div class="track">{segs}</div>'
                     f'<div class="val">{ftok(tot)}</div></div>')
    legend = "".join(f'<span><span class="key" style="background:var({var})"></span>'
                     f'{name} tokens</span>' for _, name, var in seg_defs)

    # numbers table (the accessible fallback for everything above)
    table = ("<table><tr><th>model</th><th>fresh input</th><th>cached</th>"
             "<th>thinking</th><th>visible</th><th>completion</th><th>cost</th>"
             "<th>result</th></tr>")
    for _, r in cm.iterrows():
        ok = r["correct"]
        badge = ('<span class="badge">—</span>' if pd.isna(ok) else
                 '<span class="badge good">&#10003; solved</span>' if ok else
                 '<span class="badge bad">&#10007; failed</span>')
        table += (f'<tr><td>{esc(r["model"])}</td><td>{ftok(r["prompt_tokens"])}</td>'
                  f'<td>{ftok(r["cached_tokens"])}</td>'
                  f'<td>{ftok(r["thinking_tokens"])}</td><td>{ftok(r["visible_tokens"])}</td>'
                  f'<td>{ftok(r["completion_tokens"])}</td><td>{fcost(r["cost"])}</td>'
                  f'<td>{badge}</td></tr>')
    table += "</table>"

    # task text + per-model answers / transcripts
    body_panels = ""
    if isinstance(first["task_text"], str) and first["task_text"]:
        q = first["origin_query"] if isinstance(first["origin_query"], str) else ""
        shown = q or first["task_text"]
        extra = (f'<details><summary>full prompt as sent to the model</summary>'
                 f'<pre>{esc(first["task_text"])}</pre></details>' if q else "")
        body_panels += (f'<div class="panel"><h2>The task</h2>'
                        f'<pre>{esc(shown)}</pre>{extra}</div>')
    gt = first["ground_truth"]
    if isinstance(gt, str) and gt:
        body_panels += (f'<div class="panel"><h2>Ground truth</h2>'
                        f'<pre>{esc(gt)}</pre></div>')
    for _, r in cm.iterrows():
        if isinstance(r["prediction"], str) and r["prediction"]:
            body_panels += (f'<div class="panel"><h2>{esc(r["model"])} answered</h2>'
                            f'<pre>{esc(r["prediction"])}</pre></div>')
    for m in cm["model"]:
        try:
            tr = data.transcript(benchmark, m, task_id)
        except (KeyError, RuntimeError):
            break
        turns = "".join(
            f'<div class="turn-role">turn {t.turn} &middot; {esc(str(t.role))}</div>'
            f'<pre>{esc(str(t.text))}</pre>' for t in tr.itertuples())
        body_panels += (f'<div class="panel"><h2>What {esc(m)} did</h2>'
                        f'<details><summary>{len(tr)} turns — click to expand'
                        f'</summary>{turns}</details></div>')

    note = f' &middot; {esc(title_note)}' if title_note else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(benchmark)} / {esc(str(task_id))} — price reversal case study</title>
<style>{_VIZ_CSS}</style></head>
<body class="viz-root"><div class="viz-wrap">
<h1>Same task, very different bills</h1>
<div class="sub">{esc(benchmark)} &middot; task {esc(str(task_id))}{note}</div>
<div class="tiles">{tiles}</div>
<div class="panel"><h2>What each model billed (USD)</h2>{cost_rows}</div>
<div class="panel"><h2>Where the tokens went</h2>
<div class="legend">{legend}</div>{tok_rows}{table}</div>
{body_panels}
<div class="foot">cost = platform-billed USD at the time of the run (early-2026
list prices) &middot; completion = thinking + visible &middot; data:
huggingface.co/datasets/{HF_REPO_ID}</div>
</div><div id="tip"></div><script>{_VIZ_JS}</script></body></html>"""


SPREAD_FIELD_NOTES = """\
Field definitions (price-reversal repeated-trial spread export)
---------------------------------------------------------------
one row per task: the same model run on the same task <n_runs> independent times.

task_id            benchmark task identifier
n_runs             independent repeated runs of this task by this model
n_correct          how many of those runs were graded correct (blank where the
                   benchmark records no correctness for these runs)
flip               True if the outcome changed across runs (sometimes right,
                   sometimes wrong) — identical input, different verdict
<dim>_mean         mean of the measured dimension across the runs, where
<dim>_median         dim = cost (platform-billed USD) or thinking (hidden
<dim>_min            reasoning tokens)
<dim>_max
<dim>_std          sample standard deviation across runs (ddof=1)
<dim>_cv           coefficient of variation = std / mean (unitless spread)
<dim>_ratio_maxmin max / min across runs — "the priciest run cost N x the
                   cheapest run of the same task"

Data: https://huggingface.co/datasets/price-reversal/price-reversal
"""


_SPREAD_CSS = """
.sp-axis { position:relative; height:16px; margin:2px 0 8px; }
.sp-axis span { position:absolute; transform:translateX(-50%); font-size:10px;
  color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
.sp-row { display:grid; grid-template-columns:150px 1fr 118px; align-items:center;
  gap:12px; margin:6px 0; }
.sp-row .name { font-size:11px; line-height:1.2; color:var(--ink-1); text-align:right;
  word-break:break-word; }
.sp-row .val { font-size:12px; color:var(--ink-2); font-variant-numeric:tabular-nums; }
.sp-track { position:relative; height:20px; }
.sp-line { position:absolute; top:9px; height:2px; background:var(--grid);
  border-radius:2px; }
.sp-mean { position:absolute; top:1px; height:18px; border-left:1.5px dashed var(--muted); }
.sp-dot { position:absolute; top:4px; width:12px; height:12px; margin-left:-6px;
  border-radius:50%; border:1px solid var(--surface-1); box-sizing:border-box; }
.sp-dot.good { background:var(--good); } .sp-dot.bad { background:var(--bad); }
.sp-dot.na { background:var(--muted); }
"""


def render_spread(data, benchmark, model, dimension="cost", metric="maxmin",
                  flips_only=False, top_k=10, task_id=None):
    """Readable report: how one model's cost (or thinking tokens) varies across
    repeated runs of the same tasks, plus the most variable task's per-run bills."""
    summ = data.repeated_summary(benchmark, model, dimension)
    ranked = data.find_variable(benchmark, model, dimension, metric, flips_only, top_k)

    def pct(x):
        return "—" if pd.isna(x) else f"{x * 100:.0f}%"

    def num(x, spec=".2f"):
        return "—" if pd.isna(x) else format(x, spec)

    lines = [
        f"===== repeated-trial spread: {benchmark} / {model} =====",
        f"{summ['n_tasks']} tasks x up to {summ['n_runs']} runs | "
        f"dimension={dimension} | ranked by {metric}", "",
        f"median {dimension} CV across tasks : {num(summ['median_cv'])}",
        f"tasks with >2x max/min {dimension:<8}: {pct(summ['pct_ratio_gt_2x'])}",
        f"worst max/min ratio               : {num(summ['worst_ratio'], ',.1f')}x "
        f"(task {summ['worst_task']})",
    ]
    if not pd.isna(summ["pct_flipped"]):
        lines.append(f"tasks whose outcome flipped       : {pct(summ['pct_flipped'])} "
                     f"({summ['n_flipped']} tasks)")

    lines += ["", f"---- top {len(ranked)} most variable tasks ----"]
    cols = ["task_id", "n_runs", "n_correct",
            f"{dimension}_min", f"{dimension}_median", f"{dimension}_max",
            f"{dimension}_cv", f"{dimension}_ratio_maxmin"]
    lines.append(ranked[cols].to_string(index=False) if len(ranked) else "(none)")

    tid = str(task_id) if task_id is not None else (
        ranked["task_id"].iloc[0] if len(ranked) else None)
    if tid is not None:
        recs = data.repeated_records(benchmark, model)
        g = recs[recs["task_id"] == tid].sort_values("run_index")
        if len(g):
            first = g.iloc[0]
            q = (first["origin_query"] if isinstance(first["origin_query"], str)
                 else first["task_text"])
            lines += ["", f"---- task {tid}: what varied across {len(g)} runs ----"]
            if isinstance(q, str) and q:
                lines.append(q[:1000] + ("…" if len(q) > 1000 else ""))
            dcols = [c for c in ["run_index", "thinking_tokens", "visible_tokens",
                                 "completion_tokens", "cost", "correct", "prediction"]
                     if c in g.columns]
            lines += ["", g[dcols].to_string(index=False)]
    return "\n".join(lines)


def render_spread_html(data, benchmark, model, dimension="cost", metric="maxmin",
                       flips_only=False, top_k=10, task_id=None):
    """Self-contained HTML page showing how one model's cost (or thinking tokens)
    varies across repeated independent runs of the same tasks: a log-scale strip of
    per-run dots (green solved / red failed), a min–max range bar, and a mean mark."""
    import math
    from html import escape as esc

    summ = data.repeated_summary(benchmark, model, dimension)
    ranked = data.find_variable(benchmark, model, dimension, metric, flips_only, top_k)
    recs = data.repeated_records(benchmark, model)
    col = PriceReversalData._DIMENSIONS[dimension]
    is_cost = dimension == "cost"

    def fval(v):
        if pd.isna(v):
            return "—"
        if is_cost:
            return f"${v:,.4f}" if v >= 0.01 else f"${v:.5f}"
        return f"{int(round(v)):,}"

    def pct(x):
        return "—" if pd.isna(x) else f"{x * 100:.0f}%"

    def num(x, spec=".2f"):
        return "—" if pd.isna(x) else format(x, spec)

    # shared log-x domain across the shown tasks so rows are directly comparable
    shown = recs[recs["task_id"].isin(set(ranked["task_id"]))]
    pos = shown[col][shown[col] > 0].dropna()
    gmin = float(pos.min()) if not pos.empty else 1.0
    gmax = float(pos.max()) if not pos.empty else 1.0
    lmin = math.log10(gmin)
    span = (math.log10(gmax) - lmin) or 1.0

    def xpct(v):
        if pd.isna(v) or v <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * (math.log10(v) - lmin) / span))

    tiles = "".join([
        f'<div class="tile"><div class="lbl">runs / task</div>'
        f'<div class="val">{summ["n_runs"]}</div></div>',
        f'<div class="tile"><div class="lbl">median {dimension} CV</div>'
        f'<div class="val">{num(summ["median_cv"])}</div></div>',
        f'<div class="tile"><div class="lbl">worst max/min</div>'
        f'<div class="val">{num(summ["worst_ratio"], ",.1f")}&times;</div></div>',
        f'<div class="tile"><div class="lbl">tasks &gt;2&times;</div>'
        f'<div class="val">{pct(summ["pct_ratio_gt_2x"])}</div></div>',
    ])
    if not pd.isna(summ["pct_flipped"]):
        tiles += (f'<div class="tile"><div class="lbl">outcome flipped</div>'
                  f'<div class="val">{pct(summ["pct_flipped"])}</div></div>')

    ticks = "".join(
        f'<span style="left:{100 * i / 3:.1f}%">{esc(fval(10 ** (lmin + span * i / 3)))}</span>'
        for i in range(4))
    axis = (f'<div class="sp-row"><div class="name"></div>'
            f'<div class="sp-axis">{ticks}</div><div class="val"></div></div>')

    rows = ""
    for _, s in ranked.iterrows():
        tid = s["task_id"]
        g = shown[shown["task_id"] == tid].sort_values("run_index")
        vmin, vmax = s[f"{dimension}_min"], s[f"{dimension}_max"]
        rng_tip = esc(f"{fval(vmin)}–{fval(vmax)} across {int(s['n_runs'])} runs")
        line = (f'<div class="sp-line" style="left:{xpct(vmin):.2f}%;'
                f'right:{100 - xpct(vmax):.2f}%" data-tip="{rng_tip}"></div>')
        mean = f'<div class="sp-mean" style="left:{xpct(s[f"{dimension}_mean"]):.2f}%"></div>'
        dots = ""
        for _, r in g.iterrows():
            v, c = r[col], r["correct"]
            cls = "na" if pd.isna(c) else ("good" if c > 0 else "bad")
            mark = "?" if pd.isna(c) else ("✓" if c > 0 else "✗")
            tip = f'run {int(r["run_index"])} · {fval(v)}'
            if is_cost and not pd.isna(r["thinking_tokens"]):
                tip += f' · {int(r["thinking_tokens"]):,} think tok'
            elif not is_cost and not pd.isna(r["cost"]):
                tip += f' · ${r["cost"]:.4f}'
            tip += f' · {mark}'
            dots += (f'<span class="sp-dot {cls}" style="left:{xpct(v):.2f}%" '
                     f'data-tip="{esc(tip)}"></span>')
        nc = s["n_correct"]
        badge = "" if (nc is None or pd.isna(nc)) else f' · {int(nc)}/{int(s["n_runs"])}✓'
        rlabel = f'{num(s[f"{dimension}_ratio_maxmin"], ",.1f")}&times;{badge}'
        rows += (f'<div class="sp-row"><div class="name">{esc(str(tid))}</div>'
                 f'<div class="sp-track">{line}{mean}{dots}</div>'
                 f'<div class="val">{rlabel}</div></div>')

    legend = ('<span><span class="key" style="background:var(--good)"></span>solved</span>'
              '<span><span class="key" style="background:var(--bad)"></span>failed</span>'
              '<span><span class="key" style="background:var(--muted)"></span>ungraded</span>'
              '<span>bar = min–max &middot; dashed = mean</span>')

    table = (f'<table><tr><th>task</th><th>runs</th><th>n&#10003;</th>'
             f'<th>{dimension} min</th><th>median</th><th>max</th>'
             f'<th>cv</th><th>max/min</th></tr>')
    for _, s in ranked.iterrows():
        nc = s["n_correct"]
        table += (f'<tr><td>{esc(str(s["task_id"]))}</td><td>{int(s["n_runs"])}</td>'
                  f'<td>{"—" if (nc is None or pd.isna(nc)) else int(nc)}</td>'
                  f'<td>{fval(s[f"{dimension}_min"])}</td>'
                  f'<td>{fval(s[f"{dimension}_median"])}</td>'
                  f'<td>{fval(s[f"{dimension}_max"])}</td>'
                  f'<td>{num(s[f"{dimension}_cv"])}</td>'
                  f'<td>{num(s[f"{dimension}_ratio_maxmin"], ",.1f")}&times;</td></tr>')
    table += "</table>"

    tid = str(task_id) if task_id is not None else (
        ranked["task_id"].iloc[0] if len(ranked) else None)
    detail = ""
    if tid is not None:
        g = recs[recs["task_id"] == tid]
        if len(g):
            first = g.iloc[0]
            q = (first["origin_query"] if isinstance(first["origin_query"], str)
                 else first["task_text"])
            if isinstance(q, str) and q:
                detail = (f'<div class="panel"><h2>Most variable task ({esc(str(tid))})</h2>'
                          f'<pre>{esc(q)}</pre></div>')

    mname = {"maxmin": "max/min ratio", "cv": "coefficient of variation",
             "std": "std"}[metric]
    sub = (f'{esc(benchmark)} &middot; {esc(model)} &middot; up to {summ["n_runs"]} '
           f'independent runs per task &middot; ranked by {esc(mname)}')
    headline = "bills" if is_cost else "thinking budgets"
    unit_foot = ("cost = platform-billed USD at the time of the run"
                 if is_cost else "thinking = hidden reasoning tokens")
    panel_title = "Cost" if is_cost else "Thinking-token"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(model)} on {esc(benchmark)} — repeated-trial spread</title>
<style>{_VIZ_CSS}{_SPREAD_CSS}</style></head>
<body class="viz-root"><div class="viz-wrap">
<h1>Same task, same model — unpredictable {headline}</h1>
<div class="sub">{sub}</div>
<div class="tiles">{tiles}</div>
<div class="panel"><h2>{panel_title} spread across repeated runs (log scale)</h2>
<div class="legend">{legend}</div>
{axis}{rows}</div>
<div class="panel"><h2>The numbers</h2>{table}</div>
{detail}
<div class="foot">each dot = one independent run of the same task &middot; {unit_foot}
&middot; data: huggingface.co/datasets/{HF_REPO_ID}</div>
</div><div id="tip"></div><script>{_VIZ_JS}</script></body></html>"""


def to_csv(df, path, notes=FIELD_NOTES):
    """Write the DataFrame plus a <path>_README.txt with field definitions."""
    df.to_csv(path, index=False)
    readme = str(path).rsplit(".", 1)[0] + "_README.txt"
    with open(readme, "w") as f:
        f.write(notes)
    print(f"wrote {path} ({len(df)} rows) and {readme}")


def _cli():
    p = argparse.ArgumentParser(
        prog="case_study.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Extract price-variability case studies from the price-reversal dataset\n"
            "(https://huggingface.co/datasets/price-reversal/price-reversal).\n\n"
            "The workflow:\n"
            "  0. `list`       see which benchmarks and models exist\n"
            "  1. `find`       rank tasks by the price reversal between two models\n"
            "                  (saves its selection for the next two commands)\n"
            "  2. `show`       the top task's full story as text: task, bills,\n"
            "                  answers, agent transcripts\n"
            "  3. `visualize`  the same story as a self-contained HTML page"
        ),
        epilog=(
            "examples:\n"
            "  python case_study.py list\n"
            "  python case_study.py find --model-a claude-haiku-4.5 --model-b gpt-5.4 --top-k 5\n"
            "  python case_study.py show                 # full story of the last find's top task\n"
            "  python case_study.py show --rank 2 --out case2.txt\n"
            "  python case_study.py visualize --out case.html"
        ))
    p.add_argument("--revision", default=DEFAULT_REVISION, metavar="COMMIT",
                   help="pin the HF dataset to a commit hash (default: latest on main)")
    p.add_argument("--main-zip", metavar="PATH",
                   help=f"local copy of {MAIN_ZIP} instead of downloading from HF")
    p.add_argument("--extras-zip", metavar="PATH",
                   help=f"local copy of {EXTRAS_ZIP} (task texts, transcripts, "
                        "answers) instead of downloading from HF")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def add(name, help):
        s = sub.add_parser(name, help=help, description=help)
        s.add_argument("--out", metavar="FILE.csv",
                       help="write the result to this CSV, plus field notes in "
                            "FILE_README.txt (default: print to terminal)")
        return s

    add("list", "Show what data exists: one row per (benchmark, model) with the "
        "number of tasks and, where available, the number of repeated runs.")

    s = add("show", "The full story of one task from the last `find`, no arguments "
            "needed: task text, every model's bill, final answers, and (multi-turn) "
            "the turn-by-turn transcript of what each model did. `find` saves its "
            f"selection to {LAST_FIND_FILE}; `show` reads it back.")
    s.add_argument("--rank", type=int, default=1, metavar="N",
                   help="which task from the last find to show, 1 = top (default: 1)")
    s.add_argument("--from", dest="from_file", default=LAST_FIND_FILE, metavar="FILE",
                   help="selection file written by `find` (default: %(default)s)")

    s = add("visualize", "Render one task as a self-contained HTML page comparing "
            "the models: billed cost, token breakdown (prompt/thinking/visible), "
            "answers or agent transcripts. By default takes the top task of the "
            "last `find` (like `show`); or name a task explicitly with "
            "--benchmark/--task-id/--models.")
    s.add_argument("--rank", type=int, default=1, metavar="N",
                   help="which task from the last find to render (default: 1)")
    s.add_argument("--from", dest="from_file", default=LAST_FIND_FILE, metavar="FILE",
                   help="selection file written by `find` (default: %(default)s)")
    s.add_argument("--benchmark", metavar="BENCH",
                   help="render this benchmark/task instead of the last find")
    s.add_argument("--task-id", metavar="ID")
    s.add_argument("--models", nargs="*", metavar="MODEL",
                   help="models to compare (with --benchmark; default: all 8)")

    s = add("find", "Rank tasks by the price reversal between model A and model B: "
            "reversal = (A's actual cost / B's actual cost) divided by (A's listed "
            "price / B's listed price), where listed price = input + output "
            "per-token price. 1.0 means spending matched the price sheet; larger "
            "means A overshot its listed price relative to B. Swap A and B to look "
            "for reversals in the other direction.")
    s.add_argument("--category", nargs="*", metavar="BENCH",
                   help="benchmark(s) to search, by dataset name as shown by "
                        "`list` (default: all 12)")
    s.add_argument("--model-a", required=True, metavar="MODEL",
                   help="model whose overspending you are looking for")
    s.add_argument("--model-b", required=True, metavar="MODEL",
                   help="reference model to compare against")
    s.add_argument("--query", metavar="TEXT",
                   help="only tasks whose text contains TEXT (case-insensitive), "
                        "e.g. --query email")
    s.add_argument("--correct", choices=["both", "neither", "a", "b"],
                   help="filter by outcome: both = A and B solved it (same "
                        "quality, different bill), neither = both failed, "
                        "a / b = only that model solved it (default: no filter)")
    s.add_argument("--top-k", type=int, default=10, metavar="N",
                   help="number of tasks to return (default: 10)")

    s = add("spread", "Analyze one model's repeated independent runs of the same "
            "tasks: how much the bill (or thinking tokens) swings run-to-run, and "
            "whether the outcome flips. Ranks tasks by that variability. --out "
            "FILE.html renders a log-scale distribution page, FILE.csv the per-task "
            "stats (+ notes), anything else (or no --out) prints a text report.")
    s.add_argument("--benchmark", required=True, metavar="BENCH",
                   help="benchmark with repeated trials (see `list`: repeated_runs>0)")
    s.add_argument("--model", required=True, metavar="MODEL",
                   help="model to inspect")
    s.add_argument("--dimension", choices=["cost", "thinking"], default="cost",
                   help="what to measure the spread of (default: cost; thinking = "
                        "hidden reasoning tokens, single-turn benchmarks only)")
    s.add_argument("--metric", choices=["maxmin", "cv", "std"], default="maxmin",
                   help="rank tasks by max/min ratio, coefficient of variation, or "
                        "standard deviation (default: maxmin)")
    s.add_argument("--flips-only", action="store_true",
                   help="keep only tasks whose correctness changed across runs")
    s.add_argument("--task-id", metavar="ID",
                   help="detail this task (default: the most variable one)")
    s.add_argument("--top-k", type=int, default=10, metavar="N",
                   help="number of tasks to rank (default: 10)")

    a = p.parse_args()
    data = PriceReversalData(revision=a.revision, main_zip=a.main_zip,
                             extras_zip=a.extras_zip)
    if a.cmd == "list":
        df = data.list_available()
    elif a.cmd == "show":
        b, t, models, header = _load_selection(a.from_file, a.rank)
        text = render_case(data, b, t, models, header=header)
        if a.out:
            with open(a.out, "w") as f:
                f.write(text)
            print(f"wrote {a.out}")
        else:
            print(text)
        return
    elif a.cmd == "visualize":
        if a.benchmark and a.task_id:
            b, t, models, header = a.benchmark, a.task_id, a.models, ""
        elif a.benchmark or a.task_id:
            sys.exit("pass both --benchmark and --task-id (or neither, to use "
                     "the last find)")
        else:
            b, t, models, header = _load_selection(a.from_file, a.rank)
        page = render_case_html(data, b, t, models, title_note=header)
        out = a.out or "case_study.html"
        with open(out, "w") as f:
            f.write(page)
        print(f"wrote {out} — open it in a browser")
        return
    elif a.cmd == "spread":
        try:
            if a.out and a.out.endswith(".html"):
                page = render_spread_html(data, a.benchmark, a.model,
                                          dimension=a.dimension, metric=a.metric,
                                          flips_only=a.flips_only, top_k=a.top_k,
                                          task_id=a.task_id)
                with open(a.out, "w") as f:
                    f.write(page)
                print(f"wrote {a.out} — open it in a browser")
            elif a.out and a.out.endswith(".csv"):
                st = data.find_variable(a.benchmark, a.model, a.dimension,
                                        a.metric, a.flips_only, a.top_k)
                to_csv(st, a.out, notes=SPREAD_FIELD_NOTES)
            else:
                text = render_spread(data, a.benchmark, a.model, a.dimension,
                                     a.metric, a.flips_only, a.top_k, a.task_id)
                if a.out:
                    with open(a.out, "w") as f:
                        f.write(text)
                    print(f"wrote {a.out}")
                else:
                    print(text)
        except (KeyError, RuntimeError) as e:
            sys.exit(e.args[0] if e.args else str(e))
        return
    else:
        df = data.find_examples(a.category, a.model_a, a.model_b,
                                a.query, a.top_k, a.correct)
        df.to_json(LAST_FIND_FILE, orient="records")
        print(f"(selection saved to {LAST_FIND_FILE} — run "
              f"`python case_study.py show` for the top task's full story)",
              file=sys.stderr)

    if a.out:
        to_csv(df, a.out)
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.max_colwidth", 60):
            print(df)


if __name__ == "__main__":
    _cli()
