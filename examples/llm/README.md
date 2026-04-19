# aiorch CLI — LLM pipeline catalog

15 example pipelines that exercise aiorch's LLM primitive end-to-end from the command line. No Platform mode, no MinIO, no Postgres — everything runs locally against whatever LLM provider you've configured, with run history and response cache stored in SQLite at `~/.aiorch/history.db`.

## Prerequisites

1. **aiorch CLI installed** — `aiorch --version` should return ≥ 0.4.1.
2. **An LLM provider API key exported to the shell.** The `aiorch.yaml` in this folder is pre-wired for **OpenRouter**, which routes to 100+ models under one key. Swap to your provider of choice by editing two lines.

```bash
# OpenRouter (default in the shipped aiorch.yaml):
export OPENROUTER_API_KEY=sk-or-v1-...

# Or, if you switch aiorch.yaml to a different provider:
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

3. **Run from this directory** so aiorch picks up `./aiorch.yaml`:

```bash
cd aiorch-cli/examples/llm
aiorch run 01-hello-llm.yaml
```

## How LLM provider config works

aiorch CLI walks up from the current directory looking for `aiorch.yaml` / `aiorch.yml` / `.aiorch.yaml`. This folder ships one pre-configured. Contents:

```yaml
llm:
  model: google/gemini-2.5-flash        # any litellm-supported string
  api_key: ${OPENROUTER_API_KEY}        # read from env, never hardcoded
  base_url: https://openrouter.ai/api/v1   # optional; set when routing through OpenRouter
```

That's the whole LLM config. No `provider:` field needed — it's inferred from the model string and the `base_url`. To swap providers, edit only these three lines. Per-pipeline / per-step overrides go in the pipeline YAML via `model:` / `temperature:` / `max_tokens:` on the step itself.

## Structure

```
cli_llm/
├── README.md                     ← this file
├── aiorch.yaml                   ← LLM provider + SQLite storage
├── 01-hello-llm.yaml             ← start here
├── ...
├── 15-changelog-drafter.yaml
└── inputs/
    ├── sample-prose.txt          ← for 02
    ├── headlines.json            ← for 07
    ├── sample-feedback.csv       ← for 09, 14
    ├── long-article.txt          ← for 11
    ├── sample-prs.json           ← for 12
    ├── sample-errors.log         ← for 13
    └── sample-commits.txt        ← for 15
```

---

## The 15 pipelines

### Tier 1 — Basic LLM primitives (01-04)

Start here. Each is 1-3 steps, no external data, runs in one second.

#### `01-hello-llm.yaml` — simplest possible LLM pipeline

No input, one prompt, one echo. First test of "is my provider wired?"

```bash
aiorch run 01-hello-llm.yaml
```

#### `02-summarize-text.yaml` — file input → LLM summary

Reads a text file from disk, asks the LLM for a 3-bullet summary.

```bash
aiorch run 02-summarize-text.yaml -i document=@./inputs/sample-prose.txt
```

#### `03-structured-extract.yaml` — JSON-schema-validated extraction

Extract people / organizations / dates / amounts from free text. `retry_on_invalid: 2` automatically asks the LLM to correct its output if it doesn't match the schema.

```bash
aiorch run 03-structured-extract.yaml
aiorch run 03-structured-extract.yaml -i text="Dana met Rohan in Tokyo on 2026-02-14 about a \$500K seed round."
```

#### `04-multi-model-compare.yaml` — same prompt, two models side-by-side

Compare a fast/cheap model vs a larger one. Edit the two `model:` lines to any pair your provider serves.

```bash
aiorch run 04-multi-model-compare.yaml
```

### Tier 2 — DAG shapes with LLM steps (05-08)

Every shape aiorch supports, demonstrated with LLM primitives.

#### `05-chain-refine.yaml` — chain of two LLM calls

Step 1 drafts, step 2 refines. Output of the first feeds the prompt of the second via Jinja interpolation.

```bash
aiorch run 05-chain-refine.yaml
aiorch run 05-chain-refine.yaml -i topic="onboarding docs for a new backend hire"
```

#### `06-parallel-perspectives.yaml` — fan-out + fan-in

Three parallel LLM calls each take a different perspective, a fourth synthesizes into a balanced recommendation. All three perspective calls run concurrently.

```bash
aiorch run 06-parallel-perspectives.yaml -i topic="mandatory code reviews"
```

#### `07-foreach-tagger.yaml` — foreach + LLM per item

One LLM call per list element, running in parallel.

```bash
aiorch run 07-foreach-tagger.yaml --input ./inputs/headlines.json
```

#### `08-classify-then-branch.yaml` — conditional routing

First LLM step classifies the input, then exactly one downstream branch runs based on that class. Skipped branches appear as SKIPPED (not failed) in the trace.

```bash
aiorch run 08-classify-then-branch.yaml
aiorch run 08-classify-then-branch.yaml -i message="The new dashboard looks great!"
```

### Tier 3 — Hybrid LLM + Python (09-11)

Where aiorch's "declarative DAG + tactical Python bodies" model pays off. LLM for semantic work, Python for deterministic work, both in the same pipeline.

#### `09-sentiment-scoring.yaml` — CSV → LLM per row → Python aggregation

Python extracts comments from a CSV, LLM classifies sentiment per row (parallel), Python aggregates into a positive / neutral / negative rollup with sample quotes.

```bash
aiorch run 09-sentiment-scoring.yaml -i feedback=@./inputs/sample-feedback.csv
```

#### `10-extract-then-validate.yaml` — LLM extract → Python validate

LLM extracts structured records, Python validates against domain rules the JSON schema can't express (age 0-120, valid email regex, date not in future or before 1950). Flags bad rows for human review.

```bash
aiorch run 10-extract-then-validate.yaml
```

#### `11-map-reduce-summarize.yaml` — split → parallel LLM map → LLM reduce

For documents larger than one LLM context window. Python chunks the text on paragraph boundaries, parallel foreach summarizes each chunk, a final LLM call reduces per-chunk summaries into a unified summary.

```bash
aiorch run 11-map-reduce-summarize.yaml -i document=@./inputs/long-article.txt
aiorch run 11-map-reduce-summarize.yaml -i document=@./inputs/long-article.txt -i chunk_size=800
```

### Tier 4 — Developer workflows (12-15)

Practical, re-runnable pipelines. Each ships with a hermetic sample so the default run works offline (the LLM call still needs network, obviously). To process real data, override the `*_path:` input.

#### `12-pr-triage.yaml` — LLM triage for a batch of PRs

Reads a JSON fixture shaped like GitHub's `/repos/<owner>/<repo>/pulls` response, runs LLM triage per PR (severity + area + reason), writes a sorted digest.

```bash
aiorch run 12-pr-triage.yaml
# real data:
# gh api '/repos/<owner>/<repo>/pulls' > my-prs.json
# aiorch run 12-pr-triage.yaml -i prs_path=./my-prs.json
```

Output: `inputs/pr-triage.json`

#### `13-error-log-triage.yaml` — nginx log → pattern dedupe → LLM root cause

Python parses and normalizes log lines, dedupes into the top 10 patterns by frequency. LLM assigns a root cause + severity + next action per pattern. Python writes the sorted triage report to disk.

```bash
aiorch run 13-error-log-triage.yaml
# real data:
# aiorch run 13-error-log-triage.yaml -i log_path=/var/log/nginx/error.log
```

Output: `inputs/error-triage-report.json`

#### `14-csv-enricher.yaml` — LLM-enrich every row of a CSV

"LLM as a column function." Each CSV row gets three new columns (sentiment, category, suggested_response) computed by an LLM call. Parallel foreach so all rows process concurrently. Writes the enriched CSV back to disk.

```bash
aiorch run 14-csv-enricher.yaml -i feedback=@./inputs/sample-feedback.csv
```

Output: `inputs/feedback-enriched.csv`

#### `15-changelog-drafter.yaml` — commit subjects → LLM release notes

Read a file of commit subjects, LLM groups them into conventional-commit categories (features / fixes / perf / security / refactor / docs / breaking), drafts plain-English bullets per category. Python writes a markdown changelog.

```bash
aiorch run 15-changelog-drafter.yaml
# real data:
# git log v0.3.0..HEAD --no-merges --pretty=format:'%s' > my-commits.txt
# aiorch run 15-changelog-drafter.yaml -i commits_path=my-commits.txt
```

Output: `inputs/CHANGELOG-draft.md`

---

## Useful CLI flags

```bash
aiorch run pipeline.yaml --dry           # show plan, don't execute (skips LLM calls)
aiorch run pipeline.yaml -v              # verbose — show step inputs and outputs
aiorch run pipeline.yaml --step <name>   # run a single step
aiorch run pipeline.yaml --from <name>   # resume from this step
aiorch validate pipeline.yaml            # syntax check, no run
aiorch visualize pipeline.yaml           # ASCII DAG diagram
aiorch plan pipeline.yaml                # show DAG layers + cost estimate
```

## Tips

- **LLM response caching is on by default** for CLI mode. Re-running an identical step with the same inputs hits the local SQLite cache at `~/.aiorch/history.db` instead of re-calling the model — useful when iterating on a downstream step without paying for the earlier ones again.
- **Cost estimate before running:** `aiorch plan 11-map-reduce-summarize.yaml -i document=@./inputs/long-article.txt` shows the DAG layers plus a rough token / cost estimate.
- **Verbose mode for debugging prompts:** `aiorch run ... -v` prints each LLM step's prompt and response to the terminal.
- **Override the model per step:** add `model: claude-sonnet-4-5` (or whatever) to any step's YAML. The pipeline YAML wins over `aiorch.yaml`.

## Summary table

| # | File | Inputs | Shape |
|---|---|---|---|
| 01 | `01-hello-llm.yaml` | — | 1 LLM + echo |
| 02 | `02-summarize-text.yaml` | text file | LLM summary |
| 03 | `03-structured-extract.yaml` | string | JSON-schema-validated extraction |
| 04 | `04-multi-model-compare.yaml` | string | 2 models parallel, side-by-side output |
| 05 | `05-chain-refine.yaml` | string | LLM chain of 2 |
| 06 | `06-parallel-perspectives.yaml` | string | 3 parallel + 1 reduce |
| 07 | `07-foreach-tagger.yaml` | list | foreach + LLM per item |
| 08 | `08-classify-then-branch.yaml` | string | classify + 3 conditional branches |
| 09 | `09-sentiment-scoring.yaml` | csv | foreach LLM + Python aggregate |
| 10 | `10-extract-then-validate.yaml` | string | LLM extract + Python validate |
| 11 | `11-map-reduce-summarize.yaml` | text file | split + parallel map + reduce |
| 12 | `12-pr-triage.yaml` | json | foreach LLM triage + write report |
| 13 | `13-error-log-triage.yaml` | log file | Python parse + LLM root-cause per pattern |
| 14 | `14-csv-enricher.yaml` | csv | foreach LLM + write enriched csv |
| 15 | `15-changelog-drafter.yaml` | text | LLM grouping + markdown changelog |

**Total: 15 pipelines, all CLI-runnable, SQLite storage, zero Platform dependencies.**
