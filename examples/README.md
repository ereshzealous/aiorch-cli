# aiorch — Examples

A curated catalog of **72 runnable pipelines** covering every aiorch primitive, DAG shape, and real-world developer workflow.

```
examples/
├── README.md                ← you are here
├── llm/                     ← 30 pipelines that use an LLM
│   ├── README.md            ← per-pipeline walkthroughs
│   ├── aiorch.yaml          ← LLM provider + storage config
│   ├── 01-hello-llm.yaml
│   ├── …
│   └── inputs/              ← sample fixtures
└── core/                    ← 42 pipelines with zero LLM dependency
    ├── README.md            ← per-pipeline walkthroughs
    ├── 01-smoke-test.yaml
    ├── …
    └── inputs/              ← sample fixtures
```

- **Start in `core/`** if you want to learn aiorch's DAG/primitive model first — no API key required.
- **Start in `llm/`** if you just want to see what pipelines look like with an LLM in the loop.

---

## 🔑 Handling secrets

aiorch **never wants a secret hardcoded in a YAML file.** Every example expects provider keys to come from environment variables and references them via `${VAR_NAME}` interpolation.

### Setting provider API keys

```bash
# Pick whichever provider matches your `llm.api_key:` setting in aiorch.yaml.
export OPENROUTER_API_KEY=sk-or-v1-...     # OpenRouter (multi-provider, recommended)
export OPENAI_API_KEY=sk-...                # direct OpenAI
export ANTHROPIC_API_KEY=sk-ant-...         # direct Anthropic
export GOOGLE_API_KEY=...                   # direct Google AI
```

You can also put these in a `.env` file in the directory you run from. aiorch reads it automatically if `python-dotenv` is on the path (it is, by default).

### Referencing secrets in `aiorch.yaml`

```yaml
llm:
  api_key: ${OPENROUTER_API_KEY}           # resolved at run time
  base_url: https://openrouter.ai/api/v1   # optional, when routing through OpenRouter
  model: google/gemini-2.5-flash
```

If `${OPENROUTER_API_KEY}` is unset, aiorch fails fast with a clear error rather than sending an empty Authorization header.

### Secrets inside pipelines (Python / shell steps)

`python:` and `run:` steps inherit your shell env, so `os.environ["DB_PASSWORD"]` or `$DB_PASSWORD` just works. No special secrets syntax needed in CLI mode.

---

## 🤖 Choosing a model

Three places model can be set, in precedence order (highest wins):

1. **Step-level** — `model:` on an individual pipeline step.
2. **Pipeline-level** — inherited from `aiorch.yaml`'s `llm.model` if the step doesn't override.
3. **Default** — aiorch falls back to a sensible default if neither is set.

### In `aiorch.yaml`

```yaml
llm:
  model: google/gemini-2.5-flash
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
```

Any [LiteLLM-supported](https://docs.litellm.ai/docs/providers) model string works. Common ones:

| Provider | Example model string |
|---|---|
| OpenAI | `gpt-4o-mini`, `gpt-4o` |
| Anthropic | `claude-sonnet-4-20250514`, `claude-3-5-haiku-20241022` |
| Google AI | `google/gemini-2.5-flash`, `google/gemini-2.5-pro` |
| OpenRouter | `anthropic/claude-3-5-haiku`, `meta-llama/llama-3.1-70b-instruct` |
| Ollama | `ollama/llama3`, `ollama/mistral` |
| Bedrock | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` |

### Per-step override in a pipeline

```yaml
steps:
  draft:
    prompt: "Write a first draft on {{topic}}"
    model: gpt-4o-mini              # cheap and fast for drafts

  refine:
    prompt: "Improve this draft: {{draft}}"
    model: claude-sonnet-4-20250514 # better reasoning for refinement
```

Useful for multi-model comparisons (see `llm/04-multi-model-compare.yaml`) or cost-tuning individual steps.

---

## 📥 Passing inputs

aiorch accepts inputs three ways. They compose — `-i` beats `--input` beats the YAML's `input:` defaults.

### 1. Inline scalars: `-i KEY=VALUE`

```bash
aiorch run core/02-inputs.yaml -i name=Eresh -i repeat=3 -i greeting=Hello
```

Type detection is automatic:

| You pass | aiorch sees |
|---|---|
| `-i count=42` | int `42` |
| `-i temp=0.7` | float `0.7` |
| `-i name=hello` | str `"hello"` |

**Booleans and lists must use option 3 below** — the scalar parser doesn't auto-detect those.

### 2. File inputs: `-i KEY=@./path`

The `@` prefix uploads a file to the local artifact store. Extension picks the format:

| Extension | Format | Pipeline receives |
|---|---|---|
| `.txt`, `.md`, `.rst` | text | `str` |
| `.json` | json | `dict` / `list` (parsed) |
| `.yaml`, `.yml` | json | `dict` / `list` (parsed) |
| `.csv` | csv | `list[dict]` |
| anything else | binary | `bytes` |

```bash
aiorch run core/07-input-artifact-text.yaml -i document=@./core/inputs/sample.txt
aiorch run core/12-input-artifact-csv.yaml  -i sales=@./core/inputs/sales.csv
aiorch run llm/02-summarize-text.yaml       -i document=@./llm/inputs/sample-prose.txt
```

### 3. Bulk JSON: `--input file.json` (or `--input '{...}'`)

For booleans, lists, nested dicts, or to keep inputs in version control:

```bash
echo '{"fruits": ["kiwi", "mango", "pear"]}' > /tmp/fruits.json
aiorch run core/06-input-list.yaml --input /tmp/fruits.json

aiorch run core/27-multi-input-typed-dashboard.yaml \
  --input ./core/inputs/batch-inputs.json \
  -i fiscal_year=2027
```

---

## 🚀 Running the examples

From the repo root (or anywhere you've exported the required env vars):

```bash
# Core pipelines — no LLM required
aiorch run examples/core/01-smoke-test.yaml
aiorch run examples/core/27-multi-input-typed-dashboard.yaml --input examples/core/inputs/batch-inputs.json

# LLM pipelines — cd in first so ./aiorch.yaml is picked up
cd examples/llm
aiorch run 01-hello-llm.yaml
aiorch run 11-map-reduce-summarize.yaml -i document=@./inputs/long-article.txt
```

> **Why `cd examples/llm` first?** aiorch auto-discovers `aiorch.yaml` by walking up from the current directory. The LLM examples ship their own provider config at `examples/llm/aiorch.yaml`, so running from there picks up OpenRouter + Gemini 2.5 Flash by default. Running from the repo root would miss it.

### Useful CLI flags

```bash
aiorch run pipeline.yaml --dry              # show plan, don't execute (skip LLM calls)
aiorch run pipeline.yaml -v                 # verbose — print each step's input/output
aiorch run pipeline.yaml --step <name>      # run a single step
aiorch run pipeline.yaml --from <name>      # resume from this step
aiorch validate pipeline.yaml               # schema + template lint
aiorch list-steps pipeline.yaml             # print the DAG
aiorch history                              # list recent runs
aiorch trace <run-id>                       # step-by-step timeline for one run
```

---

## 📚 Per-pipeline walkthroughs

- [`examples/llm/README.md`](llm/README.md) — every LLM pipeline explained, grouped by tier (basic primitives → DAG shapes → hybrid LLM+Python → developer workflows).
- [`examples/core/README.md`](core/README.md) — every core pipeline explained, grouped by concern (basic primitives & inputs → DAG shapes → database access → production-shaped patterns → developer utilities).

Each walkthrough has runnable commands, expected outputs, and notes on what each example demonstrates.

---

## 🧪 One-time setup for database examples

Three of the core pipelines (`33`, `34`, `35`) read from a SQLite fixture. Run the setup script once:

```bash
cd examples/core
./inputs/setup-sample-db.sh
```

This creates `examples/core/inputs/sample.db` with 8 customers, 6 products, and 15 orders. Idempotent — safe to re-run.

---

## 💡 Tips

- **LLM response caching is on by default** in CLI mode. Re-running a pipeline step with identical inputs hits `~/.aiorch/history.db` instead of re-calling the model — great for iterating on a downstream step without paying for the upstream ones again.
- **Estimate cost before a big run:** `aiorch plan 11-map-reduce-summarize.yaml -i document=@./inputs/long-article.txt` shows the DAG layers and a rough token/cost estimate.
- **Debug prompts with `-v`** — it prints each LLM step's exact prompt and raw response to the terminal.
- **Override the model per step** by adding `model: claude-sonnet-4-20250514` (or any other LiteLLM string) to the step's YAML. Pipeline YAML wins over `aiorch.yaml`.
