# aiorch CLI — non-LLM pipeline catalog

42 example pipelines that exercise every aiorch primitive and DAG shape, all without LLMs. Every pipeline in this directory runs cleanly via `aiorch run` from the command line — pipelines that needed Platform mode (MinIO connectors, webhook triggers, Postgres-backed connector store) have been moved out.

## Quick orientation

```
cli/
├── README.md                       ← this file
├── 01-smoke-test.yaml              ← simplest possible pipeline
├── 02-inputs.yaml
├── ... (39 yaml files)
├── 50-changelog-from-git.yaml
└── inputs/                         ← sample input files for -i @./path
    ├── sample.txt
    ├── prose.txt
    ├── code-with-todos.txt
    ├── order.json
    ├── sales.csv
    ├── tiny.png
    ├── batch-inputs.json
    ├── diamond-inputs.json
    ├── setup-sample-db.sh          ← creates sample.db for pipelines 33-35
    └── sample.db                   ← SQLite DB (created by setup script)
```

The numbering is sparse (jumps from 15 → 18, 18 → 21, 21 → 25, 25 → 27) because the catalog originally had 32 pipelines and the missing numbers were Platform-mode-only examples (MinIO connectors, webhook triggers) that don't run from the CLI. Those have been removed from this directory; they live in the project's `no_llm/` reference directory if you ever want to see them.

## How to run any pipeline

```bash
cd aiorch-cli/examples/core
aiorch run 01-smoke-test.yaml
```

That's the shape. The CLI reads the YAML, builds the DAG, runs every step locally in your terminal, and prints output as it goes.

## Passing inputs from the command line

aiorch's CLI takes inputs three ways:

### 1. Inline scalars with `-i KEY=VALUE`

```bash
aiorch run 02-inputs.yaml -i greeting=Hello -i repeat=3 -i name=Eresh
```

| Format | Becomes |
|---|---|
| `-i count=42` | int 42 |
| `-i temp=0.7` | float 0.7 |
| `-i name=hello` | str "hello" |

Type detection is automatic — numeric strings become numbers, everything else is a string. **Booleans and lists must use option 3 below** because the CLI doesn't auto-detect those.

### 2. File inputs with `-i KEY=@./path`

The `@` prefix uploads a file to the local artifact store. Extension picks the format:

| Extension | Format | Pipeline sees |
|---|---|---|
| `.txt`, `.md`, `.rst` | text | `inputs["doc"]` is a `str` |
| `.json` | json | `inputs["doc"]` is a `dict`/`list` |
| `.yaml`, `.yml` | json | parsed YAML as `dict`/`list` |
| `.csv` | csv | `inputs["doc"]` is `list[dict]` |
| anything else | binary | `inputs["doc"]` is `bytes` |

Examples:

```bash
aiorch run 07-input-artifact-text.yaml -i document=@./inputs/sample.txt
aiorch run 11-input-artifact-json.yaml -i order=@./inputs/order.json
aiorch run 12-input-artifact-csv.yaml -i sales=@./inputs/sales.csv
aiorch run 13-input-artifact-binary.yaml -i file=@./inputs/tiny.png
```

### 3. Bulk inputs file with `--input file.json`

For booleans, lists, deeply nested dicts, or when you want to keep your inputs in version control:

```bash
aiorch run 27-multi-input-typed-dashboard.yaml --input ./inputs/batch-inputs.json
```

The contents of `batch-inputs.json`:

```json
{
  "report_title": "Q2 2026 Dashboard",
  "fiscal_year": 2026,
  "tax_rate": 0.075,
  "show_details": true,
  "categories": ["Hardware", "Cloud", "Travel"]
}
```

You can also pass inline JSON: `--input '{"key": "value"}'`.

### Combining the three

`-i` overrides `--input` overrides defaults from the YAML's `input:` block. So the typical pattern is:

```bash
aiorch run pipeline.yaml --input ./inputs/base.json -i fiscal_year=2027
```

— load the base from a file, override one field on the command line.

## Other useful CLI flags

```bash
aiorch run pipeline.yaml --dry           # show plan, don't execute
aiorch run pipeline.yaml -v              # verbose — show step inputs and outputs
aiorch run pipeline.yaml --step name     # run a single step
aiorch run pipeline.yaml --from name     # resume from this step
aiorch validate pipeline.yaml            # syntax check, no run
aiorch list pipeline.yaml                # list all steps in the DAG
aiorch visualize pipeline.yaml           # ASCII DAG diagram
aiorch plan pipeline.yaml                # show DAG layers + cost estimate
```

---

# The 42 pipelines

## Group 1 — basic primitives and input types (01-15)

These are the smallest examples — one or two steps, one input type each. Run them in order if you're new.

### 01-smoke-test.yaml — three shell echoes, no inputs

The simplest possible pipeline. Three `run:` steps that echo strings and pass values via Jinja.

```bash
aiorch run 01-smoke-test.yaml
```

No inputs. Should finish in under a second.

---

### 02-inputs.yaml — three string inputs, shell + Jinja

Demonstrates how user inputs flow into shell commands.

```bash
# All defaults
aiorch run 02-inputs.yaml

# Override the name
aiorch run 02-inputs.yaml -i name=Eresh -i repeat=5 -i greeting=Hello
```

Required input: `name` (no default). Defaults exist for `greeting` and `repeat`.

---

### 03-input-integer.yaml — integer input with min/max bounds

Tests integer type validation with `minimum: 1`, `maximum: 10`.

```bash
aiorch run 03-input-integer.yaml -i count=5
aiorch run 03-input-integer.yaml -i count=11   # rejected — exceeds max
```

---

### 04-input-boolean.yaml — boolean input + shell branching

Boolean must come via `--input` since the CLI scalar parser doesn't auto-detect bools.

```bash
echo '{"verbose": true}'  > /tmp/yes.json
echo '{"verbose": false}' > /tmp/no.json

aiorch run 04-input-boolean.yaml --input /tmp/yes.json
aiorch run 04-input-boolean.yaml --input /tmp/no.json
```

---

### 05-input-number.yaml — float input with min/max + awk classifier

```bash
aiorch run 05-input-number.yaml -i temperature=0.7
aiorch run 05-input-number.yaml -i temperature=0.1     # deterministic zone
aiorch run 05-input-number.yaml -i temperature=0.95    # creative zone
aiorch run 05-input-number.yaml -i temperature=1.5     # rejected
```

---

### 06-input-list.yaml — list input + foreach iteration

Lists must come via `--input` (CLI scalar parser doesn't handle JSON arrays).

```bash
echo '{"fruits": ["kiwi", "mango", "pear"]}' > /tmp/fruits.json
aiorch run 06-input-list.yaml --input /tmp/fruits.json
```

The `process_each` step uses `foreach: "{{fruits}}"` and runs once per item.

---

### 07-input-artifact-text.yaml — file input as text

The first artifact pipeline. Pass a `.txt` or `.md` file via the `@` syntax.

```bash
aiorch run 07-input-artifact-text.yaml -i document=@./inputs/sample.txt
aiorch run 07-input-artifact-text.yaml -i document=@./inputs/prose.txt
```

The pipeline reads the file's content as a UTF-8 string and prints character count + the body.

---

### 08-text-document-metrics.yaml — multi-step shell document analysis

Real-world workflow: stage file → 4 parallel analysis steps (wc, top words, longest line, TODO scan) → aggregate.

```bash
aiorch run 08-text-document-metrics.yaml -i document=@./inputs/prose.txt
aiorch run 08-text-document-metrics.yaml -i document=@./inputs/code-with-todos.txt
```

All shell + standard Unix tools. No Python.

---

### 09-text-document-metrics-python.yaml — same workflow, single python step

Cleaner version of 08 using the new `python:` primitive instead of shell + awk.

```bash
aiorch run 09-text-document-metrics-python.yaml -i document=@./inputs/prose.txt
```

Compare the trace timeline against 08 — fewer steps, single primitive.

---

### 10-python-primitive-smoke.yaml — verifies `python:` works end-to-end

Three string/integer inputs feeding a Python step that prints + emits a structured result.

```bash
aiorch run 10-python-primitive-smoke.yaml -i name=Eresh -i count=7 -i phrase="hello world from aiorch"
```

---

### 11-input-artifact-json.yaml — JSON file input parsed to dict

```bash
aiorch run 11-input-artifact-json.yaml -i order=@./inputs/order.json
```

The python step receives `inputs["order"]` as a real Python dict (already parsed). Computes order subtotal, tax, total, top product, etc.

---

### 12-input-artifact-csv.yaml — CSV file input parsed to list[dict]

```bash
aiorch run 12-input-artifact-csv.yaml -i sales=@./inputs/sales.csv
```

The python step gets `inputs["sales"]` as `list[dict]` via `csv.DictReader`. Computes per-product totals + per-category totals.

**Gotcha demonstrated:** csv.DictReader returns every value as a string. The pipeline casts `quantity` to int and `price` to float at the top of the analyze step.

---

### 13-input-artifact-binary.yaml — raw bytes input

```bash
aiorch run 13-input-artifact-binary.yaml -i file=@./inputs/tiny.png
```

Pipeline computes SHA-256, detects the file type by magic-number sniffing, prints a hex dump of the first 16 bytes. Try with any binary file.

---

### 14-input-http.yaml — http input with `format: text`

Auto-resolves a URL at run start. No CLI input needed — the URL is hardcoded in the YAML.

```bash
aiorch run 14-input-http.yaml
```

Hits `https://api.github.com/zen` and computes char/word stats on the response.

**Requires outbound network access from your laptop.**

---

### 15-input-http-json.yaml — http input with `format: json`

```bash
aiorch run 15-input-http-json.yaml
```

Hits `https://api.github.com/repos/anthropics/anthropic-sdk-python` and reports stars/forks/license/days-since-push from the parsed JSON.

---

## Group 2 — DAG shapes and patterns (18, 21, 25, 27-32)

These exercise non-trivial DAG shapes (parallel, diamond, foreach, fan-out, multi-layer). All run cleanly in CLI mode.

### 18-parallel-fanout-fanin.yaml — 4 workers in parallel + join

```bash
aiorch run 18-parallel-fanout-fanin.yaml
```

One stage step, four parallel python workers each computing a different metric, one final aggregator. Total wall clock should be ~600ms (one worker's sleep), not ~2s (4 sequential sleeps). If you see ~2s, the parallel dispatch isn't working and that's a real bug to flag.

---

### 21-failure-isolation-with-retry.yaml — retry + condition + try/except

```bash
aiorch run 21-failure-isolation-with-retry.yaml -i trigger_expensive=true
```

Demonstrates three failure-handling patterns:
1. `retry: 5` with `retry_delay: 1s` for transient failures
2. Try/except inside a python step (graceful catch)
3. `condition:` to skip a step based on upstream output

Run a few times — the `flaky` step has a 70% chance of failing per attempt, so the retry count varies between runs. Boolean inputs need a JSON file:

```bash
echo '{"trigger_expensive": false}' > /tmp/no-expensive.json
aiorch run 21-failure-isolation-with-retry.yaml --input /tmp/no-expensive.json
```

---

### 25-health-check-multi.yaml — 3 parallel HTTP health checks

```bash
aiorch run 25-health-check-multi.yaml
```

Three python steps, each hits one endpoint via `httpx` directly. They run in parallel (one DAG layer). Fails the run if any endpoint is red.

**Requires outbound network access AND aiorch's web UI running locally** (one of the checks targets `http://localhost:5173/api/health`). If you're not running the UI, that check will fail and the gate will reject — that's the expected behavior, not a bug.

---

### 27-multi-input-typed-dashboard.yaml — every user-input type in one pipeline

The reference card. Uses string, integer (with bounds), number (with bounds), boolean, and list inputs together.

```bash
# All defaults
aiorch run 27-multi-input-typed-dashboard.yaml --input ./inputs/batch-inputs.json

# Override one field
aiorch run 27-multi-input-typed-dashboard.yaml --input ./inputs/batch-inputs.json -i fiscal_year=2027

# Test integer bounds (will fail with "must be >= 2020")
aiorch run 27-multi-input-typed-dashboard.yaml --input ./inputs/batch-inputs.json -i fiscal_year=1999
```

---

### 28-diamond-dependency.yaml — diamond DAG (1→2→1)

```bash
aiorch run 28-diamond-dependency.yaml -i data_size=1000
aiorch run 28-diamond-dependency.yaml --input ./inputs/diamond-inputs.json
```

Source → [normalize, summary] → merge. The two middle branches read the same source and run in parallel. Verify in trace that they overlap.

---

### 29-nested-foreach-aggregate.yaml — foreach + inner python loop

Per-region sales aggregation. Outer foreach iterates regions; inner Python loop iterates transactions in each region.

```bash
aiorch run 29-nested-foreach-aggregate.yaml
```

No external inputs needed — the source data is hardcoded in the first step.

---

### 30-map-reduce-text.yaml — split → map(parallel foreach) → reduce

```bash
aiorch run 30-map-reduce-text.yaml -i chunk_count=4
aiorch run 30-map-reduce-text.yaml -i chunk_count=8
```

Classic map/reduce. The `map` step uses `parallel: true` on its foreach so the chunks process concurrently. Toggle that flag to `false` and watch the trace timeline change from overlapping → stacked.

To pass a custom text via CLI, use the `--input` flag with a JSON file:

```bash
echo '{"text": "your custom text here ...", "chunk_count": 3}' > /tmp/text.json
aiorch run 30-map-reduce-text.yaml --input /tmp/text.json
```

---

### 31-conditional-multi-branch.yaml — string-input routing

```bash
aiorch run 31-conditional-multi-branch.yaml -i mode=fast
aiorch run 31-conditional-multi-branch.yaml -i mode=thorough
aiorch run 31-conditional-multi-branch.yaml -i mode=skip
aiorch run 31-conditional-multi-branch.yaml -i mode=invalid     # rejected
```

Three branches, exactly one runs per invocation based on the `mode` input. Other two branches show as "skipped" in the trace (NOT failed).

---

### 32-compute-graph-numerical.yaml — wide multi-layer numerical DAG

```bash
aiorch run 32-compute-graph-numerical.yaml -i a=3 -i b=4 -i c=5 -i d=2
```

Four scalar inputs feed three layers of parallel ops. Open the trace timeline — you should see three distinct layers (4 parallel L1 ops, 3 parallel L2 ops, 1 final).

---

## Group 3 — Database access (33-35)

These three pipelines read from a SQLite database via Python's stdlib `sqlite3` inside `python:` steps. **They run entirely from the CLI** — no aiorch connector, no Platform mode, no Postgres setup required.

The trick: `python:` steps run in the executor's interpreter, so anything you can `import` from the executor's environment is available. `sqlite3` is stdlib so it always works. The same pattern works for `psycopg2`/`asyncpg` (Postgres), `mysql.connector` (MySQL), `redis` (Redis), or any other client library you have installed in the executor's Python.

### One-time setup

```bash
cd aiorch-cli/examples/core
./inputs/setup-sample-db.sh
```

Creates `inputs/sample.db` with three tables (8 customers, 6 products, 15 orders). Idempotent — safe to re-run.

---

### 33-db-read-basic.yaml — simple SELECT, single python step

The warmup. One python step opens the DB, fetches recent orders, prints them.

```bash
aiorch run 33-db-read-basic.yaml
aiorch run 33-db-read-basic.yaml -i limit=5
aiorch run 33-db-read-basic.yaml -i limit=20
```

Confirms your SQLite DB is set up correctly before you try the bigger pipelines.

---

### 34-db-parallel-queries.yaml — five SQL queries running in parallel

The centerpiece. Five independent aggregation queries, each in its own `python:` step, all in the same DAG layer so the runtime dispatches them concurrently:

1. Top customers by total order volume
2. Per-region revenue (joined with customers + products)
3. Per-category sales counts (joined with products)
4. Per-day order counts (time series)
5. Top products by revenue (joined with orders)

```bash
aiorch run 34-db-parallel-queries.yaml
```

Watch the trace output — all five queries report `▶ start` before any of them report `✓ done`. That's parallel dispatch via `asyncio.gather`. Same shape as pipeline 18 (synthetic parallel workers) but with REAL work instead of `time.sleep`.

The aggregate fan-in step depends on all five queries, so it doesn't run until they all complete.

**Why each step opens its own connection:** SQLite connections aren't safely shareable across coroutines. The cleanest pattern for parallel SQLite is one connection per step. For Postgres / MySQL with proper async drivers (`asyncpg`, `aiomysql`), you'd want a connection pool — but the per-step connection pattern still works.

---

### 35-db-join-aggregate.yaml — multi-table read, in-memory join, write back to disk

Classic ETL: extract from N sources, join in Python (not SQL), aggregate, load into a JSON file.

```bash
aiorch run 35-db-join-aggregate.yaml
cat aiorch-cli/examples/core/inputs/sales-summary.json | jq
```

DAG shape:

```
read_orders ─┐
read_customers ─┼──► join_in_python ──► aggregate ──► write_summary ──► show
read_products ─┘
```

The first three reads run in parallel (one DAG layer), then `join_in_python` does a hash-join in memory across all three, `aggregate` builds a per-region per-category rollup, and `write_summary` dumps the result to `inputs/sales-summary.json` via plain `pathlib`.

**Why join in Python instead of SQL?** Sometimes you can't — the data is from different databases, mixed with API responses or uploaded files, or the join shape is awkward in SQL. Python is the universal join surface. For million-row workloads, swap out the dict-based hash-join for `polars` or `duckdb` — same step shape, same DAG.

---

## Adapting to other databases

The three DB pipelines above use SQLite for zero-setup demos, but the same pattern works for any database with a Python driver. Swap the import + connection setup:

**Postgres (asyncpg)** — note `asyncpg` is async, so you need `asyncio.run` inside the python step:

```python
import asyncio
import asyncpg

async def fetch():
    conn = await asyncpg.connect("postgresql://user:pw@localhost:5432/db")
    rows = await conn.fetch("SELECT * FROM orders LIMIT 10")
    await conn.close()
    return [dict(r) for r in rows]

result = asyncio.run(fetch())
```

**Postgres (psycopg2)** — sync, simpler:

```python
import psycopg2
import psycopg2.extras

conn = psycopg2.connect("postgresql://user:pw@localhost:5432/db")
cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
cursor.execute("SELECT * FROM orders LIMIT 10")
result = cursor.fetchall()
cursor.close()
conn.close()
```

**MySQL:**

```python
import mysql.connector
conn = mysql.connector.connect(host="localhost", user="root", password="...", database="...")
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM orders LIMIT 10")
result = cursor.fetchall()
cursor.close()
conn.close()
```

**Redis:**

```python
import redis
r = redis.Redis(host="localhost", port=6379)
result = {
    "keys": [k.decode() for k in r.keys("user:*")[:10]],
    "count": r.dbsize(),
}
```

In every case, the same parallel-queries pattern (one query per `python:` step in the same DAG layer) gives you concurrent reads. For Postgres specifically, `asyncpg` connection pools make this very efficient — you can dispatch 50 parallel queries from 50 python steps with negligible overhead.

**Connection credentials:** for production use, don't hardcode them. Either:
- Set environment variables (`os.environ["DB_PASSWORD"]`) and reference them in the step
- Read from a config file via `pathlib`
- (Platform mode only) declare `secrets:` on the step and use `os.environ["DB_PASSWORD"]` — aiorch injects workspace secrets into the subprocess env

---

## Group 4 — Production-shaped patterns (36-39)

Heavier on aiorch primitives, less Python boilerplate. Closer to what real production pipelines look like.

### 36-webhook-api-poll.yaml — pull-based polling of a REST API

The pull equivalent of an inbound webhook trigger. Hits GitHub's events API for the Anthropic SDK repo, filters to a chosen event type, processes each event in a foreach loop, and reports a summary.

```bash
aiorch run 36-webhook-api-poll.yaml
aiorch run 36-webhook-api-poll.yaml -i max_events=10
aiorch run 36-webhook-api-poll.yaml -i event_filter=PushEvent
```

Demonstrates: `python: + httpx` for the API call, `retry`/`retry_delay` for transient blips, `foreach` over the response, `condition:` to short-circuit when there's nothing to process. **Real production pattern** — schedule this with `aiorch schedule add ... --cron '*/5 * * * *'` and you have a polling worker.

**Foreach gotcha to know about:** `foreach: "{{filtered.events}}"` doesn't work because aiorch's foreach resolver only handles top-level variables, not nested paths. The pipeline includes a small `lift_events` step that re-publishes the nested list as a top-level variable so foreach can iterate it. Same workaround applies to pipelines 29 and 39.

---

### 37-multi-source-fanout.yaml — five sources joined in one pipeline

Reads from 5 different sources in parallel: a public HTTP API (`type: http`), a SQLite database (`python: + sqlite3`), a local file (`python: + pathlib`), environment variables (`python: + os.environ`), and a computed value (pure Python). All five run as one DAG layer (concurrent), then a merge step joins them into a unified dashboard.

```bash
aiorch run 37-multi-source-fanout.yaml
```

Reference card for "how do I read from X in a CLI pipeline." Copy the source step you need into your own pipeline.

**Prerequisite:** run `inputs/setup-sample-db.sh` once (for the SQLite source).

---

### 38-retry-strategies.yaml — four retry patterns side by side

Four parallel branches, each demonstrating a different resilience strategy:

1. **Aggressive fast retry** (`retry: 10, retry_delay: 100ms`) — for transient network blips
2. **Patient backoff** (`retry: 3, retry_delay: 2s`) — for slow upstream recovery
3. **Graceful try/except** (no retry, never fails) — for best-effort calls
4. **Internal retry-then-default** (Python loop with exponential backoff) — for fault-tolerant fallback

```bash
aiorch run 38-retry-strategies.yaml
```

The flaky steps use random failures, so run it 3-4 times to see different behaviors. The strategies are designed so each shows distinct trace timing.

**Aiorch's retry is fixed-delay** (no exponential backoff in the runtime). Strategy 4 demonstrates how to do exponential backoff inside a python step when you need it.

---

### 39-pipeline-as-orchestrator.yaml — production-shaped 10-step chain

Long sequential workflow that mirrors what real production pipelines look like:

1. **fetch** — pull from external API (with retry)
2. **validate_response** — schema check, fail loud
3. **extract_records** — trim to relevant fields
4. **enrich** — compute derived fields per record
5. **partition_by_type** — split into Issues vs PRs
6. **process_each_partition** — foreach + per-partition compute
7. **reconcile** — integrity check (record count round-trip)
8. **format_report** — build human-readable
9. **write_report** — persist to disk
10. **notify** — final summary echo (production: action: webhook)

```bash
aiorch run 39-pipeline-as-orchestrator.yaml
aiorch run 39-pipeline-as-orchestrator.yaml -i max_records=10
aiorch run 39-pipeline-as-orchestrator.yaml --from enrich
cat aiorch-cli/examples/core/inputs/orchestrator-report.json | jq
```

Use this as a template for your own orchestration pipelines. The shapes generalize — fetch/validate/enrich/process/reconcile/notify is the canonical chain for any "pull from upstream, transform, push downstream" workflow.

**Why so many small steps?** Each step is independently retryable, observable, and resumable. Use `--from <step>` to pick up after fixing an upstream issue without re-running everything.

---

## Group 5 — Developer utility pipelines (40-50)

Practical, reusable pipelines for developer workflows: git digests, log triage, dependency audits, SSL cert expiry, JUnit parsing, PR digests, env drift, disk walks, HTTP health, and changelog generation. These are the ones that earn their keep — schedule them, wire them to webhooks, or just run them manually when you need an answer.

Most take sensible defaults so you can just `aiorch run 40-...yaml` with no flags and see it work.

### 40-git-commit-digest.yaml — per-author standup rollup from a git-log fixture

```bash
aiorch run 40-git-commit-digest.yaml
aiorch run 40-git-commit-digest.yaml -i log_path=my-log.txt
```

Parses a git-log dump (with `--numstat` blocks), groups commits by author, and writes `inputs/git-commit-digest.json`. Hermetic — ships with `inputs/sample-git-log.txt` so the default run works on a fresh clone with no local git repo needed. To process real commits, dump the log to a file first:

```bash
git log --since='30 days ago' \
  --pretty=format:'__AIORCH_COMMIT__%H|%an|%ae|%at|%s' \
  --numstat > my-log.txt
aiorch run 40-git-commit-digest.yaml -i log_path=my-log.txt
```

Good standup fodder when you wire it to nightly log archives or CI.

---

### 41-stale-branches.yaml — find branches nobody has touched in 30+ days

```bash
aiorch run 41-stale-branches.yaml
aiorch run 41-stale-branches.yaml -i stale_days=60
```

Parses a `git for-each-ref` dump (`name|unix_ts|author|subject` per line), computes age in days off wall clock, and flags anything older than `stale_days`. Hermetic — ships with `inputs/sample-branches.txt` covering a 0d–220d age range so the default run always has a realistic mix. To process a real repo, dump the branches to a file first:

```bash
git for-each-ref \
  --format='%(refname:short)|%(committerdate:unix)|%(authorname)|%(subject)' \
  refs/heads/ > my-branches.txt
aiorch run 41-stale-branches.yaml -i branches_path=my-branches.txt
```

---

### 42-log-triage.yaml — nginx access log bucketing + top paths/IPs

```bash
aiorch run 42-log-triage.yaml
aiorch run 42-log-triage.yaml -i log_path=/var/log/nginx/access.log
```

Parses a combined-format access log, buckets responses by status class (2xx/3xx/4xx/5xx), computes error rate, and runs two parallel reducers (top paths / top IPs) against the parsed records. Ships with `inputs/sample-access.log` (~100 lines) for the default run.

---

### 43-dep-audit.yaml — parallel PyPI lookups against a requirements.txt

```bash
aiorch run 43-dep-audit.yaml
aiorch run 43-dep-audit.yaml -i requirements_path=./my-reqs.txt
```

Parses `==` pins, lifts them to a top-level list, and runs one parallel `httpx` call per package against `https://pypi.org/pypi/<pkg>/json`. Reports outdated vs current vs unknown. Per-package try/except so a network flake on one dep doesn't kill the run. Ships with `inputs/sample-requirements.txt`.

**Network-dependent.** Flaky networks may surface a few "unknown" entries — that's fine, the report still lands.

---

### 44-ssl-cert-expiry.yaml — parallel TLS cert expiry checker

```bash
aiorch run 44-ssl-cert-expiry.yaml
aiorch run 44-ssl-cert-expiry.yaml -i warn_days=60
```

Opens one TCP + TLS handshake per host in parallel via `foreach`, pulls the peer certificate, parses `notAfter`, and flags anything expiring within `warn_days`. Uses `socket.create_connection` + `ssl.wrap_socket` + `getpeercert()` from stdlib. Default host list: github.com, pypi.org, google.com, cloudflare.com, anthropic.com.

**Network-dependent.** Unreachable hosts come back with `status: "error"` and don't block the summary.

---

### 45-test-results.yaml — JUnit XML parser with failures + slowest

```bash
aiorch run 45-test-results.yaml
aiorch run 45-test-results.yaml -i junit_path=./ci/pytest.xml
```

Parses a pytest-style `--junitxml` file, rolls up counts by outcome, and runs two parallel reducers (extract failures / extract slowest 5) in the same DAG layer. Uses stdlib `xml.etree.ElementTree`. Ships with `inputs/sample-junit.xml` (15 tests, 2 failures, 1 skip).

---

### 46-pr-digest.yaml — PRs grouped three ways from a JSON fixture

```bash
aiorch run 46-pr-digest.yaml
aiorch run 46-pr-digest.yaml -i prs_path=my-prs.json
```

Parses a JSON fixture shaped like the GitHub `/repos/<owner>/<repo>/pulls` response and groups the PRs three ways in parallel: by author, by age bucket (fresh/recent/stale), and by review status (has reviews vs awaiting review). Writes the digest to `inputs/pr-digest.json`. Hermetic — ships with `inputs/sample-prs.json` (10 PRs, mixed ages, mixed reviewer state) so the default run works offline with no auth. To process real PRs, dump the API response first:

```bash
gh api '/repos/<owner>/<repo>/pulls?state=open&per_page=100' > my-prs.json
aiorch run 46-pr-digest.yaml -i prs_path=my-prs.json
```

---

### 47-env-drift.yaml — compare two dotenv files and report key drift

```bash
aiorch run 47-env-drift.yaml
aiorch run 47-env-drift.yaml -i env_path=.env -i example_path=.env.example
```

Parallel parses `inputs/sample.env` and `inputs/sample.env.example`, computes set drift (missing-in-env / extra-in-env / matched), and flags the in-sync status. Tiny hand-rolled dotenv parser — no `python-dotenv` dep. Ships with deliberately drifted sample files.

---

### 48-large-files.yaml — find the biggest files in a directory tree

```bash
aiorch run 48-large-files.yaml
aiorch run 48-large-files.yaml -i root_path=/path/to/tree -i min_bytes=1000000 -i top_n=20
```

`os.walk` with in-place `dirs[:]` pruning to skip `.git`, `inputs`, `__pycache__`, `node_modules`, `.venv`. Reports top N files above `min_bytes` sorted descending. Default `root_path` is `.` (the current working directory) so the run is hermetic — point it at any tree with `-i root_path=...`.

---

### 49-http-health.yaml — parallel HTTP health checks with latency + TLS validity

```bash
aiorch run 49-http-health.yaml
```

Fans out N URLs via `foreach` over `httpx.Client` with a 5s timeout, reports per-URL status/latency/content-length/TLS validity, then rolls up an up/degraded/down summary. Total wall clock ≈ the slowest URL, not the sum. Per-URL try/except so one failure never kills the run.

**Network-dependent.** Default list mixes `httpbin.org/status/200` (up), `httpbin.org/status/500` (degraded), and a few real sites for variety.

---

### 50-changelog-from-git.yaml — conventional-commit changelog draft from a commits fixture

```bash
aiorch run 50-changelog-from-git.yaml
aiorch run 50-changelog-from-git.yaml -i commits_path=my-commits.txt
```

Reads a plaintext file of commit subjects (one per line), matches `feat/fix/chore/docs/refactor/test/perf/ci/build/style` prefixes, groups them, and renders markdown to `inputs/CHANGELOG-draft.md`. Anything that doesn't match a prefix goes in an `other` bucket so no commit is silently dropped. Hermetic — ships with `inputs/sample-commits.txt` so the default run works on a fresh clone. To process a real range, dump the subjects first:

```bash
git log v0.3.0..HEAD --no-merges --pretty=format:'%s' > my-commits.txt
aiorch run 50-changelog-from-git.yaml -i commits_path=my-commits.txt
```

---

## Tips for working in CLI mode

### Verify a pipeline before running it

```bash
aiorch validate 27-multi-input-typed-dashboard.yaml
aiorch list 27-multi-input-typed-dashboard.yaml      # show steps
aiorch plan 27-multi-input-typed-dashboard.yaml      # show DAG layers
aiorch visualize 27-multi-input-typed-dashboard.yaml # ASCII DAG diagram
```

### Dry-run

```bash
aiorch run 27-multi-input-typed-dashboard.yaml --dry --input ./inputs/batch-inputs.json
```

Builds the DAG, validates inputs, but skips actual execution. Useful for catching input-validation errors without spending compute.

### Run a single step

```bash
aiorch run 28-diamond-dependency.yaml --step branch_normalize
```

Useful for testing one step in isolation. Other steps' outputs need to be available via cached state from a previous full run, OR the step has to be self-contained.

### Verbose output

```bash
aiorch run 09-text-document-metrics-python.yaml -v -i document=@./inputs/prose.txt
```

`-v` prints each step's inputs and outputs to the terminal as they run.

### Resume a failed run

```bash
aiorch run 18-parallel-fanout-fanin.yaml --from worker_max
```

Skips every step before `worker_max` and continues from there. Requires that earlier steps' outputs are checkpointed (which the executor does automatically if you ran the pipeline before).

---

## Summary table

| # | File | Inputs | One-line description |
|---|---|---|---|
| 01 | `01-smoke-test.yaml` | — | Three shell echoes |
| 02 | `02-inputs.yaml` | string×3 | String inputs into shell |
| 03 | `03-input-integer.yaml` | integer | Integer with min/max bounds |
| 04 | `04-input-boolean.yaml` | boolean | Boolean + shell branching (use --input) |
| 05 | `05-input-number.yaml` | number | Float with bounds + awk classifier |
| 06 | `06-input-list.yaml` | list | List + foreach (use --input) |
| 07 | `07-input-artifact-text.yaml` | text file | Read text file via @ syntax |
| 08 | `08-text-document-metrics.yaml` | text file | Shell document metrics (4 parallel) |
| 09 | `09-text-document-metrics-python.yaml` | text file | Same as 08 with python: primitive |
| 10 | `10-python-primitive-smoke.yaml` | mixed scalars | Smoke-test for python: primitive |
| 11 | `11-input-artifact-json.yaml` | JSON file | Order processing from JSON |
| 12 | `12-input-artifact-csv.yaml` | CSV file | Sales aggregation from CSV |
| 13 | `13-input-artifact-binary.yaml` | binary file | SHA-256 + magic number detection |
| 14 | `14-input-http.yaml` | http (auto) | GitHub /zen text fetch |
| 15 | `15-input-http-json.yaml` | http (auto) | GitHub repo metadata fetch |
| 18 | `18-parallel-fanout-fanin.yaml` | — | 4 parallel workers + join |
| 21 | `21-failure-isolation-with-retry.yaml` | boolean | Retry + condition + try/except |
| 25 | `25-health-check-multi.yaml` | — | 3 parallel HTTP health checks |
| 27 | `27-multi-input-typed-dashboard.yaml` | all 5 input types | Reference: every scalar input type |
| 28 | `28-diamond-dependency.yaml` | integer | Diamond DAG (1→2→1) |
| 29 | `29-nested-foreach-aggregate.yaml` | — | Per-region rollup |
| 30 | `30-map-reduce-text.yaml` | string + integer | Map/reduce word count |
| 31 | `31-conditional-multi-branch.yaml` | string | Mode-driven routing |
| 32 | `32-compute-graph-numerical.yaml` | 4 numbers | Wide multi-layer numerical DAG |
| 33 | `33-db-read-basic.yaml` | integer | SQLite SELECT via python: + sqlite3 stdlib |
| 34 | `34-db-parallel-queries.yaml` | — | 5 SQL queries running in parallel + fan-in |
| 35 | `35-db-join-aggregate.yaml` | — | 3-table read + Python join + aggregate + JSON write |
| 36 | `36-webhook-api-poll.yaml` | integer + string | Pull-based polling of GitHub events API + foreach + retry |
| 37 | `37-multi-source-fanout.yaml` | http (auto) | 5 sources joined: http + sqlite + file + env + computed |
| 38 | `38-retry-strategies.yaml` | — | Four retry patterns side by side |
| 39 | `39-pipeline-as-orchestrator.yaml` | string + integer | 10-step production-shaped workflow |
| 40 | `40-git-commit-digest.yaml` | string | Per-author standup digest parsed from a git-log fixture |
| 41 | `41-stale-branches.yaml` | string + integer | Branches untouched 30+ days parsed from a for-each-ref fixture |
| 42 | `42-log-triage.yaml` | string | nginx log bucketed by status + top paths/IPs (parallel reducers) |
| 43 | `43-dep-audit.yaml` | string | Parallel PyPI lookups for each pinned dep in a requirements.txt |
| 44 | `44-ssl-cert-expiry.yaml` | list + integer | Parallel TLS cert expiry check with per-host failure isolation |
| 45 | `45-test-results.yaml` | string | JUnit XML parser with failures + slowest (parallel reducers) |
| 46 | `46-pr-digest.yaml` | string | Open PRs grouped by author / age / review status from a JSON fixture |
| 47 | `47-env-drift.yaml` | string×2 | Compare two dotenv files and report key drift |
| 48 | `48-large-files.yaml` | string + integer×2 | Walk a dir tree and report top N files above byte threshold |
| 49 | `49-http-health.yaml` | list | Parallel HTTP health checks with latency + TLS validity |
| 50 | `50-changelog-from-git.yaml` | string | Conventional-commit changelog draft from a commits-subjects fixture |

**Total: 42 pipelines, all run from the CLI.**
