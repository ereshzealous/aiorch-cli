# aiorch

**YAML-driven LLM + Python + shell pipelines, runnable from the CLI.**

aiorch is a single-binary pipeline runner. You declare your workflow in
a YAML file, point it at a model (OpenAI, Anthropic, Gemini,
OpenRouter, Ollama — anything [LiteLLM](https://docs.litellm.ai/)
supports), and `aiorch run` executes the DAG. No server, no scheduler,
no database setup — just `pip install aiorch` and a YAML file.

```bash
pip install aiorch
export OPENROUTER_API_KEY=sk-or-v1-...
aiorch run examples/01-hello-llm.yaml
```

---

## What you can build

- **LLM primitives** — prompt, schema-extract, classify-and-branch,
  multi-model compare.
- **DAG shapes** — chain, parallel+merge, foreach, conditional flows.
- **Hybrid LLM + Python** — use an LLM for analysis, deterministic
  Python for side effects.
- **Agents** — LLM with tool-use (function calling) + MCP servers over
  stdio or HTTP.
- **Real connectors** — Postgres, S3, Kafka, SMTP, webhooks (install
  the `connectors` extra).
- **Cost tracking** — every run logs prompt / completion tokens + cost
  per provider to `~/.aiorch/history.db`.

---

## A 30-second tour

```yaml
# examples/01-hello-llm.yaml
name: hello-llm
inputs:
  question:
    type: string
    default: What is 2+2?

steps:
  answer:
    llm:
      prompt: "{{inputs.question}}"
      max_tokens: 50
```

```bash
$ aiorch run examples/01-hello-llm.yaml
[answer] 4
```

Override inputs:

```bash
$ aiorch run examples/01-hello-llm.yaml --input question="Why is the sky blue?"
```

Or pass a file:

```bash
$ aiorch run examples/20-csv-to-markdown-report.yaml \
    --input data=@./inputs/sample-projects.csv
```

---

## Configuration

aiorch looks for `aiorch.yaml` in the current directory. Minimum:

```yaml
llm:
  api_key: ${OPENROUTER_API_KEY}
  model: google/gemini-2.5-flash
  api_base: https://openrouter.ai/api/v1

storage:
  type: sqlite        # default — ~/.aiorch/history.db
```

No `aiorch.yaml`? CLI falls back to env vars (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, etc.) and a default model.

---

## What's in the box

- `aiorch run <file>` — execute a pipeline.
- `aiorch validate <file>` — dry-run checks (schema, unresolved vars).
- `aiorch list` — show recent runs + status.
- `aiorch show <run-id>` — step-by-step trace.
- `aiorch new <template>` — scaffold a pipeline from a template.

`aiorch --help` for the full list.

---

## Installation

```bash
pip install aiorch                   # core CLI
pip install 'aiorch[connectors]'     # + Postgres / S3 / Kafka / SMTP
pip install 'aiorch[metrics]'        # + Prometheus export (opt-in)
pip install 'aiorch[validation]'     # + jsonschema input validation
```

Requires Python 3.11+.

---

## MCP support

aiorch ships direct MCP client support over both stdio (subprocess) and
Streamable HTTP (the MCP 2025 spec). In your YAML:

```yaml
steps:
  analyze:
    agent:
      model: gpt-4o-mini
      tools:
        mcp:
          - server: "npx -y @modelcontextprotocol/server-filesystem"
            args: ["/tmp"]
      prompt: "List files under /tmp and summarize"
```

Pooled / cross-replica MCP session management is part of the commercial
aiorch platform — not in the CLI.

---

## Examples

See `examples/` for runnable pipelines. Each one has `--help`-style
annotations at the top explaining what inputs it takes.

---

## Roadmap

This is **v0.1 alpha**. YAML schema and CLI flags may change. Pin
versions in CI.

Planned:
- More LLM primitives (structured output schemas, streaming sinks).
- Richer connector catalog.
- Pipeline composition (one pipeline imports another).

---

## The commercial platform

If you need multi-tenant workspaces, RBAC, a web UI, scheduling,
webhook triggers, cross-replica MCP session pools, cost analytics,
audit logs, or Postgres-backed run history across a team — there's a
commercial aiorch platform built on top of this CLI. [Contact the
author](mailto:eresh.zealous@gmail.com) if that's you.

---

## License

Apache 2.0 — see `LICENSE`.
