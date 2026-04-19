# aiorch

**YAML-driven pipelines for LLMs, Python, and shell — runnable from the command line.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#-roadmap)

aiorch turns a YAML file into a runnable pipeline. Declare your steps — LLM prompts, Python snippets, shell commands, MCP tool calls — and `aiorch run` executes the DAG. No server, no scheduler, no database setup.

```bash
pip install aiorch
export OPENROUTER_API_KEY=sk-or-v1-...
aiorch run examples/01-hello-llm.yaml
```

Works with any provider [LiteLLM](https://docs.litellm.ai/) supports — OpenAI, Anthropic, Gemini, OpenRouter, Ollama, Bedrock, and more.

---

## ✨ Features

- 🤖 **LLM primitives** — prompt, schema-extract, classify-and-branch, multi-model compare
- 🧩 **DAG shapes** — chain, parallel + merge, foreach, conditional flows
- 🐍 **LLM + Python hybrid** — an LLM for reasoning, deterministic Python for side effects
- 🛠️ **Agents + MCP** — function-calling LLMs with MCP tools over stdio or Streamable HTTP
- 🔌 **Real connectors** — Postgres, S3, Kafka, SMTP, webhooks (`aiorch[connectors]`)
- 💰 **Cost tracking** — prompt / completion tokens and USD per provider per run, persisted to `~/.aiorch/history.db`
- 🧪 **Dry-run + validation** — catch schema errors and unresolved templates before spending tokens

---

## 🚀 Quick start

```yaml
# hello.yaml
name: hello
steps:
  answer:
    prompt: |
      In one sentence, what is aiorch?
    output: summary

  show:
    run: echo "{{summary}}"
    depends: [answer]
```

```bash
$ aiorch run hello.yaml
[answer]  aiorch runs declarative YAML pipelines...
[show]    aiorch runs declarative YAML pipelines...
```

Override inputs at runtime:

```bash
$ aiorch run examples/20-csv-to-markdown-report.yaml \
    --input data=@./inputs/sample-projects.csv
```

---

## 📦 Installation

```bash
pip install aiorch                    # core CLI
pip install 'aiorch[connectors]'      # + Postgres / S3 / Kafka / SMTP
pip install 'aiorch[metrics]'         # + Prometheus export (opt-in)
pip install 'aiorch[validation]'      # + jsonschema input validation
```

Requires **Python 3.11+**.

---

## 🔧 Configuration

aiorch looks for `aiorch.yaml` in the current directory.

```yaml
llm:
  api_key: ${OPENROUTER_API_KEY}
  model: google/gemini-2.5-flash
  api_base: https://openrouter.ai/api/v1

storage:
  type: sqlite        # default — ~/.aiorch/history.db
```

No `aiorch.yaml`? aiorch falls back to standard environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, etc.) and a sensible default model.

---

## 🖥️ CLI reference

| Command | Purpose |
|---|---|
| `aiorch run <file>` | Execute a pipeline |
| `aiorch validate <file>` | Schema + template lint, no execution |
| `aiorch list-steps <file>` | Inspect a pipeline's DAG |
| `aiorch init <template>` | Scaffold a new pipeline from a template |
| `aiorch history` | List recent runs and their status |
| `aiorch history <run-id>` | Show details of one run |
| `aiorch trace <run-id>` | Step-by-step trace for one run |

Run `aiorch --help` for the full list of flags.

---

## 🧠 MCP support

aiorch ships a built-in MCP client — both **stdio** (subprocess) and **Streamable HTTP** (MCP 2025 spec). Attach tools to any `agent:` step:

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

---

## 📚 Examples

**72 runnable pipelines** shipped under [`examples/`](examples), organized into two tracks:

| Directory | Count | What's inside |
|---|---|---|
| [`examples/llm/`](examples/llm) | 30 | LLM pipelines — prompts, extraction, chains, fan-out, agents, hybrid LLM + Python |
| [`examples/core/`](examples/core) | 42 | Zero-LLM pipelines — every primitive, every DAG shape, input types, DB access, developer utilities |

Each track has its own README with per-pipeline walkthroughs:

- [`examples/README.md`](examples/README.md) — **start here** for the full guide on secrets, model selection, and passing inputs
- [`examples/llm/README.md`](examples/llm/README.md) — LLM pipelines by tier (basic → DAG shapes → hybrid → developer workflows)
- [`examples/core/README.md`](examples/core/README.md) — core pipelines by group (primitives → DAG shapes → DB access → production patterns → utilities)

```bash
# Core pipelines — no API key required
aiorch run examples/core/01-smoke-test.yaml

# LLM pipelines — provider config lives at examples/llm/aiorch.yaml
export OPENROUTER_API_KEY=sk-or-v1-...
cd examples/llm && aiorch run 01-hello-llm.yaml
```

---

## 🗺️ Roadmap

This is **v0.1 alpha** — YAML schema and CLI flags may change. Pin an exact version in CI.

Planned:
- Additional LLM primitives (structured output schemas, streaming sinks)
- Broader connector catalog
- Pipeline composition (one pipeline imports another)
- First-class Windows support

---

## 🤝 Contributing

Issues and PRs welcome at [github.com/ereshzealous/aiorch-cli](https://github.com/ereshzealous/aiorch-cli).

---

## 📄 License

Apache 2.0 — see [`LICENSE`](LICENSE).
