# aiorch — AI pipeline orchestration

aiorch runs YAML-defined pipelines that mix deterministic steps (shell,
python, HTTP, SQL) with LLM steps in a single DAG. Designed for ops teams
who need to orchestrate real work — not just prototype AI demos.

## Install

```bash
pip install aiorch
```

## Quick start

Create a pipeline file `hello.yaml`:

```yaml
name: hello
steps:
  greet:
    prompt: "Say hi in 5 different languages."
    output: greetings
  show:
    run: echo {{greetings}}
    depends: [greet]
```

Then:

```bash
export OPENAI_API_KEY=sk-...
aiorch run hello.yaml
```

aiorch reads `aiorch.yaml` from the current directory (or any parent) for
LLM provider config. A default `~/.aiorch/history.db` SQLite file stores
your run history so you can resume failed runs, inspect traces, and cache
LLM responses across runs.

## Webhook triggers

External systems can trigger pipelines by POSTing to a workspace-scoped
URL. Aiorch verifies HMAC signatures, enforces per-trigger rate limits,
and writes every inbound hit (success OR rejection) to a delivery-history
table so operators can debug misconfigurations without grepping logs.

Create a trigger via the Webhooks page, paste the URL + secret into your
external system's webhook config, and aiorch will fire the mapped pipeline
on every matching event.

## Connectors

A pipeline step can read from or write to:

- **Postgres / MySQL / SQLite** — via the `database` connector type
- **S3 / MinIO / R2 / GCS** — via the `object_store` connector type
- **Kafka topics** — via the `stream` connector type
- **Slack / Discord / Teams webhooks** — via the `webhook` connector type

Secrets attached to a connector are encrypted at rest with AES-256-GCM.
The connector form understands the field types and authentication modes
of each subtype; the UI renders type-appropriate inputs.

## Cost tracking

Every LLM call records per-step cost in USD. You can see the total cost
per run in the Runs page, and break it down per step in the trace view.
Per-provider budget caps (daily + monthly) reject runs that would exceed
the budget instead of charging the card.

## Role-based access control

Workspaces have four roles: admin, editor, operator, viewer. Permissions
are managed per-action (pipeline.view, pipeline.create, secret.create,
etc.) and the catalog is visible on the Roles page.
