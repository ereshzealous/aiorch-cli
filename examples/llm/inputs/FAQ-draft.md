# Frequently Asked Questions

### 1. What is aiorch and what problem does it solve?

aiorch is a pipeline orchestration tool that combines deterministic steps (shell, Python, HTTP, SQL) with LLM steps into a single DAG, defined in YAML. It helps operations teams build and manage AI-powered workflows rather than just prototype AI models.

### 2. How do I define a pipeline in aiorch?

Pipelines are defined using YAML files. They consist of a `name` and a series of `steps`, where each step can specify a prompt for an LLM, a shell command to `run`, or other operations, with `depends` to define execution order.

### 3. How does aiorch handle LLM integration and API keys?

aiorch integrates LLMs directly within pipeline steps using prompts. It expects LLM provider API keys, like `OPENAI_API_KEY`, to be set as environment variables and can be configured via an `aiorch.yaml` file.

### 4. Can aiorch interact with external data sources and services?

Yes, aiorch uses 'connectors' to interact with various external systems. This includes databases like Postgres/MySQL/SQLite, object storage like S3/MinIO/R2/GCS, Kafka topics, and communication webhooks like Slack/Discord/Teams.

### 5. Does aiorch track the costs associated with LLM usage?

Yes, aiorch tracks the cost of each LLM call in USD, providing per-step and total run cost visibility. It also supports per-provider daily and monthly budget caps to prevent exceeding spending limits.

### 6. How can I trigger a pipeline from an external system?

Pipelines can be triggered via webhooks. External systems POST to a workspace-scoped URL, and aiorch handles signature verification and rate limiting, recording all trigger attempts for debugging.

### 7. What kind of historical data and debugging features does aiorch offer?

aiorch stores run history in a SQLite database (`~/.aiorch/history.db`), allowing users to resume failed runs, inspect execution traces, and cache LLM responses. It also logs webhook delivery history for debugging.

### 8. How does aiorch manage sensitive information like API keys or database credentials?

Secrets associated with connectors are encrypted at rest using AES-256-GCM. The UI provides type-appropriate input fields for different connector authentication modes, ensuring secure handling of credentials.

### 9. What kind of access control does aiorch provide for teams?

aiorch supports role-based access control (RBAC) with four defined roles: admin, editor, operator, and viewer. Permissions are managed granularly per-action, such as `pipeline.view` or `secret.create`.

### 10. What is the typical deployment or operational environment for aiorch?

While not explicitly stated, aiorch's focus on ops teams, webhook triggers, cost tracking, and RBAC suggests it's designed for a managed, potentially cloud-hosted environment where pipeline execution, monitoring, and team collaboration are key.
