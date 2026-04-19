# Frequently Asked Questions

### 1. What are the core differences between the three waves of pipeline orchestration?

The first wave (cron + shell) offered simplicity but lacked dependency management and error handling. The second wave (Airflow, Luigi) introduced DAGs, UIs, and robust error handling but required extensive Python and heavy infrastructure. The third wave focuses on YAML-first, integrated AI steps, and a balance between declarative configuration and scripting for broader accessibility.

### 2. Why is the third wave being developed if tools like Airflow already exist?

The third wave addresses the limitations of second-wave tools, which optimized for software engineers and excluded non-technical users like business analysts or DBAs due to their Python-centric nature and heavy deployment requirements. It aims to empower a wider range of operators.

### 3. What problem does the third wave primarily aim to solve?

The third wave primarily aims to bridge the gap between technical and non-technical operators. It wants to make pipeline orchestration accessible to non-software engineers who understand business logic but are not proficient in complex programming frameworks or distributed systems.

### 4. How does the third wave handle AI steps differently from previous methods?

The third wave integrates AI steps as first-class primitives directly within the YAML configuration, rather than requiring Python wrappers. This allows the framework to handle specific AI failure modes (e.g., invalid JSON, context window issues) natively, benefiting pipeline authors.

### 5. What is the role of YAML in the third wave of orchestration?

In the third wave, YAML is used for the declarative layer, defining the DAG shape, input/output contracts, retry policies, and branching decisions. This provides a clear, operator-friendly view of the pipeline's structure and control flow.

### 6. How does the third wave prevent repeating past failures, specifically relating to too much YAML or too much Python?

The third wave seeks a 'sweet spot' by using YAML for the declarative DAG shape and high-level concerns, and short Python or shell bodies for the actual per-step computation. This prevents encoding complex logic in YAML (a wave-one failure) and avoids making every pipeline bespoke Python (a wave-two failure).

### 7. What kind of 'steps' can be orchestrated in the third wave?

The third wave can orchestrate both deterministic steps like shell commands, SQL queries, HTTP calls, and file I/O, as well as AI steps such as LLM calls, embeddings, and agentic tool use, all within the same DAG.

### 8. What are the advantages of separating the DAG layer from the step body in the third wave?

This separation makes pipelines more readable and maintainable for a broader audience, including those who own the business process but aren't engineers. Operators can reason about the DAG, while domain logic lives within concise step bodies, improving collaboration and understanding.

### 9. What is 'aiorch' and how does it relate to the third wave?

Aiorch is mentioned as an early example of a tool in the third-wave space. It demonstrates the pattern of using YAML for the DAG shape and high-level policy, combined with short Python or shell for per-step computation, which is characteristic of this new approach.

### 10. Is the third wave of orchestration considered a fully mature or proven solution yet?

No, the documentation states that the long-term equilibrium is still an open question, and the space is young. While early evidence is encouraging, the ultimate success of the YAML-plus-short-Python-bodies pattern depends on its generalization to real-world workflows.
