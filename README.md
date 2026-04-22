# Do Frontier LLMs Adapt to Neurodivergence? A Cross-Model Audit

Code, configs, and data for an empirical audit of how GPT-5-chat and Claude Sonnet 4.6 respond to neurodivergence (ND) context supplied via the system prompt.

See [`spec.md`](spec.md) for the full experimental design and research questions.

## Layout

```
configs/        # profiles, prompts, queries — single source of truth for the experiment
ndbench/        # runner, metrics, judges, analysis code
data/responses/ # raw model outputs (JSONL), one per (model, condition, profile)
paper/          # LaTeX source
```

## Quickstart

1. `cp .env.example .env` and paste real API keys into `.env` (never commit).
2. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. `python -m ndbench.runner`  — dispatches all model calls, caches to `data/responses/`.
4. `python -m ndbench.metrics.run`  — computes structural, surface, and harm metrics.
5. `python -m ndbench.analyze`  — fits models, emits figures and tables to `paper/figures/`.

## Models under audit

- `gpt-5-chat-latest` (OpenAI, non-reasoning chat variant)
- `claude-sonnet-4-6` (Anthropic)

## License

Code: MIT. Data and paper artifacts: CC-BY-4.0.
