# LLM Eval Harness

Upload model outputs, define what good looks like, and find out whether you can trust an LLM to grade them.

An open-source web app that scores LLM outputs against a custom rubric, lets a human review the scores, and reports judge–human agreement (quadratic weighted Cohen's kappa).

> **Status:** work in progress. Live demo link, results, and demo GIF coming soon.

## Layout

- `backend/` – FastAPI + SQLAlchemy + Alembic (Postgres)
- `frontend/` – Next.js (not started)
- `seed/` – demo dataset, rubric, and human labels (not started)

## Run the backend locally

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
DATABASE_URL=postgresql+psycopg://user:pass@host/db .venv/bin/alembic upgrade head
.venv/bin/uvicorn app.main:app --reload   # docs at http://localhost:8000/docs
```

Tests use SQLite and need no database: `.venv/bin/pytest`.

## License

MIT
