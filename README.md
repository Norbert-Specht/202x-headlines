# 202x Headlines

A curated archive of real news headlines so absurd they function as accidental satire, organised by year.

Each headline is sourced, scraped, reviewed, and published — not invented. The project is a mirror: the absurdity is in the record, not the commentary.

---

## Project structure

```
202x-headlines/
├── core/               # Reusable infrastructure (models, repository, DB setup)
├── projects/headlines/ # Project-specific logic for 202x Headlines
├── tests/              # Smoke tests
└── data/               # SQLite database — created at runtime, gitignored
```

The `core/` and `projects/` split is intentional. `core/` contains no project-specific logic — every query is scoped by a `project_id` parameter. This means the same article pipeline can serve future projects (a climate-news archive is the confirmed next candidate) without duplicating the data layer.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the tests

```bash
pytest tests/
```

Tests run against an in-memory SQLite database — no setup required.

---

## Environment variables

Copy `.env.example` to `.env` and fill in real values before running any project-specific code. Never commit `.env` to version control.
