# 202x Headlines

A curated archive of real news headlines so absurd they function as accidental satire, organised by year.

Each headline is sourced, scraped, reviewed, and published — not invented. The project is a mirror: the absurdity is in the record, not the commentary.

---

## Milestone status

| Milestone | Status | Description |
|---|---|---|
| M1 | ✅ | Foundation — article models, ArticleRepository, SQLite, smoke tests |
| M2 | ✅ | RSS scraper — `scraper.py`, 4 sources (Guardian, NPR, BBC, Al Jazeera) |
| M3 | ✅ | Review CLI — `review.py`, multi-reason rejection, keyboard-driven |
| M3b | ✅ | LLM filter — `filter.py`, Ollama/mistral:7b candidate scoring |
| M4 | ✅ | Data model — filter score, rationale, flags fields; migration script |
| M5a | ✅ | Reddit scraper — `reddit_scraper.py`, r/nottheonion, public JSON API |
| M5b | 🔲 | Flask frontend — year-based archive, candidate review UI |
| M5c | 🔲 | Scheduler — automated daily scrape + filter run |
| M6 | 🔲 | Deployment — hosted archive, public URL |

---

## Project structure

```
202x-headlines/
├── core/               # Models, ArticleRepository, DB setup, session management
├── tests/              # 27 tests — run against in-memory SQLite, no setup required
├── data/               # SQLite database — created at runtime, gitignored
├── scraper.py          # RSS scraper — fetches from configured sources
├── reddit_scraper.py   # Reddit scraper — r/nottheonion via public JSON API
├── filter.py           # LLM filter — scores candidates using Ollama/mistral:7b
├── review.py           # Review CLI — keyboard-driven accept/reject interface
├── sources.yaml        # RSS and Reddit source config with enabled flags
└── filter-config.yaml  # Filter thresholds and prompt config
```

The `core/` layer contains no project-specific logic — every query is scoped by a `project_id` parameter. This means the same pipeline can serve future projects without duplicating the data layer.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the pipeline

Each step is a standalone script. Run them in order:

```bash
# 1. Scrape RSS sources
python scraper.py

# 2. Scrape Reddit (r/nottheonion, top posts from past 24h)
python reddit_scraper.py

# 3. Score candidates with the LLM filter
python filter.py

# 4. Review and accept/reject in the terminal
python review.py
```

---

## Running the tests

```bash
pytest tests/
```

27/27 tests pass against an in-memory SQLite database — no setup required.

---

## Environment variables

Copy `.env.example` to `.env` and fill in real values before running any project-specific code. Never commit `.env` to version control.
