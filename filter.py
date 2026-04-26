"""
LLM candidate filter for the 202x Headlines pipeline.

Reads unscored candidate articles from the database, sends each one to a
locally-running Ollama model, and writes the score, rationale, and flags
back to the database. Sits between the scraper and the review tool.

Usage:
    python filter.py
    python filter.py --export-rationales   # also writes filter_report.txt

Requires Ollama to be running locally with mistral:7b available:
    ollama serve
    ollama pull mistral:7b
"""

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests
import yaml

from core import get_engine, get_session, init_db
from core.prompts import build_filter_prompt
from core.repository import ArticleRepository


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "filter-config.yaml"
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "articles.db"

_DIVIDER = "─" * 43

# Divider used inside the filter report file — slightly shorter than the
# terminal divider so the file reads cleanly in a text editor at any width.
_REPORT_DIVIDER = "─" * 40

# Output path for the optional rationale export.
_REPORT_PATH = _PROJECT_ROOT / "filter_report.txt"


# --------------------------------------------------------------------------- #
# Config loading                                                               #
# --------------------------------------------------------------------------- #

def load_model_config(config_path: str | Path) -> dict:
    """
    Read and parse the filter-config.yaml file.

    Expected keys: project_id, model, ollama_url, score_threshold.

    Parameters
    ----------
    config_path : Path to filter-config.yaml.

    Returns
    -------
    dict
        Parsed config dict.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the file is present but cannot be parsed as valid YAML, or is
        missing required keys.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Filter config not found: {path}\n"
            "Create filter-config.yaml in the project root before running the filter."
        )

    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Could not parse {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")

    for required_key in ("project_id", "model", "ollama_url"):
        if required_key not in config:
            raise ValueError(f"{path} is missing required key: '{required_key}'")

    return config


# --------------------------------------------------------------------------- #
# Ollama API call                                                              #
# --------------------------------------------------------------------------- #

def score_article(
    headline: str,
    excerpt: str | None,
    model: str,
    prompt_fn: Callable[[str, str | None], str],
    ollama_url: str = "http://localhost:11434",
) -> dict:
    """
    Send one article to Ollama and return the parsed filter result.

    The prompt_fn parameter is a callable rather than a direct import of
    build_filter_prompt so that tests can inject a mock prompt function
    without patching module-level imports. This keeps the scoring logic
    testable in isolation.

    Parameters
    ----------
    headline   : Article headline string.
    excerpt    : Article excerpt string, or None if unavailable.
    model      : Ollama model name, e.g. "mistral:7b".
    prompt_fn  : Callable that takes (headline, excerpt) and returns the
                 prompt string. Pass core.prompts.build_filter_prompt in
                 production; pass a mock in tests.
    ollama_url : Base URL for the Ollama REST API.

    Returns
    -------
    dict
        Parsed model response with keys: "score", "rationale", "flags".
        On malformed JSON: {"score": None, "rationale": "parse error", "flags": []}.

    Raises
    ------
    RuntimeError
        If Ollama is not reachable (connection refused, timeout, etc.).
    """
    prompt = prompt_fn(headline, excerpt)

    # ------------------------------------------------------------------ #
    # Call the Ollama generate endpoint                                    #
    # ------------------------------------------------------------------ #

    api_url = f"{ollama_url.rstrip('/')}/api/generate"

    try:
        response = requests.post(
            url=api_url,
            json={
                "model": model,
                "prompt": prompt,
                # stream=False returns the full response in a single JSON object
                # rather than a sequence of partial token chunks.
                "stream": False,
            },
            timeout=120,  # local inference can be slow; allow up to 2 minutes
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {ollama_url}. "
            "Is Ollama running? Try: ollama serve"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama request timed out after 120s for model {model!r}."
        ) from exc

    # ------------------------------------------------------------------ #
    # Extract and parse the model's text response                         #
    # ------------------------------------------------------------------ #

    raw_text = response.json().get("response", "")

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # The model occasionally wraps its response in markdown fences or
        # adds a preamble despite instructions. Try to extract a JSON object.
        result = _extract_json(raw_text)

    if result is None:
        print(f"  WARNING: Could not parse model response as JSON: {raw_text[:120]!r}")
        return {"score": None, "rationale": "parse error", "flags": []}

    # ------------------------------------------------------------------ #
    # Normalise — ensure all expected keys are present                    #
    # ------------------------------------------------------------------ #

    return {
        "score":     result.get("score"),
        "rationale": result.get("rationale", ""),
        "flags":     result.get("flags", []),
    }


def _extract_json(text: str) -> dict | None:
    """
    Attempt to extract a JSON object from a string that may contain
    surrounding prose or markdown fences.

    Parameters
    ----------
    text : Raw string from the model response.

    Returns
    -------
    dict | None
        Parsed dict if a JSON object was found, otherwise None.
    """
    # Find the first '{' and last '}' and try parsing what's between them.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Report export                                                                #
# --------------------------------------------------------------------------- #

def write_filter_report(engine, project_id: str, report_path: Path) -> int:
    """
    Write a plain-text report of all scored candidates to report_path.

    Queries every candidate article that has a filter_score (i.e. has been
    scored by the LLM), sorts them by score descending, and writes one
    block per article showing score, source, headline, URL, flags, and
    rationale. A header line at the top records the timestamp and total count.

    Parameters
    ----------
    engine      : SQLAlchemy engine connected to the project database.
    project_id  : Project ID string used to scope the repository query.
    report_path : File path where the report will be written. Overwrites any
                  existing file at that path.

    Returns
    -------
    int
        Number of scored articles included in the report.
    """
    # ------------------------------------------------------------------ #
    # Fetch all scored candidates                                          #
    # ------------------------------------------------------------------ #

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        all_candidates = repo.get_candidates(project_id=project_id)

        # Keep only articles that have been through the filter.
        scored = [a for a in all_candidates if a.filter_score is not None]

        # Collect all display fields while the session is still open —
        # ORM objects become detached after the context manager exits.
        entries = [
            {
                "score":       a.filter_score,
                "source_name": a.source_name or "unknown",
                "headline":    a.headline,
                "source_url":  a.source_url,
                # filter_flags is stored as a comma-separated string or None.
                "flags":       a.filter_flags or "—",
                "rationale":   a.filter_rationale or "",
            }
            for a in scored
        ]

    # ------------------------------------------------------------------ #
    # Sort and format                                                      #
    # ------------------------------------------------------------------ #

    # Most promising candidates first — makes the report useful for prompt
    # calibration without having to scroll past low scorers.
    entries.sort(key=lambda e: e["score"], reverse=True)

    n = len(entries)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    noun = "article" if n == 1 else "articles"

    lines: list[str] = [
        f"Filter Report — {timestamp} — {n} {noun} scored",
        "",
    ]

    for entry in entries:
        lines.append(_REPORT_DIVIDER)
        lines.append(f"SCORE: {entry['score']:.2f}  |  SOURCE: {entry['source_name']}")
        lines.append(f"HEADLINE: {entry['headline']}")
        lines.append(f"URL: {entry['source_url']}")
        lines.append(f"FLAGS: {entry['flags']}")
        lines.append(f"RATIONALE: {entry['rationale']}")
        lines.append("")

    lines.append(_REPORT_DIVIDER)

    # ------------------------------------------------------------------ #
    # Write file                                                           #
    # ------------------------------------------------------------------ #

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return n


# --------------------------------------------------------------------------- #
# Filter run                                                                   #
# --------------------------------------------------------------------------- #

def run_filter(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    db_path: str | Path = _DEFAULT_DB_PATH,
    export_rationales: bool = False,
) -> dict:
    """
    Run the LLM filter over all unscored candidate articles for the project.

    Fetches every candidate article whose filter_score is None, sends each
    to the configured Ollama model, and writes the score, rationale, and
    flags back to the database using ArticleRepository.set_filter_result.

    When export_rationales is True, writes a plain-text report of all scored
    articles (including those scored in previous runs) to filter_report.txt
    in the project root after the filter run completes.

    Parameters
    ----------
    config_path        : Path to filter-config.yaml. Defaults to project root.
    db_path            : Path to the SQLite database. Defaults to data/articles.db.
    export_rationales  : When True, write filter_report.txt after the run.

    Returns
    -------
    dict
        Run summary: {"processed": int, "errors": int, "skipped": int}.
    """
    # ------------------------------------------------------------------ #
    # Load config                                                          #
    # ------------------------------------------------------------------ #

    config = load_model_config(config_path)
    project_id = config["project_id"]
    model = config["model"]
    ollama_url = config["ollama_url"]

    # ------------------------------------------------------------------ #
    # Set up database                                                      #
    # ------------------------------------------------------------------ #

    engine = get_engine(db_path=db_path)
    init_db(engine)

    # ------------------------------------------------------------------ #
    # Fetch unscored candidates                                            #
    # ------------------------------------------------------------------ #

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        all_candidates = repo.get_candidates(project_id=project_id)

        # Only process articles that haven't been scored yet.
        unscored = [a for a in all_candidates if a.filter_score is None]

        # Collect the data we need before the session closes — ORM objects
        # become detached after the session context exits.
        to_score = [
            {"id": a.id, "headline": a.headline, "excerpt": a.excerpt}
            for a in unscored
        ]

    total = len(to_score)

    # ------------------------------------------------------------------ #
    # Print run header                                                     #
    # ------------------------------------------------------------------ #

    title = project_id.replace("-", " ").title()
    print(f"\n{title} — Filter Run")
    print(_DIVIDER)

    if total == 0:
        print("No unscored candidates. Run the scraper first.")
        print(_DIVIDER)
        return {"processed": 0, "errors": 0, "skipped": 0}

    print(f"Scoring {total} unscored candidate{'s' if total != 1 else ''}...\n")

    processed = 0
    errors = 0
    skipped = 0

    # ------------------------------------------------------------------ #
    # Score each article                                                   #
    # ------------------------------------------------------------------ #

    for index, article_data in enumerate(to_score, start=1):
        article_id = article_data["id"]
        headline = article_data["headline"]
        excerpt = article_data["excerpt"]

        # Truncate headline display to keep output tidy.
        display_headline = headline if len(headline) <= 80 else headline[:77] + "..."
        print(f'[{index}/{total}] "{display_headline}"')

        try:
            result = score_article(
                headline=headline,
                excerpt=excerpt,
                model=model,
                prompt_fn=build_filter_prompt,
                ollama_url=ollama_url,
            )
        except RuntimeError as exc:
            # Ollama not reachable — no point continuing the run.
            print(f"  ERROR: {exc}")
            print(_DIVIDER)
            print(f"Aborted. Processed: {processed}   Errors: {errors + 1}   Skipped: {total - processed - 1}")
            return {"processed": processed, "errors": errors + 1, "skipped": total - processed - 1}

        score = result["score"]
        rationale = result["rationale"]
        flags = result["flags"]

        if score is None:
            # Model returned malformed JSON — still store what we have so
            # we don't re-attempt this article on the next run.
            errors += 1
            print(f"  ERROR — could not parse model response\n")
        else:
            processed += 1
            flags_display = ", ".join(flags) if flags else "—"
            print(f"         Score: {score:.2f} | {flags_display}")
            # Wrap rationale at 72 chars for readable terminal output.
            print(f'         "{rationale}"\n')

        # ------------------------------------------------------------------ #
        # Write result back to database                                        #
        # ------------------------------------------------------------------ #

        with get_session(engine) as session:
            repo = ArticleRepository(session)
            repo.set_filter_result(
                article_id=article_id,
                score=score,
                rationale=rationale,
                flags=flags,
            )

    # ------------------------------------------------------------------ #
    # Print summary                                                        #
    # ------------------------------------------------------------------ #

    print(_DIVIDER)
    print(f"Done. Processed: {processed}   Errors: {errors}   Skipped: {skipped}")
    print()

    # ------------------------------------------------------------------ #
    # Optional rationale export                                            #
    # ------------------------------------------------------------------ #

    if export_rationales:
        n = write_filter_report(engine, project_id, _REPORT_PATH)
        noun = "entry" if n == 1 else "entries"
        print(f"Filter report written → {_REPORT_PATH.name}  ({n} {noun})")
        print()

    return {"processed": processed, "errors": errors, "skipped": skipped}


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM candidate filter for the 202x Headlines pipeline.",
    )
    parser.add_argument(
        "--export-rationales",
        action="store_true",
        # Writes filter_report.txt to the project root after the run.
        # Includes all scored candidates (not just this run) sorted by
        # score descending — useful for prompt calibration and review.
        help="Write a scored-article report to filter_report.txt after the run.",
    )
    args = parser.parse_args()

    run_filter(export_rationales=args.export_rationales)
