"""
LLM candidate filter for the 202x Headlines pipeline.

Reads unscored candidate articles from the database, sends each one to a
locally-running Ollama model, and writes the score, rationale, and flags
back to the database. Sits between the scraper and the review tool.

Usage:
    python filter.py

Requires Ollama to be running locally with mistral:7b available:
    ollama serve
    ollama pull mistral:7b
"""

import json
import warnings
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
# Filter run                                                                   #
# --------------------------------------------------------------------------- #

def run_filter(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> dict:
    """
    Run the LLM filter over all unscored candidate articles for the project.

    Fetches every candidate article whose filter_score is None, sends each
    to the configured Ollama model, and writes the score, rationale, and
    flags back to the database using ArticleRepository.set_filter_result.

    Parameters
    ----------
    config_path : Path to filter-config.yaml. Defaults to project root.
    db_path     : Path to the SQLite database. Defaults to data/articles.db.

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

    return {"processed": processed, "errors": errors, "skipped": skipped}


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    run_filter()
