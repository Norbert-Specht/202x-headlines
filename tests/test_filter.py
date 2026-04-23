"""
Tests for filter.py and core/prompts.py.

All Ollama API calls are mocked — no real network requests are made.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core import get_engine, get_session, init_db
from core.prompts import build_filter_prompt
from core.repository import ArticleRepository
from filter import load_model_config, run_filter, score_article


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_ollama_response(payload: dict) -> MagicMock:
    """
    Build a mock requests.Response that returns the given dict as JSON text.

    The Ollama /api/generate endpoint wraps the model's text output in a
    {"response": "..."} envelope — this helper replicates that structure.
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": json.dumps(payload)}
    return mock_resp


def _seed_candidate(repo: ArticleRepository, url: str, headline: str) -> int:
    """Add a candidate article and return its id."""
    article = repo.add_candidate(
        project_id="test-project",
        headline=headline,
        source_url=url,
        source_name="Test Source",
        excerpt="A short excerpt for testing.",
        date_scraped=datetime(2025, 6, 1),
    )
    return article.id


# --------------------------------------------------------------------------- #
# build_filter_prompt tests                                                    #
# --------------------------------------------------------------------------- #

def test_build_filter_prompt_contains_headline_and_excerpt():
    """
    build_filter_prompt should return a non-empty string that includes
    both the headline and excerpt in the prompt body.
    """
    prompt = build_filter_prompt(
        headline="Airline charges extra fee to breathe recycled air",
        excerpt="Low-cost carrier introduces oxygen surcharge on flights.",
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "Airline charges extra fee to breathe recycled air" in prompt
    assert "oxygen surcharge" in prompt


def test_build_filter_prompt_handles_none_excerpt():
    """build_filter_prompt should not crash when excerpt is None."""
    prompt = build_filter_prompt(
        headline="Some headline",
        excerpt=None,
    )

    assert isinstance(prompt, str)
    assert "Some headline" in prompt


# --------------------------------------------------------------------------- #
# score_article tests                                                          #
# --------------------------------------------------------------------------- #

def test_score_article_parses_valid_json_response():
    """
    score_article should return a correctly structured dict when the model
    returns valid JSON with score, rationale, and flags.
    """
    valid_payload = {
        "score": 0.91,
        "rationale": "Classic economic grotesque — monetising a basic human need.",
        "flags": ["absurd-headline", "dark-wit", "economic-grotesque"],
    }

    with patch("filter.requests.post", return_value=_make_ollama_response(valid_payload)):
        result = score_article(
            headline="Airline charges extra fee to breathe recycled air",
            excerpt="Low-cost carrier introduces oxygen surcharge.",
            model="mistral:7b",
            prompt_fn=build_filter_prompt,
        )

    assert result["score"] == pytest.approx(0.91)
    assert result["rationale"] == "Classic economic grotesque — monetising a basic human need."
    assert "dark-wit" in result["flags"]


def test_score_article_returns_error_dict_on_malformed_json():
    """
    score_article should return {"score": None, "rationale": "parse error",
    "flags": []} without raising when the model returns unparseable text.
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    # Return plain prose instead of JSON — simulates a badly behaved model.
    mock_resp.json.return_value = {"response": "Sorry, I cannot evaluate this article."}

    with patch("filter.requests.post", return_value=mock_resp):
        result = score_article(
            headline="Some headline",
            excerpt=None,
            model="mistral:7b",
            prompt_fn=build_filter_prompt,
        )

    assert result["score"] is None
    assert result["rationale"] == "parse error"
    assert result["flags"] == []


def test_score_article_raises_runtime_error_when_ollama_unreachable():
    """
    score_article should raise RuntimeError with a helpful message when
    Ollama is not reachable.
    """
    import requests as req

    with patch("filter.requests.post", side_effect=req.exceptions.ConnectionError("refused")):
        with pytest.raises(RuntimeError, match="Could not reach Ollama"):
            score_article(
                headline="Some headline",
                excerpt=None,
                model="mistral:7b",
                prompt_fn=build_filter_prompt,
            )


# --------------------------------------------------------------------------- #
# run_filter integration test                                                  #
# --------------------------------------------------------------------------- #

def test_run_filter_processes_articles_and_returns_summary(tmp_path):
    """
    run_filter should score two unscored candidates, write results to the
    database, and return a summary dict with processed=2, errors=0.
    """
    # ------------------------------------------------------------------ #
    # Set up in-memory database with two candidate articles               #
    # ------------------------------------------------------------------ #

    engine = get_engine(db_path=":memory:")
    init_db(engine)

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        _seed_candidate(repo, url="http://example.com/1", headline="Headline One")
        _seed_candidate(repo, url="http://example.com/2", headline="Headline Two")

    # ------------------------------------------------------------------ #
    # Write a minimal filter-config.yaml                                  #
    # ------------------------------------------------------------------ #

    config_file = tmp_path / "filter-config.yaml"
    config_file.write_text(
        "project_id: test-project\n"
        "model: mistral:7b\n"
        "ollama_url: http://localhost:11434\n"
        "score_threshold: 0.6\n"
    )

    # ------------------------------------------------------------------ #
    # Mock score_article to return a fixed result for both articles       #
    # ------------------------------------------------------------------ #

    fake_result = {
        "score": 0.75,
        "rationale": "Good fit for the publication.",
        "flags": ["dark-wit", "capitalism-relevant"],
    }

    with patch("filter.score_article", return_value=fake_result), \
         patch("filter.get_engine", return_value=engine), \
         patch("filter.init_db"):

        summary = run_filter(config_path=config_file, db_path=":memory:")

    assert summary["processed"] == 2
    assert summary["errors"] == 0
    assert summary["skipped"] == 0

    # ------------------------------------------------------------------ #
    # Verify results were written to the database                         #
    # ------------------------------------------------------------------ #

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        candidates = repo.get_candidates(project_id="test-project")
        # Collect attributes inside the session to avoid DetachedInstanceError.
        scores = [a.filter_score for a in candidates]
        rationales = [a.filter_rationale for a in candidates]
        flags_stored = [a.filter_flags for a in candidates]

    # All candidates should now have a score written.
    assert all(s == pytest.approx(0.75) for s in scores)
    assert all(r == "Good fit for the publication." for r in rationales)
    assert all(f == "dark-wit,capitalism-relevant" for f in flags_stored)
