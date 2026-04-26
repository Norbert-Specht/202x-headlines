"""
Smoke tests for ArticleRepository against an in-memory SQLite database.

No mocking — every test hits a real (in-memory) database so the ORM
queries, constraints, and relationships are all exercised for real.
"""

from datetime import datetime

import pytest

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def session():
    """
    Yield a live session backed by a fresh in-memory SQLite database.

    A new engine and schema are created for each test so there is no state
    leakage between tests.
    """
    engine = get_engine(db_path=":memory:")
    init_db(engine)
    with get_session(engine) as s:
        yield s


@pytest.fixture
def repo(session):
    """Return an ArticleRepository bound to the test session."""
    return ArticleRepository(session)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def make_candidate(repo: ArticleRepository, project_id="test-project", url="http://example.com/1"):
    """Add a minimal candidate article and return it."""
    return repo.add_candidate(
        project_id=project_id,
        headline="Scientists Confirm Sun Still Hot",
        source_url=url,
        source_name="The Daily Obvious",
        excerpt="Researchers have once again verified solar temperatures.",
        date_scraped=datetime(2025, 6, 1, 12, 0, 0),
        image_url=None,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

def test_add_candidate_and_retrieve(repo):
    """Adding a candidate should make it visible in get_candidates."""
    article = make_candidate(repo)

    candidates = repo.get_candidates(project_id="test-project")

    assert len(candidates) == 1
    assert candidates[0].id == article.id
    assert candidates[0].status == "candidate"
    assert candidates[0].headline == "Scientists Confirm Sun Still Hot"


def test_accept_sets_status_and_creates_review(repo):
    """
    Accepting an article should:
    - change status to "accepted"
    - create an ArticleReview with decision="accept"
    - set date_published
    - populate publication_year from date_published
    """
    article = make_candidate(repo)
    pub_date = datetime(2025, 3, 15, 9, 0, 0)

    accepted = repo.accept(article_id=article.id, date_published=pub_date)

    assert accepted.status == "accepted"
    assert accepted.date_published == pub_date
    assert accepted.publication_year == 2025  # extracted from pub_date

    assert len(accepted.reviews) == 1
    assert accepted.reviews[0].decision == "accept"


def test_reject_sets_status_and_creates_review(repo):
    """
    Rejecting an article should change status to "rejected" and record the
    rejection reason in an ArticleReview.
    """
    article = make_candidate(repo)

    rejected = repo.reject(
        article_id=article.id,
        reason="not_absurd_enough",
        notes="This headline is merely unfortunate, not satirically absurd.",
    )

    assert rejected.status == "rejected"
    assert len(rejected.reviews) == 1
    assert rejected.reviews[0].decision == "reject"
    assert rejected.reviews[0].rejection_reason == "not_absurd_enough"
    assert "satirically absurd" in rejected.reviews[0].reviewer_notes


def test_get_article_by_url_returns_existing(repo):
    """get_article_by_url should return the article if the URL is known."""
    article = make_candidate(repo, url="http://example.com/unique-url")

    found = repo.get_article_by_url("http://example.com/unique-url")

    assert found is not None
    assert found.id == article.id


def test_get_article_by_url_returns_none_for_unknown(repo):
    """get_article_by_url should return None for an unseen URL."""
    result = repo.get_article_by_url("http://example.com/never-seen")
    assert result is None


def test_filter_score_is_persisted(repo):
    """set_filter_score should update and persist the filter_score field."""
    article = make_candidate(repo)

    updated = repo.set_filter_score(article_id=article.id, score=0.87)

    assert updated.filter_score == pytest.approx(0.87)


def test_project_id_isolation(repo):
    """
    Candidates from project A must not appear in project B's queue.
    """
    make_candidate(repo, project_id="project-a", url="http://example.com/a")
    make_candidate(repo, project_id="project-b", url="http://example.com/b")

    a_candidates = repo.get_candidates(project_id="project-a")
    b_candidates = repo.get_candidates(project_id="project-b")

    assert len(a_candidates) == 1
    assert a_candidates[0].source_url == "http://example.com/a"

    assert len(b_candidates) == 1
    assert b_candidates[0].source_url == "http://example.com/b"


def test_multi_reason_rejection(repo):
    """
    Rejecting with two categories should store both as a comma-separated
    string in rejection_reason, and persist the optional note in reviewer_notes.

    This mirrors what review.py does: it joins selected short_keys with ", "
    before passing them to repo.reject().
    """
    article = make_candidate(repo)

    rejected = repo.reject(
        article_id=article.id,
        reason="not_absurd_enough, wrong_kind_of_dark",
        notes="Depressing story about factory closures — no wit, no absurdity.",
    )

    assert rejected.status == "rejected"
    assert rejected.reviews[0].rejection_reason == "not_absurd_enough, wrong_kind_of_dark"
    assert "no wit" in rejected.reviews[0].reviewer_notes


def test_get_accepted_by_year_filters_correctly(repo):
    """
    get_accepted_by_year should return only articles whose publication_year
    matches the requested year, not articles from other years.
    """
    article_2024 = make_candidate(repo, url="http://example.com/2024")
    article_2025 = make_candidate(repo, url="http://example.com/2025")
    article_2026 = make_candidate(repo, url="http://example.com/2026")

    repo.accept(article_id=article_2024.id, date_published=datetime(2024, 11, 1))
    repo.accept(article_id=article_2025.id, date_published=datetime(2025, 5, 20))
    repo.accept(article_id=article_2026.id, date_published=datetime(2026, 2, 14))

    results_2025 = repo.get_accepted_by_year(project_id="test-project", year=2025)

    assert len(results_2025) == 1
    assert results_2025[0].publication_year == 2025
    assert results_2025[0].source_url == "http://example.com/2025"
