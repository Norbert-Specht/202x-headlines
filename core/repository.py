"""
ArticleRepository — all database operations for the article pipeline.

Every method is scoped to a project_id so the same database can serve
multiple projects without cross-contamination.
"""

from datetime import datetime, UTC

from sqlalchemy.orm import Session

from core.models import Article, ArticleReview


class ArticleRepository:
    """
    Data-access layer for Article and ArticleReview records.

    Accepts a SQLAlchemy Session at construction time so the caller controls
    transaction boundaries (useful in tests and in the web layer).
    """

    def __init__(self, session: Session):
        """
        Initialise the repository with an active SQLAlchemy session.

        Parameters
        ----------
        session : Session
            An open SQLAlchemy session. The caller is responsible for
            committing, rolling back, and closing it.
        """
        self.session = session

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def add_candidate(
        self,
        project_id: str,
        headline: str,
        source_url: str,
        source_name: str | None,
        excerpt: str | None,
        date_scraped: datetime,
        image_url: str | None = None,
    ) -> Article:
        """
        Create a new article in "candidate" status and persist it.

        Parameters
        ----------
        project_id   : Namespace for the article (e.g. "202x-headlines").
        headline     : The article headline as scraped.
        source_url   : Canonical URL — must be unique across the table.
        source_name  : Human-readable outlet name, e.g. "Reuters".
        excerpt      : Lede or first paragraph, used in the review UI.
        date_scraped : When the scraper first fetched this URL.
        image_url    : Header image URL from the feed or Reddit thumbnail.
                       None if the source did not provide an image.

        Returns
        -------
        Article
            The newly created and persisted Article instance.
        """
        article = Article(
            project_id=project_id,
            headline=headline,
            source_url=source_url,
            source_name=source_name,
            excerpt=excerpt,
            date_scraped=date_scraped,
            image_url=image_url,
            status="candidate",
        )
        self.session.add(article)
        self.session.flush()  # assigns article.id without committing
        return article

    def accept(
        self,
        article_id: int,
        date_published: datetime | None = None,
    ) -> Article:
        """
        Accept an article: set status to "accepted", record the review, and
        populate date_published and publication_year.

        Parameters
        ----------
        article_id     : Primary key of the article to accept.
        date_published : Publication date. Defaults to utcnow if not given.

        Returns
        -------
        Article
            The updated Article instance.
        """
        article = self.session.get(Article, article_id)

        if date_published is None:
            date_published = datetime.now(UTC)

        article.status = "accepted"
        article.date_published = date_published

        # Extract the year so archive queries can filter by year without
        # a full datetime comparison.
        article.publication_year = date_published.year

        review = ArticleReview(
            article_id=article_id,
            reviewed_at=datetime.now(UTC),
            decision="accept",
        )
        self.session.add(review)
        self.session.flush()
        return article

    def reject(
        self,
        article_id: int,
        reason: str | None = None,
        notes: str | None = None,
    ) -> Article:
        """
        Reject an article: set status to "rejected" and record the review.

        Parameters
        ----------
        article_id : Primary key of the article to reject.
        reason     : Short structured rejection reason (used as ML label).
        notes      : Free-form editor notes (richer ML training signal).

        Returns
        -------
        Article
            The updated Article instance.
        """
        article = self.session.get(Article, article_id)

        article.status = "rejected"

        review = ArticleReview(
            article_id=article_id,
            reviewed_at=datetime.now(UTC),
            decision="reject",
            rejection_reason=reason,
            reviewer_notes=notes,
        )
        self.session.add(review)
        self.session.flush()
        return article

    def set_filter_result(
        self,
        article_id: int,
        score: float | None,
        rationale: str | None,
        flags: list[str],
    ) -> Article:
        """
        Persist the full LLM filter result for an article in one call.

        Updates filter_score, filter_rationale, and filter_flags together
        so the three fields are always written atomically.

        Parameters
        ----------
        article_id : Primary key of the article to update.
        score      : Float in [0.0, 1.0], or None if scoring failed.
        rationale  : One or two sentence explanation from the model.
        flags      : List of flag strings from the fixed vocabulary.
                     Stored as a comma-separated string internally.

        Returns
        -------
        Article
            The updated Article instance.
        """
        article = self.session.get(Article, article_id)

        article.filter_score = score
        article.filter_rationale = rationale
        # Join the list into a comma-separated string for storage.
        # Empty list is stored as None rather than an empty string.
        article.filter_flags = ",".join(flags) if flags else None

        self.session.flush()
        return article

    def set_filter_score(self, article_id: int, score: float) -> Article:
        """
        Persist the LLM relevance score for an article.

        Parameters
        ----------
        article_id : Primary key of the article to update.
        score      : Float in [0.0, 1.0] from the LLM filter stage.

        Returns
        -------
        Article
            The updated Article instance.
        """
        article = self.session.get(Article, article_id)
        article.filter_score = score
        self.session.flush()
        return article

    # ------------------------------------------------------------------ #
    # Read operations                                                      #
    # ------------------------------------------------------------------ #

    def get_candidates(self, project_id: str) -> list[Article]:
        """
        Return all candidate articles for a project, oldest first.

        Ordered by date_scraped ascending so the review queue processes
        articles in the order they were discovered.

        Parameters
        ----------
        project_id : Project namespace to filter by.

        Returns
        -------
        list[Article]
        """
        return (
            self.session.query(Article)
            .filter(
                Article.project_id == project_id,
                Article.status == "candidate",
            )
            .order_by(Article.date_scraped.asc())
            .all()
        )

    def get_article_by_url(self, source_url: str) -> Article | None:
        """
        Look up an article by its source URL.

        Used by the scraper to avoid inserting duplicates before attempting
        an insert.

        Parameters
        ----------
        source_url : The canonical URL to look up.

        Returns
        -------
        Article | None
            The matching Article, or None if not found.
        """
        return (
            self.session.query(Article)
            .filter(Article.source_url == source_url)
            .one_or_none()
        )

    def get_accepted(self, project_id: str) -> list[Article]:
        """
        Return all accepted articles for a project, most recently published first.

        Parameters
        ----------
        project_id : Project namespace to filter by.

        Returns
        -------
        list[Article]
        """
        return (
            self.session.query(Article)
            .filter(
                Article.project_id == project_id,
                Article.status == "accepted",
            )
            .order_by(Article.date_published.desc())
            .all()
        )

    def get_accepted_by_year(self, project_id: str, year: int) -> list[Article]:
        """
        Return accepted articles for a project filtered by publication year.

        This is the primary query for the year-based archive display on the
        front end. Uses the indexed publication_year column, not a datetime
        range, for efficiency.

        Each returned Article includes all content fields: headline,
        source_url, excerpt, and image_url (nullable — not all articles have
        a header image). Callers should guard against image_url being None.

        Parameters
        ----------
        project_id : Project namespace to filter by.
        year       : Four-digit year, e.g. 2025.

        Returns
        -------
        list[Article]
            Ordered by date_published descending (newest first within year).
        """
        return (
            self.session.query(Article)
            .filter(
                Article.project_id == project_id,
                Article.status == "accepted",
                Article.publication_year == year,
            )
            .order_by(Article.date_published.desc())
            .all()
        )

    def get_rejected(self, project_id: str) -> list[Article]:
        """
        Return all rejected articles for a project, with their review records
        eagerly loaded.

        Primarily used to build the ML training dataset.

        Parameters
        ----------
        project_id : Project namespace to filter by.

        Returns
        -------
        list[Article]
        """
        return (
            self.session.query(Article)
            .filter(
                Article.project_id == project_id,
                Article.status == "rejected",
            )
            .all()
        )
