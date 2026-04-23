"""
SQLAlchemy ORM models for the article tracker.

Two tables: Article (the scraped/curated record) and ArticleReview (the
accept/reject decision log). ArticleReview is also the future ML training
signal — rejection reasons and reviewer notes are captured here.
"""

from datetime import datetime, UTC

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Article(Base):
    """
    Represents a single news article that has been scraped and is moving
    through the review pipeline (candidate → accepted | rejected).

    project_id scopes every query so the same database can serve multiple
    projects (202x-headlines, climate-hub, etc.) without table duplication.
    """

    __tablename__ = "articles"

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Namespaces articles by project — never omit this in queries.
    project_id = Column(String, nullable=False, index=True)

    # ------------------------------------------------------------------ #
    # Content                                                              #
    # ------------------------------------------------------------------ #

    headline = Column(String, nullable=False)

    # Unique constraint prevents the scraper from inserting duplicates.
    source_url = Column(String, nullable=False, unique=True)

    # Human-readable outlet name, e.g. "Reuters", "BBC News".
    source_name = Column(String, nullable=True)

    # Lede or first paragraph — surfaced in the review UI so editors don't
    # need to open the original link for every candidate.
    excerpt = Column(Text, nullable=True)

    # ------------------------------------------------------------------ #
    # Pipeline state                                                       #
    # ------------------------------------------------------------------ #

    # When the scraper first saw this URL.
    date_scraped = Column(DateTime, nullable=False)

    # One of: "candidate", "accepted", "rejected".
    status = Column(String, nullable=False)

    # Set when an article is accepted and scheduled for publication.
    date_published = Column(DateTime, nullable=True)

    # Year extracted from date_published. Indexed because the archive UI
    # queries by year constantly (e.g. "show me all 2024 articles").
    publication_year = Column(Integer, nullable=True, index=True)

    # LLM relevance score 0.0–1.0. Set during the filter stage, before
    # human review. High scores bubble up in the review queue.
    filter_score = Column(Float, nullable=True)

    # ------------------------------------------------------------------ #
    # Timestamps                                                           #
    # ------------------------------------------------------------------ #

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    # ------------------------------------------------------------------ #
    # Relationships                                                        #
    # ------------------------------------------------------------------ #

    # An article can be reviewed more than once if a decision is reversed.
    reviews = relationship("ArticleReview", back_populates="article")

    def __repr__(self):
        return (
            f"<Article id={self.id} project={self.project_id!r} "
            f"status={self.status!r} headline={self.headline[:50]!r}>"
        )


class ArticleReview(Base):
    """
    Records each accept/reject decision made on an Article.

    Kept as a separate table (rather than fields on Article) so that
    decision history is preserved if a decision is later reversed, and so
    reviewer_notes can accumulate as a training corpus.
    """

    __tablename__ = "article_reviews"

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #

    id = Column(Integer, primary_key=True, autoincrement=True)

    # The article this decision belongs to.
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)

    # ------------------------------------------------------------------ #
    # Decision                                                             #
    # ------------------------------------------------------------------ #

    reviewed_at = Column(DateTime, nullable=False)

    # "accept" or "reject".
    decision = Column(String, nullable=False)

    # Short structured reason for rejection — used as an ML label later.
    rejection_reason = Column(String, nullable=True)

    # Free-form editor notes — richer signal for fine-tuning than the
    # structured reason alone.
    reviewer_notes = Column(Text, nullable=True)

    # ------------------------------------------------------------------ #
    # Relationships                                                        #
    # ------------------------------------------------------------------ #

    article = relationship("Article", back_populates="reviews")

    def __repr__(self):
        return (
            f"<ArticleReview id={self.id} article_id={self.article_id} "
            f"decision={self.decision!r}>"
        )
