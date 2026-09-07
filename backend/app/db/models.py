import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


class Trip(Base):
    """Stores overarching trip information, metadata, and state snapshot."""

    __tablename__ = "trips"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        default="Untitled Journey",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PLANNING",
        index=True,
    )
    total_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="USD",
    )
    state_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # 1-to-many relationship with itinerary segments
    segments: Mapped[List["TripSegmentModel"]] = relationship(
        back_populates="trip",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class TripSegmentModel(Base):
    """Normalized relational storage for individual trip segments."""

    __tablename__ = "trip_segments"

    id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
    )
    trip_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("trips.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    end_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    location_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    location_iata: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        index=True,
    )
    location_lat: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    location_lng: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="USD",
    )
    is_confirmed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Inverse relationship back to Trip
    trip: Mapped["Trip"] = relationship(
        back_populates="segments",
    )


class UserPreferenceModel(Base):
    """Persistent storage for traveler preferences."""

    __tablename__ = "user_preferences"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    home_airport: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )
    preferred_currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="USD",
    )
    preferences_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AuditLog(Base):
    """Immutable audit trail for human-in-the-loop decisions and state transitions."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    trip_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )
    action_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    payload: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
