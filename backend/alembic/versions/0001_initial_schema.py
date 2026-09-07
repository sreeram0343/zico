"""Initial schema migration for trips, trip_segments, user_preferences, and audit_logs.

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-09-07 19:55:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create trips table
    op.create_table(
        "trips",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True, server_default="Untitled Journey"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="PLANNING"),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("currency", sa.String(length=10), nullable=False, server_default="USD"),
        sa.Column("state_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trips_user_id", "trips", ["user_id"], unique=False)
    op.create_index("ix_trips_status", "trips", ["status"], unique=False)

    # 2. Create trip_segments table
    op.create_table(
        "trip_segments",
        sa.Column("id", sa.String(length=100), nullable=False),
        sa.Column("trip_id", sa.String(length=36), nullable=False),
        sa.Column("segment_type", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_name", sa.String(length=255), nullable=True),
        sa.Column("location_iata", sa.String(length=10), nullable=True),
        sa.Column("location_lat", sa.Float(), nullable=True),
        sa.Column("location_lng", sa.Float(), nullable=True),
        sa.Column("cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("currency", sa.String(length=10), nullable=False, server_default="USD"),
        sa.Column("is_confirmed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trip_segments_trip_id", "trip_segments", ["trip_id"], unique=False)
    op.create_index("ix_trip_segments_segment_type", "trip_segments", ["segment_type"], unique=False)
    op.create_index("ix_trip_segments_start_time", "trip_segments", ["start_time"], unique=False)
    op.create_index("ix_trip_segments_location_iata", "trip_segments", ["location_iata"], unique=False)

    # 3. Create user_preferences table
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("home_airport", sa.String(length=10), nullable=True),
        sa.Column("preferred_currency", sa.String(length=10), nullable=False, server_default="USD"),
        sa.Column("preferences_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_preferences_user_id", "user_preferences", ["user_id"], unique=True)

    # 4. Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("trip_id", sa.String(length=36), nullable=False),
        sa.Column("action_type", sa.String(length=100), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_trip_id", "audit_logs", ["trip_id"], unique=False)
    op.create_index("ix_audit_logs_action_type", "audit_logs", ["action_type"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_action_type", table_name="audit_logs")
    op.drop_index("ix_audit_logs_trip_id", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_user_preferences_user_id", table_name="user_preferences")
    op.drop_table("user_preferences")

    op.drop_index("ix_trip_segments_location_iata", table_name="trip_segments")
    op.drop_index("ix_trip_segments_start_time", table_name="trip_segments")
    op.drop_index("ix_trip_segments_segment_type", table_name="trip_segments")
    op.drop_index("ix_trip_segments_trip_id", table_name="trip_segments")
    op.drop_table("trip_segments")

    op.drop_index("ix_trips_status", table_name="trips")
    op.drop_index("ix_trips_user_id", table_name="trips")
    op.drop_table("trips")
