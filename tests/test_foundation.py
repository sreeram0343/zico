"""
Unit tests for Core Foundation & Data Modeling:
- Pydantic Settings startup validation
- Domain Schemas: TripState, TripSegment, UserPreferences
- SQLAlchemy ORM models: Trip, TripSegmentModel, UserPreferenceModel, AuditLog
- Alembic migration scripts
"""

from datetime import datetime, timedelta, timezone
import os
import sys
import pytest

# Ensure environment does not hang on remote vector probes
os.environ["QDRANT_URL"] = ":memory:"

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.core.config import Settings
from app.graph.state import (
    ActionStatus,
    ActionType,
    Location,
    SegmentType,
    TripConstraints,
    TripSegment,
    TripState,
    UserPreferences,
    ZicoGraphState,
)
from app.db.models import AuditLog, Base, Trip, TripSegmentModel, UserPreferenceModel


# ===========================================================================
# 1. Environment Configuration & Settings Tests
# ===========================================================================


def test_settings_valid_defaults():
    """Verify default initialization and helper properties of Settings."""
    cfg = Settings()
    assert cfg.PROJECT_NAME == "ZICO Intelligent Travel Operations"
    assert cfg.APP_ENV in ["development", "staging", "production", "test"]
    assert cfg.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert cfg.DATABASE_URL.startswith("postgresql+asyncpg://") or cfg.DATABASE_URL.startswith("postgresql://")
    assert cfg.sync_database_url.startswith("postgresql://")
    assert cfg.is_development is True or cfg.is_test is True or cfg.is_production is True


def test_settings_strict_app_env_validation():
    """Verify Settings rejects invalid APP_ENV values."""
    with pytest.raises(ValueError, match="Invalid APP_ENV"):
        Settings(APP_ENV="invalid_env_name")


def test_settings_strict_log_level_validation():
    """Verify Settings rejects invalid LOG_LEVEL values."""
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        Settings(LOG_LEVEL="VERBOSE")


def test_settings_strict_database_url_validation():
    """Verify Settings rejects unsupported database connection schemes."""
    with pytest.raises(ValueError, match="unsupported scheme"):
        Settings(DATABASE_URL="mysql://user:pass@localhost/db")


def test_settings_strict_redis_url_validation():
    """Verify Settings rejects invalid Redis URIs."""
    with pytest.raises(ValueError, match="must start with redis://"):
        Settings(REDIS_URL="http://localhost:6379")


def test_settings_strict_qdrant_url_validation():
    """Verify Settings rejects invalid Qdrant endpoints."""
    with pytest.raises(ValueError, match="must start with http"):
        Settings(QDRANT_URL="ftp://qdrant.local")


def test_settings_production_validation():
    """Verify production runtime validation fails when required secrets are missing."""
    prod_settings = Settings(
        APP_ENV="production",
        OPENAI_API_KEY="",
        DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/zico",
    )
    with pytest.raises(ValueError, match="Production startup validation failed"):
        prod_settings.validate_runtime()


# ===========================================================================
# 2. Domain Schemas: UserPreferences, TripSegment, TripState
# ===========================================================================


def test_user_preferences_defaults_and_validation():
    """Verify UserPreferences default values, case-normalization, and constraints."""
    pref = UserPreferences(
        preferred_airlines=["Emirates", "Qatar Airways"],
        preferred_cabin_class="business",
        seating_preference="window",
        currency="eur",
        home_airport="dac",
    )
    assert pref.preferred_cabin_class == "BUSINESS"
    assert pref.seating_preference == "WINDOW"
    assert pref.currency == "EUR"
    assert pref.home_airport == "DAC"
    assert pref.max_layover_hours == 8.0
    assert pref.hotel_rating_min == 3.5

    # Rejection of invalid currency length
    with pytest.raises(ValueError, match="Currency must be a 3-letter"):
        UserPreferences(currency="DOLLARS")

    # Rejection of invalid home airport
    with pytest.raises(ValueError, match="Home airport must be a 3-letter"):
        UserPreferences(home_airport="DHAKA")


def test_trip_segment_chronological_validation():
    """Verify TripSegment enforces end_time > start_time."""
    loc = Location(name="John F. Kennedy Intl", iata_code="JFK")
    t_dep = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)
    t_arr = datetime(2026, 9, 15, 16, 30, tzinfo=timezone.utc)

    seg = TripSegment(
        id="seg_flight_01",
        type=SegmentType.FLIGHT,
        title="Flight BA 178",
        start_time=t_dep,
        end_time=t_arr,
        location=loc,
        cost=450.0,
        currency="USD",
    )
    assert seg.duration_minutes == 390
    assert seg.cost == 450.0

    # Invalid: end_time before start_time
    with pytest.raises(ValueError, match="end_time must be strictly after start_time"):
        TripSegment(
            id="seg_invalid",
            type=SegmentType.FLIGHT,
            title="Impossible Flight",
            start_time=t_arr,
            end_time=t_dep,
            location=loc,
        )

    # Invalid: end_time equal to start_time
    with pytest.raises(ValueError, match="end_time must be strictly after start_time"):
        TripSegment(
            id="seg_invalid_equal",
            type=SegmentType.FLIGHT,
            title="Zero Duration Flight",
            start_time=t_dep,
            end_time=t_dep,
            location=loc,
        )


def test_trip_segment_negative_cost_rejection():
    """Verify TripSegment rejects negative costs."""
    loc = Location(name="Hotel Paris")
    t1 = datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 18, 11, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        TripSegment(
            id="hotel_01",
            type=SegmentType.HOTEL,
            title="Grand Hotel",
            start_time=t1,
            end_time=t2,
            location=loc,
            cost=-50.0,
        )


def test_trip_state_aggregation_and_operations():
    """Verify TripState lifecycle, automatic cost calculation, and segment operations."""
    loc1 = Location(name="London Heathrow", iata_code="LHR")
    loc2 = Location(name="Hilton London")
    t1 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)
    t4 = datetime(2026, 10, 4, 11, 0, tzinfo=timezone.utc)

    seg1 = TripSegment(
        id="flight_01",
        type=SegmentType.FLIGHT,
        title="Flight BA 101",
        start_time=t1,
        end_time=t2,
        location=loc1,
        cost=600.0,
    )
    seg2 = TripSegment(
        id="hotel_01",
        type=SegmentType.HOTEL,
        title="Hilton Park Lane",
        start_time=t3,
        end_time=t4,
        location=loc2,
        cost=900.0,
    )

    trip = TripState(
        trip_id="trip_london_2026",
        user_id="user_john",
        itinerary=[seg1, seg2],
    )

    # Automatic cost calculation
    assert trip.total_cost == 1500.0
    assert trip.start_date == t1
    assert trip.end_date == t4
    assert len(trip.get_segments_by_type(SegmentType.FLIGHT)) == 1
    assert len(trip.get_segments_by_type(SegmentType.HOTEL)) == 1

    # Remove segment
    assert trip.remove_segment("flight_01") is True
    assert len(trip.itinerary) == 1
    assert trip.total_cost == 900.0

    # Add segment
    trip.add_segment(seg1)
    assert len(trip.itinerary) == 2
    assert trip.total_cost == 1500.0


def test_trip_state_temporal_overlap_detection():
    """Verify TripState detects concurrent/overlapping flights."""
    loc = Location(name="Airport")
    t1 = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 1, 14, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)  # Overlaps with t1-t2
    t4 = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)

    flight1 = TripSegment(id="f1", type=SegmentType.FLIGHT, title="Flight 1", start_time=t1, end_time=t2, location=loc)
    flight2 = TripSegment(id="f2", type=SegmentType.FLIGHT, title="Flight 2", start_time=t3, end_time=t4, location=loc)

    trip = TripState(trip_id="trip_conflict", user_id="user_1", itinerary=[flight1, flight2])
    overlaps = trip.has_temporal_overlap()
    assert len(overlaps) == 1
    assert overlaps[0][0].id == "f1"
    assert overlaps[0][1].id == "f2"


def test_zico_graph_state_to_trip_state():
    """Verify bridging between LangGraph runtime state and TripState."""
    loc = Location(name="Paris", iata_code="CDG")
    t1 = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    seg = TripSegment(id="fl_paris", type=SegmentType.FLIGHT, title="Air France", start_time=t1, end_time=t2, location=loc, cost=300.0)

    graph_state = ZicoGraphState(
        trip_id="trip_paris",
        user_id="user_sarah",
        itinerary=[seg],
        preferences=UserPreferences(currency="EUR"),
    )

    trip_domain = graph_state.to_trip_state(title="Autumn in Paris")
    assert isinstance(trip_domain, TripState)
    assert trip_domain.trip_id == "trip_paris"
    assert trip_domain.title == "Autumn in Paris"
    assert trip_domain.preferences.currency == "EUR"
    assert trip_domain.total_cost == 300.0


# ===========================================================================
# 3. Database ORM Models Tests
# ===========================================================================


def test_orm_models_instantiation():
    """Verify SQLAlchemy 2.0 ORM models instantiation and column attributes."""
    # 1. Trip Model
    trip = Trip(
        id="trip_orm_01",
        user_id="user_db_123",
        title="Summer Vacation",
        status="CONFIRMED",
        total_cost=1200.0,
        currency="USD",
        state_json={"trip_id": "trip_orm_01"},
    )
    assert trip.id == "trip_orm_01"
    assert trip.title == "Summer Vacation"
    assert trip.status == "CONFIRMED"
    assert trip.total_cost == 1200.0

    # 2. TripSegmentModel
    now = datetime.now(timezone.utc)
    seg = TripSegmentModel(
        id="seg_orm_01",
        trip_id="trip_orm_01",
        segment_type="FLIGHT",
        title="Flight Emirates EK 202",
        start_time=now,
        end_time=now + timedelta(hours=6),
        location_name="Dubai International Airport",
        location_iata="DXB",
        cost=850.0,
        currency="USD",
        is_confirmed=True,
    )
    assert seg.trip_id == "trip_orm_01"
    assert seg.location_iata == "DXB"
    assert seg.is_confirmed is True

    # 3. UserPreferenceModel
    pref_model = UserPreferenceModel(
        user_id="user_db_123",
        home_airport="DAC",
        preferred_currency="USD",
        preferences_json={"cabin": "BUSINESS"},
    )
    assert pref_model.user_id == "user_db_123"
    assert pref_model.home_airport == "DAC"

    # 4. AuditLog
    log = AuditLog(
        trip_id="trip_orm_01",
        action_type="BOOKING_APPROVAL",
        payload={"approved_by": "user"},
    )
    assert log.trip_id == "trip_orm_01"
    assert log.action_type == "BOOKING_APPROVAL"


# ===========================================================================
# 4. Alembic Migration Script Verification
# ===========================================================================


def test_alembic_configuration_and_migration():
    """Verify Alembic environment loads target metadata and initial migration."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
    ini_path = os.path.join(backend_dir, "alembic.ini")
    assert os.path.exists(ini_path)

    alembic_cfg = Config(ini_path)
    alembic_cfg.set_main_option("script_location", os.path.join(backend_dir, "alembic"))
    script = ScriptDirectory.from_config(alembic_cfg)
    heads = script.get_heads()

    assert len(heads) == 1
    assert heads[0] == "0001_initial_schema"

    rev = script.get_revision("0001_initial_schema")
    assert rev is not None
    assert rev.doc.startswith("Initial schema migration")


if __name__ == "__main__":
    print("Running test_settings_valid_defaults...")
    test_settings_valid_defaults()
    print("Running test_settings_strict_app_env_validation...")
    test_settings_strict_app_env_validation()
    print("Running test_settings_strict_log_level_validation...")
    test_settings_strict_log_level_validation()
    print("Running test_settings_strict_database_url_validation...")
    test_settings_strict_database_url_validation()
    print("Running test_settings_strict_redis_url_validation...")
    test_settings_strict_redis_url_validation()
    print("Running test_settings_strict_qdrant_url_validation...")
    test_settings_strict_qdrant_url_validation()
    print("Running test_settings_production_validation...")
    test_settings_production_validation()
    print("Running test_user_preferences_defaults_and_validation...")
    test_user_preferences_defaults_and_validation()
    print("Running test_trip_segment_chronological_validation...")
    test_trip_segment_chronological_validation()
    print("Running test_trip_segment_negative_cost_rejection...")
    test_trip_segment_negative_cost_rejection()
    print("Running test_trip_state_aggregation_and_operations...")
    test_trip_state_aggregation_and_operations()
    print("Running test_trip_state_temporal_overlap_detection...")
    test_trip_state_temporal_overlap_detection()
    print("Running test_zico_graph_state_to_trip_state...")
    test_zico_graph_state_to_trip_state()
    print("Running test_orm_models_instantiation...")
    test_orm_models_instantiation()
    print("Running test_alembic_configuration_and_migration...")
    test_alembic_configuration_and_migration()

    print("\n=======================================================")
    print("SUCCESS: ALL FOUNDATION & DATA MODELING TESTS PASSED!")
    print("=======================================================")
