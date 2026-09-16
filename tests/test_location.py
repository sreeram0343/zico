"""
Unit tests for ZICO's travel location and airport resolution tool.

Validates deterministic location normalization, IATA/ICAO recognition,
alias handling, multi-airport ambiguity, error handling, and serialization.
All tests run locally and offline with zero external network access.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict
from unittest.mock import patch

import pytest

from app.tools.location import (
    AirportCandidate,
    LocationResolutionError,
    LocationResolver,
    ResolvedLocation,
    resolve_location,
)


# ---------------------------------------------------------------------------
# Test 26: Empty Query Validation
# ---------------------------------------------------------------------------


def test_empty_query_validation() -> None:
    """Verify empty and whitespace-only queries fail validation before lookup."""
    for empty_input in ["", "   ", "\t\n  "]:
        result = resolve_location(empty_input)
        assert isinstance(result, ResolvedLocation)
        assert result.status == "invalid_query"
        assert result.location_type == "unknown"
        assert result.iata_code is None
        assert result.airport_name is None
        assert result.matches == []
        assert result.error_message is not None
        assert "empty" in result.error_message.lower() or "blank" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Test 27: Exact IATA Code Lookup
# ---------------------------------------------------------------------------


def test_exact_iata_code() -> None:
    """Verify standard 3-letter IATA codes are recognized deterministically."""
    result = resolve_location("TRV")
    assert result.status == "resolved"
    assert result.location_type == "airport"
    assert result.iata_code == "TRV"
    assert result.icao_code == "VOTV"
    assert result.airport_name == "Trivandrum International Airport"
    assert result.country == "India"
    assert result.country_code == "IN"
    assert result.city == "Thiruvananthapuram"
    assert result.confidence == 1.0
    assert result.match_type == "iata"
    assert len(result.matches) == 1
    assert result.matches[0].iata == "TRV"

    # Verify another international code (DXB)
    dxb_res = resolve_location("DXB")
    assert dxb_res.status == "resolved"
    assert dxb_res.iata_code == "DXB"
    assert dxb_res.airport_name == "Dubai International Airport"
    assert dxb_res.country == "United Arab Emirates"


# ---------------------------------------------------------------------------
# Test 28: Exact Airport Name Lookup
# ---------------------------------------------------------------------------


def test_exact_airport_name() -> None:
    """Verify exact airport names resolve to their authoritative IATA/ICAO data."""
    result = resolve_location("Dubai International Airport")
    assert result.status == "resolved"
    assert result.location_type == "airport"
    assert result.iata_code == "DXB"
    assert result.icao_code == "OMDB"
    assert result.airport_name == "Dubai International Airport"
    assert result.city == "Dubai"
    assert result.country == "United Arab Emirates"
    assert result.confidence == 1.0
    assert result.match_type == "airport_name"


# ---------------------------------------------------------------------------
# Test 29: City Name Resolution
# ---------------------------------------------------------------------------


def test_city_name_resolution() -> None:
    """Verify city name maps to canonical city and associated airport."""
    # Trivandrum has 1 commercial airport: TRV
    result = resolve_location("Trivandrum")
    assert result.status == "resolved"
    assert result.location_type == "city"
    assert result.normalized_name == "Thiruvananthapuram"
    assert result.city == "Thiruvananthapuram"
    assert result.airport_name == "Trivandrum International Airport"
    assert result.iata_code == "TRV"
    assert result.country == "India"
    assert len(result.matches) == 1

    # Dubai has 1 commercial passenger airport: DXB
    dubai_res = resolve_location("Dubai")
    assert dubai_res.status == "resolved"
    assert dubai_res.location_type == "city"
    assert dubai_res.city == "Dubai"
    assert dubai_res.iata_code == "DXB"
    assert dubai_res.airport_name == "Dubai International Airport"


# ---------------------------------------------------------------------------
# Test 30: City Alias Normalization
# ---------------------------------------------------------------------------


def test_city_alias_normalization() -> None:
    """Verify known city aliases normalize to canonical display names and airports."""
    # Thiruvananthapuram <-> Trivandrum
    result = resolve_location("Thiruvananthapuram")
    assert result.status == "resolved"
    assert result.normalized_name == "Thiruvananthapuram"
    assert result.city == "Thiruvananthapuram"
    assert result.iata_code == "TRV"
    assert result.airport_name == "Trivandrum International Airport"

    # Bombay <-> Mumbai
    bombay_res = resolve_location("Bombay")
    assert bombay_res.status == "resolved"
    assert bombay_res.normalized_name == "Mumbai"
    assert bombay_res.iata_code == "BOM"

    # Bangalore <-> Bengaluru
    blr_res = resolve_location("Bengaluru")
    assert blr_res.status == "resolved"
    assert blr_res.normalized_name == "Bengaluru"
    assert blr_res.iata_code == "BLR"


# ---------------------------------------------------------------------------
# Test 31: Case and Whitespace Normalization
# ---------------------------------------------------------------------------


def test_case_and_whitespace_normalization() -> None:
    """Verify case differences and leading/trailing/multiple whitespace are handled safely."""
    # Lowercase with padding
    res1 = resolve_location("  trv  ")
    assert res1.status == "resolved"
    assert res1.iata_code == "TRV"

    # Mixed case with extra internal spacing
    res2 = resolve_location("   dUbAi   InTeRnAtIoNaL   aIrPoRt   ")
    assert res2.status == "resolved"
    assert res2.iata_code == "DXB"

    # City with mixed case
    res3 = resolve_location("  tRiVaNdRuM  ")
    assert res3.status == "resolved"
    assert res3.iata_code == "TRV"


# ---------------------------------------------------------------------------
# Test 32: Unknown Location
# ---------------------------------------------------------------------------


def test_unknown_location() -> None:
    """Verify non-existent locations return not_found without raising an operational error."""
    result = resolve_location("DeliberatelyNonexistentCity999")
    assert result.status == "not_found"
    assert result.location_type == "unknown"
    assert result.iata_code is None
    assert result.airport_name is None
    assert result.matches == []
    assert result.error_message is None


# ---------------------------------------------------------------------------
# Test 33: Ambiguous Location
# ---------------------------------------------------------------------------


def test_ambiguous_location() -> None:
    """Verify queries matching multiple cities or multiple airports return ambiguous status."""
    # Springfield exists in multiple US states (MO, OH, IL, VT)
    springfield_res = resolve_location("Springfield")
    assert springfield_res.status == "ambiguous"
    assert springfield_res.location_type == "city"
    assert springfield_res.normalized_name == "Springfield"
    assert len(springfield_res.matches) >= 2
    # Do not arbitrarily select one airport code when ambiguous
    assert springfield_res.iata_code is None

    # New York has multiple commercial airports (JFK, LGA, etc.)
    ny_res = resolve_location("New York")
    assert ny_res.status == "ambiguous"
    assert ny_res.location_type == "city"
    assert ny_res.normalized_name == "New York"
    ny_iatas = [m.iata for m in ny_res.matches if m.iata]
    assert "JFK" in ny_iatas
    assert "LGA" in ny_iatas
    assert ny_res.iata_code is None


# ---------------------------------------------------------------------------
# Test 34: Invalid Input Types
# ---------------------------------------------------------------------------


def test_invalid_input_types() -> None:
    """Verify non-string or None inputs are cleanly handled with invalid_query status."""
    for invalid in [None, 12345, 99.9, [], {}, True]:
        result = resolve_location(invalid)
        assert result.status == "invalid_query"
        assert result.location_type == "unknown"
        assert result.matches == []
        assert result.error_message is not None


# ---------------------------------------------------------------------------
# Test 35: Data-Source Failure
# ---------------------------------------------------------------------------


def test_data_source_failure() -> None:
    """Verify operational loader failures raise LocationResolutionError and return error status."""
    with patch("airportsdata.load", side_effect=RuntimeError("Corrupt airport database")):
        resolver = LocationResolver(lazy_load=True)

        # Calling with raise_on_error=True raises LocationResolutionError
        with pytest.raises(LocationResolutionError) as exc_info:
            resolver.resolve("TRV", raise_on_error=True)
        assert "Data source initialization failed" in str(exc_info.value)

        # Calling with default returns error status safely without uncaught exception
        res = resolver.resolve("TRV", raise_on_error=False)
        assert res.status == "error"
        assert res.error_message is not None
        assert "Data source initialization failed" in res.error_message
        assert res.status != "not_found"


# ---------------------------------------------------------------------------
# Test 36: Malformed Provider / Data Response
# ---------------------------------------------------------------------------


def test_malformed_provider_data() -> None:
    """Verify malformed airport records are skipped or handled safely without crashes."""
    malformed_dataset: Dict[str, Any] = {
        # Missing name
        "NONAME": {"iata": "NONAME", "city": "TestCity"},
        # Non-dict record
        "NOTADICT": "This is a string not a dict",
        # Valid record
        "GOO": {
            "name": "Good International Airport",
            "iata": "GOO",
            "icao": "KGD1",
            "city": "Goodville",
            "country": "US",
            "lat": 30.0,
            "lon": -90.0,
        },
    }

    resolver = LocationResolver(iata_data=malformed_dataset)

    # Valid entry resolves
    res = resolver.resolve("GOO")
    assert res.status == "resolved"
    assert res.airport_name == "Good International Airport"

    # Malformed entries are safely omitted and not indexed
    res_bad = resolver.resolve("NONAME")
    assert res_bad.status == "not_found"

    res_str = resolver.resolve("NOTADICT")
    assert res_str.status == "not_found"


# ---------------------------------------------------------------------------
# Test 37: Optional Fields Handling
# ---------------------------------------------------------------------------


def test_optional_fields_handling() -> None:
    """Verify missing optional fields (ICAO, coordinates, country, tz) default safely to None."""
    minimal_record: Dict[str, Any] = {
        "MIN": {
            "name": "Minimal Airstrip",
            "iata": "MIN",
            "icao": None,
            "city": "Nowhere",
            "country": None,
            "lat": None,
            "lon": "invalid_string_lat",
            "tz": None,
            "subd": None,
        }
    }

    resolver = LocationResolver(iata_data=minimal_record)
    result = resolver.resolve("MIN")

    assert result.status == "resolved"
    assert result.airport_name == "Minimal Airstrip"
    assert result.iata_code == "MIN"
    assert result.icao_code is None
    assert result.country is None
    assert result.country_code is None
    assert result.latitude is None
    assert result.longitude is None
    assert result.timezone is None


# ---------------------------------------------------------------------------
# Test 38: JSON Serialization
# ---------------------------------------------------------------------------


def test_serialization() -> None:
    """Verify ResolvedLocation models serialize cleanly to JSON for LangGraph state."""
    result = resolve_location("TRV")
    assert result.status == "resolved"

    as_dict = result.to_dict()
    assert isinstance(as_dict, dict)
    assert as_dict["query"] == "TRV"
    assert as_dict["iata_code"] == "TRV"
    assert as_dict["airport_name"] == "Trivandrum International Airport"
    assert isinstance(as_dict["matches"], list)

    # Roundtrip through json.dumps and json.loads
    json_str = json.dumps(as_dict)
    reconstructed = json.loads(json_str)
    assert reconstructed["iata_code"] == "TRV"
    assert reconstructed["country"] == "India"
    assert reconstructed["status"] == "resolved"


# ---------------------------------------------------------------------------
# Test 39: Secret Protection in Logs
# ---------------------------------------------------------------------------


def test_secret_protection(caplog: pytest.LogCaptureFixture) -> None:
    """Verify logging does not expose secrets or sensitive authentication data."""
    caplog.set_level(logging.DEBUG)

    resolve_location("TRV")
    resolve_location("nonexistent_city_secret_test")

    captured = caplog.text
    assert "api_key" not in captured.lower()
    assert "token" not in captured.lower()
    assert "authorization" not in captured.lower()
    assert "password" not in captured.lower()


# ---------------------------------------------------------------------------
# Test 40: Deterministic Resolution
# ---------------------------------------------------------------------------


def test_resolution_determinism() -> None:
    """Verify same query repeatedly produces identical results."""
    res1 = resolve_location("TRV")
    res2 = resolve_location("TRV")
    res3 = resolve_location("TRV")

    assert res1.to_dict() == res2.to_dict() == res3.to_dict()


# ---------------------------------------------------------------------------
# Test 41: ICAO Code Recognition
# ---------------------------------------------------------------------------


def test_icao_code_lookup() -> None:
    """Verify standard 4-letter ICAO codes resolve to corresponding airport and IATA."""
    for icao, expected_iata, expected_name in [
        ("VOTV", "TRV", "Trivandrum International Airport"),
        ("OMDB", "DXB", "Dubai International Airport"),
        ("KJFK", "JFK", "John F Kennedy International Airport"),
        ("EGLL", "LHR", "London Heathrow Airport"),
    ]:
        res = resolve_location(icao)
        assert res.status == "resolved"
        assert res.location_type == "airport"
        assert res.icao_code == icao
        assert res.iata_code == expected_iata
        assert res.airport_name == expected_name
        assert res.match_type == "icao"


# ---------------------------------------------------------------------------
# Test 42: Compound and Natural Language Queries
# ---------------------------------------------------------------------------


def test_compound_and_natural_queries() -> None:
    """Verify natural phrases like 'New York JFK' and 'Dubai airport' resolve correctly."""
    # New York JFK -> JFK airport
    jfk_res = resolve_location("New York JFK")
    assert jfk_res.status == "resolved"
    assert jfk_res.iata_code == "JFK"
    assert jfk_res.airport_name == "John F Kennedy International Airport"
    assert jfk_res.city == "New York"

    # Dubai airport -> DXB
    dubai_res = resolve_location("Dubai airport")
    assert dubai_res.status == "resolved"
    assert dubai_res.iata_code == "DXB"
    assert dubai_res.city == "Dubai"

    # Airport nickname (Heathrow)
    lhr_res = resolve_location("Heathrow")
    assert lhr_res.status == "resolved"
    assert lhr_res.iata_code == "LHR"
    assert lhr_res.airport_name == "London Heathrow Airport"
