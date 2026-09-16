"""
Travel location and airport resolution module for ZICO.

This module resolves natural-language location and travel queries (such as
"Trivandrum", "Thiruvananthapuram", "Dubai", "Dubai airport", "New York",
"New York JFK", "TRV", "DXB") into structured, normalized geographic and
airport representations.

Architecture:
    User Input / Flight Agent -> LocationResolver -> Normalized ResolvedLocation -> AviationStack

Features:
    - Deterministic, authoritative data backed by the `airportsdata` database.
    - Full support for 3-letter IATA codes and 4-letter ICAO codes.
    - City name normalization and canonical alias support (e.g. Trivandrum -> Thiruvananthapuram).
    - Multi-airport / ambiguous location detection (e.g. "Springfield", "London", "New York").
    - Airport nickname and compound query resolution (e.g. "New York JFK", "Dubai airport").
    - Pure, JSON-serializable Pydantic models compatible with LangGraph state.
    - Fast in-memory lookup with zero external network dependencies.
    - Secret-safe logging via `app.core.logging`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import airportsdata
from pydantic import BaseModel, Field
import pycountry

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical Mappings & Aliases
# ---------------------------------------------------------------------------

# Alternate or historical city names mapped to the key present in airportsdata
CITY_ALIASES: Dict[str, str] = {
    "thiruvananthapuram": "trivandrum",
    "bengaluru": "bangalore",
    "bombay": "mumbai",
    "madras": "chennai",
    "calcutta": "kolkata",
    "cochin": "kochi",
    "calicut": "kozhikode",
    "peking": "beijing",
    "saigon": "ho chi minh city",
    "rangoon": "yangon",
    "kiev": "kyiv",
    "canton": "guangzhou",
    "leningrad": "st petersburg",
}

# Preferred canonical modern display names for cities
CITY_DISPLAY_NAMES: Dict[str, str] = {
    "trivandrum": "Thiruvananthapuram",
    "bangalore": "Bengaluru",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "madras": "Chennai",
    "chennai": "Chennai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "kozhikode": "Kozhikode",
    "peking": "Beijing",
    "beijing": "Beijing",
    "saigon": "Ho Chi Minh City",
    "yangon": "Yangon",
    "kyiv": "Kyiv",
}

# Common airport nicknames mapped directly to their primary IATA code
AIRPORT_ALIASES: Dict[str, str] = {
    "jfk": "JFK",
    "heathrow": "LHR",
    "gatwick": "LGW",
    "changi": "SIN",
    "haneda": "HND",
    "narita": "NRT",
    "schiphol": "AMS",
    "fiumicino": "FCO",
    "orly": "ORY",
    "o'hare": "ORD",
    "ohare": "ORD",
    "midway": "MDW",
    "newark": "EWR",
    "laguardia": "LGA",
}

# Substrings indicating military air bases or non-commercial facilities
MILITARY_INDICATORS = (
    "air base",
    " afb",
    "air force",
    "naval air",
    "army airfield",
    "raf ",
)


def _is_military_facility(name: Optional[str]) -> bool:
    """Return True if the airport name indicates a military or non-commercial facility."""
    if not name:
        return False
    low = name.lower()
    return any(indicator in low for indicator in MILITARY_INDICATORS)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LocationResolutionError(Exception):
    """Raised when an underlying data source failure or operational error occurs."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AirportCandidate(BaseModel):
    """Structured representation of an airport match candidate."""

    name: str
    iata: Optional[str] = None
    icao: Optional[str] = None
    city: Optional[str] = None
    subdivision: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None


class ResolvedLocation(BaseModel):
    """
    Authoritative normalized travel location result for ZICO agents.

    Statuses:
        - 'resolved': Unambiguous location match found.
        - 'ambiguous': Multiple valid airports or geographic locations match.
        - 'not_found': Valid query, but no matching airport or city exists in dataset.
        - 'invalid_query': Query is empty, blank, or of invalid type.
        - 'error': Operational error occurred during resolution.
    """

    query: str
    status: str  # "resolved" | "ambiguous" | "not_found" | "invalid_query" | "error"
    location_type: Optional[str] = None  # "airport" | "city" | "country" | "unknown"
    normalized_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    airport_name: Optional[str] = None
    iata_code: Optional[str] = None
    icao_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    confidence: Optional[float] = None
    match_type: Optional[str] = None
    matches: List[AirportCandidate] = Field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize into a JSON-compatible dictionary for LangGraph state."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Location Resolver
# ---------------------------------------------------------------------------


class LocationResolver:
    """
    Deterministic resolver for travel locations, cities, and airports.
    """

    def __init__(
        self,
        iata_data: Optional[Dict[str, Any]] = None,
        icao_data: Optional[Dict[str, Any]] = None,
        lazy_load: bool = False,
    ) -> None:
        """
        Initialize the location resolver.

        Args:
            iata_data: Optional pre-loaded IATA database mapping for tests/mocks.
            icao_data: Optional pre-loaded ICAO database mapping for tests/mocks.
            lazy_load: If True, defer index loading until first query execution.
        """
        self._raw_iata: Optional[Dict[str, Any]] = iata_data
        self._raw_icao: Optional[Dict[str, Any]] = icao_data

        self._iata_index: Dict[str, Dict[str, Any]] = {}
        self._icao_index: Dict[str, Dict[str, Any]] = {}
        self._airport_name_index: Dict[str, List[Dict[str, Any]]] = {}
        self._city_index: Dict[str, List[Dict[str, Any]]] = {}
        self._is_initialized = False

        if not lazy_load:
            self._initialize()

    def _initialize(self) -> None:
        """Load airport datasets and construct fast in-memory lookup indices."""
        try:
            # 1. Load IATA database
            if self._raw_iata is not None:
                iata_db = self._raw_iata
            else:
                iata_db = airportsdata.load("IATA")

            # 2. Load ICAO database (optional fallback for non-IATA fields)
            if self._raw_icao is not None:
                icao_db = self._raw_icao
            else:
                try:
                    icao_db = airportsdata.load("ICAO")
                except Exception:
                    icao_db = {}

            if not isinstance(iata_db, dict):
                raise LocationResolutionError("Loaded IATA dataset must be a dictionary.")

            # 3. Build indices
            self._iata_index = {}
            self._icao_index = {}
            self._airport_name_index = {}
            self._city_index = {}

            # Seed ICAO index with full ICAO dataset first
            if isinstance(icao_db, dict):
                for icao_code, apt in icao_db.items():
                    if isinstance(apt, dict) and isinstance(icao_code, str):
                        self._icao_index[icao_code.upper().strip()] = apt

            # Process authoritative IATA dataset
            for iata_code, apt in iata_db.items():
                if not isinstance(apt, dict) or not isinstance(iata_code, str):
                    continue

                clean_iata = iata_code.upper().strip()
                self._iata_index[clean_iata] = apt

                # Map ICAO if present
                raw_icao = apt.get("icao")
                if isinstance(raw_icao, str) and raw_icao.strip():
                    self._icao_index[raw_icao.upper().strip()] = apt

                # Index by airport name
                raw_name = apt.get("name")
                if isinstance(raw_name, str) and raw_name.strip():
                    clean_name = raw_name.lower().strip()
                    self._airport_name_index.setdefault(clean_name, []).append(apt)

                # Index by city
                raw_city = apt.get("city")
                if isinstance(raw_city, str) and raw_city.strip():
                    clean_city = raw_city.lower().strip()
                    self._city_index.setdefault(clean_city, []).append(apt)

            self._is_initialized = True
            logger.debug(
                "LocationResolver initialized successfully (iata=%d, icao=%d, cities=%d)",
                len(self._iata_index),
                len(self._icao_index),
                len(self._city_index),
            )
        except LocationResolutionError:
            raise
        except Exception as exc:
            logger.error("Failed to initialize LocationResolver: %s", exc)
            raise LocationResolutionError(f"Data source initialization failed: {exc}") from exc

    def _ensure_initialized(self) -> None:
        """Ensure lookup indices are populated before query execution."""
        if not self._is_initialized:
            self._initialize()

    def _normalize_record(self, raw: Any) -> Optional[AirportCandidate]:
        """
        Transform a raw airport mapping into a validated, typed AirportCandidate model.
        Returns None if the record is malformed.
        """
        if not isinstance(raw, dict):
            return None

        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            return None

        # Resolve Country Name
        country_code = raw.get("country")
        country_name: Optional[str] = None
        if isinstance(country_code, str) and country_code.strip():
            country_code = country_code.strip().upper()
            try:
                c = pycountry.countries.get(alpha_2=country_code)
                country_name = c.name if c else country_code
            except Exception:
                country_name = country_code
        else:
            country_code = None

        # Validate Coordinates
        lat = raw.get("lat")
        lon = raw.get("lon")
        latitude = float(lat) if isinstance(lat, (int, float)) else None
        longitude = float(lon) if isinstance(lon, (int, float)) else None

        # Resolve Canonical Display City
        raw_city = raw.get("city")
        display_city: Optional[str] = None
        if isinstance(raw_city, str) and raw_city.strip():
            city_key = raw_city.strip().lower()
            display_city = CITY_DISPLAY_NAMES.get(city_key, raw_city.strip())

        raw_iata = raw.get("iata")
        iata = raw_iata.strip().upper() if isinstance(raw_iata, str) and raw_iata.strip() else None

        raw_icao = raw.get("icao")
        icao = raw_icao.strip().upper() if isinstance(raw_icao, str) and raw_icao.strip() else None

        raw_tz = raw.get("tz")
        tz = raw_tz.strip() if isinstance(raw_tz, str) and raw_tz.strip() else None

        raw_subd = raw.get("subd")
        subd = raw_subd.strip() if isinstance(raw_subd, str) and raw_subd.strip() else None

        return AirportCandidate(
            name=name.strip(),
            iata=iata,
            icao=icao,
            city=display_city,
            subdivision=subd,
            country=country_name,
            country_code=country_code,
            latitude=latitude,
            longitude=longitude,
            timezone=tz,
        )

    def resolve(self, query: Any, raise_on_error: bool = False) -> ResolvedLocation:
        """
        Resolve a natural-language travel query into a normalized location result.

        Args:
            query: The location or code string to resolve.
            raise_on_error: If True, re-raises LocationResolutionError on operational failures.

        Returns:
            ResolvedLocation model containing status and normalized coordinates/codes.
        """
        # 1. Input Validation
        if query is None or not isinstance(query, str):
            return ResolvedLocation(
                query=str(query if query is not None else ""),
                status="invalid_query",
                location_type="unknown",
                error_message="Location query must be a non-empty string.",
            )

        cleaned = re.sub(r"\s+", " ", query).strip()
        if not cleaned:
            return ResolvedLocation(
                query=query,
                status="invalid_query",
                location_type="unknown",
                error_message="Location query cannot be empty or whitespace-only.",
            )

        logger.info("Resolving travel location (query=%r)", cleaned)

        try:
            self._ensure_initialized()
        except LocationResolutionError as exc:
            if raise_on_error:
                raise
            logger.error("Location resolution failed: %s", exc)
            return ResolvedLocation(
                query=cleaned,
                status="error",
                location_type="unknown",
                error_message=str(exc),
            )

        upper_query = cleaned.upper()
        lower_query = cleaned.lower()

        # 2. Exact 3-letter IATA Code Match
        if len(upper_query) == 3 and upper_query.isalpha():
            raw_apt = self._iata_index.get(upper_query)
            if raw_apt:
                cand = self._normalize_record(raw_apt)
                if cand:
                    logger.info("Location resolved via IATA code: %s -> %s", upper_query, cand.name)
                    return ResolvedLocation(
                        query=cleaned,
                        status="resolved",
                        location_type="airport",
                        normalized_name=cand.name,
                        city=cand.city,
                        country=cand.country,
                        country_code=cand.country_code,
                        airport_name=cand.name,
                        iata_code=cand.iata,
                        icao_code=cand.icao,
                        latitude=cand.latitude,
                        longitude=cand.longitude,
                        timezone=cand.timezone,
                        confidence=1.0,
                        match_type="iata",
                        matches=[cand],
                    )

        # 3. Exact 4-letter ICAO Code Match
        if len(upper_query) == 4 and upper_query.isalpha():
            raw_apt = self._icao_index.get(upper_query)
            if raw_apt:
                cand = self._normalize_record(raw_apt)
                if cand:
                    logger.info("Location resolved via ICAO code: %s -> %s", upper_query, cand.name)
                    return ResolvedLocation(
                        query=cleaned,
                        status="resolved",
                        location_type="airport",
                        normalized_name=cand.name,
                        city=cand.city,
                        country=cand.country,
                        country_code=cand.country_code,
                        airport_name=cand.name,
                        iata_code=cand.iata,
                        icao_code=cand.icao,
                        latitude=cand.latitude,
                        longitude=cand.longitude,
                        timezone=cand.timezone,
                        confidence=1.0,
                        match_type="icao",
                        matches=[cand],
                    )

        # 4. Compound Query Resolution (e.g. "New York JFK", "TRV Airport", "London LHR")
        tokens = cleaned.split()
        if len(tokens) > 1:
            for token in tokens:
                token_upper = token.upper()
                if len(token_upper) == 3 and token_upper.isalpha() and token_upper in self._iata_index:
                    raw_apt = self._iata_index[token_upper]
                    cand = self._normalize_record(raw_apt)
                    if cand:
                        cand_city = (cand.city or "").lower()
                        cand_name = cand.name.lower()
                        # Check if query's other words match the airport's city or name
                        query_without_code = re.sub(rf"\b{re.escape(token)}\b", "", cleaned, flags=re.IGNORECASE).strip().lower()
                        if cand_city in query_without_code or query_without_code in cand_city or cand_name in lower_query or query_without_code in cand_name:
                            logger.info("Location resolved via compound IATA match: %s -> %s", cleaned, cand.name)
                            return ResolvedLocation(
                                query=cleaned,
                                status="resolved",
                                location_type="airport",
                                normalized_name=cand.name,
                                city=cand.city,
                                country=cand.country,
                                country_code=cand.country_code,
                                airport_name=cand.name,
                                iata_code=cand.iata,
                                icao_code=cand.icao,
                                latitude=cand.latitude,
                                longitude=cand.longitude,
                                timezone=cand.timezone,
                                confidence=1.0,
                                match_type="compound",
                                matches=[cand],
                            )

        # 5. Exact Airport Name Match
        if lower_query in self._airport_name_index:
            matching_raw = self._airport_name_index[lower_query]
            candidates = [c for c in (self._normalize_record(r) for r in matching_raw) if c]
            if len(candidates) == 1:
                cand = candidates[0]
                logger.info("Location resolved via exact airport name: %s -> %s", cleaned, cand.name)
                return ResolvedLocation(
                    query=cleaned,
                    status="resolved",
                    location_type="airport",
                    normalized_name=cand.name,
                    city=cand.city,
                    country=cand.country,
                    country_code=cand.country_code,
                    airport_name=cand.name,
                    iata_code=cand.iata,
                    icao_code=cand.icao,
                    latitude=cand.latitude,
                    longitude=cand.longitude,
                    timezone=cand.timezone,
                    confidence=1.0,
                    match_type="airport_name",
                    matches=[cand],
                )
            elif len(candidates) > 1:
                logger.info("Location is ambiguous for airport name: %s (matches=%d)", cleaned, len(candidates))
                return ResolvedLocation(
                    query=cleaned,
                    status="ambiguous",
                    location_type="airport",
                    normalized_name=candidates[0].name,
                    city=candidates[0].city,
                    country=candidates[0].country,
                    country_code=candidates[0].country_code,
                    matches=candidates,
                    match_type="airport_name",
                )

        # 6. Airport Suffix Normalization & Airport Nicknames
        stripped_airport = re.sub(r"\s+(international\s+)?airport$", "", lower_query).strip()
        if stripped_airport in AIRPORT_ALIASES:
            alias_iata = AIRPORT_ALIASES[stripped_airport]
            raw_apt = self._iata_index.get(alias_iata)
            if raw_apt:
                cand = self._normalize_record(raw_apt)
                if cand:
                    logger.info("Location resolved via airport alias: %s -> %s", cleaned, cand.name)
                    return ResolvedLocation(
                        query=cleaned,
                        status="resolved",
                        location_type="airport",
                        normalized_name=cand.name,
                        city=cand.city,
                        country=cand.country,
                        country_code=cand.country_code,
                        airport_name=cand.name,
                        iata_code=cand.iata,
                        icao_code=cand.icao,
                        latitude=cand.latitude,
                        longitude=cand.longitude,
                        timezone=cand.timezone,
                        confidence=1.0,
                        match_type="airport_alias",
                        matches=[cand],
                    )

        # 7. City Name and City Aliases
        target_city = CITY_ALIASES.get(lower_query, CITY_ALIASES.get(stripped_airport, stripped_airport))
        city_raw_matches = self._city_index.get(target_city, [])

        if not city_raw_matches and target_city != lower_query:
            city_raw_matches = self._city_index.get(lower_query, [])

        if city_raw_matches:
            candidates = [c for c in (self._normalize_record(r) for r in city_raw_matches) if c]

            # Filter out military facilities to prioritize commercial passenger travel
            commercial_candidates = [c for c in candidates if not _is_military_facility(c.name)]
            effective_candidates = commercial_candidates if commercial_candidates else candidates

            display_city = CITY_DISPLAY_NAMES.get(target_city, effective_candidates[0].city or cleaned.title())

            # Check if all candidates belong to distinct states/countries (e.g. "Springfield")
            subdivisions = {c.subdivision for c in effective_candidates if c.subdivision}
            countries = {c.country_code for c in effective_candidates if c.country_code}

            if len(subdivisions) > 1 or len(countries) > 1 or len(effective_candidates) > 1:
                # If there's only 1 commercial candidate after filtering out military facilities:
                if len(effective_candidates) == 1:
                    cand = effective_candidates[0]
                    logger.info("Location resolved via city: %s -> %s (%s)", cleaned, display_city, cand.iata)
                    return ResolvedLocation(
                        query=cleaned,
                        status="resolved",
                        location_type="city",
                        normalized_name=display_city,
                        city=display_city,
                        country=cand.country,
                        country_code=cand.country_code,
                        airport_name=cand.name,
                        iata_code=cand.iata,
                        icao_code=cand.icao,
                        latitude=cand.latitude,
                        longitude=cand.longitude,
                        timezone=cand.timezone,
                        confidence=1.0,
                        match_type="city",
                        matches=[cand],
                    )

                logger.info("Location is ambiguous: %s (matches=%d)", cleaned, len(effective_candidates))
                return ResolvedLocation(
                    query=cleaned,
                    status="ambiguous",
                    location_type="city",
                    normalized_name=display_city,
                    city=display_city,
                    country=effective_candidates[0].country,
                    country_code=effective_candidates[0].country_code,
                    matches=effective_candidates,
                    match_type="city",
                )
            elif len(effective_candidates) == 1:
                cand = effective_candidates[0]
                logger.info("Location resolved via city: %s -> %s (%s)", cleaned, display_city, cand.iata)
                return ResolvedLocation(
                    query=cleaned,
                    status="resolved",
                    location_type="city",
                    normalized_name=display_city,
                    city=display_city,
                    country=cand.country,
                    country_code=cand.country_code,
                    airport_name=cand.name,
                    iata_code=cand.iata,
                    icao_code=cand.icao,
                    latitude=cand.latitude,
                    longitude=cand.longitude,
                    timezone=cand.timezone,
                    confidence=1.0,
                    match_type="city",
                    matches=[cand],
                )

        # 8. Not Found
        logger.info("Location not found: %s", cleaned)
        return ResolvedLocation(
            query=cleaned,
            status="not_found",
            location_type="unknown",
            normalized_name=None,
            matches=[],
            match_type="not_found",
        )


# ---------------------------------------------------------------------------
# Module-level Convenience Interface
# ---------------------------------------------------------------------------

_default_resolver: Optional[LocationResolver] = None


def get_default_resolver() -> LocationResolver:
    """Obtain or initialize the module-level singleton LocationResolver."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = LocationResolver()
    return _default_resolver


def resolve_location(query: Any, resolver: Optional[LocationResolver] = None) -> ResolvedLocation:
    """
    Resolve a travel location or airport query.

    Args:
        query: Location query string (e.g. "TRV", "Trivandrum", "Dubai", "New York JFK").
        resolver: Optional LocationResolver instance override.

    Returns:
        A ResolvedLocation object.
    """
    res = resolver if resolver is not None else get_default_resolver()
    return res.resolve(query)
