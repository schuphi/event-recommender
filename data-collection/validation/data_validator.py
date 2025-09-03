#!/usr/bin/env python3
"""
Comprehensive data validation pipeline for event recommender.
Validates, cleanses, and enriches event data from all sources.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import difflib
from urllib.parse import urlparse
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import h3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Event validation status."""
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_ENRICHMENT = "needs_enrichment"
    DUPLICATE = "duplicate"
    SUSPICIOUS = "suspicious"

@dataclass
class ValidationIssue:
    """Individual validation issue."""
    field: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggested_fix: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of data validation."""
    status: ValidationStatus
    issues: List[ValidationIssue]
    cleaned_data: Optional[Dict] = None
    confidence_score: float = 0.0
    duplicate_of: Optional[str] = None

@dataclass
class EventData:
    """Standardized event data structure."""
    # Core fields
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Venue
    venue_name: str
    venue_address: str
    venue_lat: float
    venue_lon: float
    venue_h3_index: str
    
    # Pricing
    price_min: Optional[float]
    price_max: Optional[float]
    currency: str = "DKK"
    
    # Content
    artists: List[str]
    genres: List[str]
    image_url: Optional[str]
    source_url: str
    
    # Metadata
    source: str  # 'eventbrite', 'meetup', 'instagram', 'tiktok'
    popularity_score: float = 0.0
    status: str = "active"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class EventDataValidator:
    """Comprehensive event data validator."""
    
    def __init__(self):
        """Initialize validator with configuration."""
        
        # Copenhagen geographic bounds
        self.copenhagen_center = (55.6761, 12.5683)
        self.copenhagen_radius_km = 50.0
        
        # Geocoder for address validation
        self.geocoder = Nominatim(user_agent="event-recommender")
        
        # Known venue mappings (for consistency)
        self.venue_mappings = {}
        self.load_venue_mappings()
        
        # Duplicate detection cache
        self.event_signatures = {}
        
        # Content filters
        self.profanity_patterns = self._load_profanity_patterns()
        
    def validate_event(self, raw_data: Dict) -> ValidationResult:
        """
        Validate and clean event data.
        
        Args:
            raw_data: Raw event data from scraper
            
        Returns:
            ValidationResult with status and cleaned data
        """
        
        issues = []
        confidence_score = 1.0
        
        try:
            # Step 1: Basic data structure validation
            structure_issues = self._validate_structure(raw_data)
            issues.extend(structure_issues)
            
            # Step 2: Data type and format validation
            format_issues = self._validate_formats(raw_data)
            issues.extend(format_issues)
            
            # Step 3: Content validation
            content_issues = self._validate_content(raw_data)
            issues.extend(content_issues)
            
            # Step 4: Geographic validation
            geo_issues = self._validate_geography(raw_data)
            issues.extend(geo_issues)
            
            # Step 5: Temporal validation
            time_issues = self._validate_temporal(raw_data)
            issues.extend(time_issues)
            
            # Step 6: Duplicate detection
            duplicate_result = self._check_duplicates(raw_data)
            if duplicate_result:
                return ValidationResult(
                    status=ValidationStatus.DUPLICATE,
                    issues=issues,
                    duplicate_of=duplicate_result
                )
            
            # Calculate confidence score based on issues
            error_count = sum(1 for issue in issues if issue.severity == 'error')
            warning_count = sum(1 for issue in issues if issue.severity == 'warning')
            
            confidence_score = max(0.0, 1.0 - (error_count * 0.3) - (warning_count * 0.1))
            
            # Determine status
            if error_count > 0:
                status = ValidationStatus.INVALID
            elif warning_count > 3 or confidence_score < 0.5:
                status = ValidationStatus.SUSPICIOUS
            elif any('enrichment' in issue.message.lower() for issue in issues):
                status = ValidationStatus.NEEDS_ENRICHMENT
            else:
                status = ValidationStatus.VALID
            
            # Clean and standardize data
            cleaned_data = self._clean_data(raw_data) if status != ValidationStatus.INVALID else None
            
            return ValidationResult(
                status=status,
                issues=issues,
                cleaned_data=cleaned_data,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                status=ValidationStatus.INVALID,
                issues=[ValidationIssue("system", "error", f"Validation system error: {str(e)}")],
                confidence_score=0.0
            )
    
    def _validate_structure(self, data: Dict) -> List[ValidationIssue]:
        """Validate basic data structure and required fields."""
        
        issues = []
        
        required_fields = {
            'title': str,
            'description': str,
            'start_time': (str, datetime),
            'venue_name': str,
            'source': str
        }
        
        for field, expected_type in required_fields.items():
            if field not in data:
                issues.append(ValidationIssue(
                    field=field,
                    severity="error",
                    message=f"Required field '{field}' is missing"
                ))
            elif not isinstance(data[field], expected_type):
                issues.append(ValidationIssue(
                    field=field,
                    severity="error",
                    message=f"Field '{field}' has wrong type. Expected {expected_type}, got {type(data[field])}"
                ))
        
        # Check for empty critical fields
        critical_fields = ['title', 'venue_name']
        for field in critical_fields:
            if field in data and isinstance(data[field], str) and not data[field].strip():
                issues.append(ValidationIssue(
                    field=field,
                    severity="error",
                    message=f"Critical field '{field}' is empty"
                ))
        
        return issues
    
    def _validate_formats(self, data: Dict) -> List[ValidationIssue]:
        """Validate data formats and ranges."""
        
        issues = []
        
        # Validate datetime fields
        datetime_fields = ['start_time', 'end_time']
        for field in datetime_fields:
            if field in data and data[field] is not None:
                if isinstance(data[field], str):
                    try:
                        parsed_date = self._parse_datetime(data[field])
                        data[field] = parsed_date  # Update in place
                    except ValueError as e:
                        issues.append(ValidationIssue(
                            field=field,
                            severity="error",
                            message=f"Invalid datetime format for '{field}': {str(e)}",
                            suggested_fix="Use ISO format: YYYY-MM-DD HH:MM:SS"
                        ))
        
        # Validate coordinates
        coordinate_fields = [('venue_lat', 'latitude'), ('venue_lon', 'longitude')]
        for field, coord_type in coordinate_fields:
            if field in data and data[field] is not None:
                try:
                    coord = float(data[field])
                    if coord_type == 'latitude' and not -90 <= coord <= 90:
                        issues.append(ValidationIssue(
                            field=field,
                            severity="error",
                            message=f"Invalid latitude: {coord}. Must be between -90 and 90"
                        ))
                    elif coord_type == 'longitude' and not -180 <= coord <= 180:
                        issues.append(ValidationIssue(
                            field=field,
                            severity="error",
                            message=f"Invalid longitude: {coord}. Must be between -180 and 180"
                        ))
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        field=field,
                        severity="error",
                        message=f"Invalid coordinate format for '{field}'"
                    ))
        
        # Validate price fields
        price_fields = ['price_min', 'price_max']
        for field in price_fields:
            if field in data and data[field] is not None:
                try:
                    price = float(data[field])
                    if price < 0:
                        issues.append(ValidationIssue(
                            field=field,
                            severity="error",
                            message=f"Negative price not allowed: {price}"
                        ))
                    elif price > 10000:  # Sanity check for DKK
                        issues.append(ValidationIssue(
                            field=field,
                            severity="warning",
                            message=f"Unusually high price: {price} DKK"
                        ))
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        field=field,
                        severity="error",
                        message=f"Invalid price format for '{field}'"
                    ))
        
        # Validate URLs
        url_fields = ['source_url', 'image_url']
        for field in url_fields:
            if field in data and data[field]:
                if not self._is_valid_url(data[field]):
                    issues.append(ValidationIssue(
                        field=field,
                        severity="warning",
                        message=f"Invalid URL format for '{field}': {data[field]}"
                    ))
        
        return issues
    
    def _validate_content(self, data: Dict) -> List[ValidationIssue]:
        """Validate content quality and appropriateness."""
        
        issues = []
        
        # Check title quality
        if 'title' in data:
            title = data['title'].strip()
            
            # Length checks
            if len(title) < 3:
                issues.append(ValidationIssue(
                    field='title',
                    severity="error",
                    message="Title too short (minimum 3 characters)"
                ))
            elif len(title) > 200:
                issues.append(ValidationIssue(
                    field='title',
                    severity="warning",
                    message="Title very long (over 200 characters)"
                ))
            
            # Content quality checks
            if title.isupper() and len(title) > 10:
                issues.append(ValidationIssue(
                    field='title',
                    severity="warning",
                    message="Title is all uppercase",
                    suggested_fix="Consider using proper case"
                ))
            
            # Check for spam patterns
            spam_indicators = ['!!!', 'FREE!!!', 'CLICK HERE', 'LIMITED TIME']
            for indicator in spam_indicators:
                if indicator in title.upper():
                    issues.append(ValidationIssue(
                        field='title',
                        severity="warning",
                        message=f"Potential spam indicator detected: {indicator}"
                    ))
        
        # Check description quality
        if 'description' in data and data['description']:
            desc = data['description'].strip()
            
            if len(desc) < 10:
                issues.append(ValidationIssue(
                    field='description',
                    severity="warning",
                    message="Description very short",
                    suggested_fix="Consider enriching with more event details"
                ))
            
            # Check for profanity
            if self._contains_profanity(desc):
                issues.append(ValidationIssue(
                    field='description',
                    severity="warning",
                    message="Description may contain inappropriate content"
                ))
        
        # Validate artists and genres if present
        list_fields = ['artists', 'genres']
        for field in list_fields:
            if field in data and data[field]:
                if not isinstance(data[field], list):
                    issues.append(ValidationIssue(
                        field=field,
                        severity="error",
                        message=f"Field '{field}' should be a list, got {type(data[field])}"
                    ))
                else:
                    # Check for empty or invalid entries
                    valid_entries = []
                    for entry in data[field]:
                        if isinstance(entry, str) and entry.strip():
                            valid_entries.append(entry.strip())
                    
                    if len(valid_entries) != len(data[field]):
                        issues.append(ValidationIssue(
                            field=field,
                            severity="warning",
                            message=f"Removed {len(data[field]) - len(valid_entries)} invalid entries from {field}"
                        ))
                        data[field] = valid_entries  # Clean in place
        
        return issues
    
    def _validate_geography(self, data: Dict) -> List[ValidationIssue]:
        """Validate geographic data and Copenhagen area relevance."""
        
        issues = []
        
        # Check if coordinates are provided
        has_lat = 'venue_lat' in data and data['venue_lat'] is not None
        has_lon = 'venue_lon' in data and data['venue_lon'] is not None
        
        if has_lat and has_lon:
            try:
                lat = float(data['venue_lat'])
                lon = float(data['venue_lon'])
                
                # Check if location is in Copenhagen area
                event_location = (lat, lon)
                distance_to_copenhagen = geodesic(event_location, self.copenhagen_center).kilometers
                
                if distance_to_copenhagen > self.copenhagen_radius_km:
                    issues.append(ValidationIssue(
                        field='venue_location',
                        severity="warning",
                        message=f"Event is {distance_to_copenhagen:.1f}km from Copenhagen center"
                    ))
                
                # Generate H3 index for location
                try:
                    h3_index = h3.latlng_to_cell(lat, lon, 8)
                    data['venue_h3_index'] = h3_index
                except Exception as e:
                    issues.append(ValidationIssue(
                        field='venue_h3_index',
                        severity="warning",
                        message=f"Could not generate H3 index: {str(e)}"
                    ))
                
            except (ValueError, TypeError) as e:
                issues.append(ValidationIssue(
                    field='venue_coordinates',
                    severity="error",
                    message=f"Invalid coordinates: {str(e)}"
                ))
        else:
            # Try to geocode address if coordinates missing
            if 'venue_address' in data and data['venue_address']:
                geocoded = self._geocode_address(data['venue_address'])
                if geocoded:
                    data['venue_lat'] = geocoded['lat']
                    data['venue_lon'] = geocoded['lon']
                    data['venue_h3_index'] = geocoded['h3_index']
                    issues.append(ValidationIssue(
                        field='venue_coordinates',
                        severity="info",
                        message="Coordinates geocoded from address"
                    ))
                else:
                    issues.append(ValidationIssue(
                        field='venue_coordinates',
                        severity="warning",
                        message="No coordinates provided and geocoding failed",
                        suggested_fix="Add venue coordinates or improve address"
                    ))
            else:
                issues.append(ValidationIssue(
                    field='venue_location',
                    severity="error",
                    message="No venue coordinates or address provided"
                ))
        
        return issues
    
    def _validate_temporal(self, data: Dict) -> List[ValidationIssue]:
        """Validate temporal aspects of event data."""
        
        issues = []
        
        if 'start_time' in data and data['start_time']:
            start_time = data['start_time']
            if isinstance(start_time, str):
                try:
                    start_time = self._parse_datetime(start_time)
                except ValueError:
                    return issues  # Already handled in format validation
            
            now = datetime.now()
            
            # Check if event is too far in the past
            if start_time < now - timedelta(days=1):
                issues.append(ValidationIssue(
                    field='start_time',
                    severity="warning",
                    message="Event appears to be in the past"
                ))
            
            # Check if event is too far in the future
            elif start_time > now + timedelta(days=365):
                issues.append(ValidationIssue(
                    field='start_time',
                    severity="warning",
                    message="Event is more than a year in the future"
                ))
            
            # Validate end time if present
            if 'end_time' in data and data['end_time']:
                end_time = data['end_time']
                if isinstance(end_time, str):
                    try:
                        end_time = self._parse_datetime(end_time)
                    except ValueError:
                        return issues
                
                if end_time <= start_time:
                    issues.append(ValidationIssue(
                        field='end_time',
                        severity="error",
                        message="End time must be after start time"
                    ))
                
                # Check for unreasonably long events
                duration = end_time - start_time
                if duration.total_seconds() > 86400 * 7:  # More than a week
                    issues.append(ValidationIssue(
                        field='end_time',
                        severity="warning",
                        message="Event duration is more than a week"
                    ))
        
        return issues
    
    def _check_duplicates(self, data: Dict) -> Optional[str]:
        """Check for duplicate events using multiple strategies."""
        
        # Create event signature for duplicate detection
        signature_components = []
        
        # Normalize title
        if 'title' in data:
            normalized_title = re.sub(r'[^a-z0-9]', '', data['title'].lower())
            signature_components.append(normalized_title)
        
        # Normalize venue
        if 'venue_name' in data:
            normalized_venue = re.sub(r'[^a-z0-9]', '', data['venue_name'].lower())
            signature_components.append(normalized_venue)
        
        # Add date component
        if 'start_time' in data:
            start_time = data['start_time']
            if isinstance(start_time, str):
                try:
                    start_time = self._parse_datetime(start_time)
                except ValueError:
                    start_time = None
            
            if start_time:
                date_key = start_time.strftime('%Y%m%d')
                signature_components.append(date_key)
        
        # Create signature hash
        if len(signature_components) >= 2:  # Need at least title and venue or date
            signature_string = '_'.join(signature_components)
            signature_hash = hashlib.md5(signature_string.encode()).hexdigest()
            
            # Check against existing signatures
            if signature_hash in self.event_signatures:
                return self.event_signatures[signature_hash]
            
            # Store this signature
            event_id = data.get('id', signature_hash)
            self.event_signatures[signature_hash] = event_id
        
        return None
    
    def _clean_data(self, data: Dict) -> Dict:
        """Clean and standardize event data."""
        
        cleaned = {}
        
        # Standard field mapping and cleaning
        field_mappings = {
            'id': lambda x: str(x) if x else self._generate_event_id(data),
            'title': lambda x: str(x).strip(),
            'description': lambda x: str(x).strip() if x else '',
            'venue_name': lambda x: str(x).strip(),
            'venue_address': lambda x: str(x).strip() if x else '',
            'source': lambda x: str(x).lower(),
            'source_url': lambda x: str(x).strip() if x else '',
            'image_url': lambda x: str(x).strip() if x and self._is_valid_url(x) else None,
            'artists': lambda x: [s.strip() for s in x if isinstance(s, str) and s.strip()] if isinstance(x, list) else [],
            'genres': lambda x: [s.strip() for s in x if isinstance(s, str) and s.strip()] if isinstance(x, list) else [],
            'currency': lambda x: str(x).upper() if x else 'DKK'
        }
        
        for field, cleaner in field_mappings.items():
            if field in data:
                try:
                    cleaned[field] = cleaner(data[field])
                except Exception as e:
                    logger.warning(f"Failed to clean field {field}: {e}")
                    if field in ['id', 'title', 'venue_name']:  # Critical fields
                        cleaned[field] = str(data[field])
        
        # Handle datetime fields
        datetime_fields = ['start_time', 'end_time']
        for field in datetime_fields:
            if field in data and data[field]:
                if isinstance(data[field], datetime):
                    cleaned[field] = data[field]
                else:
                    try:
                        cleaned[field] = self._parse_datetime(data[field])
                    except ValueError:
                        if field == 'start_time':  # Required field
                            cleaned[field] = datetime.now() + timedelta(hours=24)  # Default to tomorrow
        
        # Handle numeric fields
        numeric_fields = ['venue_lat', 'venue_lon', 'price_min', 'price_max', 'popularity_score']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    cleaned[field] = float(data[field])
                except (ValueError, TypeError):
                    cleaned[field] = None
        
        # Add missing fields with defaults
        defaults = {
            'status': 'active',
            'currency': 'DKK',
            'popularity_score': 0.0,
            'artists': [],
            'genres': [],
            'created_at': datetime.now()
        }
        
        for field, default_value in defaults.items():
            if field not in cleaned:
                cleaned[field] = default_value
        
        return cleaned
    
    def _parse_datetime(self, dt_string: str) -> datetime:
        """Parse datetime string with multiple format support."""
        
        # Common datetime formats
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y',
            '%d-%m-%Y %H:%M',
            '%d-%m-%Y'
        ]
        
        # Clean the string
        dt_string = dt_string.strip().replace('Z', '+00:00')
        
        for fmt in formats:
            try:
                return datetime.strptime(dt_string, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse datetime: {dt_string}")
    
    def _geocode_address(self, address: str) -> Optional[Dict]:
        """Geocode address to coordinates."""
        
        try:
            # Add Copenhagen context to improve accuracy
            full_address = f"{address}, Copenhagen, Denmark"
            
            location = self.geocoder.geocode(
                full_address, 
                timeout=10,
                exactly_one=True
            )
            
            if location:
                lat, lon = location.latitude, location.longitude
                h3_index = h3.latlng_to_cell(lat, lon, 8)
                
                return {
                    'lat': lat,
                    'lon': lon,
                    'h3_index': h3_index,
                    'formatted_address': location.address
                }
                
        except Exception as e:
            logger.warning(f"Geocoding failed for '{address}': {e}")
        
        return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _generate_event_id(self, data: Dict) -> str:
        """Generate unique event ID from data."""
        
        components = []
        
        if 'source' in data:
            components.append(data['source'])
        
        if 'title' in data and 'venue_name' in data:
            title_part = re.sub(r'[^a-z0-9]', '', data['title'].lower())[:10]
            venue_part = re.sub(r'[^a-z0-9]', '', data['venue_name'].lower())[:10]
            components.extend([title_part, venue_part])
        
        components.append(str(int(datetime.now().timestamp())))
        
        return '_'.join(components)
    
    def _contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        
        text_lower = text.lower()
        for pattern in self.profanity_patterns:
            if pattern in text_lower:
                return True
        return False
    
    def _load_profanity_patterns(self) -> List[str]:
        """Load profanity patterns."""
        
        # Basic profanity filter - in production, use a more comprehensive list
        return [
            'fuck', 'shit', 'damn', 'crap', 'stupid', 'idiot', 
            'hate', 'kill', 'die', 'murder', 'drug', 'cocaine'
        ]
    
    def load_venue_mappings(self):
        """Load venue name mappings for consistency."""
        
        # Common venue name variations in Copenhagen
        self.venue_mappings = {
            'culture box': 'Culture Box',
            'culturebox': 'Culture Box',
            'vega': 'Vega',
            'amager bio': 'Amager Bio',
            'kb hallen': 'KB Hallen',
            'royal arena': 'Royal Arena',
            'loppen': 'Loppen',
            'rust': 'RUST',
            'the standard': 'The Standard',
            'hive': 'Hive',
            'chateau motel': 'Chateau Motel'
        }
    
    def normalize_venue_name(self, venue_name: str) -> str:
        """Normalize venue name for consistency."""
        
        venue_lower = venue_name.lower().strip()
        
        # Check mappings
        if venue_lower in self.venue_mappings:
            return self.venue_mappings[venue_lower]
        
        # Basic cleanup
        venue_cleaned = venue_name.strip()
        
        # Capitalize words properly
        venue_cleaned = ' '.join(word.capitalize() for word in venue_cleaned.split())
        
        return venue_cleaned

def main():
    """Example usage of EventDataValidator."""
    
    validator = EventDataValidator()
    
    # Test data
    test_event = {
        'id': 'test_001',
        'title': 'AMAZING TECHNO NIGHT!!!',
        'description': 'Best party ever',
        'start_time': '2024-12-01T20:00:00',
        'end_time': '2024-12-02T04:00:00',
        'venue_name': 'culture box',
        'venue_address': 'Kronprinsessegade 54A, 1306 København K',
        'venue_lat': 55.6826,
        'venue_lon': 12.5941,
        'price_min': 150,
        'price_max': 200,
        'artists': ['DJ Example', '', 'Producer Test'],
        'genres': ['techno', 'electronic'],
        'source': 'eventbrite',
        'source_url': 'https://www.eventbrite.com/e/test-event-12345'
    }
    
    result = validator.validate_event(test_event)
    
    print(f"Validation Status: {result.status.value}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Issues Found: {len(result.issues)}")
    
    for issue in result.issues:
        print(f"  - [{issue.severity.upper()}] {issue.field}: {issue.message}")
    
    if result.cleaned_data:
        print("\nCleaned Data Sample:")
        for key, value in list(result.cleaned_data.items())[:5]:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()