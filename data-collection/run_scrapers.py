#!/usr/bin/env python3
"""
Unified scraper runner for all event sources.

Usage:
    # Run all scrapers
    python run_scrapers.py

    # Run specific scrapers
    python run_scrapers.py --sources eventbrite,luma

    # Run with custom limits
    python run_scrapers.py --max-results 50

    # Dry run (don't save to database)
    python run_scrapers.py --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ingest_events

# Import scrapers
from scrapers.official_apis.eventbrite import fetch_eventbrite_events
from scrapers.official_apis.ticketmaster import fetch_ticketmaster_events
from scrapers.official_apis.luma import fetch_luma_events

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Registry of available scrapers
SCRAPERS = {
    "eventbrite": {
        "name": "Eventbrite",
        "fetch": fetch_eventbrite_events,
        "requires_key": True,
        "env_var": "EVENTBRITE_API_TOKEN",
        "description": "Music, nightlife, and arts events",
    },
    "ticketmaster": {
        "name": "TicketMaster",
        "fetch": fetch_ticketmaster_events,
        "requires_key": True,
        "env_var": "TICKETMASTER_API_KEY",
        "description": "Concerts, sports events, theatre",
    },
    "luma": {
        "name": "Luma",
        "fetch": fetch_luma_events,
        "requires_key": False,
        "env_var": None,
        "description": "Tech and startup events (no API key needed)",
    },
}


def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are configured."""
    status = {}
    for source, config in SCRAPERS.items():
        if config["requires_key"]:
            env_var = config["env_var"]
            status[source] = bool(os.getenv(env_var))
        else:
            status[source] = True
    return status


def run_scraper(
    source: str,
    max_results: int = 100,
    dry_run: bool = False,
) -> Dict:
    """
    Run a single scraper and ingest results.

    Args:
        source: Scraper name (eventbrite, meetup, etc.)
        max_results: Maximum events to fetch
        dry_run: If True, fetch but don't save to database

    Returns:
        Stats dictionary
    """
    if source not in SCRAPERS:
        logger.error(f"Unknown source: {source}")
        return {"error": f"Unknown source: {source}"}

    config = SCRAPERS[source]
    logger.info(f"Running {config['name']} scraper...")

    # Check API key if required
    if config["requires_key"]:
        env_var = config["env_var"]
        if not os.getenv(env_var):
            logger.warning(f"Skipping {config['name']}: {env_var} not set")
            return {"skipped": True, "reason": f"{env_var} not configured"}

    try:
        # Fetch events
        fetch_func = config["fetch"]
        events = fetch_func(max_results=max_results)

        logger.info(f"Fetched {len(events)} events from {config['name']}")

        if dry_run:
            logger.info(f"[DRY RUN] Would ingest {len(events)} events")
            return {
                "source": source,
                "fetched": len(events),
                "dry_run": True,
            }

        # Ingest into pipeline
        if events:
            stats = ingest_events(events, source=source)
            stats["source"] = source
            return stats
        else:
            return {"source": source, "fetched": 0}

    except Exception as e:
        logger.error(f"Error running {config['name']} scraper: {e}")
        return {"source": source, "error": str(e)}


def run_all_scrapers(
    sources: Optional[List[str]] = None,
    max_results: int = 100,
    dry_run: bool = False,
) -> Dict:
    """
    Run multiple scrapers.

    Args:
        sources: List of scraper names. If None, runs all available.
        max_results: Maximum events per scraper
        dry_run: If True, fetch but don't save

    Returns:
        Combined stats dictionary
    """
    if sources is None:
        sources = list(SCRAPERS.keys())

    results = {
        "timestamp": datetime.now().isoformat(),
        "sources": {},
        "totals": {
            "fetched": 0,
            "inserted": 0,
            "duplicates": 0,
            "errors": 0,
        },
    }

    for source in sources:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {source}")
        logger.info(f"{'='*50}")

        stats = run_scraper(source, max_results=max_results, dry_run=dry_run)
        results["sources"][source] = stats

        # Update totals
        if not stats.get("skipped") and not stats.get("error"):
            results["totals"]["fetched"] += stats.get("fetched", stats.get("processed", 0))
            results["totals"]["inserted"] += stats.get("inserted", 0)
            results["totals"]["duplicates"] += stats.get("duplicates", 0)
            results["totals"]["errors"] += stats.get("errors", 0)

    return results


def print_status():
    """Print status of all scrapers and API keys."""
    print("\n" + "=" * 60)
    print("EVENT SCRAPER STATUS")
    print("=" * 60)

    key_status = check_api_keys()

    for source, config in SCRAPERS.items():
        status_icon = "✅" if key_status[source] else "❌"
        key_info = f"({config['env_var']})" if config["requires_key"] else "(no key needed)"

        print(f"\n{status_icon} {config['name']}")
        print(f"   {config['description']}")
        print(f"   API Key: {key_info}")

    print("\n" + "-" * 60)
    ready = sum(1 for v in key_status.values() if v)
    print(f"Ready: {ready}/{len(SCRAPERS)} scrapers")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run event scrapers for Copenhagen events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_scrapers.py                    # Run all available scrapers
  python run_scrapers.py --sources luma     # Run only Luma (no API key needed)
  python run_scrapers.py --status           # Show scraper status
  python run_scrapers.py --dry-run          # Fetch but don't save
        """
    )

    parser.add_argument(
        "--sources", "-s",
        type=str,
        help="Comma-separated list of sources (eventbrite,meetup,ticketmaster,luma)"
    )
    parser.add_argument(
        "--max-results", "-m",
        type=int,
        default=100,
        help="Maximum events per source (default: 100)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Fetch events but don't save to database"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of all scrapers and API keys"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.status:
        print_status()
        return

    # Parse sources
    sources = None
    if args.sources:
        sources = [s.strip().lower() for s in args.sources.split(",")]
        invalid = [s for s in sources if s not in SCRAPERS]
        if invalid:
            print(f"Error: Unknown sources: {invalid}")
            print(f"Available: {list(SCRAPERS.keys())}")
            return

    print_status()

    # Run scrapers
    print("\nStarting event collection...\n")

    results = run_all_scrapers(
        sources=sources,
        max_results=args.max_results,
        dry_run=args.dry_run,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)

    for source, stats in results["sources"].items():
        if stats.get("skipped"):
            print(f"⏭️  {source}: Skipped ({stats.get('reason', 'no API key')})")
        elif stats.get("error"):
            print(f"❌ {source}: Error - {stats.get('error')}")
        else:
            fetched = stats.get("fetched", stats.get("processed", 0))
            inserted = stats.get("inserted", 0)
            print(f"✅ {source}: {fetched} fetched, {inserted} inserted")

    print("-" * 60)
    totals = results["totals"]
    print(f"Total: {totals['fetched']} fetched, {totals['inserted']} inserted, {totals['duplicates']} duplicates")

    if args.dry_run:
        print("\n[DRY RUN] No events were saved to database")


if __name__ == "__main__":
    main()
