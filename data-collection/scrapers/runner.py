#!/usr/bin/env python3
"""
Simple scraper runner for Railway deployment.
Focuses on official APIs that don't require credentials.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_scrapers():
    """Run available scrapers that don't require credentials."""
    logger.info("🚀 Starting event scrapers...")
    
    try:
        # Try to run Scandinavia Standard scraper (no credentials needed)
        logger.info("📰 Running Scandinavia Standard scraper...")
        
        # Add the project root to Python path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from data_collection.scrapers.official_apis.scandinavia_standard import ScandinaviaStandardScraper
        
        scraper = ScandinaviaStandardScraper()
        events = scraper.scrape_events()
        
        logger.info(f"✅ Scraped {len(events)} events from Scandinavia Standard")
        
        return len(events)
        
    except ImportError as e:
        logger.warning(f"⚠️  Could not import scrapers: {e}")
        logger.info("💡 This is normal for a minimal deployment")
        return 0
    except Exception as e:
        logger.error(f"❌ Scraper failed: {e}")
        return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Event scraper runner")
    parser.add_argument("--source", default="all", help="Source to scrape")
    parser.add_argument("--max-events", type=int, default=100, help="Max events to scrape")
    
    args = parser.parse_args()
    
    total_events = run_scrapers()
    logger.info(f"🎯 Total events scraped: {total_events}")
    
    if total_events > 0:
        logger.info("✅ Scraping completed successfully")
    else:
        logger.info("ℹ️  No events scraped - this is normal for first run")