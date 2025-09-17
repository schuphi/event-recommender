#!/usr/bin/env python3
"""
Scheduler for Copenhagen Event Recommender.
Runs daily scraping at 7:00 AM using APScheduler.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EventScraperScheduler:
    """Scheduler for running event scraping tasks."""

    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.project_root = Path(__file__).parent

        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)

    def run_scraper(self):
        """Execute the daily scraper."""
        logger.info("=== Starting scheduled scraping ===")

        try:
            # Use the enhanced scraper that combines venue scraping with Eventbrite API
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "enhanced_scraper_runner.py")
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                logger.info("Scraping completed successfully")
                logger.info(f"Output: {result.stdout}")
            else:
                logger.error("Scraping failed")
                logger.error(f"Error: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to run scraper: {e}")

    def setup_schedule(self):
        """Set up the daily scraping schedule."""
        # Daily at 7:00 AM
        self.scheduler.add_job(
            func=self.run_scraper,
            trigger=CronTrigger(hour=7, minute=0),
            id='daily_scraping',
            name='Daily Event Scraping',
            replace_existing=True
        )

        logger.info("Scheduled daily scraping at 7:00 AM")

    def run_once_now(self):
        """Run scraper immediately for testing."""
        logger.info("Running scraper immediately for testing...")
        self.run_scraper()

    def start(self, run_now=False):
        """Start the scheduler."""
        try:
            self.setup_schedule()

            if run_now:
                self.run_once_now()

            logger.info("Scheduler started. Press Ctrl+C to stop.")
            logger.info("Next run scheduled for 7:00 AM daily")

            self.scheduler.start()

        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")

def main():
    """Main entry point."""
    scheduler = EventScraperScheduler()

    # Check for command line arguments
    run_now = len(sys.argv) > 1 and sys.argv[1] == "--run-now"

    if run_now:
        logger.info("Running scraper immediately and then starting scheduler...")

    scheduler.start(run_now=run_now)

if __name__ == "__main__":
    main()