# Copenhagen Social Event Scrapers (2025)

A comprehensive suite of social media scrapers designed to discover nightlife events in Copenhagen from Instagram and TikTok. Built with modern 2025 methods including async/await patterns, viral content discovery, and intelligent event detection.

## 🌟 Features

### Instagram Scraping
- **Venue Account Monitoring**: Scrapes official venue Instagram accounts
- **Viral Event Discovery**: Finds trending event content through hashtags
- **Event Detection**: Smart algorithms to identify event posts vs regular content
- **Artist & Genre Recognition**: Automatically extracts artist names and music genres
- **Date Extraction**: Attempts to parse event dates from post content

### TikTok Scraping (2025 Methods)
- **Official TikTokApi Integration**: Uses the latest TikTokApi v7+ with Playwright
- **Trending Content**: Monitors trending videos for event content
- **Hashtag Discovery**: Searches Copenhagen-specific event hashtags
- **Viral Scoring**: Advanced engagement-based scoring system
- **Multi-Strategy Approach**: Combines trending, hashtag, keyword, and profile searches

### Integrated Management
- **Multi-Platform Coordination**: Manages both Instagram and TikTok scrapers
- **Smart Deduplication**: Removes duplicate events across platforms
- **Standardized Output**: Uniform event format regardless of source platform
- **JSON Export**: Save events in structured format for further processing
- **Analytics**: Built-in event analysis and filtering capabilities

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for TikTok)
python -m playwright install
```

### Basic Usage

```python
import asyncio
from social_scraper_manager import SocialScraperManager

async def main():
    # Initialize scraper manager
    manager = SocialScraperManager()
    
    # Scrape events from all platforms
    events = await manager.scrape_all_platforms(
        days_back=7,
        max_events_per_platform=50,
        include_viral=True
    )
    
    # Save results
    manager.save_events_to_json(events)
    
    # Get top events
    top_events = manager.get_top_events(events['combined'], limit=10)
    for event in top_events:
        print(f"{event['title']} at {event['venue_name']}")
    
    await manager.close()

# Run the scraper
asyncio.run(main())
```

### Demo Script

```bash
python demo.py
```

## 🔧 Configuration

### Authentication (Optional but Recommended)

#### Instagram
- Improves rate limits and access to more content
- Set environment variables or pass to constructor:

```python
manager = SocialScraperManager(
    instagram_username="your_username",
    instagram_password="your_password"
)
```

#### TikTok
- Required for full functionality
- Get MS token from browser developer tools
- Set environment variable or pass to constructor:

```python
manager = SocialScraperManager(
    tiktok_ms_token="your_ms_token"
)
```

### Venue Configuration

The scrapers include predefined Copenhagen venues. You can customize the venue list in each scraper class:

```python
COPENHAGEN_VENUES = {
    "vega_copenhagen": "Vega",
    "rust_cph": "Rust", 
    "culturebox_cph": "Culture Box",
    # Add your venues here
}
```

## 📊 Event Data Structure

Each event includes the following standardized fields:

```json
{
  "platform": "instagram|tiktok",
  "id": "unique_event_id",
  "title": "Event title",
  "description": "Full event description",
  "venue_name": "Venue name",
  "venue_username": "Social media handle",
  "url": "Link to original post/video",
  "image_url": "Event image/thumbnail",
  "date_time": "2025-01-15T20:00:00",
  "hashtags": ["techno", "copenhagen", "underground"],
  "detected_artists": ["Artist Name"],
  "detected_genres": ["techno", "house"],
  "engagement_score": 150.5,
  "likes": 1000,
  "comments": 50,
  "views": 10000
}
```

## 🎯 Use Cases

### Event Discovery Apps
- Real-time event feed for nightlife apps
- Automatic event aggregation from social media
- Trending event detection

### Venue Analytics
- Monitor venue social media performance
- Track event engagement across platforms
- Identify popular event types and artists

### Music Industry Intelligence
- Discover emerging artists and venues
- Track music genre trends
- Monitor nightlife scene developments

### Marketing & Promotion
- Identify viral event content patterns
- Monitor competitor events
- Find influencer collaborations

## 🏗️ Architecture

### Core Components

1. **InstagramEventScraper**: Base venue account scraper
2. **InstagramViralEventScraper**: Viral content discovery
3. **ModernTikTokScraper**: TikTok scraper with 2025 methods
4. **SocialScraperManager**: Orchestrates all scrapers

### Data Flow

```
Social Platforms → Individual Scrapers → Manager → Deduplication → Standardized Events → JSON Export
```

### Key Features

- **Async/Await**: Modern Python async patterns for performance
- **Rate Limiting**: Built-in rate limiting to avoid blocks
- **Error Handling**: Robust error handling and recovery
- **Caching**: Intelligent caching for efficiency
- **Extensible**: Easy to add new platforms or venues

## ⚠️ Important Notes

### Legal & Ethical Considerations
- Respects platform rate limits
- Only accesses publicly available content
- Follows robots.txt and terms of service
- No account automation or spamming

### Rate Limiting
- Built-in delays between requests
- Respects platform API limits
- Automatic backoff on errors
- Session management for efficiency

### Data Privacy
- Only collects public event information
- No personal user data collection
- Follows GDPR principles
- Transparent data usage

## 🔍 Troubleshooting

### Common Issues

**No events found:**
- Check internet connection
- Verify venue usernames are correct
- Try increasing `days_back` parameter
- Check if platforms are blocking requests

**Import errors:**
- Ensure all dependencies are installed
- Run `pip install -r requirements.txt`
- Install Playwright: `python -m playwright install`

**Authentication issues:**
- Verify credentials are correct
- Check environment variables
- Instagram may require app-specific password

**Rate limiting:**
- Reduce `max_events_per_platform`
- Increase delays between requests
- Use authentication for better limits
- Implement proxy rotation for production

## 🚀 Advanced Usage

### Custom Event Detection

```python
def custom_event_filter(event):
    # Custom logic to filter events
    return 'techno' in event.get('detected_genres', [])

# Filter events
filtered_events = [e for e in events['combined'] if custom_event_filter(e)]
```

### Venue-Specific Analysis

```python
# Get events by specific venue
vega_events = manager.get_events_by_venue(events['combined'], 'Vega')

# Get events by genre
techno_events = manager.get_events_by_genre(events['combined'], 'techno')
```

### Real-time Monitoring

```python
import schedule
import time

def scrape_job():
    events = asyncio.run(manager.scrape_all_platforms(days_back=1))
    # Process new events...

# Schedule scraping every hour
schedule.every().hour.do(scrape_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 📈 Performance Tips

1. **Use Authentication**: Better rate limits and more data
2. **Implement Caching**: Cache venue info and user data
3. **Batch Processing**: Process multiple venues in parallel
4. **Database Integration**: Store events in database for persistence
5. **Monitoring**: Add logging and metrics for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- TikTokApi by davidteather for TikTok scraping capabilities
- Instaloader for Instagram scraping foundation
- Copenhagen nightlife venues for creating amazing events
- The Python async/await community for modern patterns

---

**Built for the Copenhagen nightlife scene** 🎉🇩🇰

For support or questions, please open an issue in the repository.