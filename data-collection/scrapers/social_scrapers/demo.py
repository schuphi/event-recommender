#!/usr/bin/env python3
"""
Demo script for Copenhagen Social Event Scrapers.
Shows how to use the Instagram and TikTok scrapers to discover events.
"""

import asyncio
import json
from social_scraper_manager import SocialScraperManager


async def demo_light_scraping():
    """Demo with light scraping (no authentication required)."""
    
    print("🎉 Copenhagen Event Scraper Demo (2025)")
    print("=" * 50)
    
    # Initialize manager without authentication
    manager = SocialScraperManager()
    
    print("📱 Starting light scraping (no authentication)...")
    
    try:
        # Light scraping with reduced limits
        events = await manager.scrape_all_platforms(
            days_back=3,  # Only last 3 days
            max_events_per_platform=10,  # Limit to 10 events per platform
            include_viral=True
        )
        
        print(f"\n📊 Scraping Results:")
        print(f"   📷 Instagram venues: {len(events['instagram_venues'])} events")
        print(f"   🔥 Instagram viral: {len(events['instagram_viral'])} events") 
        print(f"   🎵 TikTok: {len(events['tiktok'])} events")
        print(f"   🎯 Combined unique: {len(events['combined'])} events")
        
        if events['combined']:
            print(f"\n🏆 Top Events:")
            top_events = manager.get_top_events(events['combined'], limit=3)
            
            for i, event in enumerate(top_events, 1):
                platform_emoji = "📷" if event['platform'] == 'instagram' else "🎵"
                print(f"   {i}. {platform_emoji} {event['title'][:50]}...")
                print(f"      📍 {event['venue_name']}")
                print(f"      💫 Score: {event['engagement_score']:.1f}")
                if event['detected_genres']:
                    print(f"      🎶 {', '.join(event['detected_genres'][:2])}")
                print()
            
            # Save results
            filename = manager.save_events_to_json(events, "demo_events.json")
            print(f"💾 Results saved to: {filename}")
            
        else:
            print("⚠️  No events found. This could be due to:")
            print("   - Rate limiting from platforms")
            print("   - No recent events in the search criteria")
            print("   - Network connectivity issues")
            
        # Show some analytics
        if events['combined']:
            genres = {}
            platforms = {"instagram": 0, "tiktok": 0}
            
            for event in events['combined']:
                platforms[event['platform']] += 1
                for genre in event.get('detected_genres', []):
                    genres[genre] = genres.get(genre, 0) + 1
            
            print(f"\n📈 Analytics:")
            print(f"   Platform distribution: {dict(platforms)}")
            if genres:
                top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top genres: {dict(top_genres)}")
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        print("💡 Tips:")
        print("   - Make sure you have an internet connection")
        print("   - Instagram/TikTok may be blocking requests")
        print("   - Try running again later or with authentication")
        
    finally:
        await manager.close()
        print("\n✅ Demo completed!")


def demo_setup_requirements():
    """Show setup requirements and installation instructions."""
    
    print("🛠️  Setup Requirements for Full Functionality:")
    print("=" * 50)
    
    print("📦 Install dependencies:")
    print("   pip install -r requirements.txt")
    print("   python -m playwright install")
    
    print("\n🔐 Optional Authentication (for better rate limits):")
    print("   Instagram:")
    print("   - Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD env vars")
    print("   - Or pass them to SocialScraperManager()")
    
    print("\n   TikTok:")
    print("   - Get MS token from browser dev tools")
    print("   - Set TIKTOK_MS_TOKEN env var")
    print("   - Or pass to SocialScraperManager()")
    
    print("\n⚡ Enhanced Features Available:")
    print("   - Viral content discovery")
    print("   - Real-time trending detection")
    print("   - Multi-platform deduplication")
    print("   - Engagement scoring")
    print("   - Genre and artist detection")
    
    print("\n🎯 Target Use Cases:")
    print("   - Event discovery for nightlife apps")
    print("   - Music venue analytics")
    print("   - Trend monitoring")
    print("   - Social media intelligence")


if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Light scraping demo")
    print("2. Setup requirements info")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_light_scraping())
    elif choice == "2":
        demo_setup_requirements()
    else:
        print("Running light scraping demo by default...")
        asyncio.run(demo_light_scraping())