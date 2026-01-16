import { useState, useEffect } from 'react';
import { EventCard } from './EventCard';
import { FilterPanel } from './FilterPanel';
import { Search, Loader2, MapPin } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface Event {
  id: string;
  title: string;
  description: string;
  date_time: string;
  price_min: number;
  price_max: number;
  currency: string;
  topic: string;
  tags: string[];
  is_free: boolean;
  venue_name: string;
  venue_address: string;
  venue_neighborhood: string;
  genres: string[];
  popularity_score: number;
  source: string;
  source_url?: string;
}

interface FilterState {
  genres: string[];
  neighborhoods: string[];
  priceRange: [number, number];
  dateFilter: string;
}

// Topic configuration for filter chips
const TOPICS = [
  { id: 'all', label: 'All', icon: '✨' },
  { id: 'tech', label: 'Tech', icon: '💻' },
  { id: 'nightlife', label: 'Nightlife', icon: '🌙' },
  { id: 'music', label: 'Music', icon: '🎵' },
  { id: 'sports', label: 'Sports', icon: '⚽' },
] as const;

const API_BASE_URL = 'http://localhost:8000';

export function EventFeed() {
  const [events, setEvents] = useState<Event[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTopic, setSelectedTopic] = useState<string>('all');
  const [showFreeOnly, setShowFreeOnly] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    genres: [],
    neighborhoods: [],
    priceRange: [0, 1000],
    dateFilter: '',
  });
  const [filtersOpen, setFiltersOpen] = useState(false);
  const { toast } = useToast();

  // Generate user ID if not exists
  useEffect(() => {
    if (!localStorage.getItem('user_id')) {
      localStorage.setItem('user_id', `user_${Math.random().toString(36).substr(2, 9)}_${Date.now()}`);
    }
  }, []);

  // Fetch events from API
  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setLoading(true);

        // Build URL with topic filter
        let url = `${API_BASE_URL}/events?upcoming_only=true&limit=50`;
        if (selectedTopic !== 'all') {
          url += `&topic=${selectedTopic}`;
        }
        if (showFreeOnly) {
          url += `&is_free=true`;
        }

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error('Failed to fetch events');
        }
        const data = await response.json();
        // Ensure data is an array and has proper structure
        const eventsData = Array.isArray(data) ? data : [];
        const safeEvents = eventsData.map(event => ({
          ...event,
          genres: event.genres || [],
          tags: event.tags || [],
          topic: event.topic || 'music',
          is_free: event.is_free || false,
          title: event.title || 'Untitled Event',
          description: event.description || '',
          venue_name: event.venue_name || 'Unknown Venue',
          venue_neighborhood: event.venue_neighborhood || 'Unknown',
          price_min: event.price_min || 0,
          price_max: event.price_max || 0,
          currency: event.currency || 'DKK',
          popularity_score: event.popularity_score || 0,
          source: event.source || 'unknown',
          source_url: event.source_url || null,
        }));
        
        // Remove duplicates by title and venue using Map for better performance
        const uniqueEventsMap = new Map();
        safeEvents.forEach(event => {
          const key = `${event.title}-${event.venue_name}`;
          if (!uniqueEventsMap.has(key)) {
            uniqueEventsMap.set(key, event);
          }
        });
        const uniqueEvents = Array.from(uniqueEventsMap.values());
        
        setEvents(uniqueEvents);
        setFilteredEvents(uniqueEvents);

        // Check if we have fewer events than requested, meaning no more to load
        if (uniqueEvents.length < 50) {
          setHasMore(false);
        }

        // Set initial page based on events loaded
        setCurrentPage(Math.floor(uniqueEvents.length / 20));
      } catch (error) {
        console.error('Error fetching events:', error);
        toast({
          title: "Error loading events",
          description: "Could not connect to the event service. Please try again later.",
          variant: "destructive",
        });
        // Fallback to empty array
        setEvents([]);
        setFilteredEvents([]);
      } finally {
        setLoading(false);
      }
    };

    fetchEvents();
  }, [toast, selectedTopic, showFreeOnly]);

  // Load more events function
  const loadMoreEvents = async () => {
    if (loadingMore || !hasMore) return;

    try {
      setLoadingMore(true);
      const nextPage = currentPage + 1;
      const offset = nextPage * 20;

      // Build URL with topic filter
      let url = `${API_BASE_URL}/events?upcoming_only=true&limit=20&offset=${offset}`;
      if (selectedTopic !== 'all') {
        url += `&topic=${selectedTopic}`;
      }
      if (showFreeOnly) {
        url += `&is_free=true`;
      }

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Failed to fetch more events');
      }

      const data = await response.json();
      const eventsData = Array.isArray(data) ? data : [];

      if (eventsData.length === 0) {
        setHasMore(false);
        return;
      }

      const safeEvents = eventsData.map(event => ({
        ...event,
        genres: event.genres || [],
        tags: event.tags || [],
        topic: event.topic || 'music',
        is_free: event.is_free || false,
        title: event.title || 'Untitled Event',
        description: event.description || '',
        venue_name: event.venue_name || 'Unknown Venue',
        venue_neighborhood: event.venue_neighborhood || 'Unknown',
        price_min: event.price_min || 0,
        price_max: event.price_max || 0,
        currency: event.currency || 'DKK',
        popularity_score: event.popularity_score || 0,
        source: event.source || 'unknown',
        source_url: event.source_url || null,
      }));

      // Add new events to existing ones, avoiding duplicates
      setEvents(prevEvents => {
        const existingIds = new Set(prevEvents.map(e => e.id));
        const newEvents = safeEvents.filter(e => !existingIds.has(e.id));
        return [...prevEvents, ...newEvents];
      });

      setCurrentPage(nextPage);

      if (eventsData.length < 20) {
        setHasMore(false);
      }

    } catch (error) {
      console.error('Error loading more events:', error);
      toast({
        title: "Error loading more events",
        description: "Could not load additional events. Please try again later.",
        variant: "destructive",
      });
    } finally {
      setLoadingMore(false);
    }
  };

  // Apply filters and search
  useEffect(() => {
    let filtered = [...events];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(event =>
        (event.title || '').toLowerCase().includes(query) ||
        (event.description || '').toLowerCase().includes(query) ||
        (event.venue_name || '').toLowerCase().includes(query) ||
        (event.venue_neighborhood || '').toLowerCase().includes(query) ||
        (event.genres || []).some(genre => genre.toLowerCase().includes(query))
      );
    }

    // Genre filter
    if (filters.genres.length > 0) {
      filtered = filtered.filter(event =>
        (event.genres || []).some(genre => filters.genres.includes(genre))
      );
    }

    // Neighborhood filter
    if (filters.neighborhoods.length > 0) {
      filtered = filtered.filter(event =>
        filters.neighborhoods.includes(event.venue_neighborhood)
      );
    }

    // Price filter
    filtered = filtered.filter(event => {
      const minPrice = event.price_min || 0;
      const maxPrice = event.price_max || 0;
      return minPrice >= filters.priceRange[0] && maxPrice <= filters.priceRange[1];
    });

    // Date filter (simplified - would need actual date logic)
    if (filters.dateFilter) {
      // For now, just show all events regardless of date filter
      // In real implementation, would filter by actual dates
    }

    setFilteredEvents(filtered || []);
  }, [events, searchQuery, filters]);

  const handleLike = async (eventId: string) => {
    try {
      // Update local storage for match algorithm
      const userLikes = JSON.parse(localStorage.getItem('user_likes') || '[]');
      const userDislikes = JSON.parse(localStorage.getItem('user_dislikes') || '[]');

      if (!userLikes.includes(eventId)) {
        userLikes.push(eventId);
        localStorage.setItem('user_likes', JSON.stringify(userLikes));
      }

      // Remove from dislikes if present
      const updatedDislikes = userDislikes.filter(id => id !== eventId);
      localStorage.setItem('user_dislikes', JSON.stringify(updatedDislikes));

      // Record interaction with API
      await fetch(`${API_BASE_URL}/interactions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': localStorage.getItem('user_id') || 'anonymous'
        },
        body: JSON.stringify({
          event_id: eventId,
          interaction_type: 'like',
          source: 'feed'
        })
      });

      toast({
        title: "Event liked! 💖",
        description: "We'll recommend similar events for you.",
      });
    } catch (error) {
      console.error('Error recording interaction:', error);
    }
  };

  const handleDislike = async (eventId: string) => {
    try {
      // Update local storage for match algorithm
      const userLikes = JSON.parse(localStorage.getItem('user_likes') || '[]');
      const userDislikes = JSON.parse(localStorage.getItem('user_dislikes') || '[]');

      if (!userDislikes.includes(eventId)) {
        userDislikes.push(eventId);
        localStorage.setItem('user_dislikes', JSON.stringify(userDislikes));
      }

      // Remove from likes if present
      const updatedLikes = userLikes.filter(id => id !== eventId);
      localStorage.setItem('user_likes', JSON.stringify(updatedLikes));

      // Record interaction with API
      await fetch(`${API_BASE_URL}/interactions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': localStorage.getItem('user_id') || 'anonymous'
        },
        body: JSON.stringify({
          event_id: eventId,
          interaction_type: 'dislike',
          source: 'feed'
        })
      });

      // Remove from current view
      setFilteredEvents(prev => prev.filter(event => event.id !== eventId));

      toast({
        title: "Got it!",
        description: "We'll show you fewer events like this.",
      });
    } catch (error) {
      console.error('Error recording interaction:', error);
    }
  };

  const handleSave = async (eventId: string) => {
    try {
      await fetch(`${API_BASE_URL}/interactions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': localStorage.getItem('user_id') || 'anonymous'
        },
        body: JSON.stringify({
          event_id: eventId,
          interaction_type: 'save',
          source: 'feed'
        })
      });

      toast({
        title: "Event saved! 📌",
        description: "Find it later in your saved events.",
      });
    } catch (error) {
      console.error('Error recording interaction:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="glass-card text-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-4" />
          <p className="text-lg font-heading">Loading Copenhagen's hottest events...</p>
          <p className="text-sm text-muted-foreground mt-2">Finding the perfect match for you</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background-secondary to-background">
      <div className="container mx-auto px-4 py-6 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-heading font-bold mb-2 bg-gradient-to-r from-primary via-accent to-neon-purple bg-clip-text text-transparent">
            Copenhagen Events
          </h1>
          <p className="text-muted-foreground">Discover your next unforgettable experience</p>
        </div>

        {/* Topic Filter Chips */}
        <div className="flex flex-wrap gap-2 mb-6 justify-center">
          {TOPICS.map((topic) => (
            <button
              key={topic.id}
              onClick={() => {
                setSelectedTopic(topic.id);
                setCurrentPage(0);
                setHasMore(true);
              }}
              className={`
                px-4 py-2 rounded-full text-sm font-medium transition-all
                flex items-center gap-2
                ${selectedTopic === topic.id
                  ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/25'
                  : 'bg-secondary/50 text-secondary-foreground hover:bg-secondary'
                }
              `}
            >
              <span>{topic.icon}</span>
              {topic.label}
            </button>
          ))}

          {/* Free events toggle */}
          <button
            onClick={() => {
              setShowFreeOnly(!showFreeOnly);
              setCurrentPage(0);
              setHasMore(true);
            }}
            className={`
              px-4 py-2 rounded-full text-sm font-medium transition-all
              flex items-center gap-2
              ${showFreeOnly
                ? 'bg-green-500 text-white shadow-lg shadow-green-500/25'
                : 'bg-secondary/50 text-secondary-foreground hover:bg-secondary'
              }
            `}
          >
            <span>🎁</span>
            Free
          </button>
        </div>

        {/* Search and Filters */}
        <div className="space-y-4 mb-8">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search events, venues, or neighborhoods..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 glass border-primary/30 focus:ring-primary/50"
            />
          </div>

          <FilterPanel
            filters={filters}
            onFiltersChange={setFilters}
            isOpen={filtersOpen}
            onToggle={() => setFiltersOpen(!filtersOpen)}
          />
        </div>

        {/* Events Grid */}
        {filteredEvents.length === 0 ? (
          <div className="glass-card text-center py-12">
            <MapPin className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-heading font-semibold mb-2">No events found</h3>
            <p className="text-muted-foreground mb-4">
              Try adjusting your filters or search terms
            </p>
            <Button
              variant="outline"
              onClick={() => {
                setSearchQuery('');
                setSelectedTopic('all');
                setShowFreeOnly(false);
                setFilters({
                  genres: [],
                  neighborhoods: [],
                  priceRange: [0, 1000],
                  dateFilter: '',
                });
              }}
              className="border-primary/30 hover:bg-primary/10"
            >
              Clear all filters
            </Button>
          </div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {filteredEvents.map((event, index) => (
              <div
                key={event.id}
                style={{ animationDelay: `${index * 100}ms` }}
                className="animate-fade-in"
              >
                <EventCard
                  event={event}
                  onLike={handleLike}
                  onDislike={handleDislike}
                  onSave={handleSave}
                />
              </div>
            ))}
          </div>
        )}

        {/* Load more button */}
        {filteredEvents.length > 0 && hasMore && (
          <div className="text-center mt-12">
            <Button
              variant="outline"
              className="glass border-primary/30 hover:bg-primary/10"
              onClick={loadMoreEvents}
              disabled={loadingMore}
            >
              {loadingMore ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Loading more events...
                </>
              ) : (
                'Load more events'
              )}
            </Button>
          </div>
        )}

        {/* No more events message */}
        {filteredEvents.length > 0 && !hasMore && (
          <div className="text-center mt-12">
            <p className="text-muted-foreground">That's all the events we have for now!</p>
          </div>
        )}
      </div>
    </div>
  );
}