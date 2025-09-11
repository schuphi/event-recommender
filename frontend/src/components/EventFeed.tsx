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
  venue_name: string;
  venue_address: string;
  venue_neighborhood: string;
  genres: string[];
  popularity_score: number;
}

interface FilterState {
  genres: string[];
  neighborhoods: string[];
  priceRange: [number, number];
  dateFilter: string;
}

const API_BASE_URL = 'https://event-recommender-production.up.railway.app';

export function EventFeed() {
  const [events, setEvents] = useState<Event[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
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
        const response = await fetch(`${API_BASE_URL}/events?upcoming_only=false`);
        if (!response.ok) {
          throw new Error('Failed to fetch events');
        }
        const data = await response.json();
        // Ensure data is an array and has proper structure
        const eventsData = Array.isArray(data) ? data : [];
        const safeEvents = eventsData.map(event => ({
          ...event,
          genres: event.genres || [],
          title: event.title || 'Untitled Event',
          description: event.description || '',
          venue_name: event.venue_name || 'Unknown Venue',
          venue_neighborhood: event.venue_neighborhood || 'Unknown',
          price_min: event.price_min || 0,
          price_max: event.price_max || 0,
          currency: event.currency || 'DKK',
          popularity_score: event.popularity_score || 0,
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
  }, [toast]);

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
            Copenhagen Nights
          </h1>
          <p className="text-muted-foreground">Discover your next unforgettable experience</p>
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

        {/* Load more placeholder */}
        {filteredEvents.length > 0 && (
          <div className="text-center mt-12">
            <Button
              variant="outline"
              className="glass border-primary/30 hover:bg-primary/10"
            >
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Loading more events...
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}