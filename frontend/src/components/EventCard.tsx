import { Heart, MapPin, Calendar, DollarSign, Users, BookmarkPlus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

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
  genres: string[] | null;
  popularity_score: number;
  source: string;
  source_url?: string;
}

// Topic styling configuration
const topicConfig: Record<string, { label: string; className: string; icon: string }> = {
  tech: {
    label: 'Tech',
    className: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    icon: '💻'
  },
  nightlife: {
    label: 'Nightlife',
    className: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    icon: '🌙'
  },
  music: {
    label: 'Music',
    className: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
    icon: '🎵'
  },
  sports: {
    label: 'Sports',
    className: 'bg-green-500/20 text-green-400 border-green-500/30',
    icon: '⚽'
  }
};

interface EventCardProps {
  event: Event;
  onLike?: (eventId: string) => void;
  onDislike?: (eventId: string) => void;
  onSave?: (eventId: string) => void;
  onGoing?: (eventId: string) => void;
}

const neighborhoodClasses = {
  Vesterbro: 'neighborhood-vesterbro',
  Nørrebro: 'neighborhood-norrebro',
  'Indre By': 'neighborhood-indre-by',
  Christiania: 'neighborhood-christiania',
  Østerbro: 'neighborhood-osterbro',
};

export function EventCard({ event, onLike, onDislike, onSave, onGoing }: EventCardProps) {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('da-DK', {
      weekday: 'short',
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const calculateMatchPercentage = () => {
    let matchScore = 0;
    let maxScore = 0;

    // Get user preferences from localStorage or defaults
    const storedPrefs = localStorage.getItem('user_preferences');
    const userPreferences = storedPrefs ? JSON.parse(storedPrefs) : {
      preferred_genres: [],
      preferred_neighborhoods: [],
      price_cap: 1000
    };

    // Get user interaction history
    const userLikes = JSON.parse(localStorage.getItem('user_likes') || '[]');
    const userDislikes = JSON.parse(localStorage.getItem('user_dislikes') || '[]');

    // Base score from venue reputation (30% weight)
    const venueScore = getVenueScore(event.venue_name);
    matchScore += venueScore * 0.3;
    maxScore += 0.3;

    // Price preference matching (20% weight)
    const priceScore = getPriceScore(event.price_min, event.price_max, userPreferences.price_cap);
    matchScore += priceScore * 0.2;
    maxScore += 0.2;

    // Event timing (15% weight) - prefer events happening soon
    const timingScore = getTimingScore(event.date_time);
    matchScore += timingScore * 0.15;
    maxScore += 0.15;

    // User interaction history (20% weight)
    const historyScore = getHistoryScore(event, userLikes, userDislikes);
    matchScore += historyScore * 0.2;
    maxScore += 0.2;

    // Source credibility (15% weight)
    const sourceScore = getSourceScore(event.source);
    matchScore += sourceScore * 0.15;
    maxScore += 0.15;

    return Math.round((matchScore / maxScore) * 100);
  };

  const getVenueScore = (venueName: string) => {
    const premiumVenues = {
      'Store Vega': 1.0,
      'Lille Vega': 0.9,
      'Jazzhus Montmartre': 0.95,
      'Culture Box': 0.85,
      'Noma': 1.0
    };
    return premiumVenues[venueName] || 0.7;
  };

  const getPriceScore = (priceMin: number, priceMax: number, priceCap: number) => {
    if (!priceMin && !priceMax) return 1.0; // Free events
    const avgPrice = ((priceMin || 0) + (priceMax || 0)) / 2;
    if (avgPrice === 0) return 1.0; // Free
    if (avgPrice <= priceCap * 0.5) return 1.0; // Great value
    if (avgPrice <= priceCap) return 0.8; // Acceptable
    if (avgPrice <= priceCap * 1.5) return 0.5; // Expensive
    return 0.2; // Very expensive
  };

  const getTimingScore = (dateTime: string) => {
    const eventDate = new Date(dateTime);
    const now = new Date();
    const daysUntil = (eventDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);

    if (daysUntil < 1) return 1.0; // Today/tomorrow
    if (daysUntil <= 7) return 0.9; // This week
    if (daysUntil <= 30) return 0.8; // This month
    return 0.6; // Future
  };

  const getHistoryScore = (currentEvent: Event, likes: string[], dislikes: string[]) => {
    if (likes.includes(currentEvent.id)) return 1.0;
    if (dislikes.includes(currentEvent.id)) return 0.1;

    // Check venue history
    const likedVenues = likes.length > 0 ? 0.8 : 0.7; // Default neutral
    return likedVenues;
  };

  const getSourceScore = (source: string) => {
    const sourceCredibility = {
      'visit_copenhagen': 1.0,
      'jazzhus_montmartre': 0.95,
      'store_vega': 0.95,
      'lille_vega': 0.95,
      'culture_box': 0.9,
      'noma_popup': 1.0,
      'golden_days': 0.9,
      'craft_beer_cph': 0.85,
      'nordic_techkomm': 0.8
    };
    return sourceCredibility[source] || 0.7;
  };

  const getClickableUrl = () => {
    // If event has source_url, use it directly
    if (event.source_url) {
      return event.source_url;
    }

    // Enhanced venue-specific URLs (comprehensive Copenhagen venue database)
    const venueUrls: Record<string, string> = {
      // Major concert venues
      'store vega': 'https://vega.dk',
      'lille vega': 'https://vega.dk',
      'ideal bar': 'https://idealbar.dk',
      'loppen': 'https://www.facebook.com/Loppen1000',
      'rust': 'https://rust.dk',
      'amager bio': 'https://amagerbio.ax',
      'pumpehuset': 'https://pumpehuset.dk',
      'kb18': 'https://kb18.dk',

      // Electronic/techno venues
      'culture box': 'https://culture-box.com',
      'warehouse9': 'https://warehouse9.dk',
      'jolene bar': 'https://www.facebook.com/jolenebar',

      // Jazz venues
      'jazzhus montmartre': 'https://jazzhusmontmartre.dk',
      'la fontaine': 'https://lafontaine.dk',
      'tivoli concert hall': 'https://tivoli.dk/en/concerts-events',

      // Fine dining
      'noma': 'https://noma.dk',
      'geranium': 'https://geranium.dk',
      'alchemist': 'https://alchemist-copenhagen.dk',

      // Museums & cultural venues
      'arken museum of modern art': 'https://arken.dk',
      'danish photography museum': 'https://photography-museum.dk',
      'østerbro kulturhus': 'https://kk.dk/kulturtilbud',
      'danish architecture centre': 'https://dac.dk',

      // Event spaces
      'refshaleøen': 'https://refshaleoen.dk',
      'scandic sydhavnen': 'https://scandichotels.com/hotels/denmark/copenhagen/scandic-sydhavnen',
      'øksnehallen': 'https://oksnehallen.dk',
      'tap1': 'https://tap1.dk',

      // Neighborhoods (fallback)
      'vesterbro': 'https://www.visitcopenhagen.com/copenhagen/neighbourhoods/vesterbro',
      'nørrebro': 'https://www.visitcopenhagen.com/copenhagen/neighbourhoods/noerrebro',
      'østerbro': 'https://www.visitcopenhagen.com/copenhagen/neighbourhoods/oesterbro',
      'indre by': 'https://www.visitcopenhagen.com/copenhagen/neighbourhoods/inner-city',
      'christiania': 'https://www.visitcopenhagen.com/copenhagen/planning/christiania'
    };

    // Smart venue matching (normalize venue names)
    const normalizeVenue = (name: string) => {
      return name?.toLowerCase()
        .replace(/[øæå]/g, match => ({'ø': 'o', 'æ': 'ae', 'å': 'a'}[match] || match))
        .replace(/[^\w\s]/g, '')
        .trim();
    };

    const venueKey = normalizeVenue(event.venue_name);

    // Direct venue match
    if (venueKey && venueUrls[venueKey]) {
      return venueUrls[venueKey];
    }

    // Partial venue matching for compound names
    for (const [key, url] of Object.entries(venueUrls)) {
      if (venueKey?.includes(key) || key.includes(venueKey || '')) {
        return url;
      }
    }

    // Enhanced source-based fallbacks
    const source = event.source?.toLowerCase();
    const venueName = event.venue_name?.toLowerCase().replace(/\s+/g, '+');
    const eventTitle = event.title?.toLowerCase().replace(/\s+/g, '+');

    if (source?.includes('eventbrite')) {
      return `https://www.eventbrite.dk/d/copenhagen--copenhagen/events/?q=${eventTitle}`;
    }
    if (source?.includes('instagram')) {
      return `https://www.instagram.com/explore/tags/${venueName?.replace(/\+/g, '')}copenhagen`;
    }
    if (source?.includes('visit_copenhagen') || source?.includes('copenhagen')) {
      return `https://www.visitcopenhagen.com/explore/events`;
    }
    if (source?.includes('facebook')) {
      return `https://www.facebook.com/search/events/?q=${venueName}%20copenhagen`;
    }

    // Intelligent fallbacks based on event type
    if (event.title?.toLowerCase().includes('festival')) {
      return `https://www.visitcopenhagen.com/copenhagen/planning/festivals-events`;
    }
    if (event.title?.toLowerCase().includes('exhibition')) {
      return `https://www.visitcopenhagen.com/copenhagen/culture/museums-exhibitions`;
    }
    if (event.title?.toLowerCase().includes('jazz') || event.venue_name?.toLowerCase().includes('jazz')) {
      return `https://jazz.dk/venues/copenhagen`;
    }
    if (event.title?.toLowerCase().includes('techno') || event.title?.toLowerCase().includes('electronic')) {
      return `https://www.residentadvisor.net/events/dk/copenhagen`;
    }

    // Final fallback: Copenhagen events search
    return `https://www.visitcopenhagen.com/explore/events`;
  };

  const formatPrice = (min: number, max: number, currency: string) => {
    if (min === max) return `${min} ${currency}`;
    if (min === 0) return `Free - ${max} ${currency}`;
    return `${min} - ${max} ${currency}`;
  };

  const neighborhoodClass = neighborhoodClasses[event.venue_neighborhood as keyof typeof neighborhoodClasses] || '';

  const topic = topicConfig[event.topic] || topicConfig.music;

  return (
    <div className="glass-card hover-lift group animate-fade-in">
      {/* Event visual header with venue info */}
      <div className="relative h-32 mb-4 rounded-xl overflow-hidden bg-gradient-to-br from-indigo-500/80 via-purple-500/80 to-pink-500/80">
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />

        {/* Topic badge - top right */}
        <div className="absolute top-3 right-3">
          <Badge variant="outline" className={`${topic.className} border backdrop-blur-sm`}>
            <span className="mr-1">{topic.icon}</span>
            {topic.label}
          </Badge>
        </div>

        {/* Free badge - top left */}
        {event.is_free && (
          <div className="absolute top-3 left-3">
            <Badge className="bg-green-500/90 text-white border-0">
              Free
            </Badge>
          </div>
        )}

        <div className="absolute bottom-3 left-3 right-3">
          <div className="flex items-center gap-2 mb-1">
            <MapPin className="h-4 w-4 text-white/80" />
            <button
              onClick={() => window.open(getClickableUrl(), '_blank')}
              className="text-sm font-medium text-white hover:text-white/80 transition-colors underline-offset-2 hover:underline"
            >
              {event.venue_name}
            </button>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {(event.tags || []).slice(0, 2).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 text-xs font-medium bg-white/20 backdrop-blur-sm rounded-md text-white"
              >
                {tag}
              </span>
            ))}
            {(event.genres || []).slice(0, 2).map((genre) => (
              <span
                key={genre}
                className="px-2 py-0.5 text-xs font-medium bg-white/20 backdrop-blur-sm rounded-md text-white"
              >
                {genre}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Event details */}
      <div className="space-y-3">
        <div>
          <h3 className="text-lg font-heading font-semibold text-foreground group-hover:text-primary transition-colors">
            {event.title}
          </h3>
          <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
            {event.description}
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Calendar className="h-4 w-4 text-primary" />
            <span>{formatDate(event.date_time)}</span>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <MapPin className="h-4 w-4 text-accent" />
            <button
              onClick={() => window.open(getClickableUrl(), '_blank')}
              className="text-muted-foreground hover:text-foreground transition-colors underline-offset-2 hover:underline"
            >
              {event.venue_name}
            </button>
            <span className={`px-2 py-0.5 text-xs rounded-full border ${neighborhoodClass}`}>
              {event.venue_neighborhood}
            </span>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <DollarSign className="h-4 w-4 text-success" />
              <span>{formatPrice(event.price_min, event.price_max, event.currency)}</span>
            </div>
            
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Users className="h-3 w-3" />
              <span>{calculateMatchPercentage()}% match</span>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 border-destructive/30 text-destructive hover:bg-destructive/10"
            onClick={() => onDislike?.(event.id)}
          >
            <Heart className="h-4 w-4 mr-1" />
            Pass
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            className="border-primary/30 text-primary hover:bg-primary/10"
            onClick={() => onSave?.(event.id)}
          >
            <BookmarkPlus className="h-4 w-4" />
          </Button>
          
          <Button
            size="sm"
            className="flex-1 bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80"
            onClick={() => onLike?.(event.id)}
          >
            <Heart className="h-4 w-4 mr-1 fill-current" />
            Like
          </Button>
        </div>
      </div>
    </div>
  );
}