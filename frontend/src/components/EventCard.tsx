import { Heart, MapPin, Calendar, DollarSign, Users, BookmarkPlus } from 'lucide-react';
import { Button } from '@/components/ui/button';

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
  genres: string[] | null;
  popularity_score: number;
}

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

  const formatPrice = (min: number, max: number, currency: string) => {
    if (min === max) return `${min} ${currency}`;
    if (min === 0) return `Free - ${max} ${currency}`;
    return `${min} - ${max} ${currency}`;
  };

  const neighborhoodClass = neighborhoodClasses[event.venue_neighborhood as keyof typeof neighborhoodClasses] || '';

  return (
    <div className="glass-card hover-lift group animate-fade-in">
      {/* Event image placeholder with gradient */}
      <div className="relative h-48 mb-4 rounded-xl overflow-hidden bg-gradient-to-br from-primary/20 via-accent/20 to-neon-purple/20">
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
        <div className="absolute bottom-3 left-3 right-3">
          <div className="flex items-center gap-2 mb-1">
            {(event.genres || []).slice(0, 2).map((genre) => (
              <span
                key={genre}
                className="px-2 py-1 text-xs font-medium bg-black/50 backdrop-blur-sm rounded-md text-primary border border-primary/30"
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
            <span>{event.venue_name}</span>
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
              <span>{Math.round(event.popularity_score * 100)}% match</span>
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