import { useState } from 'react';
import { ChevronDown, ChevronUp, Filter, MapPin, Calendar, Music, DollarSign } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';

interface FilterState {
  genres: string[];
  neighborhoods: string[];
  priceRange: [number, number];
  dateFilter: string;
}

interface FilterPanelProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  isOpen: boolean;
  onToggle: () => void;
}

const GENRES = [
  'techno', 'house', 'electronic', 'indie', 'rock', 'jazz', 'folk', 'pop', 'hip-hop', 'ambient'
];

const NEIGHBORHOODS = [
  'Vesterbro', 'Nørrebro', 'Indre By', 'Christiania', 'Østerbro', 'Frederiksberg', 'Amager'
];

const DATE_FILTERS = [
  { label: 'Tonight', value: 'tonight' },
  { label: 'Tomorrow', value: 'tomorrow' },
  { label: 'This Weekend', value: 'weekend' },
  { label: 'Next Week', value: 'next_week' },
  { label: 'This Month', value: 'month' },
];

const neighborhoodClasses = {
  Vesterbro: 'border-vesterbro text-vesterbro hover:bg-vesterbro/10',
  Nørrebro: 'border-norrebro text-norrebro hover:bg-norrebro/10',
  'Indre By': 'border-indre-by text-indre-by hover:bg-indre-by/10',
  Christiania: 'border-christiania text-christiania hover:bg-christiania/10',
  Østerbro: 'border-osterbro text-osterbro hover:bg-osterbro/10',
  Frederiksberg: 'border-primary text-primary hover:bg-primary/10',
  Amager: 'border-accent text-accent hover:bg-accent/10',
};

export function FilterPanel({ filters, onFiltersChange, isOpen, onToggle }: FilterPanelProps) {
  const toggleGenre = (genre: string) => {
    const newGenres = filters.genres.includes(genre)
      ? filters.genres.filter(g => g !== genre)
      : [...filters.genres, genre];
    onFiltersChange({ ...filters, genres: newGenres });
  };

  const toggleNeighborhood = (neighborhood: string) => {
    const newNeighborhoods = filters.neighborhoods.includes(neighborhood)
      ? filters.neighborhoods.filter(n => n !== neighborhood)
      : [...filters.neighborhoods, neighborhood];
    onFiltersChange({ ...filters, neighborhoods: newNeighborhoods });
  };

  const updatePriceRange = (range: number[]) => {
    onFiltersChange({ ...filters, priceRange: [range[0], range[1]] });
  };

  const setDateFilter = (dateFilter: string) => {
    onFiltersChange({ ...filters, dateFilter });
  };

  const clearFilters = () => {
    onFiltersChange({
      genres: [],
      neighborhoods: [],
      priceRange: [0, 1000],
      dateFilter: '',
    });
  };

  const activeFiltersCount = filters.genres.length + filters.neighborhoods.length + 
    (filters.dateFilter ? 1 : 0) + (filters.priceRange[0] > 0 || filters.priceRange[1] < 1000 ? 1 : 0);

  return (
    <Collapsible open={isOpen} onOpenChange={onToggle}>
      <CollapsibleTrigger asChild>
        <Button
          variant="outline"
          className="w-full justify-between glass border-primary/30 hover:bg-primary/10"
        >
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4" />
            <span>Filters</span>
            {activeFiltersCount > 0 && (
              <Badge variant="outline" className="text-xs border-primary text-primary">
                {activeFiltersCount}
              </Badge>
            )}
          </div>
          {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-6 pt-4 animate-slide-up">
        <div className="glass-card">
          {/* Date Filters */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-primary" />
              <h3 className="font-heading font-medium">When</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {DATE_FILTERS.map((date) => (
                <Button
                  key={date.value}
                  variant={filters.dateFilter === date.value ? "default" : "outline"}
                  size="sm"
                  className={filters.dateFilter === date.value 
                    ? "bg-primary text-primary-foreground" 
                    : "border-primary/30 hover:bg-primary/10"
                  }
                  onClick={() => setDateFilter(filters.dateFilter === date.value ? '' : date.value)}
                >
                  {date.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Price Range */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <DollarSign className="h-4 w-4 text-success" />
              <h3 className="font-heading font-medium">Price Range</h3>
              <span className="text-sm text-muted-foreground">
                {filters.priceRange[0]} - {filters.priceRange[1]} DKK
              </span>
            </div>
            <Slider
              value={filters.priceRange}
              onValueChange={updatePriceRange}
              max={1000}
              min={0}
              step={25}
              className="w-full"
            />
          </div>

          {/* Genres */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Music className="h-4 w-4 text-accent" />
              <h3 className="font-heading font-medium">Genres</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {GENRES.map((genre) => (
                <Button
                  key={genre}
                  variant={filters.genres.includes(genre) ? "default" : "outline"}
                  size="sm"
                  className={filters.genres.includes(genre) 
                    ? "bg-accent text-accent-foreground" 
                    : "border-accent/30 hover:bg-accent/10"
                  }
                  onClick={() => toggleGenre(genre)}
                >
                  {genre}
                </Button>
              ))}
            </div>
          </div>

          {/* Neighborhoods */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <MapPin className="h-4 w-4 text-neon-green" />
              <h3 className="font-heading font-medium">Copenhagen Areas</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {NEIGHBORHOODS.map((neighborhood) => (
                <Button
                  key={neighborhood}
                  variant="outline"
                  size="sm"
                  className={`${
                    neighborhoodClasses[neighborhood as keyof typeof neighborhoodClasses] || 
                    'border-primary/30 hover:bg-primary/10'
                  } ${filters.neighborhoods.includes(neighborhood) ? 'bg-current/10' : ''}`}
                  onClick={() => toggleNeighborhood(neighborhood)}
                >
                  {neighborhood}
                </Button>
              ))}
            </div>
          </div>

          {/* Clear Filters */}
          {activeFiltersCount > 0 && (
            <Button
              variant="ghost"
              onClick={clearFilters}
              className="w-full text-muted-foreground hover:text-foreground"
            >
              Clear all filters
            </Button>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}