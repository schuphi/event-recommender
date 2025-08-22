// Example EventCard component for Lovable customization reference
'use client';

import { useState } from 'react';
import Image from 'next/image';
import { Heart, X, Calendar, MapPin, Users, ExternalLink } from 'lucide-react';
import { RecommendedEvent, InteractionType } from '@/types/api';
import { apiClient } from '@/lib/api';

interface EventCardProps {
  event: RecommendedEvent;
  onInteraction?: (eventId: string, type: InteractionType) => void;
  showExplanation?: boolean;
  position?: number;
}

export default function EventCard({ 
  event, 
  onInteraction, 
  showExplanation = false,
  position 
}: EventCardProps) {
  const [isInteracting, setIsInteracting] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const handleInteraction = async (type: InteractionType) => {
    if (isInteracting) return;
    
    setIsInteracting(true);
    
    try {
      await apiClient.recordInteraction({
        event_id: event.event.event_id,
        interaction_type: type,
        source: 'feed',
        position
      });
      
      onInteraction?.(event.event.event_id, type);
    } catch (error) {
      console.error('Failed to record interaction:', error);
    } finally {
      setIsInteracting(false);
    }
  };

  const formatPrice = (min?: number, max?: number) => {
    if (!min && !max) return 'Free';
    if (min === max) return `${min} DKK`;
    return `${min || 0}-${max || 0} DKK`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const isTomorrow = date.toDateString() === new Date(now.getTime() + 86400000).toDateString();
    
    if (isToday) return `Today ${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
    if (isTomorrow) return `Tomorrow ${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
    
    return date.toLocaleDateString('en-US', { 
      weekday: 'short',
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getNeighborhoodColor = (neighborhood?: string) => {
    const colors: Record<string, string> = {
      'Vesterbro': 'border-l-neon-pink',
      'Nørrebro': 'border-l-neon-green', 
      'Indre By': 'border-l-neon-blue',
      'Christiania': 'border-l-neon-purple',
      'Østerbro': 'border-l-neon-yellow'
    };
    return colors[neighborhood || ''] || 'border-l-primary-500';
  };

  return (
    <div className={`
      group relative bg-dark-800/90 backdrop-blur-sm rounded-2xl border border-dark-600
      hover:border-primary-500/50 transition-all duration-300 overflow-hidden
      ${getNeighborhoodColor(event.event.neighborhood)} border-l-4
      hover:shadow-lg hover:shadow-primary-500/20 hover:-translate-y-1
    `}>
      {/* Event Image */}
      <div className="relative h-48 overflow-hidden">
        <Image
          src={event.event.image_url || '/placeholder-event.jpg'}
          alt={event.event.title}
          fill
          className="object-cover transition-transform duration-300 group-hover:scale-105"
        />
        
        {/* Overlay with interaction buttons */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end">
            <div className="flex space-x-2">
              <button
                onClick={() => handleInteraction('like')}
                disabled={isInteracting}
                className="p-2 bg-neon-green/20 hover:bg-neon-green/40 border border-neon-green/50 rounded-full transition-colors duration-200 backdrop-blur-sm"
              >
                <Heart className="w-5 h-5 text-neon-green" />
              </button>
              
              <button
                onClick={() => handleInteraction('dislike')}
                disabled={isInteracting}
                className="p-2 bg-red-500/20 hover:bg-red-500/40 border border-red-500/50 rounded-full transition-colors duration-200 backdrop-blur-sm"
              >
                <X className="w-5 h-5 text-red-400" />
              </button>
              
              <button
                onClick={() => handleInteraction('going')}
                disabled={isInteracting}
                className="p-2 bg-neon-blue/20 hover:bg-neon-blue/40 border border-neon-blue/50 rounded-full transition-colors duration-200 backdrop-blur-sm"
              >
                <Calendar className="w-5 h-5 text-neon-blue" />
              </button>
            </div>
            
            {event.distance_km && (
              <span className="text-xs text-gray-300 bg-black/50 px-2 py-1 rounded-full backdrop-blur-sm">
                {event.distance_km.toFixed(1)}km
              </span>
            )}
          </div>
        </div>

        {/* Recommendation score badge */}
        <div className="absolute top-4 right-4">
          <div className="bg-primary-500/90 text-white text-xs font-medium px-2 py-1 rounded-full backdrop-blur-sm">
            {Math.round(event.recommendation_score * 100)}% match
          </div>
        </div>

        {/* Viral indicator */}
        {event.event.source?.includes('viral') && (
          <div className="absolute top-4 left-4">
            <div className="bg-neon-pink/90 text-white text-xs font-medium px-2 py-1 rounded-full backdrop-blur-sm animate-pulse-neon">
              🔥 Trending
            </div>
          </div>
        )}
      </div>

      {/* Event Details */}
      <div className="p-4 space-y-3">
        <div>
          <h3 className="font-display font-semibold text-lg text-white line-clamp-2 group-hover:text-primary-400 transition-colors">
            {event.event.title}
          </h3>
          
          <div className="flex items-center space-x-4 mt-2 text-sm text-gray-400">
            <div className="flex items-center space-x-1">
              <MapPin className="w-4 h-4" />
              <span>{event.event.venue_name}</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Calendar className="w-4 h-4" />
              <span>{formatDate(event.event.date_time)}</span>
            </div>
          </div>
        </div>

        {/* Genres */}
        {event.event.genres.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {event.event.genres.slice(0, 3).map((genre) => (
              <span
                key={genre}
                className="text-xs bg-dark-700 text-gray-300 px-2 py-1 rounded-full"
              >
                {genre}
              </span>
            ))}
            {event.event.genres.length > 3 && (
              <span className="text-xs text-gray-500">
                +{event.event.genres.length - 3} more
              </span>
            )}
          </div>
        )}

        {/* Price and attendance */}
        <div className="flex justify-between items-center">
          <div className="text-sm">
            <span className="text-primary-400 font-medium">
              {formatPrice(event.event.price_min, event.event.price_max)}
            </span>
          </div>
          
          {event.predicted_attendance && (
            <div className="flex items-center space-x-1 text-xs text-gray-400">
              <Users className="w-3 h-3" />
              <span>{event.predicted_attendance} expected</span>
            </div>
          )}
        </div>

        {/* Explanation (if enabled) */}
        {showExplanation && event.explanation && (
          <div className="mt-3 p-3 bg-dark-700/50 rounded-lg border border-dark-600">
            <h4 className="text-xs font-medium text-gray-300 mb-2">Why recommended:</h4>
            <div className="space-y-1">
              {event.explanation.reasons.slice(0, 2).map((reason, index) => (
                <div key={index} className="text-xs text-gray-400 flex items-start space-x-1">
                  <span className="text-primary-400 mt-1">•</span>
                  <span>{reason}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex space-x-2 pt-2">
          <button
            onClick={() => setShowDetails(true)}
            className="flex-1 bg-primary-600 hover:bg-primary-700 text-white text-sm font-medium py-2 px-4 rounded-lg transition-colors duration-200"
          >
            View Details
          </button>
          
          {event.event.source_url && (
            <button
              onClick={() => window.open(event.event.source_url, '_blank')}
              className="p-2 bg-dark-700 hover:bg-dark-600 text-gray-400 rounded-lg transition-colors duration-200"
            >
              <ExternalLink className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}