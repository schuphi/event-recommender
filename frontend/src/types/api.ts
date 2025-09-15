// API types for Copenhagen Event Recommender

export interface EventResponse {
  id: string;
  title: string;
  description: string;
  date_time: string;
  end_date_time?: string;
  price_min?: number;
  price_max?: number;
  currency: string;
  venue_name: string;
  venue_address: string;
  venue_neighborhood: string;
  genres?: string[];
  source: string;
  source_url?: string;
  image_url?: string;
  popularity_score: number;
}

export interface UserPreferences {
  preferred_genres?: string[];
  price_cap?: number;
  radius_km?: number;
  preferred_neighborhoods?: string[];
  preferred_times?: string[];
}

export interface RecommendationRequest {
  user_id?: string;
  user_preferences?: UserPreferences;
  num_recommendations?: number;
  location_lat?: number;
  location_lon?: number;
}

export interface RecommendationResponse {
  user_id: string;
  recommendations: EventResponse[];
  total: number;
  method: string;
  timestamp: string;
}

export interface InteractionRequest {
  user_id?: string;
  event_id: string;
  interaction_type: 'like' | 'dislike' | 'going' | 'interested' | 'share' | 'view';
  source?: string;
  context?: Record<string, any>;
}

export interface UserResponse {
  user_id: string;
  name?: string;
  preferences?: UserPreferences;
  location_lat?: number;
  location_lon?: number;
  created_at: string;
}

export interface SearchRequest {
  query: string;
  user_id?: string;
  filters?: {
    genres?: string[];
    neighborhoods?: string[];
    price_range?: [number, number];
    date_range?: [string, string];
  };
  limit?: number;
}

export interface AnalyticsResponse {
  total_events: number;
  total_users: number;
  total_interactions: number;
  popular_genres: Array<{ genre: string; count: number }>;
  popular_venues: Array<{ venue: string; count: number }>;
  interaction_trends: Array<{ date: string; count: number }>;
}