// TypeScript types for Copenhagen Event Recommender API
// Auto-generated from FastAPI Pydantic models

export interface UserPreferences {
  preferred_genres: string[];
  preferred_artists: string[];
  preferred_venues: string[];
  price_range: [number, number];
  preferred_times: number[];
  preferred_days: number[];
  max_distance_km: number;
  location_lat?: number;
  location_lon?: number;
}

export interface EventResponse {
  event_id: string;
  title: string;
  description?: string;
  date_time: string;
  end_time?: string;
  venue_name: string;
  venue_lat: number;
  venue_lon: number;
  venue_address?: string;
  neighborhood?: string;
  price_min?: number;
  price_max?: number;
  currency: string;
  genres: string[];
  artists: string[];
  popularity_score: number;
  image_url?: string;
  source?: string;
  source_url?: string;
}

export interface RecommendationExplanation {
  overall_score: number;
  components: Record<string, number>;
  reasons: string[];
  model_confidence: number;
}

export interface RecommendedEvent {
  event: EventResponse;
  recommendation_score: number;
  rank: number;
  explanation?: RecommendationExplanation;
  distance_km?: number;
  predicted_attendance?: number;
}

export interface RecommendationResponse {
  user_id?: string;
  session_id: string;
  events: RecommendedEvent[];
  total_candidates: number;
  model_version: string;
  timestamp: string;
  processing_time_ms: number;
  filters_applied?: Record<string, any>;
  cold_start: boolean;
}

export interface RecommendationFilters {
  date_filter?: 'today' | 'tomorrow' | 'this_week' | 'this_weekend' | 'next_week' | 'this_month';
  min_price?: number;
  max_price?: number;
  genres?: string[];
  venues?: string[];
  neighborhoods?: string[];
  min_popularity?: number;
  include_past_events?: boolean;
}

export interface RecommendationRequest {
  user_id?: string;
  user_preferences: UserPreferences;
  location_lat?: number;
  location_lon?: number;
  num_recommendations?: number;
  filters?: RecommendationFilters;
  use_collaborative?: boolean;
  diversity_factor?: number;
  explain?: boolean;
}

export type InteractionType = 'like' | 'dislike' | 'going' | 'went' | 'save';

export interface InteractionRequest {
  user_id?: string;
  event_id: string;
  interaction_type: InteractionType;
  rating?: number;
  source?: string;
  position?: number;
}

export interface UserResponse {
  user_id: string;
  name?: string;
  preferences?: Record<string, any>;
  location_lat?: number;
  location_lon?: number;
  created_at: string;
  last_active?: string;
  interaction_count: number;
}

export interface SearchRequest {
  query: string;
  location_lat?: number;
  location_lon?: number;
  max_distance_km?: number;
  date_filter?: string;
  limit?: number;
}

export interface AnalyticsResponse {
  period_days: number;
  total_users: number;
  active_users: number;
  total_interactions: number;
  total_recommendations: number;
  total_events: number;
  events_by_genre: Record<string, number>;
  events_by_neighborhood: Record<string, number>;
  popular_venues: Array<Record<string, any>>;
  interaction_breakdown: Record<string, number>;
  avg_session_length: number;
  recommendation_ctr: number;
  model_accuracy?: number;
  avg_recommendation_score: number;
  cold_start_percentage: number;
  timestamp: string;
}

export interface ApiError {
  error: string;
  message: string;
  timestamp: string;
  request_id?: string;
  details?: Record<string, any>;
}

// API Response wrapper
export interface ApiResponse<T> {
  data?: T;
  error?: ApiError;
  loading: boolean;
}

// Common filter options
export const COPENHAGEN_NEIGHBORHOODS = [
  'Indre By', 'Vesterbro', 'Nørrebro', 'Østerbro', 'Frederiksberg',
  'Christiania', 'Islands Brygge', 'Refshaleøen', 'Papirøen', 'Sydhavnen'
] as const;

export const MUSIC_GENRES = [
  'techno', 'house', 'electronic', 'indie', 'rock', 'pop', 'jazz', 'blues',
  'hip hop', 'alternative', 'experimental', 'classical', 'folk', 'reggae',
  'punk', 'metal', 'disco', 'funk', 'soul', 'ambient'
] as const;

export const COPENHAGEN_VENUES = [
  'Vega', 'Rust', 'Culture Box', 'Loppen', 'KB18', 'Pumpehuset',
  'Amager Bio', 'ALICE', 'BETA2300', 'BRUS', 'Jazzhouse', 'Ideal Bar'
] as const;

export type CopenhagenNeighborhood = typeof COPENHAGEN_NEIGHBORHOODS[number];
export type MusicGenre = typeof MUSIC_GENRES[number];
export type CopenhagenVenue = typeof COPENHAGEN_VENUES[number];