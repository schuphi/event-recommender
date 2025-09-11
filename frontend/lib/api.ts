// API client for Copenhagen Event Recommender
import { 
  RecommendationRequest, 
  RecommendationResponse, 
  EventResponse, 
  InteractionRequest,
  UserResponse,
  SearchRequest,
  AnalyticsResponse,
  UserPreferences
} from '@/types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://event-recommender-production.up.railway.app';

class ApiClient {
  private baseURL: string;
  private userId: string | null = null;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
    
    // Initialize user ID from localStorage
    if (typeof window !== 'undefined') {
      this.userId = localStorage.getItem('user_id') || this.generateUserId();
      localStorage.setItem('user_id', this.userId);
    }
  }

  private generateUserId(): string {
    return 'user_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    // Add user ID header for anonymous users
    if (this.userId && !headers.Authorization) {
      headers['X-User-ID'] = this.userId;
    }

    const config: RequestInit = {
      ...options,
      headers,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // User Management
  async registerUser(userData: {
    name?: string;
    preferences?: UserPreferences;
    location_lat?: number;
    location_lon?: number;
  }): Promise<UserResponse> {
    const response = await this.request<UserResponse>('/users/register', {
      method: 'POST',
      body: JSON.stringify({
        user_id: this.userId,
        ...userData
      })
    });
    return response;
  }

  async getUser(userId?: string): Promise<UserResponse> {
    const id = userId || this.userId;
    return this.request<UserResponse>(`/users/${id}`);
  }

  async updateUserPreferences(preferences: UserPreferences): Promise<{ status: string }> {
    return this.request(`/users/${this.userId}/preferences`, {
      method: 'PUT',
      body: JSON.stringify(preferences)
    });
  }

  // Event Discovery
  async getEvents(params: {
    limit?: number;
    offset?: number;
    neighborhood?: string;
    date_from?: string;
    date_to?: string;
    min_price?: number;
    max_price?: number;
    genres?: string;
  } = {}): Promise<EventResponse[]> {
    const queryParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, value.toString());
      }
    });

    const endpoint = `/events${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    return this.request<EventResponse[]>(endpoint);
  }

  async getEvent(eventId: string): Promise<EventResponse> {
    return this.request<EventResponse>(`/events/${eventId}`);
  }

  // Recommendations
  async getRecommendations(request: RecommendationRequest): Promise<RecommendationResponse> {
    return this.request<RecommendationResponse>('/recommend', {
      method: 'POST',
      body: JSON.stringify({
        ...request,
        user_id: request.user_id || this.userId
      })
    });
  }

  async getSimilarEvents(eventId: string, numRecommendations = 10): Promise<{
    event_id: string;
    similar_events: any[];
    count: number;
  }> {
    return this.request(`/recommend/similar?event_id=${eventId}&num_recommendations=${numRecommendations}`, {
      method: 'POST'
    });
  }

  // Search
  async searchEvents(request: SearchRequest): Promise<{
    query: string;
    events: EventResponse[];
    count: number;
  }> {
    return this.request('/search', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  // Interactions
  async recordInteraction(interaction: InteractionRequest): Promise<{ status: string }> {
    return this.request('/interactions', {
      method: 'POST',
      body: JSON.stringify({
        ...interaction,
        user_id: interaction.user_id || this.userId
      })
    });
  }

  async getUserInteractions(userId?: string, params: {
    limit?: number;
    interaction_type?: string;
  } = {}): Promise<{
    user_id: string;
    interactions: any[];
    count: number;
  }> {
    const id = userId || this.userId;
    const queryParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, value.toString());
      }
    });

    const endpoint = `/users/${id}/interactions${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    return this.request(endpoint);
  }

  // Analytics (for admin/dashboard)
  async getAnalytics(days = 7): Promise<AnalyticsResponse> {
    return this.request<AnalyticsResponse>(`/analytics/dashboard?days=${days}`);
  }

  async getEventAnalytics(eventId: string, days = 30): Promise<any> {
    return this.request(`/analytics/events/${eventId}?days=${days}`);
  }

  // Health Check
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    version: string;
    services?: Record<string, string>;
  }> {
    return this.request('/health');
  }

  // Utility methods
  setUserId(userId: string) {
    this.userId = userId;
    if (typeof window !== 'undefined') {
      localStorage.setItem('user_id', userId);
    }
  }

  getUserId(): string | null {
    return this.userId;
  }

  clearUser() {
    this.userId = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('user_id');
    }
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
export default apiClient;